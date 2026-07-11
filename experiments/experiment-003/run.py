#!/usr/bin/env python3
"""Run the frozen Experiment 003 observation-model comparison.

The script is experiment-local and does not modify production fitting code.

Usage:
    .venv/bin/python experiments/experiment-003/run.py --stage discovery
    .venv/bin/python experiments/experiment-003/run.py --stage all
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy import special
from scipy.optimize import minimize
from scipy.stats import poisson, qmc

from src import config
from src.losscone.chi2 import compute_lillis_chi2_batch
from src.losscone.cpu import LossConeFitter
from src.losscone.er_data import ERData
from src.losscone.model import synth_losscone, synth_losscone_batch
from src.losscone.pitch_angle import PitchAngle
from src.losscone.types import FitChunkData, FitMethod

HERE = Path(__file__).resolve().parent
EXPERIMENT_002 = HERE.parent / "experiment-002"
sys.path.insert(0, str(EXPERIMENT_002))
from compression import count_interval  # noqa: E402
from raw_telemetry import read_frames, reconstruct_blocks  # noqa: E402

METHOD_INTERVAL = "raw_interval_nb"
METHOD_CONDITIONAL = "conditional_shape_t4"
METHOD_LILLIS = "smooth_lillis_pairwise"
METHODS = (METHOD_INTERVAL, METHOD_CONDITIONAL, METHOD_LILLIS)

FIRST_RAW_SPEC = 630
U_SPACECRAFT = 11.0
BACKGROUND_RATE_CPS = 10.0
STUDENT_DF = 4.0
MODEL_FLOOR = 0.05
PARAMETER_BOUNDS = np.array([[-2000.0, 20.0], [0.3, 1.1], [0.0, 5.0]])
LHS_SAMPLES = 384
PRIMARY_TARGET_RELATIVE_V = -171.0
PRIMARY_TOLERANCE_V = 25.0


@dataclass(frozen=True)
class PanelRow:
    panel: str
    panel_index: int
    spec_no: int
    utc: str
    raw_block_index: int
    is_published_primary: bool


@dataclass(frozen=True)
class SpectrumData:
    panel: PanelRow
    ratio: FitChunkData
    ratio2: FitChunkData
    mask: np.ndarray
    response_s: np.ndarray
    count_lower: np.ndarray
    count_upper: np.ndarray
    count_code: np.ndarray


@dataclass(frozen=True)
class FitResult:
    panel: str
    spec_no: int
    utc: str
    method: str
    seed: int
    u_surface_v: float
    u_relative_v: float
    bs_over_bm: float
    beam_amp: float
    objective_per_bin: float
    median_abs_residual: float
    reflected_residual_rms: float
    incident_residual_rms: float
    n_valid: int
    boundary_hit: bool
    optimizer_success: bool
    elapsed_s: float


@dataclass
class FitOutput:
    result: FitResult
    residuals: np.ndarray
    expected: np.ndarray | None
    model: np.ndarray


def _read_panel(path: Path) -> list[PanelRow]:
    with path.open(encoding="ascii", newline="") as source:
        return [
            PanelRow(
                panel=row["panel"],
                panel_index=int(row["panel_index"]),
                spec_no=int(row["spec_no"]),
                utc=row["utc"],
                raw_block_index=int(row["raw_block_index"]),
                is_published_primary=row["is_published_primary"] == "True",
            )
            for row in csv.DictReader(source)
        ]


def _chunk_index(er_data: ERData, spec_no: int) -> int:
    specs = er_data.data[config.SPEC_NO_COLUMN].unique()
    match = np.flatnonzero(specs == spec_no)
    if match.size != 1:
        raise ValueError(f"Expected one chunk for spec {spec_no}, found {match.size}")
    return int(match[0])


def _load_response(path: Path) -> np.ndarray:
    response = np.full((15, 88), np.nan, dtype=np.float64)
    with path.open(encoding="ascii", newline="") as source:
        for row in csv.DictReader(source):
            energy = int(row["energy_index"])
            direction = int(row["direction_index"])
            response[energy, direction] = float(row["effective_response_s"])
    for energy in range(response.shape[0]):
        missing = ~np.isfinite(response[energy]) | (response[energy] <= 0.0)
        response[energy, missing] = np.nanmedian(response[energy, ~missing])
    if not np.all(np.isfinite(response) & (response > 0.0)):
        raise ValueError("Response table contains unresolved invalid bins")
    return response


def _decode_codes(codes: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    code = np.frombuffer(codes, dtype=np.uint8).reshape(15, 88).copy()
    lower = np.empty(code.shape, dtype=np.float64)
    upper = np.empty(code.shape, dtype=np.float64)
    for index in np.ndindex(code.shape):
        interval = count_interval(int(code[index]))
        lower[index] = interval.lower
        upper[index] = np.inf if interval.upper is None else interval.upper
    return code, lower, upper


class ExperimentContext:
    def __init__(
        self,
        data_path: Path,
        telemetry_path: Path,
        response_path: Path,
        panel_path: Path,
    ) -> None:
        self.panel = _read_panel(panel_path)
        self.er_data = ERData(str(data_path))
        self.pitch_angle = PitchAngle(self.er_data)
        spacecraft = np.full(len(self.er_data.data), U_SPACECRAFT, dtype=np.float64)
        self.ratio_fitter = LossConeFitter(
            self.er_data,
            pitch_angle=self.pitch_angle,
            spacecraft_potential=spacecraft,
            normalization_mode="ratio",
            incident_flux_stat="mean",
            fit_method="lillis",
        )
        self.ratio2_fitter = LossConeFitter(
            self.er_data,
            pitch_angle=self.pitch_angle,
            spacecraft_potential=spacecraft,
            normalization_mode="ratio2",
            incident_flux_stat="mean",
            fit_method="lillis",
        )
        self.response = _load_response(response_path)
        self.raw_blocks = reconstruct_blocks(read_frames(telemetry_path), 3)

    def spectrum(self, panel: PanelRow) -> SpectrumData:
        chunk = _chunk_index(self.er_data, panel.spec_no)
        ratio = self.ratio_fitter._prepare_chunk_data(chunk)
        ratio2 = self.ratio2_fitter._prepare_chunk_data(chunk)
        if ratio is None or ratio2 is None:
            raise RuntimeError(f"Could not prepare spectrum {panel.spec_no}")
        expected_block = panel.spec_no - FIRST_RAW_SPEC
        if panel.raw_block_index != expected_block:
            raise ValueError(f"Raw block mismatch for spectrum {panel.spec_no}")
        code, lower, upper = _decode_codes(self.raw_blocks[expected_block].codes)
        return SpectrumData(
            panel=panel,
            ratio=ratio,
            ratio2=ratio2,
            mask=ratio.combined_mask(FitMethod.LILLIS),
            response_s=self.response,
            count_lower=lower,
            count_upper=upper,
            count_code=code,
        )


def _lhs(seed: int, n_samples: int = LHS_SAMPLES) -> np.ndarray:
    sample = qmc.LatinHypercube(d=3, seed=seed).random(n_samples)
    return qmc.scale(sample, PARAMETER_BOUNDS[:, 0], PARAMETER_BOUNDS[:, 1])


def _models(data: FitChunkData, params: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    params = np.atleast_2d(params)
    return synth_losscone_batch(
        energy_grid=data.energies,
        pitch_grid=data.pitches,
        U_surface=params[:, 0],
        U_spacecraft=data.spacecraft_slice,
        bs_over_bm=params[:, 1],
        beam_width_eV=np.full(len(params), config.LOSS_CONE_BEAM_WIDTH_EV),
        beam_amp=params[:, 2],
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=np.full(len(params), MODEL_FLOOR),
        boundary_tau_deg=config.LOSS_CONE_BOUNDARY_TAU_DEG,
        return_mask=True,
    )


def _single_model(data: FitChunkData, params: np.ndarray) -> np.ndarray:
    return synth_losscone(
        energy_grid=data.energies,
        pitch_grid=data.pitches,
        U_surface=float(params[0]),
        U_spacecraft=data.spacecraft_slice,
        bs_over_bm=float(params[1]),
        beam_width_eV=config.LOSS_CONE_BEAM_WIDTH_EV,
        beam_amp=float(params[2]),
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=MODEL_FLOOR,
        boundary_tau_deg=config.LOSS_CONE_BOUNDARY_TAU_DEG,
    )


def _profile_count_mean(
    models: np.ndarray,
    data: SpectrumData,
) -> tuple[np.ndarray, np.ndarray]:
    response_factor = 0.02 * data.ratio.energies[:, None] * data.response_s
    background = BACKGROUND_RATE_CPS * data.response_s
    mask = data.mask
    signal = np.maximum(data.count_lower - background, 0.0)
    numerator = np.sum(np.where(mask, signal, 0.0), axis=1)
    denominator = np.sum(
        np.where(mask[None, :, :], models * response_factor[None, :, :], 0.0),
        axis=2,
    )
    amplitude = np.divide(
        numerator[None, :],
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0.0,
    )
    mu = (
        background[None, :, :]
        + amplitude[:, :, None] * models * response_factor[None, :, :]
    )
    return np.maximum(mu, np.finfo(float).tiny), amplitude


def _interval_log_probability(
    lower: np.ndarray,
    upper: np.ndarray,
    mu: np.ndarray,
    alpha: float,
) -> np.ndarray:
    lower3 = lower[None, :, :]
    upper3 = upper[None, :, :]
    if alpha <= 0.0:
        log_upper = poisson.logcdf(upper3, mu)
        log_lower = poisson.logcdf(lower3 - 1.0, mu)
        log_sf_lower = poisson.logsf(lower3 - 1.0, mu)
        log_sf_upper = poisson.logsf(upper3, mu)
        log_open = poisson.logsf(lower3 - 1.0, mu)
        log_exact = poisson.logpmf(lower3, mu)
    else:
        size = 1.0 / alpha
        probability = size / (size + mu)
        finite_upper = np.where(np.isfinite(upper3), upper3, 0.0)

        def log_beta_cdf(k: np.ndarray) -> np.ndarray:
            value = special.betainc(size, np.maximum(k, -1.0) + 1.0, probability)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(k < 0.0, -np.inf, np.log(value))

        def log_beta_sf(k: np.ndarray) -> np.ndarray:
            value = special.betaincc(size, np.maximum(k, -1.0) + 1.0, probability)
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.where(k < 0.0, 0.0, np.log(value))

        log_upper = log_beta_cdf(finite_upper)
        log_lower = log_beta_cdf(lower3 - 1.0)
        log_sf_lower = log_beta_sf(lower3 - 1.0)
        log_sf_upper = log_beta_sf(finite_upper)
        log_open = log_sf_lower
        log_exact = (
            special.gammaln(lower3 + size)
            - special.gammaln(size)
            - special.gammaln(lower3 + 1.0)
            + size * np.log(probability)
            + lower3 * np.log1p(-probability)
        )

    def log_difference(log_a: np.ndarray, log_b: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            delta = np.minimum(log_b - log_a, 0.0)
            return log_a + np.log(-np.expm1(delta))

    cdf_logp = log_difference(log_upper, log_lower)
    sf_logp = log_difference(log_sf_lower, log_sf_upper)
    finite_logp = np.where(np.isfinite(cdf_logp), cdf_logp, sf_logp)
    finite_logp = np.where(
        np.isfinite(sf_logp) & (sf_logp > finite_logp), sf_logp, finite_logp
    )
    finite_logp = np.where(lower3 == upper3, log_exact, finite_logp)
    return np.where(np.isinf(upper3), log_open, finite_logp)


def _interval_objective(
    models: np.ndarray,
    model_masks: np.ndarray,
    data: SpectrumData,
    alpha: float,
) -> np.ndarray:
    mu, _ = _profile_count_mean(models, data)
    valid = data.mask[None, :, :] & model_masks
    logp = _interval_log_probability(data.count_lower, data.count_upper, mu, alpha)
    logp = np.where(valid, logp, 0.0)
    n_valid = np.maximum(np.count_nonzero(valid, axis=(1, 2)), 1)
    objective = -np.sum(logp, axis=(1, 2)) / n_valid
    objective[~np.isfinite(objective)] = 1e30
    return objective


def _center_rows(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask, values, 0.0)
    count = np.maximum(np.count_nonzero(mask, axis=-1, keepdims=True), 1)
    mean = np.sum(masked, axis=-1, keepdims=True) / count
    return values - mean


def _conditional_residuals(
    models: np.ndarray, model_masks: np.ndarray, data: SpectrumData
) -> tuple[np.ndarray, np.ndarray]:
    valid = data.mask[None, :, :] & model_masks
    observed = np.log(np.maximum(data.ratio.raw_flux, np.finfo(float).tiny))
    model_log = np.log(np.maximum(models, np.finfo(float).tiny))
    observed_centered = _center_rows(observed[None, :, :], valid)
    model_centered = _center_rows(model_log, valid)
    return observed_centered - model_centered, valid


def _conditional_objective(
    models: np.ndarray,
    model_masks: np.ndarray,
    data: SpectrumData,
    sigma: float,
) -> np.ndarray:
    residual, valid = _conditional_residuals(models, model_masks, data)
    scaled2 = np.square(residual / sigma)
    loss = 0.5 * (STUDENT_DF + 1.0) * np.log1p(scaled2 / STUDENT_DF)
    n_valid = np.maximum(np.count_nonzero(valid, axis=(1, 2)), 1)
    objective = np.sum(np.where(valid, loss, 0.0), axis=(1, 2)) / n_valid
    objective[~np.isfinite(objective)] = 1e30
    return objective


def _lillis_objective(
    models: np.ndarray, model_masks: np.ndarray, data: SpectrumData
) -> np.ndarray:
    valid = data.mask[None, :, :] & model_masks
    return compute_lillis_chi2_batch(
        data.ratio2.norm2d,
        models,
        data.ratio2.raw_flux,
        data.ratio2.pitches,
        model_mask=valid,
    )


def _optimize(
    data: SpectrumData,
    method: str,
    alpha: float,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, float, bool]:
    physical_data = data.ratio2 if method == METHOD_LILLIS else data.ratio
    samples = _lhs(seed)
    models, model_masks = _models(physical_data, samples)

    if method == METHOD_INTERVAL:
        values = _interval_objective(models, model_masks, data, alpha)
    elif method == METHOD_CONDITIONAL:
        values = _conditional_objective(models, model_masks, data, sigma)
    elif method == METHOD_LILLIS:
        values = _lillis_objective(models, model_masks, data)
    else:
        raise ValueError(method)

    best = samples[int(np.nanargmin(values))]

    def scalar(params: np.ndarray) -> float:
        if not np.all(np.isfinite(params)):
            return 1e30
        model, model_mask = _models(physical_data, np.asarray(params)[None, :])
        if method == METHOD_INTERVAL:
            value = _interval_objective(model, model_mask, data, alpha)[0]
        elif method == METHOD_CONDITIONAL:
            value = _conditional_objective(model, model_mask, data, sigma)[0]
        else:
            value = _lillis_objective(model, model_mask, data)[0]
        return float(value) if np.isfinite(value) else 1e30

    local = minimize(
        scalar,
        best,
        method="Powell",
        bounds=[tuple(bound) for bound in PARAMETER_BOUNDS],
        options={"maxiter": 180, "xtol": 1e-3, "ftol": 1e-6},
    )
    if np.isfinite(local.fun) and local.fun < float(np.nanmin(values)):
        return np.asarray(local.x), float(local.fun), bool(local.success)
    return best, float(np.nanmin(values)), False


def fit_spectrum(
    data: SpectrumData,
    method: str,
    *,
    alpha: float,
    sigma: float,
    seed: int = 42,
) -> FitOutput:
    started = time.monotonic()
    params, objective, success = _optimize(data, method, alpha, sigma, seed)
    physical_data = data.ratio2 if method == METHOD_LILLIS else data.ratio
    model = _single_model(physical_data, params)
    mask = data.mask
    expected: np.ndarray | None = None

    if method == METHOD_INTERVAL:
        expected = _profile_count_mean(model[None, :, :], data)[0][0]
        variance = expected + alpha * np.square(expected)
        residual = (data.count_lower - expected) / np.sqrt(variance)
    elif method == METHOD_CONDITIONAL:
        residual3, valid3 = _conditional_residuals(
            model[None, :, :], np.ones((1, *model.shape), dtype=bool), data
        )
        residual = residual3[0] / sigma
        mask = valid3[0]
    else:
        residual = data.ratio2.norm2d - model

    reflected = mask & (physical_data.pitches >= 90.0)
    incident = mask & (physical_data.pitches < 90.0)
    boundary_hit = bool(
        np.any(np.isclose(params, PARAMETER_BOUNDS[:, 0], atol=(2.0, 0.01, 0.05)))
        or np.any(np.isclose(params, PARAMETER_BOUNDS[:, 1], atol=(2.0, 0.01, 0.05)))
    )
    result = FitResult(
        panel=data.panel.panel,
        spec_no=data.panel.spec_no,
        utc=data.panel.utc,
        method=method,
        seed=seed,
        u_surface_v=float(params[0]),
        u_relative_v=float(params[0] - U_SPACECRAFT),
        bs_over_bm=float(params[1]),
        beam_amp=float(params[2]),
        objective_per_bin=float(objective),
        median_abs_residual=float(np.median(np.abs(residual[mask]))),
        reflected_residual_rms=float(np.sqrt(np.mean(np.square(residual[reflected])))),
        incident_residual_rms=float(np.sqrt(np.mean(np.square(residual[incident])))),
        n_valid=int(np.count_nonzero(mask)),
        boundary_hit=boundary_hit,
        optimizer_success=success,
        elapsed_s=float(time.monotonic() - started),
    )
    return FitOutput(result=result, residuals=residual, expected=expected, model=model)


def _estimate_discrepancy(
    discovery: list[SpectrumData],
) -> tuple[float, float, list[FitOutput]]:
    preliminary = []
    count_numerator = 0.0
    count_denominator = 0.0
    conditional_residuals = []
    for data in discovery:
        count_fit = fit_spectrum(data, METHOD_INTERVAL, alpha=0.0, sigma=0.3, seed=42)
        conditional_fit = fit_spectrum(
            data, METHOD_CONDITIONAL, alpha=0.0, sigma=0.3, seed=42
        )
        preliminary.extend((count_fit, conditional_fit))
        if count_fit.expected is not None:
            mask = data.mask
            observed = data.count_lower[mask]
            expected = count_fit.expected[mask]
            count_numerator += float(np.sum(np.square(observed - expected) - expected))
            count_denominator += float(np.sum(np.square(expected)))
        valid = data.mask & np.isfinite(conditional_fit.residuals)
        conditional_residuals.extend((conditional_fit.residuals[valid] * 0.3).tolist())
    alpha = float(np.clip(count_numerator / max(count_denominator, 1e-12), 1e-6, 10.0))
    residual_array = np.asarray(conditional_residuals)
    center = float(np.median(residual_array))
    sigma = float(
        np.clip(1.4826 * np.median(np.abs(residual_array - center)), 0.05, 2.0)
    )
    return alpha, sigma, preliminary


def _write_results(path: Path, results: list[FitResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def _plot_comparison(path: Path, results: list[FitResult]) -> None:
    import matplotlib.pyplot as plt

    method_color = {
        METHOD_INTERVAL: "#2f6b5f",
        METHOD_CONDITIONAL: "#b14a3b",
        METHOD_LILLIS: "#3f5f9a",
    }
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    figure.patch.set_facecolor("white")
    for axis in axes:
        axis.set_facecolor("white")
    benchmarks = [
        result
        for result in results
        if result.panel == "benchmark" and result.seed == 42
    ]
    for offset, method in zip((-0.2, 0.0, 0.2), METHODS, strict=True):
        selected = [result for result in benchmarks if result.method == method]
        axes[0].scatter(
            np.arange(len(selected)) + offset,
            [result.u_relative_v for result in selected],
            label=method,
            color=method_color[method],
            s=42,
        )
    axes[0].axhline(PRIMARY_TARGET_RELATIVE_V, color="black", linestyle="--")
    axes[0].set_xticks(range(3), ["652", "653", "654"])
    axes[0].set_xlabel("Spectrum")
    axes[0].set_ylabel("Surface-to-spacecraft voltage [V]")
    axes[0].set_title("Benchmark neighborhood")
    axes[0].legend(fontsize=8)

    confirmation = [result for result in results if result.panel == "confirmation"]
    lillis = {
        result.spec_no: result.u_relative_v
        for result in confirmation
        if result.method == METHOD_LILLIS
    }
    for method in (METHOD_INTERVAL, METHOD_CONDITIONAL):
        differences = [
            result.u_relative_v - lillis[result.spec_no]
            for result in confirmation
            if result.method == method and result.spec_no in lillis
        ]
        if differences:
            axes[1].hist(
                differences,
                bins=16,
                alpha=0.6,
                label=method,
                color=method_color[method],
            )
    axes[1].axvline(0.0, color="black", linestyle="--")
    axes[1].set_xlabel("Voltage difference from smooth Lillis [V]")
    axes[1].set_ylabel("Confirmation spectra")
    axes[1].set_title("Out-of-sample method disagreement")
    if confirmation:
        axes[1].legend(fontsize=8)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, facecolor="white", transparent=False)
    plt.close(figure)


def _plot_residual_atlas(path: Path, outputs: list[FitOutput]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    with PdfPages(path) as pdf:
        for output in outputs:
            data = output.residuals
            figure, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
            finite = np.isfinite(data)
            artist = ax.imshow(
                np.where(finite, data, np.nan),
                aspect="auto",
                origin="upper",
                cmap="coolwarm",
                vmin=-3,
                vmax=3,
            )
            ax.set_xlabel("Direction index")
            ax.set_ylabel("Energy index (high to low)")
            ax.set_title(
                f"Spectrum {output.result.spec_no}: {output.result.method}\n"
                f"U_rel={output.result.u_relative_v:.1f} V"
            )
            figure.colorbar(artist, ax=ax, label="Scale-normalized residual")
            pdf.savefig(figure)
            plt.close(figure)


def _summary(results: list[FitResult], alpha: float, sigma: float) -> dict[str, object]:
    primary = {
        result.method: result
        for result in results
        if result.spec_no == 653 and result.seed == 42
    }
    confirmation = [
        result
        for result in results
        if result.panel == "confirmation" and result.seed == 42
    ]
    lillis = {
        result.spec_no: result.u_relative_v
        for result in confirmation
        if result.method == METHOD_LILLIS
    }
    confirmation_summary = {}
    for method in (METHOD_INTERVAL, METHOD_CONDITIONAL):
        differences = np.array(
            [
                result.u_relative_v - lillis[result.spec_no]
                for result in confirmation
                if result.method == method and result.spec_no in lillis
            ]
        )
        if differences.size:
            confirmation_summary[method] = {
                "median_abs_difference_from_lillis_v": float(
                    np.median(np.abs(differences))
                ),
                "catastrophic_over_100v_fraction": float(
                    np.mean(np.abs(differences) > 100.0)
                ),
            }
    return {
        "count_overdispersion_alpha": alpha,
        "conditional_log_scale_sigma": sigma,
        "primary": {
            method: {
                "u_relative_v": result.u_relative_v,
                "target_error_v": result.u_relative_v - PRIMARY_TARGET_RELATIVE_V,
                "passes_25v_tolerance": abs(
                    result.u_relative_v - PRIMARY_TARGET_RELATIVE_V
                )
                <= PRIMARY_TOLERANCE_V,
                "bs_over_bm": result.bs_over_bm,
                "median_abs_residual": result.median_abs_residual,
            }
            for method, result in primary.items()
        },
        "confirmation": confirmation_summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data", type=Path, default=Path("data/1999/091_120APR/3D990429.TAB")
    )
    parser.add_argument(
        "--telemetry",
        type=Path,
        default=Path("artifacts/experiments/experiment-002/source/M9911914.B"),
    )
    parser.add_argument(
        "--response",
        type=Path,
        default=Path("experiments/experiment-002/response_audit.csv"),
    )
    parser.add_argument(
        "--panel",
        type=Path,
        default=Path("experiments/experiment-003/panel_manifest.csv"),
    )
    parser.add_argument(
        "--stage",
        choices=("discovery", "confirmation", "all", "seeds"),
        default="all",
    )
    parser.add_argument("--spec-first", type=int)
    parser.add_argument("--spec-last", type=int)
    parser.add_argument("--count-alpha", type=float)
    parser.add_argument("--conditional-sigma", type=float)
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()

    context = ExperimentContext(args.data, args.telemetry, args.response, args.panel)
    spectra = {row.spec_no: context.spectrum(row) for row in context.panel}
    if args.count_alpha is None or args.conditional_sigma is None:
        discovery = [
            spectra[row.spec_no] for row in context.panel if row.panel == "discovery"
        ]
        alpha, sigma, _preliminary = _estimate_discrepancy(discovery)
    else:
        alpha = float(args.count_alpha)
        sigma = float(args.conditional_sigma)
    print(f"frozen count alpha={alpha:.8g}; conditional sigma={sigma:.8g}")

    if args.stage == "discovery":
        selected_rows = [row for row in context.panel if row.panel != "confirmation"]
    elif args.stage == "confirmation":
        selected_rows = [
            row
            for row in context.panel
            if row.panel == "confirmation"
            and (args.spec_first is None or row.spec_no >= args.spec_first)
            and (args.spec_last is None or row.spec_no <= args.spec_last)
        ]
    elif args.stage == "seeds":
        selected_rows = []
    else:
        selected_rows = context.panel

    outputs = []
    for row in selected_rows:
        data = spectra[row.spec_no]
        for method in METHODS:
            output = fit_spectrum(data, method, alpha=alpha, sigma=sigma, seed=42)
            outputs.append(output)
            gc.collect()
            result = output.result
            print(
                f"{row.panel:12s} {row.spec_no} {method:24s} "
                f"Urel={result.u_relative_v:8.2f} "
                f"B={result.bs_over_bm:.3f} obj={result.objective_per_bin:.4f}"
            )

    if args.stage == "seeds":
        primary = spectra[653]
        for seed in (42, 314, 2718):
            for method in METHODS:
                outputs.append(
                    fit_spectrum(primary, method, alpha=alpha, sigma=sigma, seed=seed)
                )
                gc.collect()

    results = [output.result for output in outputs]
    _write_results(args.output_dir / "results.csv", results)
    summary = _summary(results, alpha, sigma)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    if args.stage in ("discovery", "all"):
        _plot_comparison(args.output_dir / "comparison.png", results)
        atlas_outputs = [
            output
            for output in outputs
            if output.result.spec_no in (652, 653, 654) and output.result.seed == 42
        ]
        _plot_residual_atlas(args.output_dir / "residual_atlas.pdf", atlas_outputs)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
