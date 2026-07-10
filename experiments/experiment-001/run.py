#!/usr/bin/env python3
"""Controlled Step-1 diagnostic for the Halekas 2008 benchmark sweep.

This is a scratch experiment, not a production interface. It compares the
current Lillis result with Cash variants that change one observation-model
assumption at a time while retaining the same loss-cone model and optimizer.

Run from the repository root:

    uv run python experiments/experiment-001/run.py
    uv run python experiments/experiment-001/run.py --u-spacecraft 11
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from src import config
from src.losscone import noise
from src.losscone.chi2 import compute_cash_deviance_batch
from src.losscone.cpu import LossConeFitter
from src.losscone.er_data import ERData
from src.losscone.model import synth_losscone
from src.losscone.pitch_angle import PitchAngle
from src.losscone.types import FitMethod


@dataclass(frozen=True)
class FitSummary:
    name: str
    normalization: str
    mask: str
    background: str
    u_surface_v: float
    u_relative_v: float
    bs_over_bm: float
    beam_amp: float
    objective_reduced: float
    raw_deviance: float
    pearson_reduced: float
    median_abs_pearson: float
    n_valid: int
    incident_fraction: float
    u_width_v: float | None = None
    deviance_top1_fraction: float | None = None
    deviance_top10_fraction: float | None = None
    reflected_deviance_fraction: float | None = None
    count_median: float | None = None
    count_p90: float | None = None
    count_max: float | None = None
    max_dev_energy_ev: float | None = None
    max_dev_pitch_deg: float | None = None
    max_dev_count: float | None = None
    max_dev_mu: float | None = None
    max_dev_norm2d: float | None = None
    objective_at_lillis: float | None = None


def _chunk_index(er_data: ERData, spec_no: int) -> int:
    specs = er_data.data[config.SPEC_NO_COLUMN].unique()
    match = np.flatnonzero(specs == spec_no)
    if match.size != 1:
        raise ValueError(f"Expected one chunk for spec {spec_no}, found {match.size}")
    return int(match[0])


def _cash_metrics(
    model: np.ndarray,
    counts: np.ndarray,
    scale: np.ndarray,
    background: np.ndarray,
    mask: np.ndarray,
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
) -> dict[str, float]:
    bg2d = np.broadcast_to(background, counts.shape)
    mu = np.maximum(model * scale + bg2d, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_ln = np.where(counts > 0.0, counts * np.log(counts / mu), 0.0)
    deviance = np.where(mask, 2.0 * (mu - counts + c_ln), 0.0)
    raw = float(np.sum(deviance))
    dof = max(int(np.count_nonzero(mask)) - 3, 1)
    pearson = (counts - mu) / np.sqrt(mu)
    pearson_reduced = float(np.sum(np.square(pearson[mask])) / dof)
    median_abs = float(np.median(np.abs(pearson[mask])))
    sorted_deviance = np.sort(deviance[mask])[::-1]
    max_index = np.unravel_index(int(np.argmax(deviance)), deviance.shape)
    reflected = mask & (pitches >= 90.0)
    valid_counts = counts[mask]
    return {
        "raw": raw,
        "pearson_reduced": pearson_reduced,
        "median_abs": median_abs,
        "top1_fraction": float(sorted_deviance[0] / raw),
        "top10_fraction": float(np.sum(sorted_deviance[:10]) / raw),
        "reflected_fraction": float(np.sum(deviance[reflected]) / raw),
        "count_median": float(np.median(valid_counts)),
        "count_p90": float(np.quantile(valid_counts, 0.9)),
        "count_max": float(np.max(valid_counts)),
        "max_energy": float(energies[max_index[0]]),
        "max_pitch": float(pitches[max_index]),
        "max_count": float(counts[max_index]),
        "max_mu": float(mu[max_index]),
        "max_norm2d": float(norm2d[max_index]),
    }


def _profiled_row_scale(
    models: np.ndarray,
    counts: np.ndarray,
    base_scale: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Profile one multiplicative incident intensity per model and energy row."""
    mask3d = np.broadcast_to(mask, models.shape)
    weighted_model = models * base_scale[None, :, :]
    numerator = np.sum(counts[None, :, :] * mask3d, axis=2)
    denominator = np.sum(weighted_model * mask3d, axis=2)
    amplitude = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0.0,
    )
    return base_scale[None, :, :] * amplitude[:, :, None]


def _profiled_row_deviance(
    models: np.ndarray,
    counts: np.ndarray,
    base_scale: np.ndarray,
    mask: np.ndarray,
    *,
    reduced: bool,
) -> np.ndarray:
    profiled_scale = _profiled_row_scale(models, counts, base_scale, mask)
    mu = np.maximum(models * profiled_scale, np.finfo(float).tiny)
    with np.errstate(divide="ignore", invalid="ignore"):
        c_ln = np.where(
            counts[None, :, :] > 0.0,
            counts[None, :, :] * np.log(counts[None, :, :] / mu),
            0.0,
        )
    deviance = np.where(mask[None, :, :], 2.0 * (mu - counts[None, :, :] + c_ln), 0.0)
    total = np.sum(deviance, axis=(1, 2))
    if reduced:
        # Three physical parameters plus one fitted intensity per represented row.
        represented_rows = int(np.count_nonzero(np.any(mask, axis=1)))
        dof = max(int(np.count_nonzero(mask)) - 3 - represented_rows, 1)
        total = total / dof
    return total


def _fit_cash_variant(
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    chunk_idx: int,
    *,
    name: str,
    normalization_mode: str,
    mask_mode: str,
    background_mode: str,
    model_floor: float = config.LOSS_CONE_BACKGROUND,
    reference_params: tuple[float, float, float] | None = None,
) -> FitSummary:
    fitter = LossConeFitter(
        er_data,
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode=normalization_mode,
        incident_flux_stat="mean",
        fit_method="cash",
        loss_cone_background=model_floor,
    )
    data = fitter._prepare_chunk_data(chunk_idx)
    if data is None:
        raise RuntimeError(f"Could not prepare chunk {chunk_idx}")

    counts, scale_total, background_empirical = noise.chunk_noise(
        data.raw_flux, data.norm2d, data.energies
    )
    bg2d = np.broadcast_to(background_empirical, counts.shape)

    if background_mode == "production":
        scale = scale_total
        background = background_empirical
    elif background_mode == "none":
        scale = scale_total
        background = np.zeros_like(background_empirical)
    elif background_mode == "signal_scale":
        # The incident normalization is based on archive flux containing
        # background. Remove b from that total count scale before adding b to mu,
        # so an incident-side model value of one predicts the observed scale.
        scale = np.maximum(scale_total - bg2d, np.finfo(float).tiny)
        background = background_empirical
    elif background_mode == "fallback_signal_scale":
        background = noise.background_counts()
        fallback2d = np.broadcast_to(background, counts.shape)
        scale = np.maximum(scale_total - fallback2d, np.finfo(float).tiny)
    elif background_mode == "profile_row_amplitude":
        scale = scale_total
        background = np.zeros_like(background_empirical)
    else:
        raise ValueError(f"Unknown background mode: {background_mode}")

    current_mask = data.combined_mask(FitMethod.CASH)
    if mask_mode == "current":
        combined_mask = current_mask
    elif mask_mode == "reflected":
        combined_mask = current_mask & (data.pitches >= 90.0)
    elif mask_mode == "all_positive":
        combined_mask = (
            data.valid_energy_mask
            & np.isfinite(data.norm2d)
            & (data.norm2d > 0.0)
            & np.isfinite(counts)
        )
    else:
        raise ValueError(f"Unknown mask mode: {mask_mode}")

    def objective_batch(models: np.ndarray, model_masks: np.ndarray) -> np.ndarray:
        mask = combined_mask[None, :, :] & model_masks
        if background_mode == "profile_row_amplitude":
            return _profiled_row_deviance(
                models, counts, scale_total, mask[0], reduced=True
            )
        return compute_cash_deviance_batch(
            models,
            counts,
            scale,
            background,
            model_mask=mask,
            reduced=True,
        )

    def objective_scalar(params: np.ndarray) -> float:
        u_surface, bs_over_bm, beam_amp = params
        model = synth_losscone(
            energy_grid=data.energies,
            pitch_grid=data.pitches,
            U_surface=float(u_surface),
            U_spacecraft=data.spacecraft_slice,
            bs_over_bm=float(bs_over_bm),
            beam_width_eV=fitter.beam_width_ev,
            beam_amp=float(beam_amp),
            beam_pitch_sigma_deg=fitter.beam_pitch_sigma_deg,
            background=fitter.background,
        )
        if background_mode == "profile_row_amplitude":
            value = _profiled_row_deviance(
                model[None], counts, scale_total, combined_mask, reduced=True
            )[0]
            return float(value) if np.isfinite(value) else 1e30
        value = compute_cash_deviance_batch(
            model[None],
            counts,
            scale,
            background,
            model_mask=combined_mask,
            reduced=True,
        )[0]
        return float(value) if np.isfinite(value) else 1e30

    params, reduced, _lhs, _lhs_best = fitter._fit_chunk_two_phase(
        data, objective_batch, objective_scalar
    )
    reference_objective = (
        objective_scalar(np.asarray(reference_params, dtype=np.float64))
        if reference_params is not None
        else None
    )
    u_surface, bs_over_bm, beam_amp = (float(x) for x in params)
    model = synth_losscone(
        energy_grid=data.energies,
        pitch_grid=data.pitches,
        U_surface=u_surface,
        U_spacecraft=data.spacecraft_slice,
        bs_over_bm=bs_over_bm,
        beam_width_eV=fitter.beam_width_ev,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=fitter.beam_pitch_sigma_deg,
        background=fitter.background,
    )
    if background_mode == "profile_row_amplitude":
        scale = _profiled_row_scale(model[None], counts, scale_total, combined_mask)[0]
    metrics = _cash_metrics(
        model,
        counts,
        scale,
        background,
        combined_mask,
        data.energies,
        data.pitches,
        data.norm2d,
    )
    incident_fraction = float(
        np.count_nonzero(combined_mask & (data.pitches < 90.0))
        / max(np.count_nonzero(combined_mask), 1)
    )
    u_sc = float(np.nanmedian(np.asarray(data.spacecraft_slice)))
    return FitSummary(
        name=name,
        normalization=normalization_mode,
        mask=mask_mode,
        background=background_mode,
        u_surface_v=u_surface,
        u_relative_v=u_surface - u_sc,
        bs_over_bm=bs_over_bm,
        beam_amp=beam_amp,
        objective_reduced=float(reduced),
        raw_deviance=metrics["raw"],
        pearson_reduced=metrics["pearson_reduced"],
        median_abs_pearson=metrics["median_abs"],
        n_valid=int(np.count_nonzero(combined_mask)),
        incident_fraction=incident_fraction,
        deviance_top1_fraction=metrics["top1_fraction"],
        deviance_top10_fraction=metrics["top10_fraction"],
        reflected_deviance_fraction=metrics["reflected_fraction"],
        count_median=metrics["count_median"],
        count_p90=metrics["count_p90"],
        count_max=metrics["count_max"],
        max_dev_energy_ev=metrics["max_energy"],
        max_dev_pitch_deg=metrics["max_pitch"],
        max_dev_count=metrics["max_count"],
        max_dev_mu=metrics["max_mu"],
        max_dev_norm2d=metrics["max_norm2d"],
        objective_at_lillis=reference_objective,
    )


def _fit_lillis(
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    chunk_idx: int,
    normalization_mode: str,
) -> FitSummary:
    fitter = LossConeFitter(
        er_data,
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode=normalization_mode,
        incident_flux_stat="mean",
        fit_method="lillis",
    )
    result, width = fitter._fit_surface_potential_lillis_with_u_width_qc(chunk_idx)
    data = fitter._prepare_chunk_data(chunk_idx)
    if data is None:
        raise RuntimeError(f"Could not prepare chunk {chunk_idx}")
    mask = data.combined_mask(FitMethod.LILLIS)
    u_sc = float(np.nanmedian(np.asarray(data.spacecraft_slice)))
    return FitSummary(
        name=f"lillis_{normalization_mode}",
        normalization=normalization_mode,
        mask="current",
        background="normalized_floor_only",
        u_surface_v=float(result.u_surface),
        u_relative_v=float(result.u_surface - u_sc),
        bs_over_bm=float(result.bs_over_bm),
        beam_amp=float(result.beam_amp),
        objective_reduced=float(result.chi2),
        raw_deviance=float("nan"),
        pearson_reduced=float("nan"),
        median_abs_pearson=float("nan"),
        n_valid=int(np.count_nonzero(mask)),
        incident_fraction=float(
            np.count_nonzero(mask & (data.pitches < 90.0))
            / max(np.count_nonzero(mask), 1)
        ),
        u_width_v=float(width) if np.isfinite(width) else None,
    )


def _plot_residual_diagnostic(
    output: Path,
    er_data: ERData,
    pitch_angle: PitchAngle,
    spacecraft_potential: np.ndarray,
    chunk_idx: int,
    lillis: FitSummary,
    cash: FitSummary,
) -> None:
    import matplotlib.pyplot as plt

    fitter = LossConeFitter(
        er_data,
        pitch_angle=pitch_angle,
        spacecraft_potential=spacecraft_potential,
        normalization_mode="ratio",
        incident_flux_stat="mean",
        fit_method="cash",
    )
    data = fitter._prepare_chunk_data(chunk_idx)
    if data is None:
        raise RuntimeError(f"Could not prepare chunk {chunk_idx}")
    mask = data.combined_mask(FitMethod.CASH)
    counts, scale, background = noise.chunk_noise(
        data.raw_flux, data.norm2d, data.energies
    )

    def model(summary: FitSummary) -> np.ndarray:
        return synth_losscone(
            energy_grid=data.energies,
            pitch_grid=data.pitches,
            U_surface=summary.u_surface_v,
            U_spacecraft=data.spacecraft_slice,
            bs_over_bm=summary.bs_over_bm,
            beam_width_eV=fitter.beam_width_ev,
            beam_amp=summary.beam_amp,
            beam_pitch_sigma_deg=fitter.beam_pitch_sigma_deg,
            background=fitter.background,
        )

    lillis_model = model(lillis)
    cash_model = model(cash)
    mu = cash_model * scale + np.broadcast_to(background, counts.shape)
    pearson = (counts - mu) / np.sqrt(np.maximum(mu, np.finfo(float).tiny))
    with np.errstate(divide="ignore", invalid="ignore"):
        c_ln = np.where(counts > 0.0, counts * np.log(counts / mu), 0.0)
    deviance = 2.0 * (mu - counts + c_ln)

    energy_grid = np.broadcast_to(data.energies[:, None], data.pitches.shape)

    def scatter_panel(
        ax: object,
        values: np.ndarray,
        title: str,
        *,
        use_mask: bool,
        cmap: str,
        vmin: float | None = None,
        vmax: float | None = None,
    ) -> None:
        selected = mask if use_mask else np.isfinite(values)
        artist = ax.scatter(
            data.pitches[selected],
            energy_grid[selected],
            c=values[selected],
            s=10,
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            linewidths=0,
        )
        ax.set_yscale("log")
        ax.set_xlim(0, 180)
        ax.set_xlabel("Pitch angle [deg]")
        ax.set_ylabel("Energy [eV]")
        ax.set_title(title)
        plt.colorbar(artist, ax=ax, pad=0.01)

    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    scatter_panel(
        axes[0, 0],
        data.norm2d,
        "Observed normalized flux",
        use_mask=False,
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    scatter_panel(
        axes[0, 1],
        lillis_model,
        f"Lillis model (U_rel={lillis.u_relative_v:.0f} V)",
        use_mask=False,
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    scatter_panel(
        axes[0, 2],
        cash_model,
        f"Cash model (U_rel={cash.u_relative_v:.0f} V)",
        use_mask=False,
        cmap="viridis",
        vmin=0,
        vmax=2,
    )
    scatter_panel(
        axes[1, 0],
        data.norm2d - lillis_model,
        "Lillis normalized residual",
        use_mask=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )
    scatter_panel(
        axes[1, 1],
        pearson,
        "Cash signed Pearson residual",
        use_mask=True,
        cmap="coolwarm",
        vmin=-30,
        vmax=30,
    )
    scatter_panel(
        axes[1, 2],
        np.log10(1.0 + np.maximum(deviance, 0.0)),
        "log10(1 + Cash deviance per bin)",
        use_mask=True,
        cmap="magma",
    )
    figure.suptitle("Halekas benchmark: spectrum 653", fontsize=14)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/1999/091_120APR/3D990429.TAB"),
    )
    parser.add_argument("--spec-no", type=int, default=653)
    parser.add_argument("--u-spacecraft", type=float, default=0.0)
    parser.add_argument("--brief", action="store_true")
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args()

    er_data = ERData(str(args.input))
    pitch_angle = PitchAngle(er_data)
    chunk_idx = _chunk_index(er_data, args.spec_no)
    u_sc = np.full(len(er_data.data), args.u_spacecraft, dtype=np.float64)

    start = chunk_idx * config.SWEEP_ROWS
    stop = start + config.SWEEP_ROWS
    utc = er_data.data[config.UTC_COLUMN].iloc[start:stop].astype(str)

    summaries = [
        _fit_lillis(er_data, pitch_angle, u_sc, chunk_idx, "ratio"),
        _fit_lillis(er_data, pitch_angle, u_sc, chunk_idx, "ratio2"),
    ]
    lillis_reference = (
        summaries[0].u_surface_v,
        summaries[0].bs_over_bm,
        summaries[0].beam_amp,
    )
    variants = [
        ("cash_production", "ratio", "current", "production", 0.05),
        ("cash_no_additive_bg", "ratio", "current", "none", 0.05),
        ("cash_signal_scale", "ratio", "current", "signal_scale", 0.05),
        ("cash_floor_near_zero", "ratio", "current", "production", 1e-6),
        ("cash_floor_near_zero_no_bg", "ratio", "current", "none", 1e-6),
        (
            "cash_profile_row_amplitude",
            "ratio",
            "current",
            "profile_row_amplitude",
            0.05,
        ),
        ("cash_production_reflected", "ratio", "reflected", "production", 0.05),
        (
            "cash_signal_scale_reflected",
            "ratio",
            "reflected",
            "signal_scale",
            0.05,
        ),
        (
            "cash_profile_rows_reflected",
            "ratio",
            "reflected",
            "profile_row_amplitude",
            0.05,
        ),
        ("cash_signal_scale_ratio2", "ratio2", "current", "signal_scale", 0.05),
        ("cash_signal_scale_all", "ratio", "all_positive", "signal_scale", 0.05),
    ]
    for name, normalization_mode, mask_mode, background_mode, model_floor in variants:
        summaries.append(
            _fit_cash_variant(
                er_data,
                pitch_angle,
                u_sc,
                chunk_idx,
                name=name,
                normalization_mode=normalization_mode,
                mask_mode=mask_mode,
                background_mode=background_mode,
                model_floor=model_floor,
                reference_params=lillis_reference,
            )
        )

    if args.plot is not None:
        cash_production = next(
            summary for summary in summaries if summary.name == "cash_production"
        )
        _plot_residual_diagnostic(
            args.plot,
            er_data,
            pitch_angle,
            u_sc,
            chunk_idx,
            summaries[0],
            cash_production,
        )

    payload = {
        "input": str(args.input),
        "spec_no": args.spec_no,
        "chunk_index": chunk_idx,
        "utc_first": utc.iloc[0],
        "utc_last": utc.iloc[-1],
        "u_spacecraft_v": args.u_spacecraft,
        "published_reference": {
            "u_m_v": -160.0,
            "u_sc_v": 11.0,
            "bs_over_bm": 0.975,
        },
        "fits": [asdict(summary) for summary in summaries],
    }
    if args.brief:
        print(
            f"spec={args.spec_no} chunk={chunk_idx} utc={utc.iloc[0]}..{utc.iloc[-1]} "
            f"u_sc={args.u_spacecraft:g}"
        )
        for summary in summaries:
            print(
                f"{summary.name:32s} U_M={summary.u_surface_v:8.2f} "
                f"U_rel={summary.u_relative_v:8.2f} "
                f"B={summary.bs_over_bm:6.3f} A={summary.beam_amp:6.3f} "
                f"obj={summary.objective_reduced:10.3f} "
                f"at_L={summary.objective_at_lillis or float('nan'):10.3f}"
            )
    else:
        print(json.dumps(payload, indent=2, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
