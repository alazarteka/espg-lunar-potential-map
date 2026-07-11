#!/usr/bin/env python3
"""Run Phase A of the clean-room Halekas reconstruction.

This is experiment-local code. It intentionally does not call the production
loss-cone fitter because Phase A tests a different pitch representation,
objective domain, beam response, and optimizer.

Run from the repository root:

    uv run python experiments/experiment-004/run.py
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.losscone.er_data import ERData
from src.losscone.pitch_angle import PitchAngle

N_PITCH_BINS = 16
PITCH_EDGES_DEG = np.linspace(0.0, 180.0, N_PITCH_BINS + 1)
PITCH_CENTERS_DEG = 0.5 * (PITCH_EDGES_DEG[:-1] + PITCH_EDGES_DEG[1:])
U_GRID_V = np.arange(-2000.0, 20.0 + 0.1, 10.0)
B_GRID = np.arange(0.50, 1.05 + 0.0001, 0.005)
BEAM_AMP_GRID = np.arange(0.0, 8.0 + 0.001, 0.25)


@dataclass(frozen=True)
class BranchResult:
    normalization: str
    floor: float
    beam_amplitude_mode: str
    measured_beam_amp: float
    u_surface_v: float
    u_relative_v: float
    bs_over_bm: float
    beam_amp: float
    objective: float
    objective_reduced: float
    objective_at_published: float
    published_objective_excess_fraction: float
    n_valid: int
    median_abs_log_residual: float
    reflected_median_abs_log_residual: float
    beam_energy_observed_ev: float
    beam_energy_model_ev: float
    beam_energy_bin_difference: int
    beam_pitch_peak_observed_deg: float
    voltage_gate_pass: bool
    grid_edge: bool


@dataclass(frozen=True)
class BinnedSpectrum:
    energies_ev: np.ndarray
    flux: np.ndarray
    support: np.ndarray


def chunk_index(er_data: ERData, spec_no: int) -> int:
    """Return the post-invalid-removal chunk index for a spectrum number."""
    specs = er_data.data[config.SPEC_NO_COLUMN].unique()
    match = np.flatnonzero(specs == spec_no)
    if match.size != 1:
        raise ValueError(f"Expected one chunk for spectrum {spec_no}, got {match.size}")
    return int(match[0])


def rebin_pitch(
    flux: np.ndarray,
    pitches_deg: np.ndarray,
    energies_ev: np.ndarray,
) -> BinnedSpectrum:
    """Average the 88 angular sectors into fixed 11.25-degree pitch bins."""
    if flux.shape != pitches_deg.shape:
        raise ValueError("flux and pitch arrays must have the same shape")
    if flux.shape[0] != len(energies_ev):
        raise ValueError("energy rows must match flux rows")

    binned = np.full((len(energies_ev), N_PITCH_BINS), np.nan, dtype=np.float64)
    support = np.zeros_like(binned, dtype=np.int64)
    for energy_index in range(len(energies_ev)):
        values = flux[energy_index]
        angles = pitches_deg[energy_index]
        valid = np.isfinite(values) & (values > 0.0) & np.isfinite(angles)
        indices = np.digitize(angles[valid], PITCH_EDGES_DEG, right=False) - 1
        indices = np.clip(indices, 0, N_PITCH_BINS - 1)
        for pitch_index in range(N_PITCH_BINS):
            selected = values[valid][indices == pitch_index]
            if selected.size:
                binned[energy_index, pitch_index] = float(np.mean(selected))
                support[energy_index, pitch_index] = int(selected.size)
    return BinnedSpectrum(np.asarray(energies_ev), binned, support)


def normalize_flux(flux: np.ndarray, mode: str) -> np.ndarray:
    """Construct conjugate-bin or incident-hemisphere normalized flux."""
    normalized = np.full_like(flux, np.nan, dtype=np.float64)
    incident = PITCH_CENTERS_DEG < 90.0
    reflected = ~incident

    if mode == "conjugate":
        for energy_index, row in enumerate(flux):
            for pitch_index in np.flatnonzero(incident):
                if np.isfinite(row[pitch_index]) and row[pitch_index] > 0.0:
                    normalized[energy_index, pitch_index] = 1.0
            for pitch_index in np.flatnonzero(reflected):
                conjugate_index = N_PITCH_BINS - 1 - pitch_index
                numerator = row[pitch_index]
                denominator = row[conjugate_index]
                if (
                    np.isfinite(numerator)
                    and numerator > 0.0
                    and np.isfinite(denominator)
                    and denominator > 0.0
                ):
                    normalized[energy_index, pitch_index] = numerator / denominator
        return normalized

    if mode == "incident_mean":
        for energy_index, row in enumerate(flux):
            incident_values = row[incident]
            valid = np.isfinite(incident_values) & (incident_values > 0.0)
            if valid.any():
                normalized[energy_index] = row / float(np.mean(incident_values[valid]))
        return normalized

    raise ValueError(f"Unknown normalization mode: {mode}")


def observed_beam(
    normalized: np.ndarray, energies_ev: np.ndarray
) -> tuple[int, float, float]:
    """Detect a field-aligned excess without using the published voltage."""

    def finite_mean(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        numerator = np.sum(np.where(valid, values, 0.0), axis=1)
        denominator = np.count_nonzero(valid, axis=1)
        return np.divide(
            numerator,
            denominator,
            out=np.full(len(values), np.nan),
            where=denominator > 0,
        )

    field_aligned = finite_mean(normalized[:, -2:])
    shoulder = finite_mean(normalized[:, -4:-2])
    score = field_aligned - shoulder
    eligible = np.isfinite(score) & (energies_ev >= 20.0) & (energies_ev <= 2000.0)
    if not eligible.any():
        raise RuntimeError("No energy row is eligible for beam detection")
    masked_score = np.where(eligible, score, -np.inf)
    energy_index = int(np.argmax(masked_score))
    amplitude = max(float(field_aligned[energy_index] - shoulder[energy_index]), 0.0)
    pitch_slice = normalized[energy_index, -4:]
    pitch_index = N_PITCH_BINS - 4 + int(np.nanargmax(pitch_slice))
    return energy_index, amplitude, float(PITCH_CENTERS_DEG[pitch_index])


def losscone_base_models(
    energies_ev: np.ndarray,
    u_spacecraft_v: float,
    floor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build hard-boundary models for every point on the deterministic grid."""
    uu, bb = np.meshgrid(U_GRID_V, B_GRID, indexing="ij")
    u_flat = uu.ravel()
    b_flat = bb.ravel()
    corrected_energy = np.maximum(energies_ev[None, :] - u_spacecraft_v, 1e-12)
    sin2 = b_flat[:, None] * (1.0 + u_flat[:, None] / corrected_energy)
    alpha_c = np.degrees(np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0))))
    boundary = 180.0 - alpha_c

    model = np.ones((len(u_flat), len(energies_ev), N_PITCH_BINS), dtype=np.float64)
    reflected = PITCH_CENTERS_DEG[None, None, :] >= 90.0
    depleted = PITCH_CENTERS_DEG[None, None, :] > boundary[:, :, None]
    model[reflected & depleted] = floor
    return model, u_flat, b_flat


def beam_profiles(
    energies_ev: np.ndarray,
    u_surface_v: np.ndarray,
    u_spacecraft_v: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a one-energy-bin, two-pitch-bin beam for every voltage."""
    centers = u_spacecraft_v - u_surface_v
    profiles = np.zeros(
        (len(u_surface_v), len(energies_ev), N_PITCH_BINS), dtype=np.float64
    )
    nearest = np.argmin(np.abs(energies_ev[None, :] - centers[:, None]), axis=1)
    in_range = (centers >= np.min(energies_ev)) & (centers <= np.max(energies_ev))
    rows = np.flatnonzero(in_range)
    profiles[rows, nearest[rows], -2:] = 1.0
    return profiles, nearest


def objective(models: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, int]:
    """Unweighted squared-log residual over every finite positive observed bin."""
    valid = np.isfinite(observed) & (observed > 0.0)
    if not valid.any():
        raise RuntimeError("No valid normalized bins")
    residual = np.log(np.maximum(models[:, valid], np.finfo(float).tiny)) - np.log(
        observed[valid]
    )
    return np.sum(np.square(residual), axis=1), int(np.count_nonzero(valid))


def fit_branch(
    observed: np.ndarray,
    energies_ev: np.ndarray,
    normalization: str,
    floor: float,
    beam_mode: str,
    measured_beam_amp: float,
    observed_beam_index: int,
    observed_beam_pitch_deg: float,
    u_spacecraft_v: float,
) -> tuple[BranchResult, np.ndarray, np.ndarray]:
    """Fit one frozen ambiguity branch and return its best model."""
    base, u_flat, b_flat = losscone_base_models(energies_ev, u_spacecraft_v, floor)
    beam, model_beam_indices = beam_profiles(energies_ev, u_flat, u_spacecraft_v)

    if beam_mode == "fixed_measured":
        amplitudes = np.full(len(u_flat), measured_beam_amp)
        models = base + amplitudes[:, None, None] * beam
        objectives, n_valid = objective(models, observed)
    elif beam_mode == "profile_grid":
        objectives = np.full(len(u_flat), np.inf, dtype=np.float64)
        amplitudes = np.zeros(len(u_flat), dtype=np.float64)
        n_valid = 0
        for amplitude in BEAM_AMP_GRID:
            models = base + amplitude * beam
            candidate, n_valid = objective(models, observed)
            improved = candidate < objectives
            objectives[improved] = candidate[improved]
            amplitudes[improved] = amplitude
    else:
        raise ValueError(f"Unknown beam mode: {beam_mode}")

    best_index = int(np.argmin(objectives))
    best_model = base[best_index] + amplitudes[best_index] * beam[best_index]
    valid = np.isfinite(observed) & (observed > 0.0)
    log_residual = np.full_like(observed, np.nan)
    log_residual[valid] = np.log(best_model[valid]) - np.log(observed[valid])
    reflected = valid & (PITCH_CENTERS_DEG[None, :] >= 90.0)
    model_beam_index = int(model_beam_indices[best_index])
    published_match = np.flatnonzero((u_flat == -160.0) & np.isclose(b_flat, 0.975))
    if published_match.size != 1:
        raise RuntimeError("Published parameters are absent from the frozen grid")
    published_objective = float(objectives[int(published_match[0])])
    u_surface = float(u_flat[best_index])
    bs_over_bm = float(b_flat[best_index])
    grid_edge = u_surface in {
        float(U_GRID_V[0]),
        float(U_GRID_V[-1]),
    } or bs_over_bm in {float(B_GRID[0]), float(B_GRID[-1])}
    result = BranchResult(
        normalization=normalization,
        floor=floor,
        beam_amplitude_mode=beam_mode,
        measured_beam_amp=measured_beam_amp,
        u_surface_v=u_surface,
        u_relative_v=u_surface - u_spacecraft_v,
        bs_over_bm=bs_over_bm,
        beam_amp=float(amplitudes[best_index]),
        objective=float(objectives[best_index]),
        objective_reduced=float(objectives[best_index] / max(n_valid - 3, 1)),
        objective_at_published=published_objective,
        published_objective_excess_fraction=(
            published_objective / float(objectives[best_index]) - 1.0
        ),
        n_valid=n_valid,
        median_abs_log_residual=float(np.nanmedian(np.abs(log_residual[valid]))),
        reflected_median_abs_log_residual=float(
            np.nanmedian(np.abs(log_residual[reflected]))
        ),
        beam_energy_observed_ev=float(energies_ev[observed_beam_index]),
        beam_energy_model_ev=float(energies_ev[model_beam_index]),
        beam_energy_bin_difference=abs(model_beam_index - observed_beam_index),
        beam_pitch_peak_observed_deg=observed_beam_pitch_deg,
        voltage_gate_pass=abs((u_surface - u_spacecraft_v) + 171.0) <= 25.0,
        grid_edge=grid_edge,
    )
    profile = np.min(objectives.reshape(len(U_GRID_V), len(B_GRID)), axis=1)
    return result, best_model, profile


def write_results(path: Path, results: list[BranchResult]) -> None:
    """Write deterministic branch results."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [asdict(result) for result in results]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_profiles(
    path: Path,
    profiles: list[tuple[str, float, str, np.ndarray]],
) -> None:
    """Write each branch's field-ratio-profiled voltage objective."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "normalization",
                "floor",
                "beam_amplitude_mode",
                "u_surface_v",
                "objective",
                "delta_objective",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        for normalization, floor, beam_mode, profile in profiles:
            minimum = float(np.min(profile))
            for u_surface, value in zip(U_GRID_V, profile, strict=True):
                writer.writerow(
                    {
                        "normalization": normalization,
                        "floor": floor,
                        "beam_amplitude_mode": beam_mode,
                        "u_surface_v": u_surface,
                        "objective": float(value),
                        "delta_objective": float(value - minimum),
                    }
                )


def plot_comparison(
    path: Path,
    observed: np.ndarray,
    model: np.ndarray,
    energies_ev: np.ndarray,
    title: str,
) -> None:
    """Plot the best measured, synthetic, and residual distributions."""
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(observed) & (observed > 0.0)
    residual = np.full_like(observed, np.nan)
    residual[valid] = np.log10(model[valid]) - np.log10(observed[valid])
    log_observed = np.log10(np.maximum(observed, 0.05))
    log_model = np.log10(np.maximum(model, 0.05))

    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.8), constrained_layout=True)
    extent = [0.0, 180.0, float(np.min(energies_ev)), float(np.max(energies_ev))]
    for axis, values, panel_title, limits, cmap in (
        (axes[0], log_observed, "Measured log10 ratio", (-1.0, 1.0), "viridis"),
        (axes[1], log_model, "Synthetic log10 ratio", (-1.0, 1.0), "viridis"),
        (axes[2], residual, "Model - measured", (-1.0, 1.0), "coolwarm"),
    ):
        artist = axis.imshow(
            values,
            origin="lower",
            aspect="auto",
            extent=extent,
            vmin=limits[0],
            vmax=limits[1],
            cmap=cmap,
        )
        axis.set_yscale("log")
        axis.set_xlabel("Pitch angle [deg]")
        axis.set_title(panel_title)
        figure.colorbar(artist, ax=axis)
    axes[0].set_ylabel("Measured energy [eV]")
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def load_benchmark(data_path: Path, spec_no: int) -> BinnedSpectrum:
    """Load and pitch-bin one Level-1 benchmark spectrum."""
    er_data = ERData(str(data_path))
    pitch_angle = PitchAngle(er_data)
    chunk = chunk_index(er_data, spec_no)
    start = chunk * config.SWEEP_ROWS
    stop = start + config.SWEEP_ROWS
    rows = er_data.data.iloc[start:stop]
    flux = rows[config.FLUX_COLS].to_numpy(dtype=np.float64)
    pitches = pitch_angle.pitch_angles[start:stop]
    energies = rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
    return rebin_pitch(flux, pitches, energies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/1999/091_120APR/3D990429.TAB"),
    )
    parser.add_argument("--spec-no", type=int, default=653)
    parser.add_argument("--u-spacecraft", type=float, default=11.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/experiment-004")
    )
    args = parser.parse_args()

    spectrum = load_benchmark(args.data_path, args.spec_no)
    results: list[BranchResult] = []
    models: dict[tuple[str, float, str], np.ndarray] = {}
    observations: dict[str, np.ndarray] = {}
    profiles: list[tuple[str, float, str, np.ndarray]] = []

    for normalization in ("conjugate", "incident_mean"):
        observed = normalize_flux(spectrum.flux, normalization)
        observations[normalization] = observed
        beam_index, beam_amplitude, beam_pitch = observed_beam(
            observed, spectrum.energies_ev
        )
        for floor in (0.05, 0.10):
            for beam_mode in ("fixed_measured", "profile_grid"):
                result, model, profile = fit_branch(
                    observed=observed,
                    energies_ev=spectrum.energies_ev,
                    normalization=normalization,
                    floor=floor,
                    beam_mode=beam_mode,
                    measured_beam_amp=beam_amplitude,
                    observed_beam_index=beam_index,
                    observed_beam_pitch_deg=beam_pitch,
                    u_spacecraft_v=args.u_spacecraft,
                )
                results.append(result)
                models[(normalization, floor, beam_mode)] = model
                profiles.append((normalization, floor, beam_mode, profile))

    write_results(args.output_dir / "phase_a_results.csv", results)
    write_profiles(args.output_dir / "phase_a_profiles.csv", profiles)
    best = min(results, key=lambda item: item.objective_reduced)
    best_key = (best.normalization, best.floor, best.beam_amplitude_mode)
    plot_comparison(
        args.output_dir / "phase_a_comparison.png",
        observations[best.normalization],
        models[best_key],
        spectrum.energies_ev,
        (
            f"Spectrum {args.spec_no}: {best.normalization}, floor={best.floor:.2f}, "
            f"beam={best.beam_amplitude_mode}; Urel={best.u_relative_v:.0f} V, "
            f"B ratio={best.bs_over_bm:.3f}"
        ),
    )

    print(f"Wrote {args.output_dir / 'phase_a_results.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a_profiles.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a_comparison.png'}")
    print(
        "Best branch: "
        f"{best.normalization}, floor={best.floor:.2f}, "
        f"beam={best.beam_amplitude_mode}, Urel={best.u_relative_v:.1f} V, "
        f"B={best.bs_over_bm:.3f}, objective/dof={best.objective_reduced:.4f}"
    )


if __name__ == "__main__":
    main()
