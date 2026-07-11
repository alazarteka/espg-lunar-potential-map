#!/usr/bin/env python3
"""Run Experiment 004 Phase A2 representation and response diagnostics.

Run from the repository root:

    uv run python experiments/experiment-004/phase_a2.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from src import config
from src.losscone.er_data import ERData
from src.losscone.pitch_angle import PitchAngle


def _load_phase_a_module():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("experiment_004_phase_a", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a = _load_phase_a_module()


@dataclass(frozen=True)
class SectorSpectrum:
    energies_ev: np.ndarray
    flux: np.ndarray
    pitches_deg: np.ndarray


@dataclass(frozen=True)
class PhaseA2Result:
    observation_construction: str
    boundary_evaluation: str
    beam_energy_response: str
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
    beam_center_model_ev: float
    beam_amp_grid_edge: bool
    parameter_grid_edge: bool
    voltage_gate_pass: bool


@dataclass(frozen=True)
class ObservableCrosscheck:
    observation_construction: str
    beam_center_ev: float
    beam_u_relative_v: float
    beam_amplitude: float
    beam_baseline: float
    beam_rms_residual: float
    boundary_u_surface_v: float
    boundary_u_relative_v: float
    boundary_bs_over_bm: float
    boundary_objective: float
    boundary_n_valid: int
    beam_boundary_difference_v: float
    published_beam_error_v: float
    published_boundary_error_v: float


def load_sector_spectrum(data_path: Path, spec_no: int) -> SectorSpectrum:
    """Load the 15 by 88 flux and pitch arrays for one spectrum."""
    er_data = ERData(str(data_path))
    pitch_angle = PitchAngle(er_data)
    chunk = phase_a.chunk_index(er_data, spec_no)
    start = chunk * config.SWEEP_ROWS
    stop = start + config.SWEEP_ROWS
    rows = er_data.data.iloc[start:stop]
    return SectorSpectrum(
        energies_ev=rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64),
        flux=rows[config.FLUX_COLS].to_numpy(dtype=np.float64),
        pitches_deg=pitch_angle.pitch_angles[start:stop],
    )


def sector_ratios(flux: np.ndarray, pitches_deg: np.ndarray) -> np.ndarray:
    """Form conjugate reflected/incident ratios on the native 88 sectors."""
    ratios = np.full_like(flux, np.nan, dtype=np.float64)
    for energy_index, (row, pitches) in enumerate(zip(flux, pitches_deg, strict=True)):
        valid = np.isfinite(row) & (row > 0.0) & np.isfinite(pitches)
        incident_indices = np.flatnonzero(valid & (pitches < 90.0))
        reflected_indices = np.flatnonzero(valid & (pitches >= 90.0))
        if not incident_indices.size:
            continue
        ratios[energy_index, incident_indices] = 1.0
        for reflected_index in reflected_indices:
            target = 180.0 - pitches[reflected_index]
            nearest = incident_indices[
                np.argmin(np.abs(pitches[incident_indices] - target))
            ]
            ratios[energy_index, reflected_index] = row[reflected_index] / row[nearest]
    return ratios


def construct_observation(spectrum: SectorSpectrum, mode: str) -> np.ndarray:
    """Construct the two frozen Phase A2 normalized observations."""
    if mode == "bin_then_divide":
        binned = phase_a.rebin_pitch(
            spectrum.flux, spectrum.pitches_deg, spectrum.energies_ev
        )
        return phase_a.normalize_flux(binned.flux, "conjugate")
    if mode == "divide_then_bin":
        ratios = sector_ratios(spectrum.flux, spectrum.pitches_deg)
        return phase_a.rebin_pitch(
            ratios, spectrum.pitches_deg, spectrum.energies_ev
        ).flux
    raise ValueError(f"Unknown observation construction: {mode}")


def support_integrated_base_models(
    spectrum: SectorSpectrum,
    u_spacecraft_v: float,
    floor: float,
    support_values: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a sharp sector-level boundary over each pitch bin's support."""
    if support_values is None:
        support_values = spectrum.flux
    if support_values.shape != spectrum.pitches_deg.shape:
        raise ValueError("support values must match the native pitch array")
    uu, bb = np.meshgrid(phase_a.U_GRID_V, phase_a.B_GRID, indexing="ij")
    u_flat = uu.ravel()
    b_flat = bb.ravel()
    corrected_energy = np.maximum(spectrum.energies_ev[None, :] - u_spacecraft_v, 1e-12)
    sin2 = b_flat[:, None] * (1.0 + u_flat[:, None] / corrected_energy)
    alpha_c = np.degrees(np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0))))
    boundary = 180.0 - alpha_c

    model = np.full(
        (len(u_flat), len(spectrum.energies_ev), phase_a.N_PITCH_BINS),
        np.nan,
        dtype=np.float64,
    )
    for energy_index in range(len(spectrum.energies_ev)):
        pitches = spectrum.pitches_deg[energy_index]
        values = support_values[energy_index]
        valid = np.isfinite(pitches) & np.isfinite(values) & (values > 0.0)
        indices = np.digitize(pitches[valid], phase_a.PITCH_EDGES_DEG, right=False) - 1
        indices = np.clip(indices, 0, phase_a.N_PITCH_BINS - 1)
        for pitch_index in range(phase_a.N_PITCH_BINS):
            support = np.sort(pitches[valid][indices == pitch_index])
            if not support.size:
                continue
            if phase_a.PITCH_CENTERS_DEG[pitch_index] < 90.0:
                model[:, energy_index, pitch_index] = 1.0
                continue
            outside_fraction = np.searchsorted(
                support, boundary[:, energy_index], side="right"
            ) / float(support.size)
            model[:, energy_index, pitch_index] = (
                floor + (1.0 - floor) * outside_fraction
            )
    return model, u_flat, b_flat


def beam_profiles(
    energies_ev: np.ndarray,
    u_surface_v: np.ndarray,
    u_spacecraft_v: float,
    mode: str,
) -> np.ndarray:
    """Build the frozen nearest-row or fractional-Gaussian beam response."""
    centers = u_spacecraft_v - u_surface_v
    energy_response = np.zeros((len(u_surface_v), len(energies_ev)))
    in_range = (centers >= np.min(energies_ev)) & (centers <= np.max(energies_ev))
    if mode == "nearest_row":
        nearest = np.argmin(np.abs(energies_ev[None, :] - centers[:, None]), axis=1)
        rows = np.flatnonzero(in_range)
        energy_response[rows, nearest[rows]] = 1.0
    elif mode == "fractional_gaussian":
        sigma = np.maximum(0.48 * centers / 2.355, 1e-12)
        energy_response = np.exp(
            -0.5 * ((energies_ev[None, :] - centers[:, None]) / sigma[:, None]) ** 2
        )
        energy_response[~in_range] = 0.0
    else:
        raise ValueError(f"Unknown beam response: {mode}")

    profiles = np.zeros(
        (len(u_surface_v), len(energies_ev), phase_a.N_PITCH_BINS),
        dtype=np.float64,
    )
    profiles[:, :, -2:] = energy_response[:, :, None]
    return profiles


def objective(
    models: np.ndarray,
    observed: np.ndarray,
    extra_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    """Evaluate the complete squared-log objective with an optional exclusion."""
    valid = np.isfinite(observed) & (observed > 0.0)
    if extra_mask is not None:
        valid &= extra_mask
    model_valid = np.all(np.isfinite(models), axis=0)
    valid &= model_valid
    if not valid.any():
        raise RuntimeError("No valid cells for the Phase A2 objective")
    residual = np.log(np.maximum(models[:, valid], np.finfo(float).tiny)) - np.log(
        observed[valid]
    )
    return np.sum(np.square(residual), axis=1), int(np.count_nonzero(valid))


def fit_joint_branch(
    observed: np.ndarray,
    base: np.ndarray,
    beam: np.ndarray,
    u_flat: np.ndarray,
    b_flat: np.ndarray,
    observation_mode: str,
    boundary_mode: str,
    beam_mode: str,
    u_spacecraft_v: float,
) -> tuple[PhaseA2Result, np.ndarray, np.ndarray]:
    """Profile beam amplitude and fit one Phase A2 factorial branch."""
    objectives = np.full(len(u_flat), np.inf)
    amplitudes = np.zeros(len(u_flat))
    n_valid = 0
    for amplitude in phase_a.BEAM_AMP_GRID:
        candidate, n_valid = objective(base + amplitude * beam, observed)
        improved = candidate < objectives
        objectives[improved] = candidate[improved]
        amplitudes[improved] = amplitude

    best_index = int(np.argmin(objectives))
    best_model = base[best_index] + amplitudes[best_index] * beam[best_index]
    valid = np.isfinite(observed) & (observed > 0.0) & np.isfinite(best_model)
    residual = np.full_like(observed, np.nan)
    residual[valid] = np.log(best_model[valid]) - np.log(observed[valid])
    reflected = valid & (phase_a.PITCH_CENTERS_DEG[None, :] >= 90.0)

    published = np.flatnonzero((u_flat == -160.0) & np.isclose(b_flat, 0.975))
    if published.size != 1:
        raise RuntimeError("Published parameters are absent from the Phase A2 grid")
    published_objective = float(objectives[int(published[0])])
    u_surface = float(u_flat[best_index])
    b_ratio = float(b_flat[best_index])
    beam_amplitude = float(amplitudes[best_index])
    result = PhaseA2Result(
        observation_construction=observation_mode,
        boundary_evaluation=boundary_mode,
        beam_energy_response=beam_mode,
        u_surface_v=u_surface,
        u_relative_v=u_surface - u_spacecraft_v,
        bs_over_bm=b_ratio,
        beam_amp=beam_amplitude,
        objective=float(objectives[best_index]),
        objective_reduced=float(objectives[best_index] / max(n_valid - 3, 1)),
        objective_at_published=published_objective,
        published_objective_excess_fraction=(
            published_objective / float(objectives[best_index]) - 1.0
        ),
        n_valid=n_valid,
        median_abs_log_residual=float(np.nanmedian(np.abs(residual[valid]))),
        reflected_median_abs_log_residual=float(
            np.nanmedian(np.abs(residual[reflected]))
        ),
        beam_center_model_ev=u_spacecraft_v - u_surface,
        beam_amp_grid_edge=beam_amplitude
        in {float(phase_a.BEAM_AMP_GRID[0]), float(phase_a.BEAM_AMP_GRID[-1])},
        parameter_grid_edge=(
            u_surface in {float(phase_a.U_GRID_V[0]), float(phase_a.U_GRID_V[-1])}
            or b_ratio in {float(phase_a.B_GRID[0]), float(phase_a.B_GRID[-1])}
        ),
        voltage_gate_pass=abs((u_surface - u_spacecraft_v) + 171.0) <= 25.0,
    )
    profile = np.min(
        objectives.reshape(len(phase_a.U_GRID_V), len(phase_a.B_GRID)), axis=1
    )
    return result, best_model, profile


def fit_beam_centroid(
    observed: np.ndarray, energies_ev: np.ndarray
) -> tuple[float, float, float, float]:
    """Fit a response-limited beam centroid to field-aligned excess."""

    def finite_mean(values: np.ndarray) -> np.ndarray:
        valid = np.isfinite(values)
        count = np.count_nonzero(valid, axis=1)
        return np.divide(
            np.sum(np.where(valid, values, 0.0), axis=1),
            count,
            out=np.full(len(values), np.nan),
            where=count > 0,
        )

    field_aligned = finite_mean(observed[:, -2:])
    shoulder = finite_mean(observed[:, -4:-2])
    excess = field_aligned - shoulder
    valid = np.isfinite(excess) & (energies_ev >= 20.0) & (energies_ev <= 1000.0)
    energies = energies_ev[valid]
    values = excess[valid]
    if len(energies) < 4:
        raise RuntimeError("Insufficient energy support for beam centroid fitting")

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude, baseline, center = parameters
        sigma = 0.48 * center / 2.355
        response = np.exp(-0.5 * ((energies - center) / sigma) ** 2)
        return baseline + amplitude * response - values

    candidates = []
    for start_center in (80.0, 120.0, 170.0, 240.0, 340.0):
        fit = least_squares(
            residual,
            x0=np.array([max(float(np.nanmax(values)), 0.1), 0.0, start_center]),
            bounds=(np.array([0.0, -5.0, 40.0]), np.array([30.0, 5.0, 500.0])),
        )
        candidates.append(fit)
    best = min(candidates, key=lambda item: float(np.sum(np.square(item.fun))))
    amplitude, baseline, center = (float(value) for value in best.x)
    rms = float(np.sqrt(np.mean(np.square(best.fun))))
    return center, amplitude, baseline, rms


def fit_boundary_only(
    observed: np.ndarray,
    base: np.ndarray,
    u_flat: np.ndarray,
    b_flat: np.ndarray,
    energies_ev: np.ndarray,
    beam_center_ev: float,
    u_spacecraft_v: float,
) -> tuple[float, float, float, int]:
    """Fit the support-integrated boundary after excluding beam-sensitive cells."""
    sigma = 0.48 * beam_center_ev / 2.355
    response = np.exp(-0.5 * ((energies_ev - beam_center_ev) / sigma) ** 2)
    keep = np.ones_like(observed, dtype=bool)
    beam_rows = response >= 0.10
    keep[np.ix_(beam_rows, phase_a.PITCH_CENTERS_DEG >= 157.5)] = False
    objectives, n_valid = objective(base, observed, extra_mask=keep)
    best = int(np.argmin(objectives))
    return (
        float(u_flat[best]),
        float(b_flat[best]),
        float(objectives[best]),
        n_valid,
    )


def write_dataclasses(path: Path, rows: list[object]) -> None:
    """Write a homogeneous dataclass list as LF-terminated CSV."""
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dictionaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(dictionaries)


def write_profiles(
    path: Path,
    profiles: list[tuple[str, str, str, np.ndarray]],
) -> None:
    """Write field-ratio-profiled Phase A2 voltage objectives."""
    fields = (
        "observation_construction",
        "boundary_evaluation",
        "beam_energy_response",
        "u_surface_v",
        "objective",
        "delta_objective",
    )
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for observation_mode, boundary_mode, beam_mode, profile in profiles:
            minimum = float(np.min(profile))
            for u_surface, value in zip(phase_a.U_GRID_V, profile, strict=True):
                writer.writerow(
                    {
                        "observation_construction": observation_mode,
                        "boundary_evaluation": boundary_mode,
                        "beam_energy_response": beam_mode,
                        "u_surface_v": u_surface,
                        "objective": float(value),
                        "delta_objective": float(value - minimum),
                    }
                )


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
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectrum = load_sector_spectrum(args.data_path, args.spec_no)
    observations = {
        mode: construct_observation(spectrum, mode)
        for mode in ("bin_then_divide", "divide_then_bin")
    }
    center_base, u_flat, b_flat = phase_a.losscone_base_models(
        spectrum.energies_ev, args.u_spacecraft, 0.10
    )
    support_values = {
        "bin_then_divide": spectrum.flux,
        "divide_then_bin": sector_ratios(spectrum.flux, spectrum.pitches_deg),
    }
    support_bases = {}
    for observation_mode, values in support_values.items():
        support_base, support_u, support_b = support_integrated_base_models(
            spectrum, args.u_spacecraft, 0.10, support_values=values
        )
        np.testing.assert_allclose(support_u, u_flat)
        np.testing.assert_allclose(support_b, b_flat)
        support_bases[observation_mode] = support_base
    beams = {
        mode: beam_profiles(spectrum.energies_ev, u_flat, args.u_spacecraft, mode)
        for mode in ("nearest_row", "fractional_gaussian")
    }

    results: list[PhaseA2Result] = []
    profiles: list[tuple[str, str, str, np.ndarray]] = []
    best_models: dict[tuple[str, str, str], np.ndarray] = {}
    for observation_mode, observed in observations.items():
        bases = {
            "pitch_bin_center": center_base,
            "sector_support": support_bases[observation_mode],
        }
        for boundary_mode, base in bases.items():
            for beam_mode, beam in beams.items():
                result, model, profile = fit_joint_branch(
                    observed,
                    base,
                    beam,
                    u_flat,
                    b_flat,
                    observation_mode,
                    boundary_mode,
                    beam_mode,
                    args.u_spacecraft,
                )
                results.append(result)
                profiles.append((observation_mode, boundary_mode, beam_mode, profile))
                best_models[(observation_mode, boundary_mode, beam_mode)] = model

    crosschecks: list[ObservableCrosscheck] = []
    for observation_mode, observed in observations.items():
        beam_center, beam_amp, beam_baseline, beam_rms = fit_beam_centroid(
            observed, spectrum.energies_ev
        )
        boundary_u, boundary_b, boundary_objective, boundary_n = fit_boundary_only(
            observed,
            support_bases[observation_mode],
            u_flat,
            b_flat,
            spectrum.energies_ev,
            beam_center,
            args.u_spacecraft,
        )
        beam_u_relative = -beam_center
        boundary_u_relative = boundary_u - args.u_spacecraft
        crosschecks.append(
            ObservableCrosscheck(
                observation_construction=observation_mode,
                beam_center_ev=beam_center,
                beam_u_relative_v=beam_u_relative,
                beam_amplitude=beam_amp,
                beam_baseline=beam_baseline,
                beam_rms_residual=beam_rms,
                boundary_u_surface_v=boundary_u,
                boundary_u_relative_v=boundary_u_relative,
                boundary_bs_over_bm=boundary_b,
                boundary_objective=boundary_objective,
                boundary_n_valid=boundary_n,
                beam_boundary_difference_v=beam_u_relative - boundary_u_relative,
                published_beam_error_v=beam_u_relative + 171.0,
                published_boundary_error_v=boundary_u_relative + 171.0,
            )
        )

    write_dataclasses(args.output_dir / "phase_a2_results.csv", results)
    write_profiles(args.output_dir / "phase_a2_profiles.csv", profiles)
    write_dataclasses(args.output_dir / "beam_boundary_crosscheck.csv", crosschecks)
    best = min(results, key=lambda item: item.objective_reduced)
    key = (
        best.observation_construction,
        best.boundary_evaluation,
        best.beam_energy_response,
    )
    phase_a.plot_comparison(
        args.output_dir / "phase_a2_comparison.png",
        observations[best.observation_construction],
        best_models[key],
        spectrum.energies_ev,
        (
            f"Phase A2 spectrum {args.spec_no}: {best.observation_construction}, "
            f"{best.boundary_evaluation}, {best.beam_energy_response}; "
            f"Urel={best.u_relative_v:.0f} V, B={best.bs_over_bm:.3f}"
        ),
    )

    print(f"Wrote {args.output_dir / 'phase_a2_results.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a2_profiles.csv'}")
    print(f"Wrote {args.output_dir / 'beam_boundary_crosscheck.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a2_comparison.png'}")
    print(
        "Best Phase A2 branch: "
        f"{best.observation_construction}, {best.boundary_evaluation}, "
        f"{best.beam_energy_response}, Urel={best.u_relative_v:.1f} V, "
        f"B={best.bs_over_bm:.3f}, objective/dof={best.objective_reduced:.4f}"
    )


if __name__ == "__main__":
    main()
