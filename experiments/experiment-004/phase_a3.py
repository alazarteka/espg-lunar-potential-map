#!/usr/bin/env python3
"""Run Experiment 004 Phase A3 physical-consistency stress tests.

Run from the repository root:

    uv run python experiments/experiment-004/phase_a3.py
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


def _load_local_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a = _load_local_module("run.py", "experiment_004_phase_a_for_a3")
phase_a2 = _load_local_module("phase_a2.py", "experiment_004_phase_a2_for_a3")

A3_BEAM_AMP_GRID = np.arange(0.0, 20.0 + 0.001, 0.5)
BEAM_BROAD_FWHM_DEG = 45.0
BEAM_BROAD_SIGMA_DEG = BEAM_BROAD_FWHM_DEG / 2.355


@dataclass(frozen=True)
class PhaseA3Result:
    observation_construction: str
    beam_pitch_response: str
    boundary_shape: str
    cone_floor: float
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
    beam_amp_grid_edge: bool
    parameter_grid_edge: bool
    voltage_gate_pass: bool


@dataclass(frozen=True)
class PhaseA3Crosscheck:
    observation_construction: str
    beam_center_ev: float
    beam_u_relative_v: float
    beam_amp: float
    beam_pitch_sigma_deg: float
    beam_pitch_fwhm_deg: float
    beam_rms_residual: float
    boundary_shape: str
    cone_floor: float
    boundary_mask: str
    boundary_u_surface_v: float
    boundary_u_relative_v: float
    boundary_bs_over_bm: float
    boundary_objective: float
    boundary_n_valid: int
    beam_boundary_difference_v: float


def integrated_boundary_models(
    spectrum,
    support_values: np.ndarray,
    u_spacecraft_v: float,
    floor: float,
    tau_deg: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average a sharp or broadened physical boundary over native support."""
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
            support = pitches[valid][indices == pitch_index]
            if not support.size:
                continue
            if phase_a.PITCH_CENTERS_DEG[pitch_index] < 90.0:
                model[:, energy_index, pitch_index] = 1.0
                continue
            if tau_deg <= 0.0:
                survival = np.mean(
                    support[None, :] <= boundary[:, energy_index, None], axis=1
                )
            else:
                z = (boundary[:, energy_index, None] - support[None, :]) / tau_deg
                survival = np.mean(1.0 / (1.0 + np.exp(-np.clip(z, -60, 60))), axis=1)
            model[:, energy_index, pitch_index] = floor + (1.0 - floor) * survival
    return model, u_flat, b_flat


def integrated_beam_profiles(
    spectrum,
    support_values: np.ndarray,
    u_surface_v: np.ndarray,
    u_spacecraft_v: float,
    pitch_mode: str,
) -> np.ndarray:
    """Build fractional-energy beams averaged over actual angular support."""
    centers = u_spacecraft_v - u_surface_v
    sigma_energy = np.maximum(0.48 * centers / 2.355, 1e-12)
    energy_response = np.exp(
        -0.5
        * ((spectrum.energies_ev[None, :] - centers[:, None]) / sigma_energy[:, None])
        ** 2
    )
    in_range = (centers >= np.min(spectrum.energies_ev)) & (
        centers <= np.max(spectrum.energies_ev)
    )
    energy_response[~in_range] = 0.0

    angular = np.full((len(spectrum.energies_ev), phase_a.N_PITCH_BINS), np.nan)
    for energy_index in range(len(spectrum.energies_ev)):
        pitches = spectrum.pitches_deg[energy_index]
        values = support_values[energy_index]
        valid = np.isfinite(pitches) & np.isfinite(values) & (values > 0.0)
        indices = np.digitize(pitches[valid], phase_a.PITCH_EDGES_DEG, right=False) - 1
        indices = np.clip(indices, 0, phase_a.N_PITCH_BINS - 1)
        for pitch_index in range(phase_a.N_PITCH_BINS):
            support = pitches[valid][indices == pitch_index]
            if not support.size:
                continue
            if pitch_mode == "narrow_top_hat":
                response = support >= 155.0
            elif pitch_mode == "broad_gaussian":
                response = np.exp(
                    -0.5 * ((support - 180.0) / BEAM_BROAD_SIGMA_DEG) ** 2
                )
            else:
                raise ValueError(f"Unknown beam pitch response: {pitch_mode}")
            angular[energy_index, pitch_index] = float(np.mean(response))
    return energy_response[:, :, None] * angular[None, :, :]


def fit_joint_branch(
    observed: np.ndarray,
    base: np.ndarray,
    beam: np.ndarray,
    u_flat: np.ndarray,
    b_flat: np.ndarray,
    observation_mode: str,
    beam_mode: str,
    boundary_mode: str,
    floor: float,
    u_spacecraft_v: float,
) -> tuple[PhaseA3Result, np.ndarray, np.ndarray]:
    """Profile extended beam amplitude for one frozen A3 branch."""
    objectives = np.full(len(u_flat), np.inf)
    amplitudes = np.zeros(len(u_flat))
    n_valid = 0
    for amplitude in A3_BEAM_AMP_GRID:
        candidate, n_valid = phase_a2.objective(base + amplitude * beam, observed)
        improved = candidate < objectives
        objectives[improved] = candidate[improved]
        amplitudes[improved] = amplitude

    best = int(np.argmin(objectives))
    best_model = base[best] + amplitudes[best] * beam[best]
    valid = np.isfinite(observed) & (observed > 0.0) & np.isfinite(best_model)
    residual = np.full_like(observed, np.nan)
    residual[valid] = np.log(best_model[valid]) - np.log(observed[valid])
    reflected = valid & (phase_a.PITCH_CENTERS_DEG[None, :] >= 90.0)
    published = np.flatnonzero((u_flat == -160.0) & np.isclose(b_flat, 0.975))
    if published.size != 1:
        raise RuntimeError("Published parameters are absent from the Phase A3 grid")

    u_surface = float(u_flat[best])
    b_ratio = float(b_flat[best])
    amplitude = float(amplitudes[best])
    objective_at_published = float(objectives[int(published[0])])
    result = PhaseA3Result(
        observation_construction=observation_mode,
        beam_pitch_response=beam_mode,
        boundary_shape=boundary_mode,
        cone_floor=floor,
        u_surface_v=u_surface,
        u_relative_v=u_surface - u_spacecraft_v,
        bs_over_bm=b_ratio,
        beam_amp=amplitude,
        objective=float(objectives[best]),
        objective_reduced=float(objectives[best] / max(n_valid - 3, 1)),
        objective_at_published=objective_at_published,
        published_objective_excess_fraction=(
            objective_at_published / float(objectives[best]) - 1.0
        ),
        n_valid=n_valid,
        median_abs_log_residual=float(np.nanmedian(np.abs(residual[valid]))),
        reflected_median_abs_log_residual=float(
            np.nanmedian(np.abs(residual[reflected]))
        ),
        beam_amp_grid_edge=amplitude
        in {float(A3_BEAM_AMP_GRID[0]), float(A3_BEAM_AMP_GRID[-1])},
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


def fit_beam_2d(
    observed: np.ndarray, energies_ev: np.ndarray
) -> tuple[float, float, float, float]:
    """Fit beam center and pitch width with one baseline per energy row."""
    energy_rows = np.flatnonzero((energies_ev >= 20.0) & (energies_ev <= 1000.0))
    pitch_indices = np.flatnonzero(phase_a.PITCH_CENTERS_DEG >= 90.0)
    values = observed[np.ix_(energy_rows, pitch_indices)]
    valid = np.isfinite(values) & (values > 0.0)
    energies = energies_ev[energy_rows]
    pitches = phase_a.PITCH_CENTERS_DEG[pitch_indices]
    initial_baseline = np.array(
        [
            float(np.nanmedian(row[:4])) if np.isfinite(row[:4]).any() else 0.2
            for row in values
        ]
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude, center, pitch_sigma = parameters[:3]
        baselines = parameters[3:]
        energy_sigma = 0.48 * center / 2.355
        energy_response = np.exp(-0.5 * ((energies - center) / energy_sigma) ** 2)
        pitch_response = np.exp(-0.5 * ((pitches - 180.0) / pitch_sigma) ** 2)
        model = baselines[:, None] + amplitude * (
            energy_response[:, None] * pitch_response[None, :]
        )
        return (model - values)[valid]

    candidates = []
    for start_center in (100.0, 150.0, 200.0, 300.0):
        initial = np.concatenate(
            ([10.0, start_center, BEAM_BROAD_SIGMA_DEG], initial_baseline)
        )
        lower = np.concatenate(([0.0, 40.0, 3.0], np.zeros(len(energy_rows))))
        upper = np.concatenate(([40.0, 500.0, 50.0], np.full(len(energy_rows), 5.0)))
        candidates.append(least_squares(residual, initial, bounds=(lower, upper)))
    best = min(candidates, key=lambda fit: float(np.sum(np.square(fit.fun))))
    amplitude, center, pitch_sigma = (float(value) for value in best.x[:3])
    rms = float(np.sqrt(np.mean(np.square(best.fun))))
    return center, amplitude, pitch_sigma, rms


def boundary_mask(
    observed: np.ndarray,
    energies_ev: np.ndarray,
    beam_center_ev: float,
    beam_pitch_sigma_deg: float,
    mode: str,
) -> np.ndarray:
    """Construct broad 2-D beam exclusion or high-energy control masks."""
    keep = np.ones_like(observed, dtype=bool)
    if mode == "fitted_beam_2d":
        energy_sigma = 0.48 * beam_center_ev / 2.355
        energy_response = np.exp(
            -0.5 * ((energies_ev - beam_center_ev) / energy_sigma) ** 2
        )
        pitch_response = np.exp(
            -0.5 * ((phase_a.PITCH_CENTERS_DEG - 180.0) / beam_pitch_sigma_deg) ** 2
        )
        contaminated = energy_response[:, None] * pitch_response[None, :] >= 0.05
        keep[contaminated] = False
    elif mode == "energy_above_340":
        keep &= energies_ev[:, None] >= 340.0
    else:
        raise ValueError(f"Unknown boundary mask: {mode}")
    return keep


def fit_boundary_only(
    observed: np.ndarray,
    base: np.ndarray,
    u_flat: np.ndarray,
    b_flat: np.ndarray,
    keep: np.ndarray,
    u_spacecraft_v: float,
) -> tuple[float, float, float, int]:
    """Fit a beam-excluded boundary model."""
    objectives, n_valid = phase_a2.objective(base, observed, extra_mask=keep)
    best = int(np.argmin(objectives))
    return (
        float(u_flat[best]),
        float(b_flat[best]),
        float(objectives[best]),
        n_valid,
    )


def write_dataclasses(path: Path, rows: list[object]) -> None:
    dictionaries = [asdict(row) for row in rows]
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(dictionaries[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(dictionaries)


def write_profiles(
    path: Path,
    profiles: list[tuple[str, str, str, float, np.ndarray]],
) -> None:
    fields = (
        "observation_construction",
        "beam_pitch_response",
        "boundary_shape",
        "cone_floor",
        "u_surface_v",
        "objective",
        "delta_objective",
    )
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for observation, beam_mode, boundary_mode, floor, profile in profiles:
            minimum = float(np.min(profile))
            for u_surface, value in zip(phase_a.U_GRID_V, profile, strict=True):
                writer.writerow(
                    {
                        "observation_construction": observation,
                        "beam_pitch_response": beam_mode,
                        "boundary_shape": boundary_mode,
                        "cone_floor": floor,
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

    spectrum = phase_a2.load_sector_spectrum(args.data_path, args.spec_no)
    observations = {
        mode: phase_a2.construct_observation(spectrum, mode)
        for mode in ("bin_then_divide", "divide_then_bin")
    }
    support_values = {
        "bin_then_divide": spectrum.flux,
        "divide_then_bin": phase_a2.sector_ratios(spectrum.flux, spectrum.pitches_deg),
    }

    results: list[PhaseA3Result] = []
    profiles: list[tuple[str, str, str, float, np.ndarray]] = []
    crosschecks: list[PhaseA3Crosscheck] = []
    best_result: PhaseA3Result | None = None
    best_model: np.ndarray | None = None
    best_observed: np.ndarray | None = None

    for observation_mode, observed in observations.items():
        values = support_values[observation_mode]
        beam_center, beam_amp, beam_pitch_sigma, beam_rms = fit_beam_2d(
            observed, spectrum.energies_ev
        )
        bases = {}
        u_flat = b_flat = None
        for boundary_mode, tau in (("sharp", 0.0), ("tau_5deg", 5.0)):
            for floor in (0.10, 0.25):
                base, candidate_u, candidate_b = integrated_boundary_models(
                    spectrum, values, args.u_spacecraft, floor, tau
                )
                bases[(boundary_mode, floor)] = base
                if u_flat is None:
                    u_flat, b_flat = candidate_u, candidate_b
                else:
                    np.testing.assert_allclose(u_flat, candidate_u)
                    np.testing.assert_allclose(b_flat, candidate_b)
        assert u_flat is not None
        assert b_flat is not None

        beams = {
            mode: integrated_beam_profiles(
                spectrum, values, u_flat, args.u_spacecraft, mode
            )
            for mode in ("narrow_top_hat", "broad_gaussian")
        }
        for beam_mode, beam in beams.items():
            for (boundary_mode, floor), base in bases.items():
                result, model, profile = fit_joint_branch(
                    observed,
                    base,
                    beam,
                    u_flat,
                    b_flat,
                    observation_mode,
                    beam_mode,
                    boundary_mode,
                    floor,
                    args.u_spacecraft,
                )
                results.append(result)
                profiles.append(
                    (observation_mode, beam_mode, boundary_mode, floor, profile)
                )
                if (
                    best_result is None
                    or result.objective_reduced < best_result.objective_reduced
                ):
                    best_result = result
                    best_model = model
                    best_observed = observed

        for (boundary_mode, floor), base in bases.items():
            for mask_mode in ("fitted_beam_2d", "energy_above_340"):
                keep = boundary_mask(
                    observed,
                    spectrum.energies_ev,
                    beam_center,
                    beam_pitch_sigma,
                    mask_mode,
                )
                boundary_u, boundary_b, boundary_objective, boundary_n = (
                    fit_boundary_only(
                        observed,
                        base,
                        u_flat,
                        b_flat,
                        keep,
                        args.u_spacecraft,
                    )
                )
                beam_u_relative = -beam_center
                boundary_u_relative = boundary_u - args.u_spacecraft
                crosschecks.append(
                    PhaseA3Crosscheck(
                        observation_construction=observation_mode,
                        beam_center_ev=beam_center,
                        beam_u_relative_v=beam_u_relative,
                        beam_amp=beam_amp,
                        beam_pitch_sigma_deg=beam_pitch_sigma,
                        beam_pitch_fwhm_deg=2.355 * beam_pitch_sigma,
                        beam_rms_residual=beam_rms,
                        boundary_shape=boundary_mode,
                        cone_floor=floor,
                        boundary_mask=mask_mode,
                        boundary_u_surface_v=boundary_u,
                        boundary_u_relative_v=boundary_u_relative,
                        boundary_bs_over_bm=boundary_b,
                        boundary_objective=boundary_objective,
                        boundary_n_valid=boundary_n,
                        beam_boundary_difference_v=(
                            beam_u_relative - boundary_u_relative
                        ),
                    )
                )

    write_dataclasses(args.output_dir / "phase_a3_results.csv", results)
    write_profiles(args.output_dir / "phase_a3_profiles.csv", profiles)
    write_dataclasses(args.output_dir / "phase_a3_crosscheck.csv", crosschecks)
    assert best_result is not None
    assert best_model is not None
    assert best_observed is not None
    phase_a.plot_comparison(
        args.output_dir / "phase_a3_comparison.png",
        best_observed,
        best_model,
        spectrum.energies_ev,
        (
            f"Phase A3 spectrum {args.spec_no}: "
            f"{best_result.observation_construction}, "
            f"{best_result.beam_pitch_response}, {best_result.boundary_shape}, "
            f"floor={best_result.cone_floor:.2f}; "
            f"Urel={best_result.u_relative_v:.0f} V, "
            f"B={best_result.bs_over_bm:.3f}"
        ),
    )
    print(f"Wrote {args.output_dir / 'phase_a3_results.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a3_profiles.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a3_crosscheck.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a3_comparison.png'}")
    print(
        "Best Phase A3 branch: "
        f"{best_result.observation_construction}, "
        f"{best_result.beam_pitch_response}, {best_result.boundary_shape}, "
        f"floor={best_result.cone_floor:.2f}, "
        f"Urel={best_result.u_relative_v:.1f} V, "
        f"B={best_result.bs_over_bm:.3f}, beam={best_result.beam_amp:.1f}, "
        f"objective/dof={best_result.objective_reduced:.4f}"
    )


if __name__ == "__main__":
    main()
