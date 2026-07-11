#!/usr/bin/env python3
"""Run Experiment 004 Phase A4 loss-cone boundary forensics.

Run from the repository root:

    uv run python experiments/experiment-004/phase_a4.py
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.optimize import least_squares

from src import config
from src.losscone.er_data import ERData
from src.losscone.pitch_angle import PitchAngle
from src.utils.thetas import get_thetas


def _load_local_module(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a = _load_local_module("run.py", "experiment_004_phase_a_for_a4")
phase_a2 = _load_local_module("phase_a2.py", "experiment_004_phase_a2_for_a4")

BOOTSTRAP_REPLICATES = 100
BOOTSTRAP_SEED = 20260711
BOUNDARY_RESOLUTION_FLOOR_DEG = 5.625
BOUNDARY_ENERGY_MIN_EV = 300.0
BOUNDARY_ENERGY_MAX_EV = 10000.0


@dataclass(frozen=True)
class SpectrumGeometry:
    spec_no: int
    energies_ev: np.ndarray
    flux: np.ndarray
    pitches_row_field_deg: np.ndarray
    pitches_sweep_field_deg: np.ndarray
    magnetic_field: np.ndarray


@dataclass(frozen=True)
class PanelIdentityResult:
    spec_no: int
    normalization: str
    magnetic_field_mode: str
    n_valid: int
    full_log_rmse: float
    reflected_log_rmse: float
    reflected_median_abs_log_residual: float
    reflected_log_correlation: float
    candidate_beam_peak: float
    published_beam_peak: float


@dataclass(frozen=True)
class BoundaryPoint:
    spec_no: int
    normalization: str
    magnetic_field_mode: str
    sector_weighting: str
    energy_ev: float
    boundary_pitch_deg: float
    boundary_sigma_deg: float
    transition_width_deg: float
    cone_level: float
    outside_contrast: float
    n_sectors: int
    bootstrap_successes: int
    fit_rms_log_residual: float
    nuisance_parameter_edge: bool
    accepted: bool


@dataclass(frozen=True)
class BoundaryFit:
    spec_no: int
    normalization: str
    magnetic_field_mode: str
    sector_weighting: str
    boundary_qc: str
    constraint_mode: str
    u_surface_v: float
    u_relative_v: float
    bs_over_bm: float
    objective: float
    n_boundaries: int
    parameter_grid_edge: bool


@dataclass(frozen=True)
class GeometryResult:
    spec_no: int
    magnetic_magnitude_mean_nt: float
    magnetic_magnitude_relative_range: float
    direction_rms_deg: float
    direction_max_deg: float
    direction_energy_correlation: float
    row_sweep_pitch_max_difference_deg: float


def load_spectra(data_path: Path, spec_nos: tuple[int, ...]) -> list[SpectrumGeometry]:
    """Load adjacent spectra and calculate row-field and sweep-field pitches."""
    er_data = ERData(str(data_path))
    pitch_angle = PitchAngle(er_data)
    spectra = []
    for spec_no in spec_nos:
        chunk = phase_a.chunk_index(er_data, spec_no)
        start = chunk * config.SWEEP_ROWS
        stop = start + config.SWEEP_ROWS
        rows = er_data.data.iloc[start:stop]
        magnetic = rows[config.MAG_COLS].to_numpy(dtype=np.float64)
        mean_field = -np.mean(magnetic, axis=0)
        mean_field /= np.linalg.norm(mean_field)
        coordinates = pitch_angle.cartesian_coords[start:stop]
        dot = np.einsum("ijk,k->ij", coordinates, mean_field)
        sweep_pitches = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
        spectra.append(
            SpectrumGeometry(
                spec_no=spec_no,
                energies_ev=rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64),
                flux=rows[config.FLUX_COLS].to_numpy(dtype=np.float64),
                pitches_row_field_deg=pitch_angle.pitch_angles[start:stop],
                pitches_sweep_field_deg=sweep_pitches,
                magnetic_field=magnetic,
            )
        )
    return spectra


def pitches_for_mode(spectrum: SpectrumGeometry, mode: str) -> np.ndarray:
    if mode == "row_field":
        return spectrum.pitches_row_field_deg
    if mode == "sweep_mean_field":
        return spectrum.pitches_sweep_field_deg
    raise ValueError(f"Unknown magnetic-field mode: {mode}")


def normalized_sectors(
    spectrum: SpectrumGeometry,
    pitches_deg: np.ndarray,
    normalization: str,
) -> np.ndarray:
    """Normalize native sectors without pitch aggregation."""
    if normalization == "conjugate_sector":
        return phase_a2.sector_ratios(spectrum.flux, pitches_deg)
    if normalization == "incident_mean":
        normalized = np.full_like(spectrum.flux, np.nan)
        for energy_index, (flux, pitches) in enumerate(
            zip(spectrum.flux, pitches_deg, strict=True)
        ):
            incident = (
                np.isfinite(flux)
                & (flux > 0.0)
                & np.isfinite(pitches)
                & (pitches < 90.0)
            )
            if incident.any():
                normalized[energy_index] = flux / float(np.mean(flux[incident]))
        return normalized
    raise ValueError(f"Unknown normalization: {normalization}")


def binned_observation(
    spectrum: SpectrumGeometry,
    pitches_deg: np.ndarray,
    normalization: str,
) -> np.ndarray:
    """Build a 16-pitch-bin panel candidate for Figure 5 comparison."""
    if normalization == "incident_mean":
        binned = phase_a.rebin_pitch(spectrum.flux, pitches_deg, spectrum.energies_ev)
        return phase_a.normalize_flux(binned.flux, "incident_mean")
    sectors = normalized_sectors(spectrum, pitches_deg, normalization)
    return phase_a.rebin_pitch(sectors, pitches_deg, spectrum.energies_ev).flux


def sample_published_panel(
    values: np.ndarray,
    white_mask: np.ndarray,
    energies_ev: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the Figure 5 raster at the experiment's energy and pitch centers."""
    sampled = np.full((len(energies_ev), phase_a.N_PITCH_BINS), np.nan)
    valid = np.zeros_like(sampled, dtype=bool)
    for energy_index, energy in enumerate(energies_ev):
        if energy < 10.0 or energy > 10000.0:
            continue
        column = int(
            np.clip(
                np.rint((np.log10(energy) - 1.0) / 3.0 * 44.0 - 0.5),
                0,
                43,
            )
        )
        for pitch_index, pitch in enumerate(phase_a.PITCH_CENTERS_DEG):
            row = int(np.clip(np.rint(pitch / 180.0 * 89.0 - 0.5), 0, 88))
            if not white_mask[row, column]:
                sampled[energy_index, pitch_index] = values[row, column]
                valid[energy_index, pitch_index] = True
    return sampled, valid


def compare_panel(
    observed: np.ndarray,
    published: np.ndarray,
    published_valid: np.ndarray,
    spec_no: int,
    normalization: str,
    magnetic_field_mode: str,
) -> PanelIdentityResult:
    valid = (
        published_valid
        & np.isfinite(observed)
        & (observed > 0.0)
        & np.isfinite(published)
        & (published > 0.0)
    )
    reflected = valid & (phase_a.PITCH_CENTERS_DEG[None, :] >= 90.0)
    log_residual = np.log(observed[valid]) - np.log(published[valid])
    reflected_residual = np.log(observed[reflected]) - np.log(published[reflected])
    reflected_observed = np.log(observed[reflected])
    reflected_published = np.log(published[reflected])
    correlation = float("nan")
    if (
        len(reflected_observed) >= 2
        and np.std(reflected_observed) > 0.0
        and np.std(reflected_published) > 0.0
    ):
        correlation = float(np.corrcoef(reflected_observed, reflected_published)[0, 1])
    beam_region = reflected
    return PanelIdentityResult(
        spec_no=spec_no,
        normalization=normalization,
        magnetic_field_mode=magnetic_field_mode,
        n_valid=int(np.count_nonzero(valid)),
        full_log_rmse=float(np.sqrt(np.mean(np.square(log_residual)))),
        reflected_log_rmse=float(np.sqrt(np.mean(np.square(reflected_residual)))),
        reflected_median_abs_log_residual=float(np.median(np.abs(reflected_residual))),
        reflected_log_correlation=correlation,
        candidate_beam_peak=float(np.nanmax(np.where(beam_region, observed, np.nan))),
        published_beam_peak=float(np.nanmax(np.where(beam_region, published, np.nan))),
    )


def angular_acceptance_weights() -> np.ndarray:
    """Approximate each sector's solid-angle support from its latitude band."""
    theta = get_thetas()
    bins = np.array([config.BINS_BY_LATITUDE[float(value)] for value in theta])
    weights = np.cos(np.deg2rad(theta)) / bins
    return weights / np.mean(weights)


def fit_transition(
    pitches_deg: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, float] | None:
    """Fit a floor-free reflected-side logistic transition in log space."""
    valid = (
        np.isfinite(pitches_deg)
        & np.isfinite(values)
        & (values > 0.0)
        & np.isfinite(weights)
        & (weights > 0.0)
        & (pitches_deg >= 90.0)
    )
    pitches = pitches_deg[valid]
    ratios = values[valid]
    fit_weights = weights[valid]
    if len(pitches) < 10:
        return None
    fit_weights = fit_weights / np.mean(fit_weights)

    def residual(parameters: np.ndarray) -> np.ndarray:
        cone_level, contrast, boundary, width = parameters
        transition = 1.0 / (
            1.0 + np.exp(-np.clip((boundary - pitches) / width, -60, 60))
        )
        model = cone_level + contrast * transition
        return np.sqrt(fit_weights) * (np.log(model) - np.log(ratios))

    starts = (
        [initial]
        if initial is not None
        else [
            np.array([0.10, 0.60, boundary, 8.0])
            for boundary in (105.0, 125.0, 145.0, 165.0)
        ]
    )
    fits = [
        least_squares(
            residual,
            x0=start,
            bounds=(
                np.array([0.01, 0.01, 90.0, 0.5]),
                np.array([2.0, 3.0, 180.0, 45.0]),
            ),
            loss="soft_l1",
            f_scale=0.3,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: float(np.sum(np.square(fit.fun))))
    rms = float(np.sqrt(np.mean(np.square(best.fun))))
    return np.asarray(best.x), rms


def bootstrap_boundary(
    pitches_deg: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    parameters: np.ndarray,
    rng: np.random.Generator,
) -> tuple[float, int]:
    valid = (
        np.isfinite(pitches_deg)
        & np.isfinite(values)
        & (values > 0.0)
        & np.isfinite(weights)
        & (weights > 0.0)
        & (pitches_deg >= 90.0)
    )
    pitches = pitches_deg[valid]
    ratios = values[valid]
    fit_weights = weights[valid]
    boundaries = []
    for _ in range(BOOTSTRAP_REPLICATES):
        indices = rng.integers(0, len(pitches), size=len(pitches))
        fitted = fit_transition(
            pitches[indices],
            ratios[indices],
            fit_weights[indices],
            initial=parameters,
        )
        if fitted is not None:
            boundary = float(fitted[0][2])
            if 90.5 < boundary < 179.5:
                boundaries.append(boundary)
    if len(boundaries) < BOOTSTRAP_REPLICATES // 2:
        return float("nan"), len(boundaries)
    sigma = float(np.std(boundaries, ddof=1))
    return max(sigma, BOUNDARY_RESOLUTION_FLOOR_DEG), len(boundaries)


def extract_boundaries(
    spectrum: SpectrumGeometry,
    pitches_deg: np.ndarray,
    normalization: str,
    magnetic_field_mode: str,
    sector_weighting: str,
    rng: np.random.Generator,
) -> list[BoundaryPoint]:
    normalized = normalized_sectors(spectrum, pitches_deg, normalization)
    if sector_weighting == "equal":
        weights = np.ones(normalized.shape[1])
    elif sector_weighting == "angular_acceptance":
        weights = angular_acceptance_weights()
    else:
        raise ValueError(f"Unknown sector weighting: {sector_weighting}")

    points = []
    for energy_index, energy in enumerate(spectrum.energies_ev):
        if energy < BOUNDARY_ENERGY_MIN_EV or energy > BOUNDARY_ENERGY_MAX_EV:
            continue
        fitted = fit_transition(
            pitches_deg[energy_index], normalized[energy_index], weights
        )
        if fitted is None:
            continue
        parameters, rms = fitted
        sigma, successes = bootstrap_boundary(
            pitches_deg[energy_index],
            normalized[energy_index],
            weights,
            parameters,
            rng,
        )
        cone_level, contrast, boundary, width = (float(value) for value in parameters)
        nuisance_edge = (
            cone_level <= 0.0101
            or cone_level >= 1.999
            or contrast <= 0.0101
            or contrast >= 2.999
            or width <= 0.501
            or width >= 44.99
        )
        accepted = (
            np.isfinite(sigma)
            and contrast >= 0.10
            and 91.0 < boundary < 179.0
            and width < 44.0
        )
        points.append(
            BoundaryPoint(
                spec_no=spectrum.spec_no,
                normalization=normalization,
                magnetic_field_mode=magnetic_field_mode,
                sector_weighting=sector_weighting,
                energy_ev=float(energy),
                boundary_pitch_deg=boundary,
                boundary_sigma_deg=sigma,
                transition_width_deg=width,
                cone_level=cone_level,
                outside_contrast=contrast,
                n_sectors=int(
                    np.count_nonzero(
                        np.isfinite(normalized[energy_index])
                        & (normalized[energy_index] > 0.0)
                        & (pitches_deg[energy_index] >= 90.0)
                    )
                ),
                bootstrap_successes=successes,
                fit_rms_log_residual=rms,
                nuisance_parameter_edge=nuisance_edge,
                accepted=accepted,
            )
        )
    return points


def predicted_boundary(
    energies_ev: np.ndarray,
    u_surface_v: np.ndarray,
    b_ratio: np.ndarray,
    u_spacecraft_v: float,
) -> np.ndarray:
    corrected = np.maximum(energies_ev[None, :] - u_spacecraft_v, 1e-12)
    sin2 = b_ratio[:, None] * (1.0 + u_surface_v[:, None] / corrected)
    alpha = np.degrees(np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0))))
    return 180.0 - alpha


def fit_physical_boundary(
    points: list[BoundaryPoint],
    u_spacecraft_v: float,
    boundary_qc: str = "preregistered",
) -> tuple[list[BoundaryFit], np.ndarray]:
    if boundary_qc == "preregistered":
        accepted = [point for point in points if point.accepted]
    elif boundary_qc == "no_nuisance_edges":
        accepted = [
            point
            for point in points
            if point.accepted and not point.nuisance_parameter_edge
        ]
    else:
        raise ValueError(f"Unknown boundary QC mode: {boundary_qc}")
    if len(accepted) < 3:
        return [], np.full(len(phase_a.U_GRID_V), np.nan)
    energies = np.array([point.energy_ev for point in accepted])
    boundaries = np.array([point.boundary_pitch_deg for point in accepted])
    sigma = np.array([point.boundary_sigma_deg for point in accepted])

    uu, bb = np.meshgrid(phase_a.U_GRID_V, phase_a.B_GRID, indexing="ij")
    u_flat = uu.ravel()
    b_flat = bb.ravel()
    predicted = predicted_boundary(energies, u_flat, b_flat, u_spacecraft_v)
    objective = np.sum(np.square((predicted - boundaries[None, :]) / sigma), axis=1)
    free_best = int(np.argmin(objective))
    profile = np.min(
        objective.reshape(len(phase_a.U_GRID_V), len(phase_a.B_GRID)), axis=1
    )

    high = energies >= 1000.0
    alpha_high = np.deg2rad(180.0 - boundaries[high])
    high_weights = 1.0 / np.square(sigma[high])
    if high.any():
        asymptotic_b = float(
            np.average(np.square(np.sin(alpha_high)), weights=high_weights)
        )
    else:
        asymptotic_b = 0.975
    asymptotic_b = float(np.clip(asymptotic_b, phase_a.B_GRID[0], phase_a.B_GRID[-1]))

    fits = []
    modes = (("free", None), ("published_b", 0.975), ("high_energy_b", asymptotic_b))
    first = accepted[0]
    for mode, fixed_b in modes:
        if fixed_b is None:
            best_u = float(u_flat[free_best])
            best_b = float(b_flat[free_best])
            best_objective = float(objective[free_best])
        else:
            predicted_fixed = predicted_boundary(
                energies,
                phase_a.U_GRID_V,
                np.full(len(phase_a.U_GRID_V), fixed_b),
                u_spacecraft_v,
            )
            fixed_objective = np.sum(
                np.square((predicted_fixed - boundaries[None, :]) / sigma), axis=1
            )
            best_index = int(np.argmin(fixed_objective))
            best_u = float(phase_a.U_GRID_V[best_index])
            best_b = fixed_b
            best_objective = float(fixed_objective[best_index])
        fits.append(
            BoundaryFit(
                spec_no=first.spec_no,
                normalization=first.normalization,
                magnetic_field_mode=first.magnetic_field_mode,
                sector_weighting=first.sector_weighting,
                boundary_qc=boundary_qc,
                constraint_mode=mode,
                u_surface_v=best_u,
                u_relative_v=best_u - u_spacecraft_v,
                bs_over_bm=best_b,
                objective=best_objective,
                n_boundaries=len(accepted),
                parameter_grid_edge=(
                    best_u in {float(phase_a.U_GRID_V[0]), float(phase_a.U_GRID_V[-1])}
                    or best_b in {float(phase_a.B_GRID[0]), float(phase_a.B_GRID[-1])}
                ),
            )
        )
    return fits, profile


def geometry_result(spectrum: SpectrumGeometry) -> GeometryResult:
    magnetic = spectrum.magnetic_field
    magnitude = np.linalg.norm(magnetic, axis=1)
    unit = magnetic / magnitude[:, None]
    mean_unit = np.mean(unit, axis=0)
    mean_unit /= np.linalg.norm(mean_unit)
    angles = np.degrees(
        np.arccos(np.clip(np.einsum("ij,j->i", unit, mean_unit), -1.0, 1.0))
    )
    correlation = float("nan")
    if np.std(angles) > 0.0:
        correlation = float(np.corrcoef(np.log10(spectrum.energies_ev), angles)[0, 1])
    return GeometryResult(
        spec_no=spectrum.spec_no,
        magnetic_magnitude_mean_nt=float(np.mean(magnitude)),
        magnetic_magnitude_relative_range=float(
            (np.max(magnitude) - np.min(magnitude)) / np.mean(magnitude)
        ),
        direction_rms_deg=float(np.sqrt(np.mean(np.square(angles)))),
        direction_max_deg=float(np.max(angles)),
        direction_energy_correlation=correlation,
        row_sweep_pitch_max_difference_deg=float(
            np.max(
                np.abs(
                    spectrum.pitches_row_field_deg - spectrum.pitches_sweep_field_deg
                )
            )
        ),
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
    profiles: list[tuple[int, str, str, str, str, np.ndarray]],
) -> None:
    fields = (
        "spec_no",
        "normalization",
        "magnetic_field_mode",
        "sector_weighting",
        "boundary_qc",
        "u_surface_v",
        "objective",
        "delta_objective",
    )
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for (
            spec_no,
            normalization,
            field_mode,
            weighting,
            boundary_qc,
            profile,
        ) in profiles:
            minimum = float(np.nanmin(profile))
            for u_surface, value in zip(phase_a.U_GRID_V, profile, strict=True):
                writer.writerow(
                    {
                        "spec_no": spec_no,
                        "normalization": normalization,
                        "magnetic_field_mode": field_mode,
                        "sector_weighting": weighting,
                        "boundary_qc": boundary_qc,
                        "u_surface_v": u_surface,
                        "objective": float(value),
                        "delta_objective": float(value - minimum),
                    }
                )


def plot_boundaries(
    path: Path,
    spectra: list[SpectrumGeometry],
    points: list[BoundaryPoint],
    fits: list[BoundaryFit],
    selected_normalization: str,
    selected_field_mode: str,
) -> None:
    figure, axes = plt.subplots(
        1, len(spectra), figsize=(14, 4.6), constrained_layout=True
    )
    for axis, spectrum in zip(axes, spectra, strict=True):
        selected = [
            point
            for point in points
            if point.spec_no == spectrum.spec_no
            and point.normalization == selected_normalization
            and point.magnetic_field_mode == selected_field_mode
            and point.sector_weighting == "equal"
            and point.accepted
        ]
        if selected:
            energies = np.array([point.energy_ev for point in selected])
            boundaries = np.array([point.boundary_pitch_deg for point in selected])
            errors = np.array([point.boundary_sigma_deg for point in selected])
            axis.errorbar(energies, boundaries, yerr=errors, fmt="o", label="Extracted")
        free = next(
            (
                fit
                for fit in fits
                if fit.spec_no == spectrum.spec_no
                and fit.normalization == selected_normalization
                and fit.magnetic_field_mode == selected_field_mode
                and fit.sector_weighting == "equal"
                and fit.boundary_qc == "preregistered"
                and fit.constraint_mode == "free"
            ),
            None,
        )
        if free is not None:
            energy_grid = np.geomspace(300.0, 10000.0, 200)
            curve = predicted_boundary(
                energy_grid,
                np.array([free.u_surface_v]),
                np.array([free.bs_over_bm]),
                11.0,
            )[0]
            axis.plot(energy_grid, curve, label=f"Free fit: {free.u_relative_v:.0f} V")
        axis.set_xscale("log")
        axis.set_xlim(300.0, 10000.0)
        axis.set_ylim(85.0, 185.0)
        axis.set_title(f"Spectrum {spectrum.spec_no}")
        axis.set_xlabel("Energy [eV]")
        axis.legend(fontsize=8)
    axes[0].set_ylabel("Reflected-side boundary pitch [deg]")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_panel_comparison(
    path: Path,
    published_values: np.ndarray,
    white_mask: np.ndarray,
    spectra: list[SpectrumGeometry],
    observations: dict[tuple[int, str, str], np.ndarray],
    normalization: str,
    field_mode: str,
) -> None:
    """Compare the published raster with all three candidate spectra."""
    figure, axes = plt.subplots(
        1, len(spectra) + 1, figsize=(16, 4.6), constrained_layout=True
    )
    published = np.where(white_mask, np.nan, published_values)
    artist = axes[0].imshow(
        np.log10(published),
        origin="lower",
        aspect="auto",
        extent=(1.0, 4.0, 0.0, 180.0),
        vmin=-1.0,
        vmax=1.0,
        cmap="viridis",
    )
    axes[0].set_title("Published Figure 5")
    for axis, spectrum in zip(axes[1:], spectra, strict=True):
        observed = observations[(spectrum.spec_no, normalization, field_mode)]
        energy_mask = (spectrum.energies_ev >= 10.0) & (spectrum.energies_ev <= 10000.0)
        energies = spectrum.energies_ev[energy_mask]
        values = observed[energy_mask]
        order = np.argsort(energies)
        axis.imshow(
            np.log10(np.maximum(values[order].T, 0.1)),
            origin="lower",
            aspect="auto",
            extent=(
                float(np.log10(energies[order][0])),
                float(np.log10(energies[order][-1])),
                0.0,
                180.0,
            ),
            vmin=-1.0,
            vmax=1.0,
            cmap="viridis",
        )
        axis.set_title(f"Spectrum {spectrum.spec_no}")
    for axis in axes:
        axis.set_xlim(1.0, 4.0)
        axis.set_xticks((1.0, 2.0, 3.0, 4.0), labels=("10", "100", "1000", "10000"))
        axis.set_xlabel("Energy [eV]")
    axes[0].set_ylabel("Pitch angle [deg]")
    figure.colorbar(artist, ax=axes, label="log10 normalized flux")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/1999/091_120APR/3D990429.TAB"),
    )
    parser.add_argument("--spec-nos", type=int, nargs="+", default=[652, 653, 654])
    parser.add_argument("--u-spacecraft", type=float, default=11.0)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("experiments/experiment-004")
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    spectra = load_spectra(args.data_path, tuple(args.spec_nos))
    provenance = args.output_dir / "provenance"
    published_values = np.load(provenance / "figure5-data-values.npy")
    published_rgb = np.asarray(
        Image.open(provenance / "figure5-data-native.png").convert("RGB")
    )
    white_mask = np.all(published_rgb > 245, axis=2)

    panel_results = []
    observations = {}
    for spectrum in spectra:
        published, published_valid = sample_published_panel(
            published_values, white_mask, spectrum.energies_ev
        )
        for normalization in ("incident_mean", "conjugate_sector"):
            for field_mode in ("row_field", "sweep_mean_field"):
                pitches = pitches_for_mode(spectrum, field_mode)
                observed = binned_observation(spectrum, pitches, normalization)
                observations[(spectrum.spec_no, normalization, field_mode)] = observed
                panel_results.append(
                    compare_panel(
                        observed,
                        published,
                        published_valid,
                        spectrum.spec_no,
                        normalization,
                        field_mode,
                    )
                )

    boundary_points = []
    for spectrum in spectra:
        for normalization_index, normalization in enumerate(
            ("incident_mean", "conjugate_sector")
        ):
            for field_mode in ("row_field", "sweep_mean_field"):
                pitches = pitches_for_mode(spectrum, field_mode)
                for weighting_index, weighting in enumerate(
                    ("equal", "angular_acceptance")
                ):
                    rng = np.random.default_rng(
                        BOOTSTRAP_SEED
                        + spectrum.spec_no * 100
                        + normalization_index * 10
                        + weighting_index
                    )
                    boundary_points.extend(
                        extract_boundaries(
                            spectrum,
                            pitches,
                            normalization,
                            field_mode,
                            weighting,
                            rng,
                        )
                    )

    boundary_fits = []
    profiles = []
    group_keys = sorted(
        {
            (
                point.spec_no,
                point.normalization,
                point.magnetic_field_mode,
                point.sector_weighting,
            )
            for point in boundary_points
        }
    )
    for key in group_keys:
        grouped = [
            point
            for point in boundary_points
            if (
                point.spec_no,
                point.normalization,
                point.magnetic_field_mode,
                point.sector_weighting,
            )
            == key
        ]
        for boundary_qc in ("preregistered", "no_nuisance_edges"):
            fits, profile = fit_physical_boundary(
                grouped, args.u_spacecraft, boundary_qc=boundary_qc
            )
            boundary_fits.extend(fits)
            profiles.append((*key, boundary_qc, profile))

    geometry = [geometry_result(spectrum) for spectrum in spectra]
    write_dataclasses(args.output_dir / "phase_a4_panel_identity.csv", panel_results)
    write_dataclasses(args.output_dir / "phase_a4_boundary_points.csv", boundary_points)
    write_dataclasses(args.output_dir / "phase_a4_fits.csv", boundary_fits)
    write_profiles(args.output_dir / "phase_a4_profiles.csv", profiles)
    write_dataclasses(args.output_dir / "phase_a4_geometry.csv", geometry)

    selected = min(panel_results, key=lambda result: result.reflected_log_rmse)
    plot_panel_comparison(
        args.output_dir / "phase_a4_panel_comparison.png",
        published_values,
        white_mask,
        spectra,
        observations,
        selected.normalization,
        selected.magnetic_field_mode,
    )
    plot_boundaries(
        args.output_dir / "phase_a4_boundaries.png",
        spectra,
        boundary_points,
        boundary_fits,
        selected.normalization,
        selected.magnetic_field_mode,
    )
    print(f"Wrote {args.output_dir / 'phase_a4_panel_identity.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a4_boundary_points.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a4_fits.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a4_profiles.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a4_geometry.csv'}")
    print(f"Wrote {args.output_dir / 'phase_a4_panel_comparison.png'}")
    print(f"Wrote {args.output_dir / 'phase_a4_boundaries.png'}")
    print(
        "Closest Figure 5 candidate: "
        f"spectrum {selected.spec_no}, {selected.normalization}, "
        f"{selected.magnetic_field_mode}, "
        f"reflected log RMSE={selected.reflected_log_rmse:.4f}, "
        f"beam peak={selected.candidate_beam_peak:.2f}"
    )


if __name__ == "__main__":
    main()
