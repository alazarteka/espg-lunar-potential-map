#!/usr/bin/env python3
"""Response-aware upward-beam estimator for Experiment 005.

This module is experiment-local. It operates on native LP ER sector support and
does not modify the production loss-cone fitter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from src import config
from src.losscone.er_data import ERData
from src.losscone.pitch_angle import PitchAngle


@dataclass(frozen=True)
class Spectrum:
    """One 15-row LP ER sweep on native directional support."""

    spec_no: int
    utc_start: str
    utc_end: str
    energies_ev: np.ndarray
    flux: np.ndarray
    pitches_deg: np.ndarray
    magnetic_field: np.ndarray


@dataclass(frozen=True)
class BeamObservation:
    """Flattened reflected-side normalized samples eligible for beam fitting."""

    normalization: str
    energies_ev: np.ndarray
    energy_indices: np.ndarray
    row_ids: np.ndarray
    sample_energies_ev: np.ndarray
    pitches_deg: np.ndarray
    values: np.ndarray
    incident_counts: np.ndarray
    reflected_counts: np.ndarray


@dataclass(frozen=True)
class BeamFit:
    """One fitted beam and the diagnostics needed for resampling."""

    success: bool
    center_ev: float
    amplitude: float
    pitch_sigma_deg: float
    objective: float
    rms_residual: float
    n_valid: int
    n_energy_rows: int
    multistart_center_spread_ev: float
    amplitude_at_bound: bool
    pitch_sigma_at_bound: bool
    parameters: np.ndarray
    model_values: np.ndarray
    residuals: np.ndarray


@dataclass(frozen=True)
class BootstrapSummary:
    """Empirical uncertainty summary from within-energy residual resampling."""

    requested: int
    successes: int
    center_q025_ev: float
    center_q16_ev: float
    center_median_ev: float
    center_q84_ev: float
    center_q975_ev: float
    amplitude_median: float
    amplitude_std: float


@dataclass(frozen=True)
class BeamAssessment:
    """Primary and sensitivity fits with the frozen Experiment 005 QC result."""

    primary: BeamFit
    sensitivity: BeamFit
    bootstrap: BootstrapSummary
    positive_excess_rows: int
    leave_one_energy_out_max_shift_ev: float
    normalization_difference_ev: float
    response_fwhm_ev: float
    beam_valid: bool
    reason_codes: tuple[str, ...]


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen Experiment 005 configuration."""
    if path is None:
        path = Path(__file__).with_name("configuration.json")
    return json.loads(path.read_text(encoding="ascii"))


def load_spectra(data_path: Path, spec_nos: range | list[int]) -> dict[int, Spectrum]:
    """Load requested structurally valid spectra in one archive pass."""
    er_data = ERData(str(data_path))
    if er_data.data.empty:
        return {}
    pitch_angle = PitchAngle(er_data)
    requested = set(spec_nos)
    spectra: dict[int, Spectrum] = {}
    grouped = er_data.data.groupby(config.SPEC_NO_COLUMN, sort=False)
    for spec_no_raw, rows in grouped:
        spec_no = int(spec_no_raw)
        if spec_no not in requested or len(rows) != config.SWEEP_ROWS:
            continue
        indices = rows.index.to_numpy(dtype=np.int64)
        spectra[spec_no] = Spectrum(
            spec_no=spec_no,
            utc_start=str(rows.iloc[0][config.UTC_COLUMN]),
            utc_end=str(rows.iloc[-1][config.UTC_COLUMN]),
            energies_ev=rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64),
            flux=rows[config.FLUX_COLS].to_numpy(dtype=np.float64),
            pitches_deg=pitch_angle.pitch_angles[indices],
            magnetic_field=rows[config.MAG_COLS].to_numpy(dtype=np.float64),
        )
    return spectra


def normalize_sectors(
    flux: np.ndarray, pitches_deg: np.ndarray, normalization: str
) -> np.ndarray:
    """Normalize native sectors by incident mean or nearest conjugate sector."""
    normalized = np.full_like(flux, np.nan, dtype=np.float64)
    for row_index, (row_flux, row_pitch) in enumerate(
        zip(flux, pitches_deg, strict=True)
    ):
        valid = np.isfinite(row_flux) & (row_flux > 0.0) & np.isfinite(row_pitch)
        incident = np.flatnonzero(valid & (row_pitch < 90.0))
        reflected = np.flatnonzero(valid & (row_pitch >= 90.0))
        if incident.size == 0:
            continue
        if normalization == "incident_hemisphere_mean":
            normalized[row_index, valid] = row_flux[valid] / float(
                np.mean(row_flux[incident])
            )
        elif normalization == "conjugate_sector":
            normalized[row_index, incident] = 1.0
            for reflected_index in reflected:
                target = 180.0 - row_pitch[reflected_index]
                nearest = incident[np.argmin(np.abs(row_pitch[incident] - target))]
                normalized[row_index, reflected_index] = (
                    row_flux[reflected_index] / row_flux[nearest]
                )
        else:
            raise ValueError(f"Unknown normalization: {normalization}")
    return normalized


def build_observation(
    spectrum: Spectrum, normalization: str, beam_config: dict[str, Any]
) -> BeamObservation:
    """Build the reflected native-sector beam-fitting observation."""
    normalized = normalize_sectors(spectrum.flux, spectrum.pitches_deg, normalization)
    energy_min, energy_max = beam_config["fit_domain_energy_ev"]
    pitch_min, pitch_max = beam_config["fit_domain_pitch_deg"]
    incident_counts = np.count_nonzero(
        np.isfinite(spectrum.flux)
        & (spectrum.flux > 0.0)
        & np.isfinite(spectrum.pitches_deg)
        & (spectrum.pitches_deg < 90.0),
        axis=1,
    )
    reflected_counts = np.count_nonzero(
        np.isfinite(normalized)
        & (normalized > 0.0)
        & np.isfinite(spectrum.pitches_deg)
        & (spectrum.pitches_deg >= pitch_min)
        & (spectrum.pitches_deg <= pitch_max),
        axis=1,
    )
    eligible = (
        (spectrum.energies_ev >= energy_min)
        & (spectrum.energies_ev <= energy_max)
        & (incident_counts >= int(beam_config["minimum_incident_sectors_per_energy"]))
        & (reflected_counts >= int(beam_config["minimum_reflected_sectors_per_energy"]))
    )
    energy_indices = np.flatnonzero(eligible)
    row_ids: list[np.ndarray] = []
    sample_energies: list[np.ndarray] = []
    sample_pitches: list[np.ndarray] = []
    sample_values: list[np.ndarray] = []
    for row_id, energy_index in enumerate(energy_indices):
        valid = (
            np.isfinite(normalized[energy_index])
            & (normalized[energy_index] > 0.0)
            & np.isfinite(spectrum.pitches_deg[energy_index])
            & (spectrum.pitches_deg[energy_index] >= pitch_min)
            & (spectrum.pitches_deg[energy_index] <= pitch_max)
        )
        count = int(np.count_nonzero(valid))
        row_ids.append(np.full(count, row_id, dtype=np.int64))
        sample_energies.append(
            np.full(count, spectrum.energies_ev[energy_index], dtype=np.float64)
        )
        sample_pitches.append(spectrum.pitches_deg[energy_index, valid])
        sample_values.append(normalized[energy_index, valid])

    def concatenate(arrays: list[np.ndarray], dtype: np.dtype) -> np.ndarray:
        if arrays:
            return np.concatenate(arrays).astype(dtype, copy=False)
        return np.array([], dtype=dtype)

    return BeamObservation(
        normalization=normalization,
        energies_ev=spectrum.energies_ev[energy_indices],
        energy_indices=energy_indices,
        row_ids=concatenate(row_ids, np.int64),
        sample_energies_ev=concatenate(sample_energies, np.float64),
        pitches_deg=concatenate(sample_pitches, np.float64),
        values=concatenate(sample_values, np.float64),
        incident_counts=incident_counts[energy_indices],
        reflected_counts=reflected_counts[energy_indices],
    )


def beam_shape(
    sample_energies_ev: np.ndarray,
    pitches_deg: np.ndarray,
    center_ev: float,
    pitch_sigma_deg: float,
    energy_fwhm_fraction: float,
) -> np.ndarray:
    """Evaluate the separable instrument-response beam shape."""
    energy_sigma = max(energy_fwhm_fraction * center_ev / 2.355, 1e-12)
    energy = np.exp(-0.5 * np.square((sample_energies_ev - center_ev) / energy_sigma))
    pitch = np.exp(-0.5 * np.square((pitches_deg - 180.0) / pitch_sigma_deg))
    return energy * pitch


def _initial_baselines(observation: BeamObservation) -> np.ndarray:
    baselines = np.empty(len(observation.energies_ev), dtype=np.float64)
    for row_id in range(len(baselines)):
        selected = observation.values[observation.row_ids == row_id]
        baselines[row_id] = float(np.median(selected)) if selected.size else 1.0
    return baselines


def _detected_center(observation: BeamObservation) -> float:
    scores = []
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        pitches = observation.pitches_deg[selected]
        values = observation.values[selected]
        aligned = values[pitches >= 157.5]
        shoulder = values[(pitches >= 112.5) & (pitches < 146.25)]
        if aligned.size and shoulder.size:
            scores.append(float(np.mean(aligned) - np.mean(shoulder)))
        else:
            scores.append(float("-inf"))
    if not scores or not np.isfinite(scores).any():
        return float(np.median(observation.energies_ev))
    return float(observation.energies_ev[int(np.argmax(scores))])


def _start_parameters(
    observation: BeamObservation, beam_config: dict[str, Any]
) -> list[np.ndarray]:
    count = int(beam_config["multistart_count"])
    center_low, center_high = beam_config["center_energy_ev_bounds"]
    sigma_low, sigma_high = beam_config["pitch_sigma_deg_bounds"]
    baselines = _initial_baselines(observation)
    amplitude_high = float(beam_config["amplitude_bounds"][1])
    aligned = observation.values[observation.pitches_deg >= 157.5]
    amplitude = float(np.percentile(aligned, 90) - np.median(observation.values))
    amplitude = float(np.clip(amplitude, 0.5, 0.5 * amplitude_high))
    sigma_starts = np.linspace(sigma_low + 5.0, sigma_high - 5.0, 3)
    n_centers = max(1, int(np.ceil(count / len(sigma_starts))))
    center_starts = np.geomspace(
        max(center_low * 1.5, 30.0), min(center_high * 0.75, 1500.0), n_centers
    )
    starts = [
        np.concatenate(([amplitude, center, sigma], baselines))
        for center in center_starts
        for sigma in sigma_starts
    ][:count]
    detected = _detected_center(observation)
    starts[0] = np.concatenate(
        ([amplitude, detected, float(np.median(sigma_starts))], baselines)
    )
    return starts


def fit_beam(
    observation: BeamObservation,
    beam_config: dict[str, Any],
    starts: list[np.ndarray] | None = None,
) -> BeamFit:
    """Fit a response-convolved beam with one baseline per energy row."""

    def failed_fit() -> BeamFit:
        empty = np.array([], dtype=np.float64)
        return BeamFit(
            False,
            float("nan"),
            float("nan"),
            float("nan"),
            float("inf"),
            float("nan"),
            int(observation.values.size),
            len(observation.energies_ev),
            float("nan"),
            False,
            False,
            empty,
            empty,
            empty,
        )

    n_rows = len(observation.energies_ev)
    if n_rows < 3 or observation.values.size == 0:
        return failed_fit()
    fwhm_fraction = float(beam_config["energy_response_fwhm_fraction"])

    def residual(parameters: np.ndarray) -> np.ndarray:
        amplitude, center, pitch_sigma = parameters[:3]
        baselines = parameters[3:]
        shape = beam_shape(
            observation.sample_energies_ev,
            observation.pitches_deg,
            center,
            pitch_sigma,
            fwhm_fraction,
        )
        model = baselines[observation.row_ids] + amplitude * shape
        return model - observation.values

    if starts is None:
        starts = _start_parameters(observation, beam_config)
    amplitude_bounds = beam_config["amplitude_bounds"]
    center_bounds = beam_config["center_energy_ev_bounds"]
    sigma_bounds = beam_config["pitch_sigma_deg_bounds"]
    baseline_bounds = beam_config["per_energy_baseline_bounds"]
    lower = np.concatenate(
        (
            [amplitude_bounds[0], center_bounds[0], sigma_bounds[0]],
            np.full(n_rows, baseline_bounds[0]),
        )
    )
    upper = np.concatenate(
        (
            [amplitude_bounds[1], center_bounds[1], sigma_bounds[1]],
            np.full(n_rows, baseline_bounds[1]),
        )
    )
    fits = []
    for start in starts:
        clipped_start = np.clip(start, lower + 1e-8, upper - 1e-8)
        fitted = least_squares(
            residual,
            clipped_start,
            bounds=(lower, upper),
            max_nfev=int(beam_config["optimizer_max_nfev"]),
        )
        if fitted.success and np.all(np.isfinite(fitted.x)):
            fits.append(fitted)
    if not fits:
        return failed_fit()
    objectives = np.array([float(np.sum(np.square(fit.fun))) for fit in fits])
    best_index = int(np.argmin(objectives))
    best = fits[best_index]
    best_objective = float(objectives[best_index])
    near_fraction = float(beam_config["near_best_objective_fraction"])
    near = objectives <= best_objective * (1.0 + near_fraction) + 1e-12
    centers = np.array([fit.x[1] for fit in fits])[near]
    center_spread = float(np.max(centers) - np.min(centers))
    amplitude, center, pitch_sigma = (float(value) for value in best.x[:3])
    model_values = observation.values + best.fun
    amplitude_at_bound = (
        amplitude <= float(amplitude_bounds[0]) + 1e-4
        or amplitude >= float(amplitude_bounds[1]) - 1e-3
    )
    pitch_sigma_at_bound = (
        pitch_sigma <= float(sigma_bounds[0]) + 1e-3
        or pitch_sigma >= float(sigma_bounds[1]) - 1e-3
    )
    return BeamFit(
        success=True,
        center_ev=center,
        amplitude=amplitude,
        pitch_sigma_deg=pitch_sigma,
        objective=best_objective,
        rms_residual=float(np.sqrt(np.mean(np.square(best.fun)))),
        n_valid=int(observation.values.size),
        n_energy_rows=n_rows,
        multistart_center_spread_ev=center_spread,
        amplitude_at_bound=amplitude_at_bound,
        pitch_sigma_at_bound=pitch_sigma_at_bound,
        parameters=np.asarray(best.x),
        model_values=np.asarray(model_values),
        residuals=np.asarray(-best.fun),
    )


def bootstrap_beam(
    observation: BeamObservation,
    fitted: BeamFit,
    beam_config: dict[str, Any],
    rng: np.random.Generator,
    replicates: int | None = None,
) -> BootstrapSummary:
    """Estimate beam uncertainty by residual resampling within energy rows."""
    requested = int(
        beam_config["bootstrap_replicates"] if replicates is None else replicates
    )
    centers: list[float] = []
    amplitudes: list[float] = []
    if not fitted.success:
        return BootstrapSummary(
            requested,
            0,
            *([float("nan")] * 7),
        )
    for _ in range(requested):
        sampled_residuals = np.empty_like(fitted.residuals)
        for row_id in range(len(observation.energies_ev)):
            selected = np.flatnonzero(observation.row_ids == row_id)
            sampled_residuals[selected] = rng.choice(
                fitted.residuals[selected], size=len(selected), replace=True
            )
        values = np.maximum(fitted.model_values + sampled_residuals, 1e-8)
        sampled_observation = replace(observation, values=values)
        sampled_fit = fit_beam(
            sampled_observation, beam_config, starts=[fitted.parameters]
        )
        if sampled_fit.success:
            centers.append(sampled_fit.center_ev)
            amplitudes.append(sampled_fit.amplitude)
    if not centers:
        return BootstrapSummary(
            requested,
            0,
            *([float("nan")] * 7),
        )
    center_array = np.asarray(centers)
    amplitude_array = np.asarray(amplitudes)
    quantiles = np.quantile(center_array, [0.025, 0.16, 0.5, 0.84, 0.975])
    return BootstrapSummary(
        requested=requested,
        successes=len(centers),
        center_q025_ev=float(quantiles[0]),
        center_q16_ev=float(quantiles[1]),
        center_median_ev=float(quantiles[2]),
        center_q84_ev=float(quantiles[3]),
        center_q975_ev=float(quantiles[4]),
        amplitude_median=float(np.median(amplitude_array)),
        amplitude_std=float(np.std(amplitude_array, ddof=1)),
    )


def count_positive_excess_rows(observation: BeamObservation, fitted: BeamFit) -> int:
    """Count energy rows with positive measured field-aligned excess."""
    if not fitted.success:
        return 0
    count = 0
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        pitches = observation.pitches_deg[selected]
        values = observation.values[selected]
        aligned = values[pitches >= 157.5]
        shoulder = values[(pitches >= 112.5) & (pitches < 146.25)]
        if aligned.size and shoulder.size and np.mean(aligned) > np.mean(shoulder):
            count += 1
    return count


def leave_one_energy_out_shift(
    observation: BeamObservation, fitted: BeamFit, beam_config: dict[str, Any]
) -> float:
    """Return the largest center shift after dropping one energy row."""
    if not fitted.success or len(observation.energies_ev) < 4:
        return float("inf")
    shifts = []
    for dropped_row in range(len(observation.energies_ev)):
        keep = observation.row_ids != dropped_row
        kept_energy_indices = np.flatnonzero(
            np.arange(len(observation.energies_ev)) != dropped_row
        )
        remap = np.full(len(observation.energies_ev), -1, dtype=np.int64)
        remap[kept_energy_indices] = np.arange(len(kept_energy_indices))
        reduced = BeamObservation(
            normalization=observation.normalization,
            energies_ev=observation.energies_ev[kept_energy_indices],
            energy_indices=observation.energy_indices[kept_energy_indices],
            row_ids=remap[observation.row_ids[keep]],
            sample_energies_ev=observation.sample_energies_ev[keep],
            pitches_deg=observation.pitches_deg[keep],
            values=observation.values[keep],
            incident_counts=observation.incident_counts[kept_energy_indices],
            reflected_counts=observation.reflected_counts[kept_energy_indices],
        )
        reduced_parameters = np.concatenate(
            (fitted.parameters[:3], fitted.parameters[3:][kept_energy_indices])
        )
        reduced_fit = fit_beam(reduced, beam_config, starts=[reduced_parameters])
        if reduced_fit.success:
            shifts.append(abs(reduced_fit.center_ev - fitted.center_ev))
    return max(shifts, default=float("inf"))


def assess_beam(
    primary_observation: BeamObservation,
    sensitivity_observation: BeamObservation,
    beam_config: dict[str, Any],
    qc_config: dict[str, Any],
    rng: np.random.Generator,
    bootstrap_replicates: int | None = None,
) -> BeamAssessment:
    """Fit both normalizations and apply every frozen beam QC gate."""
    primary = fit_beam(primary_observation, beam_config)
    sensitivity = fit_beam(sensitivity_observation, beam_config)
    bootstrap = bootstrap_beam(
        primary_observation,
        primary,
        beam_config,
        rng,
        replicates=bootstrap_replicates,
    )
    response_fwhm = (
        float(beam_config["energy_response_fwhm_fraction"]) * primary.center_ev
        if primary.success
        else float("nan")
    )
    positive_rows = count_positive_excess_rows(primary_observation, primary)
    leave_one_out = leave_one_energy_out_shift(
        primary_observation, primary, beam_config
    )
    normalization_difference = (
        abs(primary.center_ev - sensitivity.center_ev)
        if primary.success and sensitivity.success
        else float("inf")
    )
    reasons: list[str] = []
    if not primary.success:
        reasons.append("primary_fit_failed")
    if not sensitivity.success:
        reasons.append("sensitivity_fit_failed")
    if primary.success:
        center_low, center_high = beam_config["center_energy_ev_bounds"]
        center_margin = (
            float(qc_config["center_bound_margin_response_fwhm"]) * response_fwhm
        )
        sigma_low, sigma_high = beam_config["pitch_sigma_deg_bounds"]
        sigma_margin = float(qc_config["pitch_sigma_bound_margin_deg"])
        if primary.amplitude_at_bound:
            reasons.append("amplitude_bound")
        if not (
            primary.center_ev - center_low > center_margin
            and center_high - primary.center_ev > center_margin
        ):
            reasons.append("center_bound_margin")
        if not (
            primary.pitch_sigma_deg - sigma_low > sigma_margin
            and sigma_high - primary.pitch_sigma_deg > sigma_margin
        ):
            reasons.append("pitch_sigma_bound_margin")
        if (
            primary.multistart_center_spread_ev
            > float(qc_config["multistart_center_spread_response_fwhm_max"])
            * response_fwhm
        ):
            reasons.append("multistart_instability")
    if positive_rows < int(beam_config["minimum_positive_excess_energy_rows"]):
        reasons.append("insufficient_positive_excess_rows")
    finite_fraction = bootstrap.successes / max(bootstrap.requested, 1)
    if finite_fraction < float(qc_config["require_finite_bootstrap_fraction"]):
        reasons.append("bootstrap_failure")
    if np.isfinite(bootstrap.amplitude_std) and bootstrap.amplitude_std > 0.0:
        amplitude_snr = bootstrap.amplitude_median / bootstrap.amplitude_std
    else:
        amplitude_snr = float("inf") if bootstrap.amplitude_median > 0.0 else 0.0
    if amplitude_snr < float(qc_config["amplitude_bootstrap_signal_to_noise_min"]):
        reasons.append("amplitude_not_resolved")
    bootstrap_width = bootstrap.center_q975_ev - bootstrap.center_q025_ev
    if not np.isfinite(bootstrap_width) or bootstrap_width > (
        float(qc_config["bootstrap_95_width_response_fwhm_max"]) * response_fwhm
    ):
        reasons.append("centroid_uncertainty_unresolved")
    if normalization_difference > (
        float(qc_config["normalization_difference_response_fwhm_max"]) * response_fwhm
    ):
        reasons.append("normalization_instability")
    if leave_one_out > (
        float(qc_config["leave_one_energy_out_shift_response_fwhm_max"]) * response_fwhm
    ):
        reasons.append("leave_one_energy_out_instability")
    return BeamAssessment(
        primary=primary,
        sensitivity=sensitivity,
        bootstrap=bootstrap,
        positive_excess_rows=positive_rows,
        leave_one_energy_out_max_shift_ev=leave_one_out,
        normalization_difference_ev=normalization_difference,
        response_fwhm_ev=response_fwhm,
        beam_valid=not reasons,
        reason_codes=tuple(reasons),
    )
