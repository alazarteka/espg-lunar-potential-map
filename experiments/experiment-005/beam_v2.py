#!/usr/bin/env python3
"""Flexible-continuum beam estimator for Experiment 005 Attempt 02."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import beam
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import csr_matrix, lil_matrix

CONTINUUM_PARAMETERS = 4


@dataclass(frozen=True)
class FlexibleBeamFit:
    """Beam fit with an independent logistic continuum at every energy."""

    success: bool
    center_ev: float
    amplitude: float
    pitch_sigma_deg: float
    center_standard_error_ev: float
    amplitude_standard_error: float
    objective: float
    rms_log_residual: float
    n_valid: int
    n_energy_rows: int
    covariance_rank: int
    covariance_dimension: int
    multistart_center_spread_ev: float
    amplitude_at_bound: bool
    pitch_sigma_at_bound: bool
    continuum_bound_contacts: int
    parameters: np.ndarray
    model_values: np.ndarray
    log_residuals: np.ndarray


@dataclass(frozen=True)
class FlexibleBeamAssessment:
    """Both normalization fits and the pre-calibration QC result."""

    primary: FlexibleBeamFit
    sensitivity: FlexibleBeamFit
    positive_excess_rows: int
    leave_one_energy_out_max_shift_ev: float
    normalization_difference_ev: float
    response_fwhm_ev: float
    base_valid: bool
    reason_codes: tuple[str, ...]


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name("configuration-attempt-02.json")
    return json.loads(path.read_text(encoding="ascii"))


def continuum_values(
    pitches_deg: np.ndarray,
    row_ids: np.ndarray,
    continuum_parameters: np.ndarray,
) -> np.ndarray:
    """Evaluate independent logistic transitions on flattened sector support."""
    row_parameters = continuum_parameters.reshape(-1, CONTINUUM_PARAMETERS)
    selected = row_parameters[row_ids]
    cone_level = selected[:, 0]
    contrast = selected[:, 1]
    boundary = selected[:, 2]
    width = selected[:, 3]
    transition = 1.0 / (
        1.0 + np.exp(-np.clip((boundary - pitches_deg) / width, -60.0, 60.0))
    )
    return cone_level + contrast * transition


def model_values(
    observation: beam.BeamObservation,
    parameters: np.ndarray,
    beam_config: dict[str, Any],
) -> np.ndarray:
    amplitude, center, pitch_sigma = parameters[:3]
    continuum = continuum_values(
        observation.pitches_deg, observation.row_ids, parameters[3:]
    )
    shape = beam.beam_shape(
        observation.sample_energies_ev,
        observation.pitches_deg,
        center,
        pitch_sigma,
        float(beam_config["energy_response_fwhm_fraction"]),
    )
    return continuum + amplitude * shape


def _continuum_bounds(
    beam_config: dict[str, Any], n_rows: int
) -> tuple[np.ndarray, np.ndarray]:
    lower_row = np.array(
        [
            beam_config["continuum_cone_level_bounds"][0],
            beam_config["continuum_contrast_bounds"][0],
            beam_config["continuum_boundary_deg_bounds"][0],
            beam_config["continuum_width_deg_bounds"][0],
        ],
        dtype=np.float64,
    )
    upper_row = np.array(
        [
            beam_config["continuum_cone_level_bounds"][1],
            beam_config["continuum_contrast_bounds"][1],
            beam_config["continuum_boundary_deg_bounds"][1],
            beam_config["continuum_width_deg_bounds"][1],
        ],
        dtype=np.float64,
    )
    return np.tile(lower_row, n_rows), np.tile(upper_row, n_rows)


def _fit_row_continuum(
    pitches_deg: np.ndarray, values: np.ndarray, beam_config: dict[str, Any]
) -> np.ndarray:
    initialization = pitches_deg <= 150.0
    if np.count_nonzero(initialization) >= 10:
        pitches_deg = pitches_deg[initialization]
        values = values[initialization]
    lower, upper = _continuum_bounds(beam_config, 1)

    def residual(parameters: np.ndarray) -> np.ndarray:
        continuum = continuum_values(
            pitches_deg, np.zeros(len(pitches_deg), dtype=np.int64), parameters
        )
        return np.log(continuum) - np.log(values)

    starts = [
        np.array([0.15, 0.85, boundary, width])
        for boundary in (105.0, 135.0, 165.0)
        for width in (5.0, 15.0)
    ]
    fits = [
        least_squares(
            residual,
            np.clip(start, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            max_nfev=500,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fitted: float(np.sum(np.square(fitted.fun))))
    return np.asarray(best.x)


def _initial_continuum(
    observation: beam.BeamObservation,
    beam_config: dict[str, Any],
    values: np.ndarray | None = None,
) -> np.ndarray:
    if values is None:
        values = observation.values
    parameters = []
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        parameters.append(
            _fit_row_continuum(
                observation.pitches_deg[selected],
                values[selected],
                beam_config,
            )
        )
    return np.concatenate(parameters)


def _detected_center(observation: beam.BeamObservation) -> float:
    scores = []
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        pitches = observation.pitches_deg[selected]
        values = observation.values[selected]
        aligned = values[pitches >= 157.5]
        shoulder = values[(pitches >= 112.5) & (pitches < 146.25)]
        score = (
            float(np.mean(aligned) - np.mean(shoulder))
            if aligned.size and shoulder.size
            else float("-inf")
        )
        scores.append(score)
    return float(observation.energies_ev[int(np.argmax(scores))])


def _start_parameters(
    observation: beam.BeamObservation, beam_config: dict[str, Any]
) -> list[np.ndarray]:
    aligned = observation.pitches_deg >= 157.5
    shoulder = (observation.pitches_deg >= 112.5) & (
        observation.pitches_deg < 146.25
    )
    amplitude = float(
        np.percentile(observation.values[aligned], 90)
        - np.median(observation.values[shoulder])
    )
    amplitude = float(
        np.clip(
            amplitude,
            0.5,
            0.5 * float(beam_config["amplitude_bounds"][1]),
        )
    )
    detected = _detected_center(observation)
    center_low, center_high = beam_config["center_energy_ev_bounds"]
    centers = [detected * factor for factor in (0.75, 1.0, 1.25, 1.75)]
    centers = [
        float(np.clip(value, center_low + 1.0, center_high - 1.0)) for value in centers
    ]
    sigma_low, sigma_high = beam_config["pitch_sigma_deg_bounds"]
    sigmas = np.linspace(sigma_low + 5.0, sigma_high - 5.0, 3)
    starts = []
    for center in centers:
        for sigma in sigmas:
            shape = beam.beam_shape(
                observation.sample_energies_ev,
                observation.pitches_deg,
                center,
                sigma,
                float(beam_config["energy_response_fwhm_fraction"]),
            )
            continuum_values_start = np.maximum(
                observation.values - amplitude * shape, 1e-6
            )
            continuum = _initial_continuum(
                observation, beam_config, values=continuum_values_start
            )
            starts.append(
                np.concatenate(([amplitude, center, sigma], continuum))
            )
    return starts[: int(beam_config["multistart_count"])]


def _failed_fit(observation: beam.BeamObservation) -> FlexibleBeamFit:
    empty = np.array([], dtype=np.float64)
    return FlexibleBeamFit(
        False,
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("nan"),
        float("inf"),
        float("nan"),
        int(observation.values.size),
        len(observation.energies_ev),
        0,
        0,
        float("nan"),
        False,
        False,
        0,
        empty,
        empty,
        empty,
    )


def _jacobian_sparsity(observation: beam.BeamObservation) -> csr_matrix:
    """Return exact parameter dependence for sparse finite differences."""
    n_parameters = 3 + CONTINUUM_PARAMETERS * len(observation.energies_ev)
    sparsity = lil_matrix((len(observation.values), n_parameters), dtype=np.int8)
    sparsity[:, :3] = 1
    for sample_index, row_id in enumerate(observation.row_ids):
        start = 3 + CONTINUUM_PARAMETERS * int(row_id)
        sparsity[sample_index, start : start + CONTINUUM_PARAMETERS] = 1
    return sparsity.tocsr()


def fit_flexible_beam(
    observation: beam.BeamObservation,
    beam_config: dict[str, Any],
    starts: list[np.ndarray] | None = None,
) -> FlexibleBeamFit:
    """Jointly fit a shared beam and row-independent logistic continua."""
    n_rows = len(observation.energies_ev)
    if n_rows < 3 or observation.values.size == 0:
        return _failed_fit(observation)
    amplitude_bounds = beam_config["amplitude_bounds"]
    center_bounds = beam_config["center_energy_ev_bounds"]
    sigma_bounds = beam_config["pitch_sigma_deg_bounds"]
    continuum_lower, continuum_upper = _continuum_bounds(beam_config, n_rows)
    lower = np.concatenate(
        (
            [amplitude_bounds[0], center_bounds[0], sigma_bounds[0]],
            continuum_lower,
        )
    )
    upper = np.concatenate(
        (
            [amplitude_bounds[1], center_bounds[1], sigma_bounds[1]],
            continuum_upper,
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        modeled = model_values(observation, parameters, beam_config)
        return np.log(modeled) - np.log(observation.values)

    if starts is None:
        starts = _start_parameters(observation, beam_config)
    jacobian_sparsity = _jacobian_sparsity(observation)
    fits = []
    for start in starts:
        if len(start) != len(lower):
            continue
        fitted = least_squares(
            residual,
            np.clip(start, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            x_scale="jac",
            jac_sparsity=jacobian_sparsity,
            max_nfev=int(beam_config["optimizer_max_nfev"]),
        )
        if fitted.success and np.all(np.isfinite(fitted.x)):
            fits.append(fitted)
    if not fits:
        return _failed_fit(observation)
    objectives = np.array([float(np.sum(np.square(fit.fun))) for fit in fits])
    best_index = int(np.argmin(objectives))
    best = fits[best_index]
    best_objective = float(objectives[best_index])
    near = (
        objectives
        <= best_objective * (1.0 + float(beam_config["near_best_objective_fraction"]))
        + 1e-12
    )
    centers = np.array([fit.x[1] for fit in fits])[near]
    center_spread = float(np.max(centers) - np.min(centers))

    covariance_dimension = len(best.x)
    jacobian = (
        best.jac.toarray() if hasattr(best.jac, "toarray") else np.asarray(best.jac)
    )
    covariance_rank = int(np.linalg.matrix_rank(jacobian))
    degrees_freedom = max(len(best.fun) - covariance_dimension, 1)
    residual_variance = float(np.sum(np.square(best.fun)) / degrees_freedom)
    information = jacobian.T @ jacobian
    covariance = residual_variance * np.linalg.pinv(information, rcond=1e-10)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    amplitude, center, pitch_sigma = (float(value) for value in best.x[:3])
    modeled = model_values(observation, best.x, beam_config)

    continuum_parameters = best.x[3:]
    continuum_at_lower = np.isclose(
        continuum_parameters, continuum_lower, rtol=0.0, atol=1e-3
    )
    continuum_at_upper = np.isclose(
        continuum_parameters, continuum_upper, rtol=0.0, atol=1e-3
    )
    return FlexibleBeamFit(
        success=True,
        center_ev=center,
        amplitude=amplitude,
        pitch_sigma_deg=pitch_sigma,
        center_standard_error_ev=float(standard_errors[1]),
        amplitude_standard_error=float(standard_errors[0]),
        objective=best_objective,
        rms_log_residual=float(np.sqrt(np.mean(np.square(best.fun)))),
        n_valid=int(observation.values.size),
        n_energy_rows=n_rows,
        covariance_rank=covariance_rank,
        covariance_dimension=covariance_dimension,
        multistart_center_spread_ev=center_spread,
        amplitude_at_bound=(
            amplitude <= float(amplitude_bounds[0]) + 1e-4
            or amplitude >= float(amplitude_bounds[1]) - 1e-3
        ),
        pitch_sigma_at_bound=(
            pitch_sigma <= float(sigma_bounds[0]) + 1e-3
            or pitch_sigma >= float(sigma_bounds[1]) - 1e-3
        ),
        continuum_bound_contacts=int(
            np.count_nonzero(continuum_at_lower | continuum_at_upper)
        ),
        parameters=np.asarray(best.x),
        model_values=modeled,
        log_residuals=np.asarray(best.fun),
    )


def count_positive_excess_rows(
    observation: beam.BeamObservation, fitted: FlexibleBeamFit
) -> int:
    if not fitted.success:
        return 0
    continuum = continuum_values(
        observation.pitches_deg, observation.row_ids, fitted.parameters[3:]
    )
    count = 0
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        aligned = selected & (observation.pitches_deg >= 157.5)
        if (
            np.any(aligned)
            and float(np.mean(observation.values[aligned] - continuum[aligned])) > 0.0
        ):
            count += 1
    return count


def leave_one_energy_out_shift(
    observation: beam.BeamObservation,
    fitted: FlexibleBeamFit,
    beam_config: dict[str, Any],
) -> float:
    if not fitted.success or len(observation.energies_ev) < 4:
        return float("inf")
    shifts = []
    continuum = fitted.parameters[3:].reshape(-1, CONTINUUM_PARAMETERS)
    for dropped_row in range(len(observation.energies_ev)):
        keep_rows = np.arange(len(observation.energies_ev)) != dropped_row
        keep_samples = observation.row_ids != dropped_row
        remap = np.full(len(observation.energies_ev), -1, dtype=np.int64)
        remap[keep_rows] = np.arange(np.count_nonzero(keep_rows))
        reduced = beam.BeamObservation(
            normalization=observation.normalization,
            energies_ev=observation.energies_ev[keep_rows],
            energy_indices=observation.energy_indices[keep_rows],
            row_ids=remap[observation.row_ids[keep_samples]],
            sample_energies_ev=observation.sample_energies_ev[keep_samples],
            pitches_deg=observation.pitches_deg[keep_samples],
            values=observation.values[keep_samples],
            incident_counts=observation.incident_counts[keep_rows],
            reflected_counts=observation.reflected_counts[keep_rows],
        )
        start = np.concatenate((fitted.parameters[:3], continuum[keep_rows].ravel()))
        reduced_fit = fit_flexible_beam(reduced, beam_config, starts=[start])
        if reduced_fit.success:
            shifts.append(abs(reduced_fit.center_ev - fitted.center_ev))
    return max(shifts, default=float("inf"))


def assess_flexible_beam(
    primary_observation: beam.BeamObservation,
    sensitivity_observation: beam.BeamObservation,
    beam_config: dict[str, Any],
    qc_config: dict[str, Any],
) -> FlexibleBeamAssessment:
    """Fit both normalizations and apply pre-calibration QC."""
    primary = fit_flexible_beam(primary_observation, beam_config)
    sensitivity = fit_flexible_beam(sensitivity_observation, beam_config)
    response_fwhm = (
        float(beam_config["energy_response_fwhm_fraction"]) * primary.center_ev
        if primary.success
        else float("nan")
    )
    normalization_difference = (
        abs(primary.center_ev - sensitivity.center_ev)
        if primary.success and sensitivity.success
        else float("inf")
    )
    positive_rows = count_positive_excess_rows(primary_observation, primary)
    leave_one_out = leave_one_energy_out_shift(
        primary_observation, primary, beam_config
    )
    reasons = []
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
        covariance_finite = (
            np.isfinite(primary.center_standard_error_ev)
            and np.isfinite(primary.amplitude_standard_error)
            and primary.center_standard_error_ev > 0.0
            and primary.amplitude_standard_error > 0.0
        )
        if not covariance_finite:
            reasons.append("covariance_unresolved")
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
        if (
            covariance_finite
            and primary.amplitude / primary.amplitude_standard_error
            < float(qc_config["amplitude_jacobian_signal_to_noise_min"])
        ):
            reasons.append("amplitude_not_resolved")
        if (
            covariance_finite
            and primary.center_standard_error_ev
            > float(qc_config["center_standard_error_response_fwhm_max"])
            * response_fwhm
        ):
            reasons.append("centroid_scale_unresolved")
    if positive_rows < int(beam_config["minimum_positive_excess_energy_rows"]):
        reasons.append("insufficient_positive_excess_rows")
    if (
        normalization_difference
        > float(qc_config["normalization_difference_response_fwhm_max"]) * response_fwhm
    ):
        reasons.append("normalization_instability")
    if (
        leave_one_out
        > float(qc_config["leave_one_energy_out_shift_response_fwhm_max"])
        * response_fwhm
    ):
        reasons.append("leave_one_energy_out_instability")
    return FlexibleBeamAssessment(
        primary=primary,
        sensitivity=sensitivity,
        positive_excess_rows=positive_rows,
        leave_one_energy_out_max_shift_ev=leave_one_out,
        normalization_difference_ev=normalization_difference,
        response_fwhm_ev=response_fwhm,
        base_valid=not reasons,
        reason_codes=tuple(reasons),
    )
