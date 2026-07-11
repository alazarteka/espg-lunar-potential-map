#!/usr/bin/env python3
"""Smooth-continuum beam estimator under development for Attempt 03."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import beam
import beam_v2
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

N_CONTINUUM_QUANTITIES = 4


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    if path is None:
        path = Path(__file__).with_name("configuration-attempt-03.json")
    return json.loads(path.read_text(encoding="ascii"))


def energy_basis(energies_ev: np.ndarray, beam_config: dict[str, Any]) -> np.ndarray:
    """Quadratic basis on a fixed normalized log-energy domain."""
    energy_low, energy_high = beam_config["fit_domain_energy_ev"]
    scaled = (
        2.0
        * (
            (np.log(energies_ev) - np.log(energy_low))
            / (np.log(energy_high) - np.log(energy_low))
        )
        - 1.0
    )
    return np.column_stack((np.ones_like(scaled), scaled, np.square(scaled)))


def _physical_bounds(beam_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lower = np.array(
        [
            beam_config["continuum_cone_level_bounds"][0],
            beam_config["continuum_contrast_bounds"][0],
            beam_config["continuum_boundary_deg_bounds"][0],
            beam_config["continuum_width_deg_bounds"][0],
        ],
        dtype=np.float64,
    )
    upper = np.array(
        [
            beam_config["continuum_cone_level_bounds"][1],
            beam_config["continuum_contrast_bounds"][1],
            beam_config["continuum_boundary_deg_bounds"][1],
            beam_config["continuum_width_deg_bounds"][1],
        ],
        dtype=np.float64,
    )
    return lower, upper


def continuum_row_parameters(
    energies_ev: np.ndarray,
    coefficients: np.ndarray,
    beam_config: dict[str, Any],
) -> np.ndarray:
    """Map smooth unconstrained coefficients into bounded row parameters."""
    basis = energy_basis(energies_ev, beam_config)
    coefficient_matrix = coefficients.reshape(N_CONTINUUM_QUANTITIES, 3)
    latent = basis @ coefficient_matrix.T
    lower, upper = _physical_bounds(beam_config)
    return lower + (upper - lower) * expit(latent)


def continuum_values(
    observation: beam.BeamObservation,
    coefficients: np.ndarray,
    beam_config: dict[str, Any],
) -> np.ndarray:
    row_parameters = continuum_row_parameters(
        observation.energies_ev, coefficients, beam_config
    )
    selected = row_parameters[observation.row_ids]
    transition = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (selected[:, 2] - observation.pitches_deg) / selected[:, 3],
                -60.0,
                60.0,
            )
        )
    )
    return selected[:, 0] + selected[:, 1] * transition


def model_values(
    observation: beam.BeamObservation,
    parameters: np.ndarray,
    beam_config: dict[str, Any],
) -> np.ndarray:
    amplitude, center, pitch_sigma = parameters[:3]
    continuum = continuum_values(observation, parameters[3:], beam_config)
    shape = beam.beam_shape(
        observation.sample_energies_ev,
        observation.pitches_deg,
        center,
        pitch_sigma,
        float(beam_config["energy_response_fwhm_fraction"]),
    )
    return continuum + amplitude * shape


def _fit_smooth_coefficients(
    observation: beam.BeamObservation,
    beam_config: dict[str, Any],
    values: np.ndarray,
) -> np.ndarray:
    row_parameters = []
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        row_parameters.append(
            beam_v2._fit_row_continuum(
                observation.pitches_deg[selected], values[selected], beam_config
            )
        )
    row_parameters_array = np.asarray(row_parameters)
    lower, upper = _physical_bounds(beam_config)
    fractions = np.clip(
        (row_parameters_array - lower) / (upper - lower), 1e-5, 1.0 - 1e-5
    )
    latent = logit(fractions)
    basis = energy_basis(observation.energies_ev, beam_config)
    coefficients = [
        np.linalg.lstsq(basis, latent[:, quantity], rcond=None)[0]
        for quantity in range(N_CONTINUUM_QUANTITIES)
    ]
    coefficient_low, coefficient_high = beam_config.get(
        "smooth_continuum_coefficient_bounds", [-12.0, 12.0]
    )
    return np.clip(np.asarray(coefficients).ravel(), coefficient_low, coefficient_high)


def _detected_center(observation: beam.BeamObservation) -> float:
    scores = []
    for row_id in range(len(observation.energies_ev)):
        selected = observation.row_ids == row_id
        pitches = observation.pitches_deg[selected]
        values = observation.values[selected]
        aligned = values[pitches >= 157.5]
        shoulder = values[(pitches >= 112.5) & (pitches < 146.25)]
        scores.append(
            float(np.mean(aligned) - np.mean(shoulder))
            if aligned.size and shoulder.size
            else float("-inf")
        )
    return float(observation.energies_ev[int(np.argmax(scores))])


def _start_parameters(
    observation: beam.BeamObservation, beam_config: dict[str, Any]
) -> list[np.ndarray]:
    aligned = observation.pitches_deg >= 157.5
    shoulder = (observation.pitches_deg >= 112.5) & (observation.pitches_deg < 146.25)
    amplitude = float(
        np.percentile(observation.values[aligned], 90)
        - np.median(observation.values[shoulder])
    )
    amplitude = float(np.clip(amplitude, 0.5, 0.5 * beam_config["amplitude_bounds"][1]))
    detected = _detected_center(observation)
    center_low, center_high = beam_config["center_energy_ev_bounds"]
    centers = [
        float(np.clip(detected * factor, center_low + 1.0, center_high - 1.0))
        for factor in (0.75, 1.0, 1.25, 1.75)
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
            continuum_observation = np.maximum(
                observation.values - amplitude * shape, 1e-6
            )
            coefficients = _fit_smooth_coefficients(
                observation, beam_config, continuum_observation
            )
            starts.append(np.concatenate(([amplitude, center, sigma], coefficients)))
    return starts[: int(beam_config["multistart_count"])]


def _failed_fit(observation: beam.BeamObservation) -> beam_v2.FlexibleBeamFit:
    empty = np.array([], dtype=np.float64)
    return beam_v2.FlexibleBeamFit(
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


def fit_smooth_beam(
    observation: beam.BeamObservation,
    beam_config: dict[str, Any],
    starts: list[np.ndarray] | None = None,
) -> beam_v2.FlexibleBeamFit:
    """Fit a shared beam over a log-energy-smooth logistic continuum."""
    if len(observation.energies_ev) < 3 or observation.values.size == 0:
        return _failed_fit(observation)
    amplitude_bounds = beam_config["amplitude_bounds"]
    center_bounds = beam_config["center_energy_ev_bounds"]
    sigma_bounds = beam_config["pitch_sigma_deg_bounds"]
    coefficient_bounds = beam_config.get(
        "smooth_continuum_coefficient_bounds", [-12.0, 12.0]
    )
    lower = np.concatenate(
        (
            [amplitude_bounds[0], center_bounds[0], sigma_bounds[0]],
            np.full(12, coefficient_bounds[0]),
        )
    )
    upper = np.concatenate(
        (
            [amplitude_bounds[1], center_bounds[1], sigma_bounds[1]],
            np.full(12, coefficient_bounds[1]),
        )
    )

    def residual(parameters: np.ndarray) -> np.ndarray:
        modeled = model_values(observation, parameters, beam_config)
        return np.log(modeled) - np.log(observation.values)

    if starts is None:
        starts = _start_parameters(observation, beam_config)
    fits = []
    for start in starts:
        fitted = least_squares(
            residual,
            np.clip(start, lower + 1e-8, upper - 1e-8),
            bounds=(lower, upper),
            x_scale="jac",
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
    covariance_rank = int(np.linalg.matrix_rank(best.jac))
    degrees_freedom = max(len(best.fun) - covariance_dimension, 1)
    residual_variance = float(np.sum(np.square(best.fun)) / degrees_freedom)
    covariance = residual_variance * np.linalg.pinv(best.jac.T @ best.jac, rcond=1e-10)
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    amplitude, center, pitch_sigma = (float(value) for value in best.x[:3])
    modeled = model_values(observation, best.x, beam_config)
    coefficients = best.x[3:]
    coefficient_contacts = np.count_nonzero(
        np.isclose(coefficients, coefficient_bounds[0], atol=1e-3)
        | np.isclose(coefficients, coefficient_bounds[1], atol=1e-3)
    )
    return beam_v2.FlexibleBeamFit(
        success=True,
        center_ev=center,
        amplitude=amplitude,
        pitch_sigma_deg=pitch_sigma,
        center_standard_error_ev=float(standard_errors[1]),
        amplitude_standard_error=float(standard_errors[0]),
        objective=best_objective,
        rms_log_residual=float(np.sqrt(np.mean(np.square(best.fun)))),
        n_valid=int(observation.values.size),
        n_energy_rows=len(observation.energies_ev),
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
        continuum_bound_contacts=int(coefficient_contacts),
        parameters=np.asarray(best.x),
        model_values=modeled,
        log_residuals=np.asarray(best.fun),
    )


def count_positive_excess_rows(
    observation: beam.BeamObservation,
    fitted: beam_v2.FlexibleBeamFit,
    beam_config: dict[str, Any],
) -> int:
    if not fitted.success:
        return 0
    continuum = continuum_values(observation, fitted.parameters[3:], beam_config)
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
    fitted: beam_v2.FlexibleBeamFit,
    beam_config: dict[str, Any],
) -> float:
    if not fitted.success or len(observation.energies_ev) < 4:
        return float("inf")
    shifts = []
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
        reduced_fit = fit_smooth_beam(reduced, beam_config, starts=[fitted.parameters])
        if reduced_fit.success:
            shifts.append(abs(reduced_fit.center_ev - fitted.center_ev))
    return max(shifts, default=float("inf"))


def assess_smooth_beam(
    primary_observation: beam.BeamObservation,
    sensitivity_observation: beam.BeamObservation,
    beam_config: dict[str, Any],
    qc_config: dict[str, Any],
    reuse_identical_sensitivity: bool = False,
) -> beam_v2.FlexibleBeamAssessment:
    """Fit both normalizations and apply the Attempt 02 base-QC vocabulary."""
    primary = fit_smooth_beam(primary_observation, beam_config)
    sensitivity = (
        primary
        if reuse_identical_sensitivity
        else fit_smooth_beam(sensitivity_observation, beam_config)
    )
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
    positive_rows = count_positive_excess_rows(
        primary_observation, primary, beam_config
    )
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
    return beam_v2.FlexibleBeamAssessment(
        primary=primary,
        sensitivity=sensitivity,
        positive_excess_rows=positive_rows,
        leave_one_energy_out_max_shift_ev=leave_one_out,
        normalization_difference_ev=normalization_difference,
        response_fwhm_ev=response_fwhm,
        base_valid=not reasons,
        reason_codes=tuple(reasons),
    )
