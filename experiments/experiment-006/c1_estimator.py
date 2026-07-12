"""Executable CPU estimator and frozen base QC for Experiment 006 C1."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import beam_model
import fit_reference
import numpy as np
import start_parameters

INCIDENT_MEAN = "incident_mean"
CONJUGATE = "conjugate"

# These values are output data. Keep them stable across refactors.
FIT_FAILED = "fit_failed"
AMPLITUDE_SNR = "amplitude_snr"
CENTER_BOUND_MARGIN = "center_bound_margin"
PITCH_SIGMA_BOUND_MARGIN = "pitch_sigma_bound_margin"
MULTISTART_CENTER_SPREAD = "multistart_center_spread"
CENTER_STANDARD_ERROR = "center_standard_error"
NORMALIZATION_DIFFERENCE = "normalization_difference"
LEAVE_ONE_ROW_OUT_FAILED = "leave_one_energy_row_out_failed"
LEAVE_ONE_ROW_OUT_SHIFT = "leave_one_energy_row_out_shift"
POSITIVE_EXCESS_ROWS = "positive_excess_energy_rows"
AXIS_SUPPORT_ROWS = "response_contributing_axis_support_rows"
SINGLE_ROW_CENTROID = "single_row_created_centroid"
COVARIANCE = "finite_full_rank_covariance"


@dataclass(frozen=True)
class BranchObservation:
    """One normalized branch and the native row/sector masks behind it."""

    name: str
    observation: fit_reference.Observation
    sample_sector_ids: np.ndarray
    native_valid: np.ndarray
    pitches_by_row_deg: np.ndarray


@dataclass(frozen=True)
class BranchAssessment:
    """Complete multistart result and branch-local QC evidence."""

    name: str
    fit: fit_reference.FitResult | None
    reason_codes: tuple[str, ...]
    contributing_rows: np.ndarray
    axis_rows: np.ndarray


@dataclass(frozen=True)
class C1Estimate:
    """Both C1 branches, leave-one-row-out fits, and nonexclusive base QC."""

    incident_mean: BranchAssessment
    conjugate: BranchAssessment
    leave_one_row_out: tuple[fit_reference.FitResult | None, ...]
    leave_one_row_ids: np.ndarray
    reason_codes: tuple[str, ...]

    @property
    def base_valid(self) -> bool:
        return not self.reason_codes


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    """Load the frozen C1 configuration."""
    source = path or Path(__file__).with_name("configuration-c1.json")
    return json.loads(source.read_text(encoding="ascii"))


def construct_observations(
    energies_ev: np.ndarray,
    channel_low_ev: np.ndarray,
    channel_high_ev: np.ndarray,
    pitches_deg: np.ndarray,
    native_flux: np.ndarray,
    native_valid: np.ndarray,
    incident_mean: np.ndarray,
    conjugate: np.ndarray,
    *,
    configuration: dict[str, Any] | None = None,
) -> dict[str, BranchObservation]:
    """Construct reflected-sample observations from one final native realization."""
    config = configuration or load_configuration()
    energies = np.asarray(energies_ev, dtype=np.float64)
    low = np.asarray(channel_low_ev, dtype=np.float64)
    high = np.asarray(channel_high_ev, dtype=np.float64)
    pitch = np.asarray(pitches_deg, dtype=np.float64)
    flux = np.asarray(native_flux, dtype=np.float64)
    valid = np.asarray(native_valid, dtype=bool)
    products = {
        INCIDENT_MEAN: np.asarray(incident_mean, dtype=np.float64),
        CONJUGATE: np.asarray(conjugate, dtype=np.float64),
    }
    if (
        energies.ndim != 1
        or low.shape != energies.shape
        or high.shape != energies.shape
    ):
        raise ValueError("Energy centers and channel edges must be matching vectors")
    expected = (energies.size, pitch.shape[1] if pitch.ndim == 2 else -1)
    if pitch.shape != expected or flux.shape != expected or valid.shape != expected:
        raise ValueError(
            "Pitch, native flux, and native mask must be row x sector arrays"
        )
    if any(product.shape != expected for product in products.values()):
        raise ValueError("Normalized products must match the native array shape")

    minimum_incident = int(config["normalization"]["minimum_incident_sectors_per_row"])
    minimum_reflected = int(
        config["normalization"]["minimum_reflected_sectors_per_row"]
    )
    native = valid & np.isfinite(pitch) & np.isfinite(flux) & (flux > 0.0)
    incident = native & (pitch < 90.0)
    reflected = native & (pitch >= 90.0)
    row_ok = (np.count_nonzero(incident, axis=1) >= minimum_incident) & (
        np.count_nonzero(reflected, axis=1) >= minimum_reflected
    )

    result: dict[str, BranchObservation] = {}
    for name, product in products.items():
        selected = reflected & row_ok[:, None] & np.isfinite(product) & (product > 0.0)
        row_ids, sector_ids = np.nonzero(selected)
        observation = fit_reference.Observation(
            energies,
            low,
            high,
            row_ids.astype(np.int64),
            pitch[selected],
            product[selected],
        )
        result[name] = BranchObservation(
            name, observation, sector_ids.astype(np.int64), native, pitch
        )
    return result


def _fit(branch: BranchObservation, *, max_nfev: int = 2000) -> fit_reference.FitResult:
    lower, upper = fit_reference.parameter_bounds()
    starts = start_parameters.construct_starts(branch.observation, lower, upper)
    if starts.shape != (12, fit_reference.N_PARAMETERS):
        raise RuntimeError("C1 start construction must produce exactly 12 starts")
    return fit_reference.fit_reference(branch.observation, starts, max_nfev=max_nfev)


def _row_evidence(
    branch: BranchObservation,
    fit: fit_reference.FitResult,
    qc: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    observation = branch.observation
    parameters = fit.parameters
    response = beam_model.integrated_energy_response(
        observation.channel_low_ev,
        observation.channel_high_ev,
        float(parameters[1]),
        observation.fwhm_fraction,
    )
    beam = (
        parameters[0]
        * response[observation.row_ids]
        * np.exp(-0.5 * np.square((observation.pitches_deg - 180.0) / parameters[2]))
    )
    threshold = float(qc["positive_beam_component_fraction"]) * parameters[0]
    contributing = np.unique(observation.row_ids[beam > threshold]).astype(np.int64)
    half_width = float(qc["surface_axis_support_half_width_deg"])
    minimum = int(qc["minimum_surface_axis_sectors_per_row"])
    axis = []
    for row in contributing:
        samples = observation.row_ids == row
        if (
            np.count_nonzero(observation.pitches_deg[samples] >= 180.0 - half_width)
            >= minimum
        ):
            axis.append(int(row))
    return contributing, np.asarray(axis, dtype=np.int64)


def _branch_assessment(
    branch: BranchObservation, qc: dict[str, Any], *, max_nfev: int
) -> BranchAssessment:
    try:
        fitted = _fit(branch, max_nfev=max_nfev)
    except (RuntimeError, ValueError, FloatingPointError):
        return BranchAssessment(
            branch.name,
            None,
            (FIT_FAILED,),
            np.array([], dtype=int),
            np.array([], dtype=int),
        )
    reasons: list[str] = []
    width = fitted.parameters[1] * branch.observation.fwhm_fraction
    lower, upper = fit_reference.parameter_bounds()
    amplitude_se = fitted.standard_errors[0]
    center_se = fitted.standard_errors[1]
    if (
        not np.isfinite(amplitude_se)
        or amplitude_se <= 0.0
        or fitted.parameters[0] / amplitude_se
        < float(qc["amplitude_jacobian_signal_to_noise_min"])
    ):
        reasons.append(AMPLITUDE_SNR)
    margin = float(qc["center_bound_margin_response_fwhm"]) * width
    if (
        fitted.parameters[1] - lower[1] < margin
        or upper[1] - fitted.parameters[1] < margin
    ):
        reasons.append(CENTER_BOUND_MARGIN)
    sigma_margin = float(qc["pitch_sigma_bound_margin_deg"])
    if (
        fitted.parameters[2] - lower[2] < sigma_margin
        or upper[2] - fitted.parameters[2] < sigma_margin
    ):
        reasons.append(PITCH_SIGMA_BOUND_MARGIN)
    near = [
        trajectory.parameters[1]
        for trajectory in fitted.fits
        if trajectory.success
        and np.isfinite(trajectory.objective)
        and trajectory.objective
        <= fitted.objective * (1.0 + float(qc["near_best_objective_fraction"]))
    ]
    if (
        near
        and np.ptp(near)
        > float(qc["multistart_center_spread_response_fwhm_max"]) * width
    ):
        reasons.append(MULTISTART_CENTER_SPREAD)
    if (
        not np.isfinite(center_se)
        or center_se > float(qc["center_standard_error_response_fwhm_max"]) * width
    ):
        reasons.append(CENTER_STANDARD_ERROR)
    if (
        fitted.jacobian_rank != fit_reference.N_PARAMETERS
        or not np.isfinite(fitted.covariance).all()
    ):
        reasons.append(COVARIANCE)
    contributing, axis = _row_evidence(branch, fitted, qc)
    if contributing.size < int(qc["minimum_positive_excess_energy_rows"]):
        reasons.append(POSITIVE_EXCESS_ROWS)
    if axis.size < int(qc["minimum_response_contributing_rows_with_axis_support"]):
        reasons.append(AXIS_SUPPORT_ROWS)
    if bool(qc["reject_single_cell_created_centroid"]) and contributing.size < 2:
        reasons.append(SINGLE_ROW_CENTROID)
    return BranchAssessment(branch.name, fitted, tuple(reasons), contributing, axis)


def _without_row(branch: BranchObservation, omitted: int) -> BranchObservation:
    source = branch.observation
    keep = source.row_ids != omitted
    observation = fit_reference.Observation(
        source.energies_ev,
        source.channel_low_ev,
        source.channel_high_ev,
        source.row_ids[keep],
        source.pitches_deg[keep],
        source.observed[keep],
        source.energy_domain_ev,
        source.fwhm_fraction,
        source.physical_lower,
        source.physical_upper,
    )
    return BranchObservation(
        branch.name,
        observation,
        branch.sample_sector_ids[keep],
        branch.native_valid,
        branch.pitches_by_row_deg,
    )


def fit_c1(
    energies_ev: np.ndarray,
    channel_low_ev: np.ndarray,
    channel_high_ev: np.ndarray,
    pitches_deg: np.ndarray,
    native_flux: np.ndarray,
    native_valid: np.ndarray,
    incident_mean: np.ndarray,
    conjugate: np.ndarray,
    *,
    configuration: dict[str, Any] | None = None,
) -> C1Estimate:
    """Fit both C1 branches and execute every frozen base-QC diagnostic."""
    config = configuration or load_configuration()
    branches = construct_observations(
        energies_ev,
        channel_low_ev,
        channel_high_ev,
        pitches_deg,
        native_flux,
        native_valid,
        incident_mean,
        conjugate,
        configuration=config,
    )
    max_nfev = int(config["estimator"]["max_nfev"])
    primary = _branch_assessment(
        branches[INCIDENT_MEAN], config["qc"], max_nfev=max_nfev
    )
    secondary = _branch_assessment(branches[CONJUGATE], config["qc"], max_nfev=max_nfev)
    reasons = list(primary.reason_codes)
    if secondary.fit is None:
        reasons.append(FIT_FAILED)
    if primary.fit is not None and secondary.fit is not None:
        width = (
            primary.fit.parameters[1]
            * branches[INCIDENT_MEAN].observation.fwhm_fraction
        )
        if (
            abs(primary.fit.parameters[1] - secondary.fit.parameters[1])
            > float(config["qc"]["normalization_difference_response_fwhm_max"]) * width
        ):
            reasons.append(NORMALIZATION_DIFFERENCE)

    omitted_rows = np.unique(branches[INCIDENT_MEAN].observation.row_ids)
    loo: list[fit_reference.FitResult | None] = []
    for row in omitted_rows:
        try:
            loo.append(
                _fit(
                    _without_row(branches[INCIDENT_MEAN], int(row)),
                    max_nfev=max_nfev,
                )
            )
        except (RuntimeError, ValueError, FloatingPointError):
            loo.append(None)
    if any(result is None for result in loo):
        reasons.append(LEAVE_ONE_ROW_OUT_FAILED)
    if primary.fit is not None:
        width = (
            primary.fit.parameters[1]
            * branches[INCIDENT_MEAN].observation.fwhm_fraction
        )
        limit = (
            float(config["qc"]["leave_one_energy_out_shift_response_fwhm_max"]) * width
        )
        if any(
            result is not None
            and abs(result.parameters[1] - primary.fit.parameters[1]) > limit
            for result in loo
        ):
            reasons.append(LEAVE_ONE_ROW_OUT_SHIFT)
        if bool(config["qc"]["reject_single_cell_created_centroid"]):
            contributing = set(primary.contributing_rows.tolist())
            for row, result in zip(omitted_rows, loo, strict=True):
                if int(row) not in contributing or result is None:
                    continue
                reduced = _without_row(branches[INCIDENT_MEAN], int(row))
                reduced_rows, _ = _row_evidence(reduced, result, config["qc"])
                if reduced_rows.size == 0:
                    reasons.append(SINGLE_ROW_CENTROID)
                    break
    return C1Estimate(
        primary, secondary, tuple(loo), omitted_rows, tuple(dict.fromkeys(reasons))
    )


fit_case = fit_c1
