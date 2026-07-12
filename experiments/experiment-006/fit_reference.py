"""CPU reference fitting and hybrid certification for Experiment 006.

This module deliberately does not construct starts.  Callers supply the frozen
``[starts, 15]`` matrix so the CPU and candidate backends use identical trials.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import beam_model
import numpy as np
from scipy.optimize import least_squares

if TYPE_CHECKING:
    from collections.abc import Sequence


N_PARAMETERS = 15


@dataclass(frozen=True)
class Observation:
    """The arrays and fixed model settings for one beam fit."""

    energies_ev: np.ndarray
    channel_low_ev: np.ndarray
    channel_high_ev: np.ndarray
    row_ids: np.ndarray
    pitches_deg: np.ndarray
    observed: np.ndarray
    energy_domain_ev: tuple[float, float] = (20.0, 2000.0)
    fwhm_fraction: float = 0.48
    physical_lower: tuple[float, ...] = (0.01, 0.01, 90.0, 0.5)
    physical_upper: tuple[float, ...] = (2.0, 3.0, 180.0, 45.0)


@dataclass(frozen=True)
class StartFit:
    """One independently polished SciPy trajectory."""

    start_index: int
    initial_parameters: np.ndarray
    parameters: np.ndarray
    objective: float
    success: bool
    status: int
    message: str
    jacobian: np.ndarray


@dataclass(frozen=True)
class FitResult:
    """All trajectories and CPU-only inference for their canonical winner."""

    fits: tuple[StartFit, ...]
    winner_index: int
    parameters: np.ndarray
    objective: float
    jacobian: np.ndarray
    jacobian_rank: int
    covariance: np.ndarray
    standard_errors: np.ndarray


@dataclass(frozen=True)
class CertificationResult:
    """A CPU-certified result plus candidate-search provenance."""

    result: FitResult
    ranked_candidate_indices: np.ndarray
    polished_candidate_indices: np.ndarray
    used_fallback: bool
    fallback_reason: str | None


def parameter_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return the frozen D1 bounds in canonical parameter order."""
    lower = np.array([0.0, 20.0, 3.0, *([-12.0] * 12)], dtype=np.float64)
    upper = np.array([40.0, 2000.0, 50.0, *([12.0] * 12)], dtype=np.float64)
    return lower, upper


def _residual(observation: Observation, parameters: np.ndarray) -> np.ndarray:
    modeled = beam_model.model_values(
        observation.energies_ev,
        observation.channel_low_ev,
        observation.channel_high_ev,
        observation.row_ids,
        observation.pitches_deg,
        parameters,
        energy_domain_ev=observation.energy_domain_ev,
        fwhm_fraction=observation.fwhm_fraction,
        physical_lower=np.asarray(observation.physical_lower, dtype=np.float64),
        physical_upper=np.asarray(observation.physical_upper, dtype=np.float64),
    )
    return beam_model.log_residuals(observation.observed, modeled)


def _validate_starts(starts: np.ndarray) -> np.ndarray:
    values = np.asarray(starts, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != N_PARAMETERS or values.shape[0] == 0:
        raise ValueError("Starts must have shape [starts, 15]")
    lower, upper = parameter_bounds()
    if not np.isfinite(values).all():
        raise ValueError("Starts must be finite")
    if np.any(values < lower) or np.any(values > upper):
        raise ValueError("Starts must lie within the D1 bounds")
    return values


def _fit_starts(observation: Observation, starts: np.ndarray) -> tuple[StartFit, ...]:
    lower, upper = parameter_bounds()
    fits: list[StartFit] = []
    for index, start in enumerate(starts):
        fitted = least_squares(
            lambda parameters: _residual(observation, parameters),
            start,
            bounds=(lower, upper),
            jac="3-point",
            method="trf",
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
            max_nfev=5000,
        )
        objective = float(np.dot(fitted.fun, fitted.fun))
        fits.append(
            StartFit(
                start_index=index,
                initial_parameters=start.copy(),
                parameters=np.asarray(fitted.x, dtype=np.float64).copy(),
                objective=objective,
                success=bool(fitted.success),
                status=int(fitted.status),
                message=str(fitted.message),
                jacobian=np.asarray(fitted.jac, dtype=np.float64).copy(),
            )
        )
    return tuple(fits)


def _summarize(fits: tuple[StartFit, ...]) -> FitResult:
    objectives = np.asarray([fit.objective for fit in fits])
    eligible = np.isfinite(objectives) & np.asarray([fit.success for fit in fits])
    if not eligible.any():
        raise RuntimeError("No successful CPU trajectory produced a finite objective")
    winner_index = int(np.flatnonzero(eligible)[np.argmin(objectives[eligible])])
    winner = fits[winner_index]
    jacobian = winner.jacobian
    rank = int(np.linalg.matrix_rank(jacobian))
    sample_count, _ = jacobian.shape
    degrees_of_freedom = sample_count - rank
    covariance = np.linalg.pinv(jacobian.T @ jacobian, hermitian=True)
    if degrees_of_freedom > 0:
        covariance *= winner.objective / degrees_of_freedom
    else:
        covariance[:] = np.nan
    standard_errors = np.sqrt(np.maximum(np.diag(covariance), 0.0))
    return FitResult(
        fits=fits,
        winner_index=winner_index,
        parameters=winner.parameters.copy(),
        objective=winner.objective,
        jacobian=jacobian.copy(),
        jacobian_rank=rank,
        covariance=covariance,
        standard_errors=standard_errors,
    )


def fit_reference(observation: Observation, starts: np.ndarray) -> FitResult:
    """Fit every explicit start in float64 and select the first best objective."""
    return _summarize(_fit_starts(observation, _validate_starts(starts)))


def _contacts_bounds(parameters: np.ndarray, tolerance: float) -> bool:
    lower, upper = parameter_bounds()
    distance = np.minimum(parameters - lower, upper - parameters)
    return bool(np.any(distance <= tolerance * np.maximum(upper - lower, 1.0)))


def certify_candidates(
    observation: Observation,
    candidate_parameters: np.ndarray,
    candidate_objectives: Sequence[float] | np.ndarray,
    reference_starts: np.ndarray,
    *,
    top_k: int = 3,
    ambiguity_rtol: float = 1e-8,
    bound_tolerance: float = 1e-8,
) -> CertificationResult:
    """Rank GPU candidates, CPU-polish them, and fall back when uncertifiable.

    Candidates tied with the Kth objective are included (top-K expansion).
    Nonfinite candidate data, bound contacts, or ambiguous polished minima force
    the complete all-start CPU reference path.
    """
    candidates = np.asarray(candidate_parameters, dtype=np.float64)
    objectives = np.asarray(candidate_objectives, dtype=np.float64)
    starts = _validate_starts(reference_starts)
    if candidates.ndim != 2 or candidates.shape[1] != N_PARAMETERS:
        raise ValueError("Candidate parameters must have shape [candidates, 15]")
    if objectives.shape != (candidates.shape[0],):
        raise ValueError("Candidate objectives must match the candidate count")
    if not 1 <= top_k <= candidates.shape[0]:
        raise ValueError("top_k must be between one and the candidate count")

    lower, upper = parameter_bounds()
    finite = (
        np.isfinite(objectives)
        & np.isfinite(candidates).all(axis=1)
        & np.all(candidates >= lower, axis=1)
        & np.all(candidates <= upper, axis=1)
    )
    ranked = np.concatenate(
        (
            np.flatnonzero(finite)[np.argsort(objectives[finite], kind="stable")],
            np.flatnonzero(~finite),
        )
    )
    reason: str | None = None
    if not finite.all():
        reason = "nonfinite or out-of-bounds candidate"
        selected = ranked[:top_k]
    else:
        cutoff = objectives[ranked[top_k - 1]]
        scale = max(abs(cutoff), np.finfo(np.float64).tiny)
        selected = ranked[objectives[ranked] <= cutoff + ambiguity_rtol * scale]

    selected_finite = selected[finite[selected]]
    if selected_finite.size:
        polished = _summarize(_fit_starts(observation, candidates[selected_finite]))
        polished_objectives = np.array([fit.objective for fit in polished.fits])
        best = polished.objective
        close = np.isclose(polished_objectives, best, rtol=ambiguity_rtol, atol=0.0)
        centers = np.array([fit.parameters[1] for fit in polished.fits])[close]
        centers_separated = centers.size > 1 and np.ptp(centers) > ambiguity_rtol * max(
            abs(centers[0]), 1.0
        )
        if centers_separated:
            reason = reason or "ambiguous polished basins"
        contact = any(
            _contacts_bounds(fit.parameters, bound_tolerance) for fit in polished.fits
        )
        if contact:
            reason = reason or "bound contact"
        if not np.isfinite(polished.jacobian).all():
            reason = reason or "nonfinite CPU Jacobian"
    else:
        reason = reason or "no finite candidate"

    used_fallback = reason is not None
    result = fit_reference(observation, starts) if used_fallback else polished
    return CertificationResult(
        result=result,
        ranked_candidate_indices=ranked,
        polished_candidate_indices=selected_finite,
        used_fallback=used_fallback,
        fallback_reason=reason,
    )
