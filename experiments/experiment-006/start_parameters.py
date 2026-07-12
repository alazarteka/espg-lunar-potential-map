"""Deterministic observation-only multistart construction for Experiment 006."""

from __future__ import annotations

from typing import TYPE_CHECKING

import beam_model
import numpy as np
from scipy.optimize import least_squares
from scipy.special import expit, logit

if TYPE_CHECKING:
    import fit_reference

N_STARTS = 12
CENTER_FACTORS = (0.75, 1.0, 1.25, 1.75)


def _canonical_rows(
    observation: fit_reference.Observation,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Return energy-sorted rows with deterministically sorted valid samples."""
    energies = np.asarray(observation.energies_ev, dtype=np.float64)
    low = np.asarray(observation.channel_low_ev, dtype=np.float64)
    high = np.asarray(observation.channel_high_ev, dtype=np.float64)
    row_ids = np.asarray(observation.row_ids, dtype=np.int64)
    pitches = np.asarray(observation.pitches_deg, dtype=np.float64)
    observed = np.asarray(observation.observed, dtype=np.float64)
    if low.shape != energies.shape or high.shape != energies.shape:
        raise ValueError("Channel edges must match the energy rows")
    if row_ids.shape != pitches.shape or pitches.shape != observed.shape:
        raise ValueError("Sample arrays must have matching shapes")

    order = np.argsort(energies, kind="stable")
    rows: list[tuple[np.ndarray, np.ndarray]] = []
    for original_row in order:
        selected = row_ids == original_row
        valid = selected & np.isfinite(pitches) & np.isfinite(observed) & (observed > 0)
        row_pitch = pitches[valid]
        row_values = observed[valid]
        sample_order = np.lexsort((row_values, row_pitch))
        rows.append((row_pitch[sample_order], row_values[sample_order]))
    return energies[order], low[order], high[order], rows


def _field_aligned_excess(rows: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    scores = np.full(len(rows), np.nan, dtype=np.float64)
    for index, (pitch, values) in enumerate(rows):
        aligned = values[pitch >= 157.5]
        shoulder = values[(pitch >= 112.5) & (pitch < 146.25)]
        if aligned.size and shoulder.size:
            scores[index] = np.median(aligned) - np.median(shoulder)
    return scores


def _fit_row_logistic(
    pitches: np.ndarray,
    values: np.ndarray,
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
) -> np.ndarray | None:
    if values.size == 0:
        return None
    # The beam lives at the field-aligned edge. Prefer the uncontaminated support
    # when it contains enough samples, as in Experiment 005's constructor.
    support = pitches <= 150.0
    if np.count_nonzero(support) >= 4:
        pitches = pitches[support]
        values = values[support]
    values = np.maximum(values, np.finfo(np.float64).tiny)

    def residual(parameters: np.ndarray) -> np.ndarray:
        transition = expit((parameters[2] - pitches) / parameters[3])
        return np.log(parameters[0] + parameters[1] * transition) - np.log(values)

    midpoint = 0.5 * (physical_lower + physical_upper)
    starts = (
        midpoint,
        np.array([0.15, 0.85, 105.0, 5.0]),
        np.array([0.15, 0.85, 135.0, 15.0]),
        np.array([0.15, 0.85, 165.0, 15.0]),
    )
    fits = [
        least_squares(
            residual,
            np.clip(start, physical_lower + 1e-8, physical_upper - 1e-8),
            bounds=(physical_lower, physical_upper),
            max_nfev=300,
        )
        for start in starts
    ]
    best = min(fits, key=lambda fit: float(np.dot(fit.fun, fit.fun)))
    return np.asarray(best.x, dtype=np.float64)


def _smooth_coefficients(
    energies: np.ndarray,
    rows: list[tuple[np.ndarray, np.ndarray]],
    beam_rows: list[np.ndarray],
    observation: fit_reference.Observation,
) -> np.ndarray:
    physical_lower = np.asarray(observation.physical_lower, dtype=np.float64)
    physical_upper = np.asarray(observation.physical_upper, dtype=np.float64)
    fitted_rows: list[np.ndarray] = []
    fitted_indices: list[int] = []
    paired_rows = zip(rows, beam_rows, strict=True)
    for index, ((pitches, values), beam) in enumerate(paired_rows):
        fitted = _fit_row_logistic(
            pitches,
            np.maximum(values - beam, np.finfo(np.float64).tiny),
            physical_lower,
            physical_upper,
        )
        if fitted is not None:
            fitted_indices.append(index)
            fitted_rows.append(fitted)
    if not fitted_rows:
        return np.zeros(12, dtype=np.float64)

    fractions = (np.asarray(fitted_rows) - physical_lower) / (
        physical_upper - physical_lower
    )
    latent = logit(np.clip(fractions, 1e-5, 1.0 - 1e-5))
    basis = beam_model.energy_basis(
        energies[np.asarray(fitted_indices)], observation.energy_domain_ev
    )
    projected = [
        np.linalg.lstsq(basis, latent[:, quantity], rcond=None)[0]
        for quantity in range(4)
    ]
    coefficients = np.column_stack(projected).T.ravel()
    return np.clip(coefficients, -12.0, 12.0)


def construct_starts(
    observation: fit_reference.Observation,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    """Construct exactly 12 finite starts from observations and frozen bounds."""
    lower = np.asarray(lower_bounds, dtype=np.float64)
    upper = np.asarray(upper_bounds, dtype=np.float64)
    if lower.shape != (15,) or upper.shape != (15,) or np.any(lower >= upper):
        raise ValueError("Frozen bounds must each have shape (15,) and be ordered")
    energies, channel_low, channel_high, rows = _canonical_rows(observation)
    if energies.size == 0 or not np.isfinite(energies).all():
        raise ValueError("At least one finite energy row is required")

    scores = _field_aligned_excess(rows)
    positive = np.isfinite(scores) & (scores > 0.0)
    if positive.any():
        detected = float(energies[np.nanargmax(np.where(positive, scores, np.nan))])
    else:
        detected = float(np.sqrt(lower[1] * upper[1]))
    amplitude = float(np.max(scores[positive])) if positive.any() else 0.0
    amplitude = float(np.clip(amplitude, lower[0], upper[0]))
    centers = np.clip(detected * np.asarray(CENTER_FACTORS), lower[1], upper[1])
    margin = min(5.0, 0.25 * (upper[2] - lower[2]))
    sigmas = np.linspace(lower[2] + margin, upper[2] - margin, 3)

    starts = []
    for center in centers:
        energy_response = beam_model.integrated_energy_response(
            channel_low, channel_high, float(center), observation.fwhm_fraction
        )
        for sigma in sigmas:
            beam_rows = [
                amplitude
                * energy_response[index]
                * np.exp(-0.5 * np.square((pitch - 180.0) / sigma))
                for index, (pitch, _) in enumerate(rows)
            ]
            coefficients = _smooth_coefficients(energies, rows, beam_rows, observation)
            starts.append(np.concatenate(([amplitude, center, sigma], coefficients)))
    result = np.asarray(starts, dtype=np.float64)
    return np.clip(np.nan_to_num(result, nan=0.0), lower, upper)
