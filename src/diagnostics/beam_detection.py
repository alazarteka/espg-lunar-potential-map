"""Beam detection utilities for loss-cone normalized flux analysis."""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeamDetectionResult:
    has_peak: bool
    peak_idx: int | None
    peak_value: float | None
    beam_energy: float | None


# Default thresholds matching losscone_peak_scan.py
DEFAULT_PITCH_MIN = 150.0
DEFAULT_PITCH_MAX = 180.0
DEFAULT_MIN_BAND_POINTS = 5
DEFAULT_MIN_PEAK = 2.0
DEFAULT_MIN_NEIGHBOR = 1.5
DEFAULT_PEAK_CONTRAST = 1.2
DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_EDGE_SKIP = 1
DEFAULT_ENERGY_MIN = 20.0
DEFAULT_ENERGY_MAX = 500.0
DEFAULT_HIGH_ENERGY_FLOOR = 400.0
DEFAULT_HIGH_ENERGY_FACTOR = 2.0
DEFAULT_HIGH_ENERGY_RATIO_MAX = 0.5
DEFAULT_HIGH_ENERGY_MIN_POINTS = 2
DEFAULT_PEAK_HALF_FRACTION = 0.5
DEFAULT_PEAK_WIDTH_MAX = 4
DEFAULT_CONTIGUITY_MIN_BINS = 3


def _build_energy_profile(
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
    pitch_min: float = DEFAULT_PITCH_MIN,
    pitch_max: float = DEFAULT_PITCH_MAX,
    min_band_points: int = DEFAULT_MIN_BAND_POINTS,
    energy_min: float | None = DEFAULT_ENERGY_MIN,
    energy_max: float | None = DEFAULT_ENERGY_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an energy profile by averaging normalized flux in a pitch band.

    Args:
        energies: Energy values (n_energies,)
        pitches: Pitch angle grid (n_energies, n_pitch_bins)
        norm2d: Normalized flux grid (n_energies, n_pitch_bins)
        pitch_min: Lower bound of pitch band (degrees)
        pitch_max: Upper bound of pitch band (degrees)
        min_band_points: Minimum valid pitch points per energy row
        energy_min: Minimum energy to include (eV)
        energy_max: Maximum energy to include (eV)

    Returns:
        Tuple of (sorted_energies, profile_values) where profile_values contains
        the mean normalized flux in the pitch band for each energy.
    """
    band_mask = (pitches >= pitch_min) & (pitches <= pitch_max)
    values = np.full(energies.shape, np.nan, dtype=float)

    for idx in range(len(energies)):
        row_mask = band_mask[idx] & np.isfinite(norm2d[idx])
        if min_band_points > 0 and np.count_nonzero(row_mask) < min_band_points:
            continue
        if np.any(row_mask):
            values[idx] = float(np.nanmean(norm2d[idx][row_mask]))

    order = np.argsort(energies)
    energies_sorted = energies[order]
    values_sorted = values[order]
    if energy_min is not None:
        values_sorted[energies_sorted < energy_min] = np.nan
    if energy_max is not None:
        values_sorted[energies_sorted > energy_max] = np.nan
    return energies_sorted, values_sorted


def _peak_width_bins(
    profile: np.ndarray,
    peak_idx: int,
    peak_value: float,
    half_fraction: float,
) -> int:
    threshold = peak_value * half_fraction
    width = 1
    for idx in range(peak_idx - 1, -1, -1):
        value = profile[idx]
        if np.isfinite(value) and value >= threshold:
            width += 1
        else:
            break
    for idx in range(peak_idx + 1, len(profile)):
        value = profile[idx]
        if np.isfinite(value) and value >= threshold:
            width += 1
        else:
            break
    return width


def _passes_high_energy_deficit(
    profile: np.ndarray,
    energies: np.ndarray,
    peak_idx: int,
    peak_value: float,
    *,
    high_energy_floor: float,
    high_energy_factor: float,
    max_high_ratio: float,
    min_points: int,
) -> bool:
    peak_energy = energies[peak_idx]
    threshold = max(high_energy_floor, high_energy_factor * peak_energy)
    high_mask = (energies >= threshold) & np.isfinite(profile)
    if np.count_nonzero(high_mask) < min_points:
        return True
    high_mean = float(np.nanmean(profile[high_mask]))
    if not np.isfinite(high_mean):
        return True
    return high_mean <= max_high_ratio * peak_value


def _max_true_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            if current > best:
                best = current
        else:
            current = 0
    return best


def _passes_pitch_contiguity(
    norm2d: np.ndarray,
    pitches: np.ndarray,
    peak_idx: int,
    *,
    pitch_min: float,
    value_floor: float,
    min_bins: int,
) -> bool:
    if peak_idx < 0 or peak_idx >= norm2d.shape[0]:
        return True
    row = norm2d[peak_idx]
    pitch_row = pitches[peak_idx]
    valid = np.isfinite(row) & np.isfinite(pitch_row)
    if np.count_nonzero(valid) < min_bins:
        return True
    order = np.argsort(pitch_row[valid])
    pitch_sorted = pitch_row[valid][order]
    row_sorted = row[valid][order]
    mask = (pitch_sorted >= pitch_min) & (row_sorted >= value_floor)
    return _max_true_run(mask) >= min_bins


def detect_peak(
    profile: np.ndarray,
    *,
    energies: np.ndarray | None = None,
    norm2d: np.ndarray | None = None,
    pitches: np.ndarray | None = None,
    contrast: float = DEFAULT_PEAK_CONTRAST,
    min_peak: float = DEFAULT_MIN_PEAK,
    neighbor_window: int = DEFAULT_NEIGHBOR_WINDOW,
    edge_skip: int = DEFAULT_EDGE_SKIP,
    min_neighbor: float = DEFAULT_MIN_NEIGHBOR,
    check_high_energy: bool = True,
    high_energy_floor: float = DEFAULT_HIGH_ENERGY_FLOOR,
    high_energy_factor: float = DEFAULT_HIGH_ENERGY_FACTOR,
    high_energy_ratio_max: float = DEFAULT_HIGH_ENERGY_RATIO_MAX,
    high_energy_min_points: int = DEFAULT_HIGH_ENERGY_MIN_POINTS,
    check_peak_width: bool = True,
    peak_half_fraction: float = DEFAULT_PEAK_HALF_FRACTION,
    peak_width_max: int = DEFAULT_PEAK_WIDTH_MAX,
    check_pitch_contiguity: bool = False,
    contiguity_pitch_min: float = DEFAULT_PITCH_MIN,
    contiguity_min_value: float | None = None,
    contiguity_min_bins: int = DEFAULT_CONTIGUITY_MIN_BINS,
) -> BeamDetectionResult:
    """Detect if the energy profile has a valid peak.

    Args:
        profile: Energy profile (mean normalized flux per energy)
        energies: Sorted energy values matching profile (optional)
        energies: Sorted energy values matching profile (optional)
        norm2d: Normalized flux grid for pitch contiguity checks (optional)
        pitches: Pitch angle grid for pitch contiguity checks (optional)
        contrast: Peak must exceed neighbors by this multiplicative factor
        min_peak: Minimum normalized value to qualify as a peak
        neighbor_window: Number of energy bins on each side for peak comparison
        edge_skip: Skip this many energy bins at both ends
        min_neighbor: Minimum normalized value for adjacent bin to confirm peak
        check_high_energy: Enforce high-energy deficit check
        high_energy_floor: Minimum high-energy threshold (eV)
        high_energy_factor: Scale factor applied to peak energy for deficit check
        high_energy_ratio_max: Max allowed high-energy mean relative to peak
        high_energy_min_points: Minimum points required for deficit check
        check_peak_width: Enforce peak width constraint
        peak_half_fraction: Fraction of peak used for width measurement
        peak_width_max: Maximum contiguous bins above half-peak
        check_pitch_contiguity: Enforce pitch contiguity check
        contiguity_pitch_min: Minimum pitch angle for contiguity check
        contiguity_min_value: Minimum normalized value for contiguity check
        contiguity_min_bins: Minimum contiguous bins required

    Returns:
        BeamDetectionResult with detection status and peak details.
    """
    if profile.size == 0:
        return BeamDetectionResult(False, None, None, None)

    window = max(1, neighbor_window)
    start = max(edge_skip, window)
    end = profile.size - max(edge_skip, window)
    if end <= start:
        return BeamDetectionResult(False, None, None, None)

    best_idx: int | None = None
    best_value: float = -np.inf

    for idx in range(start, end):
        value = profile[idx]
        if not np.isfinite(value) or value < min_peak:
            continue
        left = profile[idx - window : idx]
        right = profile[idx + 1 : idx + 1 + window]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            left_max = np.nanmax(left) if left.size else np.nan
            right_max = np.nanmax(right) if right.size else np.nan
        left_finite = np.isfinite(left_max)
        right_finite = np.isfinite(right_max)
        if not left_finite and not right_finite:
            continue

        if left_finite and value < left_max * contrast:
            continue
        if right_finite and value < right_max * contrast:
            continue

        if not (
            (left_finite and left_max >= min_neighbor)
            or (right_finite and right_max >= min_neighbor)
        ):
            continue

        if check_peak_width:
            width = _peak_width_bins(profile, idx, value, peak_half_fraction)
            if width > peak_width_max:
                continue

        if (
            check_high_energy
            and energies is not None
            and len(energies) > idx
            and not _passes_high_energy_deficit(
                profile,
                energies,
                idx,
                value,
                high_energy_floor=high_energy_floor,
                high_energy_factor=high_energy_factor,
                max_high_ratio=high_energy_ratio_max,
                min_points=high_energy_min_points,
            )
        ):
            continue

        if check_pitch_contiguity and norm2d is not None and pitches is not None:
            floor = (
                float(contiguity_min_value)
                if contiguity_min_value is not None
                else min_neighbor
            )
            if not _passes_pitch_contiguity(
                norm2d,
                pitches,
                idx,
                pitch_min=contiguity_pitch_min,
                value_floor=floor,
                min_bins=contiguity_min_bins,
            ):
                continue

        if value > best_value:
            best_value = value
            best_idx = idx

    if best_idx is not None:
        beam_energy = None
        if energies is not None and len(energies) > best_idx:
            beam_energy = float(energies[best_idx])
        return BeamDetectionResult(True, best_idx, float(best_value), beam_energy)
    return BeamDetectionResult(False, None, None, None)
