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


def _build_energy_profile(
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
    pitch_min: float = 150.0,
    pitch_max: float = 180.0,
    min_band_points: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Build an energy profile by averaging normalized flux in a pitch band.

    Args:
        energies: Energy values (n_energies,)
        pitches: Pitch angle grid (n_energies, n_pitch_bins)
        norm2d: Normalized flux grid (n_energies, n_pitch_bins)
        pitch_min: Lower bound of pitch band (degrees)
        pitch_max: Upper bound of pitch band (degrees)
        min_band_points: Minimum valid pitch points per energy row

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
    return energies[order], values[order]


def detect_peak(
    profile: np.ndarray,
    *,
    contrast: float = 1.2,
    min_peak: float = 2.0,
    neighbor_window: int = 1,
    edge_skip: int = 1,
    min_neighbor: float = 1.5,
) -> BeamDetectionResult:
    """Detect if the energy profile has a valid peak.

    Args:
        profile: Energy profile (mean normalized flux per energy)
        contrast: Peak must exceed neighbors by this multiplicative factor
        min_peak: Minimum normalized value to qualify as a peak
        neighbor_window: Number of energy bins on each side for peak comparison
        edge_skip: Skip this many energy bins at both ends
        min_neighbor: Minimum normalized value for adjacent bin to confirm peak

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
        if not np.isfinite(left_max) or not np.isfinite(right_max):
            continue

        if not (value >= left_max * contrast and value >= right_max * contrast):
            continue

        if left_max < min_neighbor and right_max < min_neighbor:
            continue

        if value > best_value:
            best_value = value
            best_idx = idx

    if best_idx is not None:
        return BeamDetectionResult(True, best_idx, float(best_value), None)
    return BeamDetectionResult(False, None, None, None)


# Default thresholds matching losscone_peak_scan.py
DEFAULT_PITCH_MIN = 150.0
DEFAULT_PITCH_MAX = 180.0
DEFAULT_MIN_BAND_POINTS = 5
DEFAULT_MIN_PEAK = 2.0
DEFAULT_MIN_NEIGHBOR = 1.5
DEFAULT_PEAK_CONTRAST = 1.2
DEFAULT_NEIGHBOR_WINDOW = 1
DEFAULT_EDGE_SKIP = 1
