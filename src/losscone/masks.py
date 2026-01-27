from __future__ import annotations

import numpy as np

from src import config


def build_lillis_mask(
    raw_flux: np.ndarray, pitches: np.ndarray | None = None
) -> np.ndarray:
    """
    Build Lillis-style mask based on relative flux thresholds.

    The mask uses raw (unnormalized) flux with per-energy maxima (incident-only
    when pitches are provided) to exclude low/zero bins and bins near the max,
    reducing bias from saturation or background.
    """
    raw_flux = np.asarray(raw_flux)
    if raw_flux.size == 0:
        return np.zeros_like(raw_flux, dtype=bool)

    original_ndim = raw_flux.ndim
    if original_ndim == 1:
        raw_flux = raw_flux[None, :]

    if pitches is not None:
        pitches = np.asarray(pitches)
        if pitches.shape != raw_flux.shape:
            raise ValueError("pitches must match raw_flux shape for Lillis mask")
        incident = pitches < 90.0
        flux_for_max = np.where(incident, raw_flux, np.nan)
    else:
        flux_for_max = raw_flux

    row_max = np.nanmax(flux_for_max, axis=1, keepdims=True)
    row_max = np.where((row_max > 0) & np.isfinite(row_max), row_max, np.nan)
    relative = raw_flux / row_max
    mask = (
        np.isfinite(raw_flux)
        & (raw_flux > 0)
        & (relative > config.LILLIS_RELATIVE_FLUX_MIN)
        & (relative < config.LILLIS_RELATIVE_FLUX_MAX)
    )
    return mask if original_ndim > 1 else mask[0]
