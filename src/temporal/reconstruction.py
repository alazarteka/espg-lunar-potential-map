"""Reconstruction helpers for temporal spherical harmonic datasets.

These helpers evaluate the fitted a_lm(t) coefficients on a lat/lon grid.
That joint space-time fit is the identifiability / sampling-limits analysis
behind the paper's negative result — Lunar Prospector's instantaneous
spatial coverage is too sparse to jointly identify the surface potential in
space and time, so the evaluated field is not a recoverable global potential
map. These functions exist to probe and visualize that non-identifiability,
not to deliver a validated map product.
"""

from __future__ import annotations

import numpy as np

from .coefficients import _build_harmonic_design


def format_timestamp(ts: np.datetime64) -> str:
    """Format numpy datetime64 for UI labels."""
    return np.datetime_as_string(ts, unit="m")


def reconstruct_global_map(
    coeffs: np.ndarray,
    lmax: int,
    lat_steps: int = 181,
    lon_steps: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate fitted spherical harmonic coefficients on a lat/lon grid.

    This produces a single time slice of the joint space-time reconstruction
    used to test spatiotemporal identifiability (see module docstring); it is
    not a validated global potential map.
    """
    latitudes = np.linspace(-90.0, 90.0, lat_steps)
    longitudes = np.linspace(-180.0, 180.0, lon_steps)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    design = _build_harmonic_design(lat_grid.ravel(), lon_grid.ravel(), lmax)

    potential_flat = np.real(design @ coeffs)
    potential_map = potential_flat.reshape(lat_grid.shape)
    return latitudes, longitudes, potential_map


def compute_potential_series(
    coeffs: np.ndarray,
    lmax: int,
    lat_steps: int,
    lon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate the reconstructed (non-identifiable) field for each time index.

    See `reconstruct_global_map`: this stacks that per-slice evaluation over
    all coefficient rows for diagnostic/visualization use, not as a
    recoverable global potential map.
    """
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    maps = np.empty((coeffs.shape[0], lat_steps, lon_steps), dtype=np.float32)

    for idx in range(coeffs.shape[0]):
        lats, lons, potential = reconstruct_global_map(
            coeffs[idx], lmax, lat_steps=lat_steps, lon_steps=lon_steps
        )
        if latitudes is None:
            latitudes = lats
        if longitudes is None:
            longitudes = lons
        maps[idx] = potential
    assert latitudes is not None
    assert longitudes is not None
    return latitudes, longitudes, maps


def compute_cell_edges(
    values: np.ndarray, clamp_min: float, clamp_max: float
) -> np.ndarray:
    """Derive cell-edge coordinates from monotonically increasing centers."""
    if values.size == 0:
        raise ValueError("values must contain at least one entry")
    if values.size == 1:
        span = abs(clamp_max - clamp_min)
        half_step = max(1.0, 0.5 * span * 0.01)
        edges = np.array(
            [values[0] - half_step, values[0] + half_step],
            dtype=values.dtype,
        )
        return np.clip(edges, clamp_min, clamp_max)

    diffs = np.diff(values) / 2.0
    edges = np.empty(values.size + 1, dtype=values.dtype)
    edges[1:-1] = values[:-1] + diffs
    edges[0] = values[0] - diffs[0]
    edges[-1] = values[-1] + diffs[-1]
    return np.clip(edges, clamp_min, clamp_max)


def compute_color_limits(
    maps: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    symmetric_percentile: float | None,
) -> tuple[float, float]:
    """Determine global color scale for animations/plots."""
    if (
        symmetric_percentile is not None
        and 0.0 < symmetric_percentile <= 100.0
        and (vmin is None or vmax is None)
    ):
        percentile_value = float(np.nanpercentile(np.abs(maps), symmetric_percentile))
        if percentile_value > 0:
            if vmin is None:
                vmin = -percentile_value
            if vmax is None:
                vmax = percentile_value

    if vmin is None:
        vmin = float(np.nanmin(maps))
    if vmax is None:
        vmax = float(np.nanmax(maps))
    if np.isclose(vmin, vmax):
        delta = max(1.0, abs(vmin) * 0.1 + 1.0)
        vmin -= delta
        vmax += delta
    return vmin, vmax
