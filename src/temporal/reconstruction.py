"""Reconstruction helpers for temporal spherical harmonic datasets."""

from __future__ import annotations

import numpy as np
from scipy.special import sph_harm_y


def _sph_harm(m: int, l: int, phi, theta):
    """Evaluate spherical harmonics with explicit argument order."""
    return sph_harm_y(l, m, theta, phi)


def format_timestamp(ts: np.datetime64) -> str:
    """Format numpy datetime64 for UI labels."""
    return np.datetime_as_string(ts, unit="m")


def reconstruct_global_map(
    coeffs: np.ndarray,
    lmax: int,
    lat_steps: int = 181,
    lon_steps: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct global potential map from spherical harmonic coefficients."""
    latitudes = np.linspace(-90.0, 90.0, lat_steps)
    longitudes = np.linspace(-180.0, 180.0, lon_steps)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    lat_rad = np.deg2rad(lat_grid.ravel())
    lon_rad = np.deg2rad(lon_grid.ravel())
    colatitudes = (np.pi / 2.0) - lat_rad

    n_points = lat_rad.size
    n_coeffs = coeffs.size
    design = np.empty((n_points, n_coeffs), dtype=np.complex128)

    col_idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            design[:, col_idx] = _sph_harm(m, l, lon_rad, colatitudes)
            col_idx += 1

    potential_flat = np.real(design @ coeffs)
    potential_map = potential_flat.reshape(lat_grid.shape)
    return latitudes, longitudes, potential_map


def compute_potential_series(
    coeffs: np.ndarray,
    lmax: int,
    lat_steps: int,
    lon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reconstructed potential map for each time index."""
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
    assert latitudes is not None and longitudes is not None
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
