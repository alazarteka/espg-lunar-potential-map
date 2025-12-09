"""Core analysis logic for ESPG maps and statistics."""

import logging
from dataclasses import dataclass

import numpy as np

from ..temporal.dataset import TemporalDataset
from ..temporal.reconstruction import compute_potential_series
from .sites import Site

# Default current density for ESPG power calculation (A/m^2)
# Using 1 micro-Amp per square meter as a representative value
DEFAULT_CURRENT_DENSITY = 1e-6


@dataclass(slots=True)
class GlobalStats:
    """Global statistical maps."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    mean_potential: np.ndarray
    p95_potential: np.ndarray
    frac_500V: np.ndarray
    frac_1kV: np.ndarray
    frac_2kV: np.ndarray
    mean_power: np.ndarray
    p95_power: np.ndarray


@dataclass(slots=True)
class SiteStats:
    """Statistics for a specific site."""

    site: Site
    mean_potential: float
    p95_potential: float
    frac_500V: float
    frac_1kV: float
    frac_2kV: float
    mean_power: float
    p95_power: float
    risk_assessment: str


def compute_global_stats(
    dataset: TemporalDataset,
    current_density: float = DEFAULT_CURRENT_DENSITY,
    lat_steps: int = 180,
    lon_steps: int = 360,
) -> GlobalStats:
    """
    Compute global statistical maps from temporal coefficients.

    Args:
        dataset: Loaded TemporalDataset with coefficients.
        current_density: Representative current density (A/m^2) for power calc.
        lat_steps: Number of latitude steps for the grid.
        lon_steps: Number of longitude steps for the grid.

    Returns:
        GlobalStats object containing 2D maps.
    """
    logging.info("Computing potential series on %dx%d grid...", lat_steps, lon_steps)
    lats, lons, maps = compute_potential_series(
        dataset.coeffs, dataset.lmax, lat_steps, lon_steps
    )

    # Use absolute potential |U|
    abs_maps = np.abs(maps)

    logging.info("Computing statistics across %d time steps...", maps.shape[0])

    mean_potential = np.mean(abs_maps, axis=0)
    p95_potential = np.percentile(abs_maps, 95, axis=0)

    # Fraction of time above thresholds
    frac_500V = np.mean(abs_maps > 500.0, axis=0)
    frac_1kV = np.mean(abs_maps > 1000.0, axis=0)
    frac_2kV = np.mean(abs_maps > 2000.0, axis=0)

    # Power calculations: P = I * |U|
    # Result is W/m^2 if I is A/m^2 and U is V
    mean_power = mean_potential * current_density
    p95_power = p95_potential * current_density

    return GlobalStats(
        latitudes=lats,
        longitudes=lons,
        mean_potential=mean_potential,
        p95_potential=p95_potential,
        frac_500V=frac_500V,
        frac_1kV=frac_1kV,
        frac_2kV=frac_2kV,
        mean_power=mean_power,
        p95_power=p95_power,
    )


def extract_site_stats(
    dataset: TemporalDataset,
    site: Site,
    current_density: float = DEFAULT_CURRENT_DENSITY,
) -> SiteStats:
    """
    Compute statistics for a single specific site.
    Evaluating spherical harmonics exactly at the site coordinates.
    """
    from ..temporal.reconstruction import _sph_harm

    # Reconstruct time series at specific point
    # U(t) = sum(a_lm(t) * Y_lm(phi, theta))
    # theta = colatitude, phi = longitude (radians)

    lat_rad = np.deg2rad(site.lat)
    lon_rad = np.deg2rad(site.lon)
    colat = (np.pi / 2.0) - lat_rad

    lmax = dataset.lmax
    n_coeffs = (lmax + 1) ** 2
    basis_vec = np.empty(n_coeffs, dtype=np.complex128)

    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            basis_vec[idx] = _sph_harm(m, l, lon_rad, colat)
            idx += 1

    # Matrix multiply: (n_times, n_coeffs) @ (n_coeffs,) -> (n_times,)
    potential_series = np.real(dataset.coeffs @ basis_vec)
    abs_series = np.abs(potential_series)

    mean_u = float(np.mean(abs_series))
    p95_u = float(np.percentile(abs_series, 95))
    frac_500 = float(np.mean(abs_series > 500.0))
    frac_1k = float(np.mean(abs_series > 1000.0))
    frac_2k = float(np.mean(abs_series > 2000.0))

    mean_p = mean_u * current_density
    p95_p = p95_u * current_density

    # Simple assessment logic
    if mean_p > 1e-3:  # > 1 mW/m^2 average
        assessment = "High Resource"
    elif p95_p > 5e-3:  # > 5 mW/m^2 peak
        assessment = "Moderate Resource / High Peak"
    elif frac_1k > 0.1:
        assessment = "High Risk / Variable"
    else:
        assessment = "Low Resource / Low Risk"

    return SiteStats(
        site=site,
        mean_potential=mean_u,
        p95_potential=p95_u,
        frac_500V=frac_500,
        frac_1kV=frac_1k,
        frac_2kV=frac_2k,
        mean_power=mean_p,
        p95_power=p95_p,
        risk_assessment=assessment,
    )
