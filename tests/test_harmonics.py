import numpy as np

from src.potential_mapper.harmonics import (
    HarmonicFit,
    build_real_spherical_harmonic_matrix,
    fit_surface_harmonics,
    _enumerate_modes,
)
from src.potential_mapper.results import PotentialResults


def _make_results(lat, lon, potential):
    n = len(lat)
    return PotentialResults(
        spacecraft_latitude=np.zeros(n),
        spacecraft_longitude=np.zeros(n),
        projection_latitude=lat,
        projection_longitude=lon,
        spacecraft_potential=np.zeros(n),
        projected_potential=potential,
        spacecraft_in_sun=np.ones(n, dtype=bool),
        projection_in_sun=np.ones(n, dtype=bool),
    )


def test_fit_surface_harmonics_recovers_coefficients():
    rng = np.random.default_rng(42)
    l_max = 3
    num_points = 200
    lat = rng.uniform(-80.0, 80.0, size=num_points)
    lon = rng.uniform(-180.0, 180.0, size=num_points)
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)

    modes = tuple(_enumerate_modes(l_max))
    design = build_real_spherical_harmonic_matrix(theta, phi, modes)
    coeffs_true = rng.normal(scale=0.5, size=design.shape[1])
    potentials = design @ coeffs_true

    results = _make_results(lat, lon, potentials)

    fit = fit_surface_harmonics(results, l_max)

    assert isinstance(fit, HarmonicFit)
    assert fit.l_max == l_max
    np.testing.assert_allclose(fit.coefficients, coeffs_true, atol=1e-8)
    assert fit.residuals.size == num_points
    assert fit.rms < 1e-10


def test_fit_surface_harmonics_handles_invalid_rows():
    rng = np.random.default_rng(7)
    l_max = 2
    lat = rng.uniform(-60.0, 60.0, size=50)
    lon = rng.uniform(-180.0, 180.0, size=50)
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)

    modes = tuple(_enumerate_modes(l_max))
    design = build_real_spherical_harmonic_matrix(theta, phi, modes)
    coeffs_true = np.linspace(-0.2, 0.2, design.shape[1])
    potentials = design @ coeffs_true

    potentials_with_nan = potentials.copy()
    potentials_with_nan[::7] = np.nan
    lat_with_nan = lat.copy()
    lat_with_nan[5] = np.nan

    results = _make_results(lat_with_nan, lon, potentials_with_nan)

    fit = fit_surface_harmonics(results, l_max)

    mask = np.isfinite(potentials_with_nan) & np.isfinite(lat_with_nan)
    np.testing.assert_allclose(fit.coefficients, coeffs_true, atol=1e-8)
    assert np.count_nonzero(fit.valid_mask) == np.count_nonzero(mask)


def test_fit_surface_harmonics_supports_weights_and_regularization():
    rng = np.random.default_rng(21)
    l_max = 2
    lat = rng.uniform(-70.0, 70.0, size=80)
    lon = rng.uniform(-180.0, 180.0, size=80)
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)

    modes = tuple(_enumerate_modes(l_max))
    design = build_real_spherical_harmonic_matrix(theta, phi, modes)
    coeffs_true = rng.normal(scale=0.3, size=design.shape[1])
    potentials = design @ coeffs_true

    weights = rng.uniform(0.5, 2.0, size=lat.shape[0])
    results = _make_results(lat, lon, potentials)

    fit = fit_surface_harmonics(results, l_max, weights=weights, regularization=1e-6)

    np.testing.assert_allclose(fit.coefficients, coeffs_true, atol=1e-6)
    predicted = fit.predict(lat, lon)
    np.testing.assert_allclose(predicted, potentials, atol=1e-6)
