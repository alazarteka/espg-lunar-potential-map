# Tests for src/kappa.py

import numpy as np
import pytest
from pint import Quantity
from scipy.integrate import simpson

from src.physics.kappa import (
    KappaParams,
    directional_flux,
    kappa_distribution,
    omnidirectional_flux,
    omnidirectional_flux_integrated,
)
from src.utils.units import *


@pytest.fixture
def base_kappa_params():
    """Fixture to provide a base set of KappaParams."""
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    return KappaParams(density=density, kappa=kappa, theta=theta)


@pytest.fixture(
    params=[
        (
            1e6 * ureg.particle / ureg.meter**3,
            5.0,
            1e3 * ureg.meter / ureg.second,
            (1e6, 5.0, 1e3),
        ),
        (
            2e6 * ureg.particle / ureg.liter,
            3.5,
            36.0 * ureg.kilometer / ureg.hour,
            (2e9, 3.5, 10.0),
        ),
        (
            1e9 * ureg.particle / ureg.centimeter**3,
            4.0,
            500 * ureg.meter / ureg.second,
            (1e15, 4.0, 500),
        ),
    ]
)
def kappa_params_set(request):
    """Fixture to provide different sets of KappaParams and their expected tuple representation."""
    density, kappa, theta, expected_tuple = request.param
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    return params, expected_tuple


def test_kappa_params_conversion(kappa_params_set):
    """Test conversion of KappaParams to tuple and unit conversion."""
    params, expected_result = kappa_params_set

    # Test to_tuple()
    result = params.to_tuple()
    assert isinstance(result, tuple)
    np.testing.assert_allclose(result, expected_result, rtol=1e-10)

    # Test unit conversion
    assert params.density.units == ureg.particle / ureg.meter**3
    assert params.theta.units == ureg.meter / ureg.second
    np.testing.assert_allclose(params.density.magnitude, expected_result[0], rtol=1e-10)
    np.testing.assert_allclose(params.kappa, expected_result[1], rtol=1e-10)
    np.testing.assert_allclose(params.theta.magnitude, expected_result[2], rtol=1e-10)


def test_kappa_params_invalid_types():
    """Test KappaParams with invalid types."""
    with pytest.raises(TypeError, match="density must be a pint Quantity"):
        KappaParams(density=1e6, kappa=5.0, theta=1e3 * ureg.meter / ureg.second)

    with pytest.raises(TypeError, match="kappa must be a float"):
        KappaParams(
            density=1e6 * ureg.particle / ureg.meter**3,
            kappa="5.0",
            theta=1e3 * ureg.meter / ureg.second,
        )

    with pytest.raises(TypeError, match="theta must be a pint Quantity"):
        KappaParams(density=1e6 * ureg.particle / ureg.meter**3, kappa=5.0, theta=1e3)


def test_kappa_distribution_basic(base_kappa_params):
    """Test basic kappa distribution calculation."""

    velocity = 500 * ureg.meter / ureg.second
    result = kappa_distribution(base_kappa_params, velocity)

    assert isinstance(result, Quantity)
    assert result.magnitude > 0
    assert result.units == ureg.particle / (
        ureg.meter**3 * (ureg.meter / ureg.second) ** 3
    )


def test_kappa_distribution_invalid_velocity(base_kappa_params):
    """Test kappa distribution with invalid velocity type."""

    with pytest.raises(TypeError, match="velocity must be a pint Quantity"):
        kappa_distribution(base_kappa_params, 500)


@pytest.mark.parametrize(
    "velocity_magnitude_range, theta, kappa",
    [
        ((0, 100) * ureg.meter / ureg.second, 1e3 * ureg.meter / ureg.second, 2.0),
        ((8e4, 8e5) * ureg.meter / ureg.second, 0.5 * ureg.meter / ureg.second, 8.0),
        ((2000, 2000) * ureg.meter / ureg.second, 0.25 * ureg.meter / ureg.second, 5.0),
    ],
)
def test_kappa_distribution_velocity_dependence(
    velocity_magnitude_range: tuple[Speed, Speed],
    theta: Speed,
    kappa: float,
):
    """
    Test that kappa distribution decreases with increasing velocity, with a characteristic ratio.
    """
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = kappa
    theta = theta
    params = KappaParams(density=density, kappa=kappa, theta=theta)

    velocities = np.linspace(*velocity_magnitude_range, num=10)
    results = kappa_distribution(params, velocities)

    assert isinstance(results, Quantity)
    assert np.all(results.magnitude >= 0)

    delta_v_per_delta_t = results[1:] - results[:-1]
    assert np.all(delta_v_per_delta_t <= 0)

    for i, v_i in enumerate(velocities):
        for j, v_j in enumerate(velocities[i:]):
            expected_result_ratio = (
                (kappa * theta**2 + v_i**2) / (kappa * theta**2 + v_j**2)
            ) ** -(kappa + 1)
            assert np.isclose(
                results[i].magnitude / results[j + i].magnitude,
                expected_result_ratio,
                rtol=1e-2,
            ), f"Failed for velocities {v_i} and {v_j}"


def test_kappa_distribution_density_dependence(kappa_params_set):
    """Test that the integral of the velocity distribution over all velocities equals the density."""
    params, _ = kappa_params_set

    velocities = np.geomspace(1e-2, 1e50, num=1001) * ureg.meter / ureg.second
    distribution = kappa_distribution(params, velocities)

    fv2 = velocities**2 * distribution

    integral = simpson(
        fv2.to(ureg.particle / (ureg.meter**3 * (ureg.meter / ureg.second))).magnitude,
        velocities.to(ureg.meter / ureg.second).magnitude,
    ) * (ureg.particle / ureg.meter**3)
    expected_density = 4 * np.pi * integral
    assert np.isclose(
        expected_density.magnitude, params.density.magnitude, rtol=1e-2
    ), f"Expected density {params.density.magnitude}, got {expected_density.magnitude}"


def test_kappa_directional_flux_basic(base_kappa_params):
    """Test basic isotropic kappa distribution directional flux calculation."""

    energy = 1e6 * ureg.electron_volt
    result = directional_flux(base_kappa_params, energy)

    assert isinstance(result, Quantity)
    assert result.magnitude > 0
    assert result.units == ureg.particle / (
        ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt
    )


def test_kappa_omnidirectional_flux_basic(base_kappa_params):
    """Test basic isotropic kappa distribution omnidirectional flux calculation."""

    energy = 1e6 * ureg.electron_volt
    result = omnidirectional_flux(base_kappa_params, energy)

    assert isinstance(result, Quantity)
    assert result.magnitude > 0
    assert result.units == ureg.particle / (
        ureg.centimeter**2 * ureg.second * ureg.electron_volt
    )


def test_omnidirectional_flux_integrated_basic(base_kappa_params):

    energy_centers = np.linspace(1e1, 1e4, num=16) * ureg.electron_volt
    energy_bounds = np.column_stack([0.75 * energy_centers, 1.25 * energy_centers])

    result = omnidirectional_flux_integrated(base_kappa_params, energy_bounds)

    assert isinstance(result, Quantity)
    assert np.all(result.magnitude >= 0)
    assert result.units == ureg.particle / (ureg.centimeter**2 * ureg.second)
