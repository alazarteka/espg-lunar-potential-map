# Tests for src/kappa.py

import numpy as np
import pytest

import pint
from pint import Quantity

from src.physics.kappa import KappaParams, kappa_distribution, directional_flux, omnidirectional_flux, omnidirectional_flux_integrated
from src.utils.units import *

def test_kappa_params_to_tuple():
    """Test conversion of KappaParams to tuple."""
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    expected_result = (1e6, 5.0, 1e3)

    params = KappaParams(density=density, kappa=kappa, theta=theta)
    result = params.to_tuple()
    
    assert isinstance(result, tuple)
    np.testing.assert_allclose(result, expected_result, rtol=1e-10)

def test_kappa_params_invalid_types():
    """Test KappaParams with invalid types."""
    with pytest.raises(TypeError, match="density must be a pint Quantity"):
        KappaParams(density=1e6, kappa=5.0, theta=1e3 * ureg.meter / ureg.second)
    
    with pytest.raises(TypeError, match="kappa must be a float"):
        KappaParams(density=1e6 * ureg.particle / ureg.meter**3, kappa="5.0", theta=1e3 * ureg.meter / ureg.second)
    
    with pytest.raises(TypeError, match="theta must be a pint Quantity"):
        KappaParams(density=1e6 * ureg.particle / ureg.meter**3, kappa=5.0, theta=1e3)

def test_kappa_params_unit_conversion():
    """Test that KappaParams correctly converts units."""
    density = 2e6 * ureg.particle / ureg.liter
    kappa = 3.5
    theta = 36.0 * ureg.kilometer / ureg.hour
    
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    
    # Check that units are converted to base units
    assert params.density.units == ureg.particle / ureg.meter**3
    assert params.theta.units == ureg.meter / ureg.second
    
    # Check magnitudes after conversion
    np.testing.assert_allclose(params.density.magnitude, 2e9, rtol=1e-10)
    np.testing.assert_allclose(params.theta.magnitude, 10.0, rtol=1e-10)

def test_kappa_distribution_basic():
    """Test basic kappa distribution calculation."""
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    
    velocity = 500 * ureg.meter / ureg.second
    result = kappa_distribution(params, velocity)

    assert isinstance(result, Quantity)
    assert result.magnitude > 0
    assert result.units == ureg.particle / (ureg.meter**3 * (ureg.meter / ureg.second) ** 3)

def test_kappa_distribution_invalid_velocity():
    """Test kappa distribution with invalid velocity type."""
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    
    with pytest.raises(TypeError, match="velocity must be a pint Quantity"):
        kappa_distribution(params, 500)

@pytest.mark.parametrize(
    "velocity_magnitude_range, theta, kappa",
    [
        ((0, 100) * ureg.meter / ureg.second, 1e3 * ureg.meter / ureg.second, 2.0),
        ((8e4, 8e5) * ureg.meter / ureg.second, 0.5 * ureg.meter / ureg.second, 8.0),
        ((2000, 2000) * ureg.meter / ureg.second, 0.25 * ureg.meter / ureg.second, 5.0),
    ]
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
                (kappa * theta ** 2 + v_i ** 2)  / (kappa * theta ** 2 + v_j ** 2)
            ) ** -(kappa + 1)
            assert np.isclose(
                results[i].magnitude / results[j + i].magnitude,
                expected_result_ratio,
                rtol=1e-2
            ), f"Failed for velocities {v_i} and {v_j}"


def test_kappa_directional_flux_basic():
    """Test basic isotropic kappa distribution directional flux calculation."""
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    
    energy = 1e6 * ureg.electron_volt
    result = directional_flux(params, energy)
    
    assert isinstance(result, Quantity)
    assert result.magnitude > 0
    assert result.units == ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)

def test_kappa_omnidirectional_flux_basic():
    """Test basic isotropic kappa distribution omnidirectional flux calculation."""
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    
    energy = 1e6 * ureg.electron_volt
    result = omnidirectional_flux(params, energy)
    
    assert isinstance(result, Quantity)
    assert result.magnitude > 0
    assert result.units == ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)

def test_omnidirectional_flux_integrated_basic():
    density = 1e6 * ureg.particle / ureg.meter**3
    kappa = 5.0
    theta = 1e3 * ureg.meter / ureg.second
    params = KappaParams(density=density, kappa=kappa, theta=theta)

    energy_centers = np.linspace(1e1, 1e4, num=16) * ureg.electron_volt
    energy_bounds = np.column_stack([0.75 * energy_centers, 1.25 * energy_centers])

    result = omnidirectional_flux_integrated(params, energy_bounds)

    assert isinstance(result, Quantity)
    assert np.all(result.magnitude >= 0)
    assert result.units == ureg.particle / (ureg.centimeter**2 * ureg.second)

# def test_kappa_fit_result_initialization():
#     """Test KappaFitResult initialization."""
#     density = 1e6 * ureg.particle / ureg.meter**3
#     kappa = 5.0
#     theta = 1e3 * ureg.meter / ureg.second
#     params = KappaParams(density=density, kappa=kappa, theta=theta)
    
#     result = KappaFitResult(
#         params=params,
#         chi2=1.5,
#         success=True,
#         message="Optimization successful"
#     )
    
#     assert result.params == params
#     assert result.chi2 == 1.5
#     assert result.success is True
#     assert result.message == "Optimization successful"
#     assert result.sigma is None

# def test_kappa_fit_result_with_sigma():
#     """Test KappaFitResult with sigma parameter."""
#     density = 1e6 * ureg.particle / ureg.meter**3
#     kappa = 5.0
#     theta = 1e3 * ureg.meter / ureg.second
#     params = KappaParams(density=density, kappa=kappa, theta=theta)
    
#     sigma_density = 1e5 * ureg.particle / ureg.meter**3
#     sigma_kappa = 0.5
#     sigma_theta = 1e2 * ureg.meter / ureg.second
#     sigma = KappaParams(density=sigma_density, kappa=sigma_kappa, theta=sigma_theta)
    
#     result = KappaFitResult(
#         params=params,
#         chi2=1.5,
#         success=True,
#         message="Optimization successful",
#         sigma=sigma
#     )
    
#     assert result.sigma == sigma