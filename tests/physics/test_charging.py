
import math
import numpy as np
import pytest
from scipy.integrate import simpson

import src.config as cfg
from src.physics.charging import (
    electron_current_density,
    electron_current_density_magnitude,
    kappa_current_density_analytic,
)
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux_magnitude,
    theta_to_temperature_ev,
)


@pytest.fixture
def sample_params():
    """Create a realistic set of plasma parameters for testing."""
    return KappaParams(
        density=1e6 * cfg.ureg.particle / cfg.ureg.meter**3,
        kappa=3.0,
        theta=1e5 * cfg.ureg.meter / cfg.ureg.second,
    )


def test_electron_current_density_units(sample_params):
    cd = electron_current_density(sample_params)
    assert isinstance(cd, cfg.ureg.Quantity)
    assert cd.units == cfg.ureg.ampere / cfg.ureg.meter**2
    assert cd.magnitude > 0


def test_electron_current_density_magnitude(sample_params):
    mag = electron_current_density_magnitude(
        density=1e6,
        kappa=3.0,
        theta=1e5,
        E_min=0.1,
        E_max=1000,
        n_steps=100,
    )
    assert isinstance(mag, float)
    qty = electron_current_density(sample_params).magnitude
    np.testing.assert_allclose(mag, qty, rtol=1e-2)


def test_kappa_current_density_analytic_matches_numeric(sample_params):
    analytic = kappa_current_density_analytic(sample_params).magnitude
    numeric = electron_current_density_magnitude(
        density=sample_params.density.magnitude,
        kappa=sample_params.kappa,
        theta=sample_params.theta.magnitude,
        E_min=1e-3,
        E_max=2e4,
        n_steps=800,
    )
    np.testing.assert_allclose(numeric, analytic, rtol=3e-2)


def test_isotropic_factor(sample_params):
    """
    Verify 0.25 isotropic factor (∫cos(θ)dΩ/4π = 1/4).

    The current density J is related to omnidirectional flux F_omni by:
    J = q * 0.25 * ∫ F_omni(E) dE

    This test verifies that electron_current_density produces a result
    consistent with this relation.
    """
    # Define integration range
    E_min = 0.1
    E_max = 1000.0
    n_steps = 100

    # Calculate current density using the function under test
    # We use the magnitude version for easier numerical comparison
    current_density = electron_current_density_magnitude(
        density=sample_params.density.magnitude,
        kappa=sample_params.kappa,
        theta=sample_params.theta.magnitude,
        E_min=E_min,
        E_max=E_max,
        n_steps=n_steps,
    )

    # Calculate the integral of omnidirectional flux manually
    energies = np.geomspace(E_min, E_max, num=n_steps)
    F_omni = omnidirectional_flux_magnitude(
        density_mag=sample_params.density.magnitude,
        kappa=sample_params.kappa,
        theta_mag=sample_params.theta.magnitude,
        energy_mag=energies,
    )  # particles / (cm^2 s eV)

    # Integrate over energy
    total_flux_cm2 = simpson(F_omni, energies)  # particles / (cm^2 s)
    total_flux_m2 = total_flux_cm2 * 1e4  # particles / (m^2 s)

    # Expected current density = q * 0.25 * total_flux
    expected_current = cfg.ELECTRON_CHARGE_MAGNITUDE * 0.25 * total_flux_m2

    # Verify the factor
    np.testing.assert_allclose(current_density, expected_current, rtol=1e-10)


def test_linear_scaling_density(sample_params):
    """Test linear scaling with density: J(2n) = 2*J(n)."""
    # Calculate J(n)
    j1 = electron_current_density(sample_params)

    # Calculate J(2n)
    params_2n = KappaParams(
        density=sample_params.density * 2,
        kappa=sample_params.kappa,
        theta=sample_params.theta,
    )
    j2 = electron_current_density(params_2n)

    np.testing.assert_allclose(j2.magnitude, 2 * j1.magnitude, rtol=1e-10)


def test_maxwellian_limit():
    """Compare to analytic Maxwellian limit (κ→∞)."""
    # Large kappa to approximate Maxwellian
    # Note: math.gamma overflows for kappa > ~171, so we use a safe large value
    kappa_large = 100.0
    density = 1e6  # particles/m^3
    theta = 1e5  # m/s

    # Analytic Maxwellian current density: J = n * q * sqrt(k_B * T / (2 * pi * m))
    # First, derive T from theta for the kappa distribution
    temp_ev = theta_to_temperature_ev(theta, kappa_large)
    kT_joule = temp_ev * cfg.ELECTRON_CHARGE_MAGNITUDE

    # Calculate expected Maxwellian current
    thermal_velocity = np.sqrt(kT_joule / (2 * np.pi * cfg.ELECTRON_MASS_MAGNITUDE))
    expected_j = density * cfg.ELECTRON_CHARGE_MAGNITUDE * thermal_velocity

    # Calculate using kappa analytic formula
    params = KappaParams(
        density=density * cfg.ureg.particle / cfg.ureg.meter**3,
        kappa=kappa_large,
        theta=theta * cfg.ureg.meter / cfg.ureg.second,
    )
    kappa_j = kappa_current_density_analytic(params).magnitude

    # Compare
    # The factor correction approaches 1 as kappa -> infinity
    np.testing.assert_allclose(kappa_j, expected_j, rtol=5e-3)
