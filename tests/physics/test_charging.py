# TODO: Add physics validation tests:
# - Verify 0.25 isotropic factor (∫cos(θ)dΩ/4π = 1/4)
# - Test linear scaling with density: J(2n) = 2*J(n)
# - Compare to analytic Maxwellian limit (κ→∞)

import numpy as np
import pytest

import src.config as cfg
from src.physics.charging import (
    electron_current_density,
    electron_current_density_magnitude,
    kappa_current_density_analytic,
)
from src.physics.kappa import KappaParams


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
