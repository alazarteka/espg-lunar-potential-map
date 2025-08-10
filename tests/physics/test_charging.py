import numpy as np
import pytest
from src.physics.charging import (
    electron_current_density,
    electron_current_density_magnitude,
)
from src.physics.kappa import KappaParams
import src.config as cfg


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
