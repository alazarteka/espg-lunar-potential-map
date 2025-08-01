# Tests for src/kappa.py

import numpy as np
import pandas as pd
import pytest

import src.config as config
from src.flux import ERData
from src.kappa import Kappa
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux,
)
from src.utils.units import (
    ureg,
)


@pytest.fixture(
    params=[
        (
            1e6 * ureg.particle / ureg.meter**3,
            5.0,
            1e7 * ureg.meter / ureg.second,
            (1e6, 5.0, 1e7),
        ),
    ]
)
def kappa_params_set(request):
    """Fixture to provide different sets of KappaParams and their expected tuple representation."""
    density, kappa, theta, expected_tuple = request.param
    params = KappaParams(density=density, kappa=kappa, theta=theta)
    return params, expected_tuple


def prepare_phis():
    phis = []
    phis_by_latitude = {
        78.75: ([], 4, 0.119570),
        56.25: ([], 8, 0.170253),
        33.75: ([], 16, 0.127401),
        11.25: ([], 16, 0.150279),
        -11.25: ([], 16, 0.150279),
        -33.75: ([], 16, 0.127401),
        -56.25: ([], 8, 0.170253),
        -78.75: ([], 4, 0.119570),
    }
    thetas = []
    for key in phis_by_latitude.keys():
        for i in range(phis_by_latitude[key][1]):
            thetas.append(key)

    # Sort the thetas
    # There is a small dependency on the order of thetas, so we sort them
    # to ensure that the test passes consistently but also remove the need
    # to download the data files.
    thetas = sorted(np.array(thetas), key=lambda x: abs(x))
    solid_angles = np.array([phis_by_latitude[theta][2] for theta in thetas])

    # Fix the phi calculation - this was creating inconsistent data
    phi_counter = {}
    for theta in thetas:
        if theta not in phi_counter:
            phi_counter[theta] = 0
        n_channels = phis_by_latitude[theta][1]
        phi_value = phi_counter[theta] / n_channels * 360
        phis.append(phi_value)
        phi_counter[theta] += 1

    return phis, solid_angles


def prepare_flux(density=1e6, kappa=5.0, theta=1.1e5):

    params = KappaParams(
        density=density * ureg.particle / ureg.m**3,
        kappa=kappa,
        theta=theta * ureg.meter / ureg.second,
    )

    energy_centers = np.geomspace(2e1, 2e4, config.SWEEP_ROWS) * ureg.electron_volt
    energy_bounds = np.column_stack([energy_centers * 0.75, energy_centers * 1.25])

    omnidirectional_particle_flux = omnidirectional_flux(
        parameters=params, energy=energy_centers
    )
    return omnidirectional_particle_flux, energy_centers


def prepare_synthetic_er(density=1e6, kappa=5.0, theta=1.1e5):
    phis, solid_angles = prepare_phis()
    omnidirectional_particle_flux, energy_centers = prepare_flux(
        density=density, kappa=kappa, theta=theta
    )

    synthetic_er_data = pd.DataFrame(columns=config.ALL_COLS)
    directional = omnidirectional_particle_flux / (4 * np.pi * ureg.steradian)  # J / sr
    directional = directional.to(
        ureg.particle
        / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)
    )
    synthetic_er_data[config.FLUX_COLS] = np.repeat(
        directional.magnitude[:, None], config.CHANNELS, axis=1
    )
    # If needed, refer to previous flux calculation logic in documentation or version control.

    synthetic_er_data[config.PHI_COLS] = phis
    synthetic_er_data["UTC"] = "2025-07-25T12:30:00"
    synthetic_er_data["time"] = (
        pd.to_datetime(synthetic_er_data["UTC"]).astype(np.int64) // 10**9
    )
    synthetic_er_data["energy"] = energy_centers.to(ureg.electron_volt).magnitude
    synthetic_er_data["spec_no"] = 1

    np.random.seed(42)
    synthetic_er_data[config.MAG_COLS] = np.random.rand(3)

    return ERData.from_dataframe(synthetic_er_data, "NULL")

@pytest.mark.skip_ci
def test_density_estimate(kappa_params_set):
    """Test the density estimate calculation."""
    params, _ = kappa_params_set
    synthetic_er = prepare_synthetic_er(
        density=params.density.to(ureg.particle / ureg.meter**3).magnitude,
        kappa=params.kappa,
        theta=params.theta.to(ureg.meter / ureg.second).magnitude,
    )
    kappa_fitter = Kappa(synthetic_er, 1)

    assert (
        kappa_fitter.density_estimate.magnitude > 0
    ), "Density estimate should be positive."
    assert (
        kappa_fitter.density_estimate.units == ureg.particle / ureg.meter**3
    ), "Density estimate should have correct units."
    assert np.isclose(
        kappa_fitter.density_estimate.magnitude,
        params.density.magnitude / 2.0,
        rtol=1e-2,
    ), f"Expected density {params.density.magnitude / 2.0}, got {kappa_fitter.density_estimate.magnitude}"

@pytest.mark.skip_ci
def test_objective_functions(kappa_params_set):
    """Test the standard and fast objective functions."""
    params, _ = kappa_params_set
    synthetic_er = prepare_synthetic_er(
        density=params.density.to(ureg.particle / ureg.meter**3).magnitude,
        kappa=params.kappa,
        theta=params.theta.to(ureg.meter / ureg.second).magnitude,
    )

    # Create a kappa-logtheta
    kappa = np.linspace(2.0, 10.0, 20)
    logtheta = np.linspace(4, 8, 20)
    kappa_logtheta = np.column_stack((kappa, logtheta))
    kappa_fitter = Kappa(synthetic_er, 1)

    _standard_objective = np.array(
        [kappa_fitter._objective_function(params) for params in kappa_logtheta]
    )
    _fast_objective = np.array(
        [kappa_fitter._objective_function_fast(params) for params in kappa_logtheta]
    )

    assert np.allclose(
        _standard_objective, _fast_objective, rtol=1e-2
    ), "Standard and fast objective functions should match within tolerance."

@pytest.mark.skip_ci
def test_objective_functions_in_fitter(kappa_params_set):
    """Test the Kappa fitter performance with the standard and fast objective functions."""
    params, _ = kappa_params_set
    synthetic_er = prepare_synthetic_er(
        density=params.density.to(ureg.particle / ureg.meter**3).magnitude,
        kappa=params.kappa,
        theta=params.theta.to(ureg.meter / ureg.second).magnitude,
    )

    kappa_fitter = Kappa(synthetic_er, 1)
    standard_fit, _ = kappa_fitter.fit(use_fast=False)
    fast_fit, _ = kappa_fitter.fit(use_fast=True)

    assert np.isclose(
        standard_fit.kappa, fast_fit.kappa, rtol=1e-2
    ), "Kappa values from standard and fast fitters should match within tolerance."
    assert np.isclose(
        standard_fit.theta.to(ureg.meter / ureg.second).magnitude,
        fast_fit.theta.to(ureg.meter / ureg.second).magnitude,
        rtol=1e-2,
    ), "Theta values from standard and fast fitters should match within tolerance."

@pytest.mark.skip_ci
def test_kappa_fitter(kappa_params_set):
    """Test the Kappa distribution fitting functionality."""

    params, _ = kappa_params_set
    synthetic_er = prepare_synthetic_er(
        density=params.density.to(ureg.particle / ureg.meter**3).magnitude,
        kappa=params.kappa,
        theta=params.theta.to(ureg.meter / ureg.second).magnitude,
    )
    kappa_fitter = Kappa(synthetic_er, 1)
    fitted_params, _ = kappa_fitter.fit()

    assert isinstance(
        fitted_params, KappaParams
    ), "Fitted parameters should be an instance of KappaParams"
    assert fitted_params.kappa > 1.5, "Fitted kappa should be greater than 1.5"
    assert fitted_params.theta.magnitude > 0, "Fitted theta should be positive."
    assert np.isclose(
        fitted_params.kappa, params.kappa, rtol=1e-2
    ), f"Expected kappa {params.kappa}, got {fitted_params.kappa}"
    assert np.isclose(
        fitted_params.theta.magnitude,
        params.theta.to(ureg.meter / ureg.second).magnitude,
        rtol=1e-2,
    ), f"Expected theta {params.theta.to(ureg.meter / ureg.second).magnitude}, got {fitted_params.theta.magnitude}"
