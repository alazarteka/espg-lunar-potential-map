# Tests for src/kappa.py

import numpy as np
import pytest

from src.kappa import Kappa
from src.physics.kappa import KappaParams
from src.utils.synthetic import prepare_synthetic_er
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

    assert kappa_fitter.density_estimate.magnitude > 0, (
        "Density estimate should be positive."
    )
    assert kappa_fitter.density_estimate.units == ureg.particle / ureg.meter**3, (
        "Density estimate should have correct units."
    )
    assert np.isclose(
        kappa_fitter.density_estimate.magnitude,
        params.density.magnitude,
        rtol=1e-2,
    ), (
        f"Expected density {params.density.magnitude}, got {kappa_fitter.density_estimate.magnitude}"
    )


@pytest.mark.skip_ci
def test_objective_functions(kappa_params_set):
    """Test that the standard and fast objective functions produce consistent results."""
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
        [
            kappa_fitter._objective_function(params, use_weights=False)
            for params in kappa_logtheta
        ]
    )
    _fast_objective = np.array(
        [
            kappa_fitter._objective_function_fast(params, use_weights=False)
            for params in kappa_logtheta
        ]
    )

    assert np.allclose(_standard_objective, _fast_objective, rtol=1e-2), (
        "Standard and fast objective functions should match within tolerance."
    )


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
    standard_results = kappa_fitter.fit(use_fast=False)
    fast_results = kappa_fitter.fit(use_fast=True)

    assert np.isclose(
        standard_results.params.kappa, fast_results.params.kappa, rtol=1e-2
    ), "Kappa values from standard and fast fitters should match within tolerance."
    assert np.isclose(
        standard_results.params.theta.to(ureg.meter / ureg.second).magnitude,
        fast_results.params.theta.to(ureg.meter / ureg.second).magnitude,
        rtol=1e-2,
    ), "Theta values from standard and fast fitters should match within tolerance."


@pytest.mark.skip_ci
def test_kappa_fitter(kappa_params_set):
    """Test the end-to-end Kappa distribution fitting functionality.

    This test uses synthetic data generated from known parameters and asserts
    that the fitter can recover those original parameters within a reasonable
    tolerance.
    """

    params, _ = kappa_params_set
    synthetic_er = prepare_synthetic_er(
        density=params.density.to(ureg.particle / ureg.meter**3).magnitude,
        kappa=params.kappa,
        theta=params.theta.to(ureg.meter / ureg.second).magnitude,
    )
    kappa_fitter = Kappa(synthetic_er, 1)
    # The synthetic data is generated without energy convolution, so we disable it in the fit
    fit_results = kappa_fitter.fit(use_convolution=False)

    assert isinstance(fit_results.params, KappaParams), (
        "Fitted parameters should be an instance of KappaParams"
    )
    assert fit_results.params.kappa > 1.5, "Fitted kappa should be greater than 1.5"
    assert fit_results.params.theta.magnitude > 0, "Fitted theta should be positive."
    assert np.isclose(fit_results.params.kappa, params.kappa, rtol=1e-2), (
        f"Expected kappa {params.kappa}, got {fit_results.params.kappa}"
    )
    assert np.isclose(
        fit_results.params.theta.magnitude,
        params.theta.to(ureg.meter / ureg.second).magnitude,
        rtol=1e-2,
    ), (
        f"Expected theta {params.theta.to(ureg.meter / ureg.second).magnitude}, got {fit_results.params.theta.magnitude}"
    )
