# Tests for src/kappa.py

import numpy as np
import pandas as pd
import pytest

from src import config
from src.kappa import FitResults, Kappa
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


# ==============================================================================
# Edge Case Tests
# ==============================================================================


class TestKappaEdgeCases:
    """Tests for Kappa class edge cases and error handling."""

    def test_invalid_spec_no_raises_in_constructor(self, caplog):
        """Kappa with non-existent spec_no raises ValueError during construction.
        
        The constructor calls _prepare_data() which logs a warning, then
        _get_density_estimate() raises ValueError because flux data is None.
        """
        er = prepare_synthetic_er()
        # Use spec_no that doesn't exist in synthetic data (which uses spec_no=1)
        with pytest.raises(ValueError, match="Data not prepared"):
            Kappa(er, spec_no=999)

        # Warning should have been logged before the error
        assert any("not found" in rec.message for rec in caplog.records)


class TestLogEnergyResponseMatrix:
    """Tests for the log-energy response matrix builder."""

    def test_response_matrix_shape(self):
        """Response matrix is square with size matching energy bins."""
        n_energies = 15
        energy_centers = np.geomspace(20, 20000, n_energies)

        W = Kappa.build_log_energy_response_matrix(energy_centers)

        assert W.shape == (n_energies, n_energies)

    def test_response_matrix_row_normalized(self):
        """Each row sums to 1 (probability distribution)."""
        energy_centers = np.geomspace(20, 20000, 15)

        W = Kappa.build_log_energy_response_matrix(energy_centers)

        row_sums = np.sum(W, axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-10)

    def test_response_matrix_diagonal_dominant(self):
        """Diagonal elements should be largest in each row."""
        energy_centers = np.geomspace(20, 20000, 15)

        W = Kappa.build_log_energy_response_matrix(energy_centers)

        for i in range(len(energy_centers)):
            assert W[i, i] == np.max(W[i, :])


# ==============================================================================
# corrected_fit Tests with SPICE Mocking
# ==============================================================================


class TestCorrectedFitDaylight:
    """Tests for corrected_fit daylight (illuminated) branch."""

    @pytest.fixture(autouse=True)
    def mock_spice_daylight(self, monkeypatch):
        """Mock SPICE functions to simulate daylight geometry.
        
        We must patch at src.kappa module level since that's where the
        functions are imported and used.
        """
        import spiceypy
        import src.kappa as kappa_module

        # Mock str2et to return some ephemeris time
        monkeypatch.setattr(spiceypy, "str2et", lambda s: 0.0)

        # Mock geometry functions at the kappa module level
        lunar_radius = config.LUNAR_RADIUS.to(ureg.meter).magnitude

        # Position spacecraft on dayside: +X direction, sun also in +X
        # This means spacecraft can see sun directly (no lunar obstruction)
        monkeypatch.setattr(
            kappa_module,
            "get_lp_position_wrt_moon",
            lambda et: np.array([lunar_radius + 100e3, 0.0, 0.0]),
        )
        monkeypatch.setattr(
            kappa_module,
            "get_lp_vector_to_sun_in_lunar_frame",
            lambda et: np.array([1.0, 0.0, 0.0]),
        )
        # For daylight, get_intersection_or_none should return None (no intersection)
        monkeypatch.setattr(
            kappa_module,
            "get_intersection_or_none",
            lambda pos, dir, r: None,
        )

    @pytest.mark.skip_ci
    def test_corrected_fit_daylight_returns_fit_and_potential(self):
        """Daylight corrected_fit returns (FitResults, U) tuple."""
        er = prepare_synthetic_er(
            density=1e6,
            kappa=5.0,
            theta=1e7,
        )
        kappa_fitter = Kappa(er, 1)

        result, U = kappa_fitter.corrected_fit(n_starts=3, use_convolution=False)

        # Should return valid fit
        assert result is not None
        assert isinstance(result, FitResults)
        # Potential should be a float
        assert isinstance(U, float)


class TestCorrectedFitNightside:
    """Tests for corrected_fit nightside (shadowed) branch."""

    @pytest.fixture(autouse=True)
    def mock_spice_nightside(self, monkeypatch):
        """Mock SPICE functions to simulate nightside geometry.
        
        We must patch at src.kappa module level since that's where the
        functions are imported and used.
        """
        import spiceypy
        import src.kappa as kappa_module

        monkeypatch.setattr(spiceypy, "str2et", lambda s: 0.0)

        lunar_radius = config.LUNAR_RADIUS.to(ureg.meter).magnitude

        # Spacecraft in -X (behind moon relative to sun)
        monkeypatch.setattr(
            kappa_module,
            "get_lp_position_wrt_moon",
            lambda et: np.array([-lunar_radius - 100e3, 0.0, 0.0]),
        )
        # Sun in +X direction → spacecraft occluded by moon
        monkeypatch.setattr(
            kappa_module,
            "get_lp_vector_to_sun_in_lunar_frame",
            lambda et: np.array([1.0, 0.0, 0.0]),
        )
        # For nightside, get_intersection_or_none should return an intersection point
        # (meaning line-of-sight to sun is blocked by moon)
        monkeypatch.setattr(
            kappa_module,
            "get_intersection_or_none",
            lambda pos, dir, r: np.array([lunar_radius, 0.0, 0.0]),
        )

    @pytest.mark.skip_ci
    def test_corrected_fit_nightside_returns_fit_and_potential(self):
        """Nightside corrected_fit returns (FitResults, U) tuple."""
        er = prepare_synthetic_er(
            density=1e6,
            kappa=5.0,
            theta=1e7,
        )
        kappa_fitter = Kappa(er, 1)

        result, U = kappa_fitter.corrected_fit(n_starts=3, use_convolution=False)

        # Should return valid fit (or None if balance fails)
        if result is not None:
            assert isinstance(result, FitResults)
            assert isinstance(U, float)


class TestCorrectedFitSpiceFailure:
    """Tests for corrected_fit when SPICE fails."""

    @pytest.fixture
    def mock_spice_failure(self, monkeypatch):
        """Mock SPICE to raise an exception."""
        import spiceypy

        def fail_str2et(s):
            raise Exception("SPICE kernel not loaded")

        monkeypatch.setattr(spiceypy, "str2et", fail_str2et)

    def test_corrected_fit_returns_none_on_spice_failure(self, mock_spice_failure):
        """SPICE failure returns (None, None)."""
        er = prepare_synthetic_er()
        kappa_fitter = Kappa(er, 1)

        result, U = kappa_fitter.corrected_fit()

        assert result is None
        assert U is None

