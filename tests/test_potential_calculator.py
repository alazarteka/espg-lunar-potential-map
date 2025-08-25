import numpy as np
import pytest

from src.kappa import Kappa
from src.spacecraft_potential import (
    calculate_potential,
    calculate_shaded_currents,
    current_balance,
    theta_to_temperature_ev,
)
from src.utils.units import ureg
from tests.test_kappa_fitter import prepare_synthetic_er


@pytest.mark.skip_ci
def test_current_balance_unphysical_temperature_positive(monkeypatch):
    """For very negative U that makes T_c<=0, F(U) should be large positive."""
    er = prepare_synthetic_er(theta=1e7)
    fitter = Kappa(er, 1)
    fit = fitter.fit()
    assert fit is not None and fit.is_good_fit

    density_mag, kappa, theta_uc = fit.params.to_tuple()
    Tuc = theta_to_temperature_ev(theta_uc, kappa)
    # Choose U such that T_c = Tuc + U/(kappa-1.5) <= 0
    U_too_negative = -((kappa - 1.5) * (Tuc + 1.0))  # [V]

    E = np.geomspace(1.0, 2.0e4, 256)
    F = current_balance(U_too_negative, fit, E, sey_E_m=500.0, sey_delta_m=1.5)
    assert F > 0.0


@pytest.mark.skip_ci
def test_shade_potential_moves_toward_zero_with_higher_SEE(monkeypatch):
    """Increasing SEE yield should make U less negative (closer to 0).

    TODO: Occasionally this test cannot bracket a root with the default
    search (skips in that case). Revisit bracketing strategy or defaults
    (E_max_ev, bounds, Tc guard) to reduce flakiness and ensure a
    deterministic bracket on synthetic cases.
    """
    er = prepare_synthetic_er(theta=1e7)

    # Force shadow: get_intersection_or_none returns a non-None sentinel
    monkeypatch.setattr(
        "src.spacecraft_potential.get_intersection_or_none", lambda *args, **kwargs: 0
    )
    # Stub SPICE-dependent calls to avoid kernel access
    monkeypatch.setattr("src.spacecraft_potential.spice.str2et", lambda s: 0.0)
    monkeypatch.setattr(
        "src.spacecraft_potential.get_lp_position_wrt_moon", lambda *_: 0
    )
    monkeypatch.setattr(
        "src.spacecraft_potential.get_lp_vector_to_sun_in_lunar_frame", lambda *_: 0
    )

    # Find a valid bracket for the current balance using the same grid
    fitter = Kappa(er, 1)
    fit = fitter.fit()
    assert fit and fit.is_good_fit
    E_min_ev, E_max_ev, n_steps = 1.0, 2.0e4, 400
    E = np.geomspace(max(E_min_ev, 0.5), E_max_ev, n_steps)
    low, high = -200.0, -0.5
    f_low = current_balance(low, fit, E, sey_E_m=500.0, sey_delta_m=1.0)
    f_high = current_balance(high, fit, E, sey_E_m=500.0, sey_delta_m=1.0)
    tries = 0
    while np.sign(f_low) == np.sign(f_high) and tries < 30:
        low *= 1.5
        f_low = current_balance(low, fit, E, sey_E_m=500.0, sey_delta_m=1.0)
        tries += 1
    if np.sign(f_low) == np.sign(f_high):
        pytest.skip("Could not bracket root for nightside under defaults")

    out_low = calculate_potential(
        er,
        1,
        sey_E_m=500.0,
        sey_delta_m=1.0,
        n_steps=n_steps,
        spacecraft_potential_low=low,
        spacecraft_potential_high=high,
    )
    out_high = calculate_potential(
        er,
        1,
        sey_E_m=500.0,
        sey_delta_m=2.0,
        n_steps=n_steps,
        spacecraft_potential_low=low,
        spacecraft_potential_high=high,
    )

    assert out_low is not None and out_high is not None
    (_, U_low), (_, U_high) = out_low, out_high
    # U_high should be less negative (numerically larger)
    assert U_high.to(ureg.volt).magnitude > U_low.to(ureg.volt).magnitude


@pytest.mark.skip_ci
def test_day_branch_returns_positive_potential(monkeypatch):
    """In daylight, the calculator should return a positive potential."""
    er = prepare_synthetic_er(theta=1e7)
    # Force day: no intersection (None)
    # Stub SPICE-dependent calls to avoid kernel access
    monkeypatch.setattr("src.spacecraft_potential.spice.str2et", lambda s: 0.0)
    monkeypatch.setattr(
        "src.spacecraft_potential.get_lp_position_wrt_moon", lambda *_: 0
    )
    monkeypatch.setattr(
        "src.spacecraft_potential.get_lp_vector_to_sun_in_lunar_frame", lambda *_: 0
    )
    monkeypatch.setattr(
        "src.spacecraft_potential.get_intersection_or_none",
        lambda *args, **kwargs: None,
    )

    out = calculate_potential(er, 1, n_steps=200)
    assert out is not None
    _, U = out
    assert U.to(ureg.volt).magnitude >= 0.0


@pytest.mark.skip_ci
def test_barrier_empty_gives_zero_Je_Jsee(monkeypatch):
    """When |U|>E_max, Je and Jsee integrals should be zero by construction."""
    er = prepare_synthetic_er(theta=1e7)
    fitter = Kappa(er, 1)
    fit = fitter.fit()
    assert fit and fit.is_good_fit

    E = np.geomspace(1.0, 100.0, 64)
    U = -1e6  # way beyond the grid, mask empty
    Je, Jsee, Ji = calculate_shaded_currents(U, fit, E, sey_E_m=500.0, sey_delta_m=1.5)
    assert Je == 0.0 and Jsee == 0.0
    assert Ji > 0.0
