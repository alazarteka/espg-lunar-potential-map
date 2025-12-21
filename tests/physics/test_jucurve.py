import math

import pytest

from src.physics.jucurve import (
    J_of_U,
    J_of_U_ref,
    JUCoefficients,
    U_from_J,
    U_from_J_ref,
)
from src.utils.units import ureg


def test_coefficients_match_halekas_2008():
    """
    Verify coefficients match Halekas 2008 (doi:10.1029/2008JA013194) paragraph 23.

    Paper states: A = 1.07 μA/m², B = 5 V, C = 0.016 μA/m², D = 60 V
    (Paper has typo listing A,B,A,B instead of A,B,C,D)
    """
    coeff = JUCoefficients()
    assert coeff.A == 1.07e-6  # 1.07 μA/m²
    assert coeff.B == 5.0  # 5 V
    assert coeff.C == 1.6e-8  # 0.016 μA/m²
    assert coeff.D == 60.0  # 60 V


def test_J_of_U_default():
    coeff = JUCoefficients()
    # At U = 0 the exponential terms are 1
    expected = coeff.A + coeff.C
    assert math.isclose(J_of_U(0.0), expected, rel_tol=1e-12)


def test_J_of_U_monotonic():
    # J should decrease as U increases
    assert J_of_U(0) > J_of_U(10) > J_of_U(50) > J_of_U(150)


def test_J_of_U_known_value():
    coeff = JUCoefficients()
    expected = coeff.A * math.exp(-10 / coeff.B) + coeff.C * math.exp(-10 / coeff.D)
    assert math.isclose(J_of_U(10), expected, rel_tol=1e-12)


def test_U_from_J_roundtrip():
    target_U = 20.0
    J_target = J_of_U(target_U)
    recovered_U = U_from_J(J_target)
    assert math.isclose(recovered_U, target_U, rel_tol=1e-6)


def test_U_from_J_out_of_range():
    max_J = J_of_U(0.0)  # maximum possible current density
    with pytest.raises(ValueError):
        # Request a J larger than the maximum; brentq should fail
        U_from_J(max_J * 1.1)


# --- Cross-validation: pint reference vs fast implementation ---


@pytest.mark.parametrize("U_val", [0.0, 5.0, 10.0, 50.0, 100.0, 150.0])
def test_J_of_U_ref_matches_fast(U_val):
    """Verify pint reference matches fast implementation."""
    fast_result = J_of_U(U_val)
    ref_result = J_of_U_ref(U_val * ureg.volt)

    assert ref_result.units == ureg.ampere / ureg.meter**2
    assert math.isclose(ref_result.magnitude, fast_result, rel_tol=1e-12)


@pytest.mark.parametrize("U_target", [5.0, 20.0, 50.0, 100.0])
def test_U_from_J_ref_matches_fast(U_target):
    """Verify pint reference matches fast implementation for inversion."""
    J_target = J_of_U(U_target)

    fast_result = U_from_J(J_target)
    ref_result = U_from_J_ref(J_target * ureg.ampere / ureg.meter**2)

    assert ref_result.units == ureg.volt
    assert math.isclose(ref_result.magnitude, fast_result, rel_tol=1e-6)
