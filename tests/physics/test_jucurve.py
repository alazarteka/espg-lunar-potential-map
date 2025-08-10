import math
import pytest
from src.physics.jucurve import J_of_U, U_from_J, JUCoefficients


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
