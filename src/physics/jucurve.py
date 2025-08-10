import math
from dataclasses import dataclass

from scipy.optimize import brentq


@dataclass(frozen=True)
class JUCoefficients:
    """
    Coefficients for the J-U curve.

    The curve is a double exponential function: J(U) = A * exp(-U / B) + C * exp(-U / D)
    See https://doi.org/10.1029/2008JA013194 for more details. Equation and values described in paragraph 23 (page 4).
    """

    A: float = 1.07e-3  # A/m^2
    B: float = 5.0  # V
    C: float = 1.6e-5  # A/m^2
    D: float = 60.0  # V


def J_of_U(U: float, coefficients=JUCoefficients()):
    """
    Calculate the current density from the spacecraft J for a given electric spacecraft potential U.

    The spacecraft emits electrons by the photoelectric effect, and this function uses the J-U curve coefficients to compute the current density.

    Args:
        U (float): The electric field strength in volts.

    Returns:
        float: The current density in A/m^2.
    """
    return coefficients.A * math.exp(-U / coefficients.B) + coefficients.C * math.exp(
        -U / coefficients.D
    )


def U_from_J(
    J_target: float,
    coefficients=JUCoefficients(),
    U_min: float = 0.0,
    U_max: float = 150.0,
):
    """
    Invert the J-U curve to find the electric potential U for a given current density J.

    Args:
        J_target (float): The target current density in A/m^2.
        coefficients (JUCoefficients): The coefficients for the J-U curve.
        U_min (float): The minimum electric potential to consider.
        U_max (float): The maximum electric potential to consider.

    Returns:
        float: The electric potential U in volts.
    """

    f = lambda U: J_of_U(U, coefficients) - J_target
    return brentq(f, U_min, U_max)
