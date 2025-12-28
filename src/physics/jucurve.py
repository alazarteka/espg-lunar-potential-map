import math
from dataclasses import dataclass

from scipy.optimize import brentq

from src.utils.units import CurrentDensityType, VoltageType, ureg


@dataclass(frozen=True)
class JUCoefficients:
    """
    Coefficients for the J-U curve.

    The curve is a double exponential function: J(U) = A * exp(-U / B) + C * exp(-U / D)
    See https://doi.org/10.1029/2008JA013194 for more details. Equation and
    values described in paragraph 23 (page 4).

    Paper mistakenly lists A, B, A, B instead of A, B, C, D. Corrected here.
    """

    A: float = 1.07e-6  # A/m^2 (paper says A = 1.07 μA/m²)
    B: float = 5.0  # V         (paper says B = 5 V)
    C: float = 1.6e-8  # A/m^2  (paper says C = 0.016 μA/m²)
    D: float = 60.0  # V        (paper says D = 60 V)


DEFAULT_JU_COEFFICIENTS = JUCoefficients()


def J_of_U(U: float, coefficients: JUCoefficients | None = None):
    """
    Calculate the current density from the spacecraft J for a given electric
    spacecraft potential U.

    The spacecraft emits electrons by the photoelectric effect, and this
    function uses the J-U curve coefficients to compute the current density.

    Args:
        U (float): The electric field strength in volts.
        coefficients: J-U curve coefficients. Uses default if None.

    Returns:
        float: The current density in A/m^2.
    """
    coefficients = coefficients or DEFAULT_JU_COEFFICIENTS
    return coefficients.A * math.exp(-U / coefficients.B) + coefficients.C * math.exp(
        -U / coefficients.D
    )


def U_from_J(
    J_target: float,
    coefficients: JUCoefficients | None = None,
    U_min: float = 0.0,
    U_max: float = 150.0,
) -> float:
    """
    Invert the J-U curve to find the electric potential U for a given current density J.

    Args:
        J_target (float): The target current density in A/m^2.
        coefficients: J-U curve coefficients. Uses default if None.
        U_min (float): The minimum electric potential to consider.
        U_max (float): The maximum electric potential to consider.

    Returns:
        float: The electric potential U in volts.
    """
    coefficients = coefficients or DEFAULT_JU_COEFFICIENTS

    def f(U):
        return J_of_U(U, coefficients) - J_target

    return brentq(f, U_min, U_max)


# --- Pint reference implementations (slow, but verifiably correct) ---


def J_of_U_ref(
    U: VoltageType,
    coefficients: JUCoefficients | None = None,
) -> CurrentDensityType:
    """
    Reference implementation of J_of_U with explicit units.

    Args:
        U: Spacecraft potential with units (e.g., ureg.volt).
        coefficients: J-U curve coefficients. Uses default if None.

    Returns:
        Current density with units (A/m²).
    """
    coefficients = coefficients or DEFAULT_JU_COEFFICIENTS

    U_V = U.to(ureg.volt).magnitude
    A = coefficients.A * ureg.ampere / ureg.meter**2
    B = coefficients.B * ureg.volt
    C = coefficients.C * ureg.ampere / ureg.meter**2
    D = coefficients.D * ureg.volt

    return A * math.exp(-U_V / B.magnitude) + C * math.exp(-U_V / D.magnitude)


def U_from_J_ref(
    J_target: CurrentDensityType,
    coefficients: JUCoefficients | None = None,
    U_min: VoltageType = 0.0 * ureg.volt,
    U_max: VoltageType = 150.0 * ureg.volt,
) -> VoltageType:
    """
    Reference implementation of U_from_J with explicit units.

    Args:
        J_target: Target current density with units (e.g., ureg.ampere / ureg.meter**2).
        coefficients: J-U curve coefficients. Uses default if None.
        U_min: Minimum potential to search.
        U_max: Maximum potential to search.

    Returns:
        Spacecraft potential with units (V).
    """
    coefficients = coefficients or DEFAULT_JU_COEFFICIENTS

    J_target_mag = J_target.to(ureg.ampere / ureg.meter**2).magnitude
    U_min_mag = U_min.to(ureg.volt).magnitude
    U_max_mag = U_max.to(ureg.volt).magnitude

    U_mag = U_from_J(J_target_mag, coefficients, U_min_mag, U_max_mag)
    return U_mag * ureg.volt
