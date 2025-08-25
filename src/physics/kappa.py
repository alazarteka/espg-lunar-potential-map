import math
from dataclasses import dataclass

import numpy as np
from numba import jit
from pint import Quantity
from scipy.integrate import simpson
from scipy.special import gamma

from src import config
from src.utils.units import (
    EnergyType,
    FluxType,
    IntegratedFluxType,
    NumberDensityType,
    OmnidirectionalFluxType,
    PhaseSpaceDensityType,
    SpeedType,
    ureg,
)


@dataclass(slots=True, frozen=True)
class KappaParams:
    """
    Represents the parameters for the kappa distribution.

    Attributes:
        density (NumberDensity): Particle number density in m^-3.
        kappa (float): Kappa parameter.
        theta (Speed): Thermal speed in m/s.
    """

    density: NumberDensityType
    kappa: float
    theta: SpeedType

    def __post_init__(self):
        """Ensure that the parameters are of the correct type."""
        if __debug__:
            if not isinstance(self.density, Quantity):
                raise TypeError("density must be a pint Quantity (NumberDensity)")
            if not isinstance(self.kappa, float):
                raise TypeError("kappa must be a float")
            if not isinstance(self.theta, Quantity):
                raise TypeError("theta must be a pint Quantity (Speed)")

        object.__setattr__(
            self, "density", self.density.to(ureg.particle / ureg.meter**3)
        )
        object.__setattr__(self, "theta", self.theta.to(ureg.meter / ureg.second))

    def to_tuple(self) -> tuple[float, float, float]:
        """
        Convert the parameters to a tuple.

        Returns:
            tuple: A tuple containing (density_mag, kappa, theta_mag) where:
                - density_mag: density in particles/m^3 (unitless)
                - kappa: kappa parameter (unitless)
                - theta_mag: thermal speed in m/s (unitless)
        """
        return self.density.magnitude, self.kappa, self.theta.magnitude


def velocity_from_energy(energy: EnergyType) -> SpeedType:
    """
    Convert energy to speed.

    Args:
        energy (Energy): Energy in electron volts.

    Returns:
        Speed: Speed in m/s.
    """
    if __debug__:
        if not isinstance(energy, Quantity) or not energy.is_compatible_with(
            ureg.electron_volt
        ):
            raise TypeError("energy must be a pint Quantity (Energy)")
    return np.sqrt(2 * energy / config.ELECTRON_MASS).to(ureg.meter / ureg.second)


def energy_from_velocity(velocity: SpeedType) -> EnergyType:
    """
    Convert speed to energy.

    Args:
        velocity (Speed): Speed in m/s.

    Returns:
        Energy: Energy in electron volts.
    """
    if __debug__:
        if not isinstance(velocity, Quantity) or not velocity.is_compatible_with(
            ureg.meter / ureg.second
        ):
            raise TypeError("velocity must be a pint Quantity (Speed)")
        if not isinstance(
            config.ELECTRON_MASS, Quantity
        ) or not config.ELECTRON_MASS.is_compatible_with(ureg.kilogram):
            raise TypeError("config.ELECTRON_MASS must be a pint Quantity (Mass)")
    return (0.5 * config.ELECTRON_MASS * velocity**2).to(ureg.electron_volt)


def kappa_distribution(
    parameters: KappaParams, velocity: SpeedType
) -> PhaseSpaceDensityType:
    """
    Calculate the kappa **phase-space distribution function** *f(v)*.

    Args:
        parameters (KappaParams): Kappa distribution parameters.
        velocity (Speed): Speed at which to evaluate the distribution.

    Returns:
        PhaseSpaceDensity: The kappa distribution evaluated at the given velocity.
    """
    if __debug__:
        if not isinstance(velocity, Quantity) or not velocity.is_compatible_with(
            ureg.meter / ureg.second
        ):
            raise TypeError("velocity must be a pint Quantity (Speed)")
        if not isinstance(parameters, KappaParams):
            raise TypeError("parameters must be an instance of KappaParams")
    assert (
        parameters.kappa > 1.5
    ), "kappa must be greater than 1.5 for a valid kappa distribution"
    assert (
        parameters.theta.magnitude > 0
    ), "theta must be greater than 0 for a valid kappa distribution"

    density = parameters.density
    kappa = parameters.kappa
    theta = parameters.theta

    prefactor: float = gamma(kappa + 1) / (
        np.power(np.pi * kappa, 1.5) * gamma(kappa - 0.5)
    )
    core: PhaseSpaceDensityType = density / theta**3
    tail: float = (1 + (velocity / theta) ** 2 / kappa) ** (-kappa - 1)

    return (prefactor * core * tail).to(
        ureg.particle / (ureg.meter**3 * (ureg.meter / ureg.second) ** 3)
    )


def directional_flux(parameters: KappaParams, energy: EnergyType) -> FluxType:
    """
    Calculate the directional flux for a kappa distribution.

    Args:
        parameters (KappaParams): Kappa distribution parameters.
        energy (Energy): Energy at which to evaluate the flux.

    Returns:
        Flux: The directional flux evaluated at the given energy.
    """
    if __debug__:
        if not isinstance(energy, Quantity) or not energy.is_compatible_with(
            ureg.electron_volt
        ):
            raise TypeError("energy must be a pint Quantity (Energy)")
        if not isinstance(parameters, KappaParams):
            raise TypeError("parameters must be an instance of KappaParams")

    velocity = velocity_from_energy(energy)
    if __debug__:
        if not isinstance(velocity, Quantity) or not velocity.is_compatible_with(
            ureg.meter / ureg.second
        ):
            raise TypeError("velocity must be a pint Quantity (Speed)")

    distribution = kappa_distribution(parameters, velocity)
    if __debug__:
        if not isinstance(
            distribution, Quantity
        ) or not distribution.is_compatible_with(
            ureg.particle / (ureg.meter**3 * (ureg.meter / ureg.second) ** 3)
        ):
            raise TypeError("distribution must be a pint Quantity (PhaseSpaceDensity)")

    return (distribution * velocity**2 / config.ELECTRON_MASS).to(
        ureg.particle
        / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)
    )


def omnidirectional_flux(
    parameters: KappaParams, energy: EnergyType
) -> OmnidirectionalFluxType:
    """
    Calculate the omnidirectional flux for a kappa distribution.

    Args:
        parameters (KappaParams): Kappa distribution parameters.
        energy (Energy): Energy at which to evaluate the flux.

    Returns:
        Flux: The omnidirectional flux evaluated at the given energy.
    """
    if __debug__:
        if not isinstance(energy, Quantity) or not energy.is_compatible_with(
            ureg.electron_volt
        ):
            raise TypeError("energy must be a pint Quantity (Energy)")
        if not isinstance(parameters, KappaParams):
            raise TypeError("parameters must be an instance of KappaParams")

    directional_flux_value = directional_flux(parameters, energy)
    omnidirectional_flux_units = ureg.particle / (
        ureg.centimeter**2 * ureg.second * ureg.electron_volt * ureg.steradian
    )
    if __debug__:
        if not isinstance(
            directional_flux_value, Quantity
        ) or not directional_flux_value.is_compatible_with(omnidirectional_flux_units):
            raise TypeError("directional_flux_value must be a pint Quantity (Flux)")

    return (4 * np.pi * ureg.steradian) * directional_flux_value.to(
        omnidirectional_flux_units
    )


def omnidirectional_flux_integrated(
    parameters: KappaParams, energy_ranges: EnergyType, n_samples: int = 101
) -> IntegratedFluxType:
    """
    Calculate the integrated omnidirectional flux over a specified energy range.

    Args:
        parameters (KappaParams): Kappa distribution parameters.
        energy_bounds (Energy): Energy bounds for integration, should be a 2D array-like
            with shape (n, 2) where n is the number of energy ranges.
        n_samples (int): Number of samples to use for integration. Must be an
            odd positive integer.

    Returns:
        IntegratedFlux: The integrated omnidirectional flux over the
        specified energy range.
    """
    if __debug__:
        if not isinstance(
            energy_ranges, Quantity
        ) or not energy_ranges.is_compatible_with(ureg.electron_volt):
            raise TypeError("energy_ranges must be a pint Quantity (Energy)")
        if not isinstance(parameters, KappaParams):
            raise TypeError("parameters must be an instance of KappaParams")
        if not isinstance(n_samples, int):
            raise TypeError("n_samples must be an integer")

    assert (
        n_samples > 0 and n_samples % 2 == 1
    ), "n_samples must be an odd positive integer"

    energy_grid = (
        np.geomspace(
            energy_ranges[:, 0].to(ureg.electron_volt).magnitude,
            energy_ranges[:, 1].to(ureg.electron_volt).magnitude,
            num=n_samples,
            axis=1,
        )
        * ureg.electron_volt
    )
    if __debug__:
        if not isinstance(energy_grid, Quantity) or not energy_grid.is_compatible_with(
            ureg.electron_volt
        ):
            raise TypeError("energy_grid must be a pint Quantity (Energy)")

    flux_values = omnidirectional_flux(parameters, energy_grid)
    if __debug__:
        if not isinstance(flux_values, Quantity) or not flux_values.is_compatible_with(
            ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
        ):
            raise TypeError(
                "flux_values must be a pint Quantity (Omnidirectional Flux)"
            )

    integrated_flux = (
        simpson(
            flux_values.to(
                ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
            ).magnitude,
            energy_grid.to(ureg.electron_volt).magnitude,
            axis=1,
        )
        * ureg.particle
        / (ureg.centimeter**2 * ureg.second)
    )
    if __debug__:
        if not isinstance(
            integrated_flux, Quantity
        ) or not integrated_flux.is_compatible_with(
            ureg.particle / (ureg.centimeter**2 * ureg.second)
        ):
            raise TypeError("integrated_flux must be a pint Quantity (IntegratedFlux)")

    return integrated_flux


@jit(nopython=True, cache=True)
def _gamma_ratio(kappa: float) -> float:
    """Numba-optimized gamma ratio calculation."""

    return math.gamma(kappa + 1) / (
        math.pow(math.pi * kappa, 1.5) * math.gamma(kappa - 0.5)
    )


@jit(nopython=True, cache=True, fastmath=True)
def omnidirectional_flux_magnitude(
    density_mag: float,
    kappa: float,
    theta_mag: float,
    energy_mag: np.ndarray,
) -> np.ndarray:
    """
    Numba-JIT compiled version of omnidirectional_flux_magnitude.

    Args:
        density_mag: Density in particles/m^3
        kappa: Kappa parameter
        theta_mag: Theta in m/s
        energy_mag: Energy magnitudes in eV

    Returns:
        Omnidirectional flux magnitude in particles/(cm^2 s eV)
    """
    ELECTRON_MASS_EV_S2_M2 = 5.685630e-12

    velocity_mag = np.sqrt(2.0 * energy_mag / ELECTRON_MASS_EV_S2_M2)

    prefactor = _gamma_ratio(kappa)
    core = density_mag / (theta_mag * theta_mag * theta_mag)  # More efficient than **3
    velocity_ratio_sq = (velocity_mag / theta_mag) * (velocity_mag / theta_mag)
    tail = np.power(1.0 + velocity_ratio_sq / kappa, -kappa - 1.0)

    distribution_mag = prefactor * core * tail

    velocity_sq = velocity_mag * velocity_mag
    directional_flux_mag = distribution_mag * velocity_sq / ELECTRON_MASS_EV_S2_M2

    return 4.0 * math.pi * 1e-4 * directional_flux_mag


def omnidirectional_flux_fast(
    parameters: KappaParams,
    energy: EnergyType,
) -> OmnidirectionalFluxType:
    """Fast version of omnidirectional_flux using magnitude calculation."""
    density_mag, kappa, theta_mag = parameters.to_tuple()
    energy_mag = energy.to(ureg.electron_volt).magnitude

    result_mag = omnidirectional_flux_magnitude(
        density_mag,
        kappa,
        theta_mag,
        energy_mag,
    )
    return (
        result_mag
        * ureg.particle
        / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
    )
