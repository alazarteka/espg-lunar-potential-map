from dataclasses import dataclass

import numpy as np
from pint import Quantity
from scipy.integrate import simpson
from scipy.special import gamma

from src import config
from src.utils.units import ureg, Length, Speed, Mass, Energy, NumberDensity, Dimensionless, Charge, Time, Voltage, Angle, Flux, PhaseSpaceDensity

@dataclass(slots=True, frozen=True)
class KappaParams:
    """
    Represents the parameters for the kappa distribution.
    
    Attributes:
        density (NumberDensity): Particle number density in m^-3.
        kappa (Dimensionless): Kappa parameter, dimensionless.
        theta (Speed): Thermal speed in m/s.
    """
    density: NumberDensity
    kappa: float
    theta: Speed

    def __post_init__(self):
        """Ensure that the parameters are of the correct type."""
        if not isinstance(self.density, Quantity):
            raise TypeError("density must be a pint Quantity")
        if not isinstance(self.kappa, float):
            raise TypeError("kappa must be a float")
        if not isinstance(self.theta, Quantity):
            raise TypeError("theta must be a pint Quantity")
        
        object.__setattr__(self, 'density', self.density.to(ureg.particle / ureg.meter**3))
        object.__setattr__(self, 'theta', self.theta.to(ureg.meter / ureg.second))

    def to_tuple(self) -> tuple[float, float, float]:
        """
        Convert the parameters to a tuple.

        Returns:
            tuple: A tuple containing the density (in m^-3), kappa, and theta (in m/s) without units.
        """
        return self.density.magnitude, self.kappa, self.theta.magnitude


def kappa_distribution(
        parameters: KappaParams,
        velocity: Speed
    ) -> PhaseSpaceDensity:
    """
    Calculate the kappa **phase-space distribution function** *f(v)*.

    Args:
        parameters (KappaParams): Kappa distribution parameters.
        velocity (Speed): Speed at which to evaluate the distribution.

    Returns:
        PhaseSpaceDensity: The kappa distribution evaluated at the given velocity.
    """

    if not isinstance(velocity, Quantity):
        raise TypeError("velocity must be a pint Quantity")

    density = parameters.density
    kappa = parameters.kappa
    theta = parameters.theta

    prefactor = gamma(kappa + 1) / (
        np.power(np.pi * kappa, 1.5) * gamma(kappa - 0.5)
    )
    core = density / theta**3
    tail = (1 + (velocity / theta) ** 2 / kappa) ** (-kappa - 1)
    return (prefactor * core * tail).to(ureg.particle / (ureg.meter ** 3 * (ureg.meter / ureg.second) ** 3))

def directional_flux(
        parameters: KappaParams,
        energy: Energy
    ) -> Flux:
    """
    Calculate the directional flux for a kappa distribution.

    Args:
        parameters (KappaParams): Kappa distribution parameters.
        energy (Energy): Energy at which to evaluate the flux.

    Returns:
        Flux: The directional flux evaluated at the given energy.
    """

    if not isinstance(energy, Quantity):
        raise TypeError("energy must be a pint Quantity")

    velocity = np.sqrt((2 * energy / config.ELECTRON_MASS)).to(ureg.meter / ureg.second)

    distribution = kappa_distribution(
        parameters, velocity
    )
    return (distribution * velocity**2 / config.ELECTRON_MASS).to(ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt))

def omnidirectional_flux(
        parameters: KappaParams,
        energy: Energy
    ):
    return (4 * np.pi * ureg.steradian) * directional_flux(parameters, energy)

def omnidirectional_flux_integrated(
        parameters: KappaParams,
        energy_bounds: Energy,
        n_samples: int = 101
    ):
    assert n_samples > 0 and n_samples % 2 == 1, "n_samples must be an odd positive integer"

    energy_grid = np.geomspace(
        energy_bounds[:, 0].to(ureg.electron_volt).magnitude, 
        energy_bounds[:, 1].to(ureg.electron_volt).magnitude, 
        num=n_samples, 
        axis=1
    ) * ureg.electron_volt
    flux_values = omnidirectional_flux(parameters, energy_grid)
    integrated_flux =  simpson(
        flux_values.to(ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)).magnitude,
        energy_grid.to(ureg.electron_volt).magnitude,
        axis=1
    ) * ureg.particle / (ureg.centimeter**2 * ureg.second)

    return integrated_flux