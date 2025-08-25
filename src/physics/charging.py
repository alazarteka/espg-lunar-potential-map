import numpy as np
from scipy.integrate import simpson

import src.config as config
from src.physics.kappa import KappaParams, omnidirectional_flux_magnitude
from src.utils.units import CurrentDensityType, EnergyType, ureg

ELECTRON_CHARGE_mag = (
    config.ELECTRON_CHARGE.magnitude
)  # Charge of an electron in Coulombs
CHARGE_PER_ELECTRON = (
    ELECTRON_CHARGE_mag * ureg.coulomb / ureg.particle
)  # Charge per electron in Coulombs


def electron_current_density(
    params: KappaParams,
    E_min: EnergyType = 0.1 * ureg.electron_volt,
    E_max: EnergyType = 1000 * ureg.electron_volt,
    n_steps: int = 100,
) -> CurrentDensityType:
    """
    Calculate the electron current density for a given set of parameters
    describing the plasma.

    Args:
        params (KappaParams): The parameters describing the plasma.
        E_min (EnergyType): The minimum energy to consider.
        E_max (EnergyType): The maximum energy to consider.
        n_steps (int): The number of steps to use in the energy integration.

    Returns:
        CurrentDensityType: The electron current density in A/m^2.
    """

    E_min = E_min.to(ureg.electron_volt)
    E_max = E_max.to(ureg.electron_volt)

    energies = (
        np.geomspace(E_min.magnitude, E_max.magnitude, num=n_steps) * ureg.electron_volt
    )
    F_omni = omnidirectional_flux_magnitude(
        density_mag=params.density.magnitude,
        kappa=params.kappa,
        theta_mag=params.theta.magnitude,
        energy_mag=energies.magnitude,
    ) * (ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt))

    energies_mag = energies.magnitude
    F_omni_mag = F_omni.to(
        ureg.particle / (ureg.meter**2 * ureg.second * ureg.electron_volt)
    ).magnitude

    number_flux = (
        0.25
        * simpson(F_omni_mag, energies_mag)
        * (ureg.particle / (ureg.meter**2 * ureg.second))
    )
    current_flux = number_flux * CHARGE_PER_ELECTRON
    return current_flux.to(ureg.ampere / ureg.meter**2)


def electron_current_density_magnitude(
    density: float,
    kappa: float,
    theta: float,
    E_min: float,
    E_max: float,
    n_steps: int = 100,
) -> float:
    """
    Calculate the electron current density for a given set of parameters
    describing the plasma.

    Meant to be used with scalar values. Expects magnitudes in the following units:
    - Density: particles / m^3
    - Kappa: dimensionless
    - Theta: m / s
    - E_min: eV
    - E_max: eV

    Args:
        density (float): The plasma density in particles / m^3.
        kappa (float): The kappa parameter (dimensionless).
        theta (float): The theta parameter in m / s.
        E_min (float): The minimum energy in eV.
        E_max (float): The maximum energy in eV.
        n_steps (int): The number of steps for the energy integration.

    Returns:
        float: The electron current density in A/m^2.
    """

    energies = np.geomspace(E_min, E_max, num=n_steps)
    F_omni = omnidirectional_flux_magnitude(
        density_mag=density, kappa=kappa, theta_mag=theta, energy_mag=energies
    )
    number_flux_cm2 = 0.25 * simpson(F_omni, energies)  # particles / (cm^2 s)
    number_flux = number_flux_cm2 * 1e4  # particles / (m^2 s)
    current_flux = number_flux * config.ELECTRON_CHARGE_MAGNITUDE  # A / m^2
    return current_flux
