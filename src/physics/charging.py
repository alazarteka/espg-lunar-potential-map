import math
import numpy as np
from scipy.integrate import simpson

import src.config as config
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux_magnitude,
    theta_to_temperature_ev,
)
from src.utils.units import CurrentDensityType, EnergyType, ureg


def sternglass_secondary_yield(
    impact_energy_ev: np.ndarray,
    peak_energy_ev: float = 500.0,
    peak_yield: float = 1.5,
) -> np.ndarray:
    """
    Calculate the Sternglass secondary electron yield.

    The Sternglass model describes secondary electron emission (SEE) from surfaces
    bombarded by primary electrons. The yield depends on the impact energy
    following an empirical formula that accounts for the penetration depth
    and secondary electron escape probability.

    Args:
        impact_energy_ev: The impact energy at the surface in eV (array).
        peak_energy_ev: The energy where the yield peaks in eV (default 500).
        peak_yield: The maximum number of secondaries per incident primary
            (default 1.5, typical for lunar regolith).

    Returns:
        The Sternglass secondary electron yield (same shape as impact_energy_ev).

    Notes:
        - For E <= 0, the yield is 0 (no impact, no secondaries)
        - The formula is: delta = 7.4 * delta_m * (E/E_m) * exp(-2 * sqrt(E/E_m))
        - This peaks at E = E_m with value approximately delta_m
    """
    out = (
        7.4
        * peak_yield
        * (impact_energy_ev / peak_energy_ev)
        * np.exp(-2.0 * np.sqrt(np.maximum(impact_energy_ev, 0.0) / peak_energy_ev))
    )
    out[impact_energy_ev <= 0.0] = 0.0
    return out


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


def _kappa_current_factor(kappa: float) -> float:
    """
    Return the kappa correction factor for the one-sided thermal flux.

    Factor: sqrt(kappa - 3/2) * Gamma(kappa - 1) / Gamma(kappa - 1/2)
    """
    if kappa <= 1.5:
        raise ValueError("kappa must be greater than 1.5 for a valid distribution")
    return math.sqrt(kappa - 1.5) * math.gamma(kappa - 1.0) / math.gamma(kappa - 0.5)


def kappa_current_density_analytic(params: KappaParams) -> CurrentDensityType:
    """
    Analytic one-sided current density for a kappa distribution.

    J_kappa = n q sqrt(kT_kappa / (2π m)) * sqrt(kappa - 3/2)
              * Gamma(kappa - 1) / Gamma(kappa - 1/2)
    """
    if __debug__ and not isinstance(params, KappaParams):
        raise TypeError("params must be an instance of KappaParams")

    density_mag = params.density.to(ureg.particle / ureg.meter**3).magnitude
    kappa = params.kappa
    theta_mag = params.theta.to(ureg.meter / ureg.second).magnitude

    temperature_ev = theta_to_temperature_ev(theta_mag, kappa)
    kT_joule = temperature_ev * config.ELECTRON_CHARGE_MAGNITUDE
    thermal_term = math.sqrt(
        kT_joule / (2.0 * math.pi * config.ELECTRON_MASS_MAGNITUDE)
    )
    correction = _kappa_current_factor(kappa)
    current_mag = (
        density_mag
        * config.ELECTRON_CHARGE_MAGNITUDE
        * thermal_term
        * correction
    )
    return current_mag * (ureg.ampere / ureg.meter**2)


def kappa_current_density_analytic_magnitude(
    density: float,
    kappa: float,
    theta: float,
) -> float:
    """
    Analytic one-sided current density for a kappa distribution (magnitudes).

    Expects:
        density: particles / m^3
        kappa: dimensionless
        theta: m / s
    """
    temperature_ev = theta_to_temperature_ev(theta, kappa)
    kT_joule = temperature_ev * config.ELECTRON_CHARGE_MAGNITUDE
    thermal_term = math.sqrt(
        kT_joule / (2.0 * math.pi * config.ELECTRON_MASS_MAGNITUDE)
    )
    correction = _kappa_current_factor(kappa)
    return (
        density * config.ELECTRON_CHARGE_MAGNITUDE * thermal_term * correction
    )
