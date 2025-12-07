"""
Spacecraft potential estimation for Lunar Prospector ER spectra.

This module determines the floating spacecraft potential U [V] per spectrum by
branching on illumination:

- Sunlight (dayside): invert a JU (photoemission) curve using the κ-fit electron
  current, pre-shift energies by −U (in eV), refit, and return corrected κ and U.
- Shade (nightside): solve current balance Je(U) + Ji(U) − Jsee(U) = 0 with a
  Sternglass-like SEE model and an OML-like ion current. The κ temperature is
  mapped from the uncorrected fit to the ambient value at the trial U.

Conventions
- Energies and temperatures are in eV; potentials in V (numerically equal for e±).
- `omnidirectional_flux_magnitude` yields 4π-integrated flux in cm^-2 s^-1 eV^-1.
  Map to a plane with a 1/4 factor (isotropic cosine-weighted incidence) and
  convert to m^-2 via ×1e4.

See docs/analysis/spacecraft_potential_analysis.md for a deeper discussion.
"""

import math

import numpy as np
import spiceypy as spice
from scipy.optimize import brentq

from src import config
from src.flux import ERData
from src.kappa import FitResults, Kappa
from src.physics.charging import electron_current_density_magnitude
from src.physics.jucurve import U_from_J
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux_magnitude,
)
from src.utils.geometry import get_intersection_or_none
from src.utils.spice_ops import (
    get_lp_position_wrt_moon,
    get_lp_vector_to_sun_in_lunar_frame,
)
from src.utils.units import (
    VoltageType,
    ureg,
)

CM2_TO_M2_FACTOR = 1.0e4  # unit conversion: cm^-2 → m^-2


def theta_to_temperature_ev(theta: float, kappa: float) -> float:
    """
    Helper function to convert the thermal spread parameter theta to electron
    temperature in eV.

    Args:
        theta (float): The thermal spread parameter in m/s.
        kappa (float): The kappa parameter.

    Returns:
        float: The electron temperature in eV.
    """
    prefactor = 0.5 * kappa / (kappa - 1.5)
    return (
        prefactor
        * theta
        * theta
        * config.ELECTRON_MASS_MAGNITUDE
        / config.ELECTRON_CHARGE_MAGNITUDE
    )


def temperature_ev_to_theta(temperature_ev: float, kappa: float) -> float:
    """
    Helper function to convert electron temperature in eV to the thermal
    spread parameter theta.

    Args:
        temperature_ev (float): The electron temperature in eV.
        kappa (float): The kappa parameter.

    Returns:
        float: The thermal spread parameter theta.
    """
    prefactor = 2.0 * (kappa - 1.5) / kappa
    return math.sqrt(
        prefactor
        * config.ELECTRON_CHARGE_MAGNITUDE
        * temperature_ev
        / config.ELECTRON_MASS_MAGNITUDE
    )


def sternglass_secondary_yield(
    impact_energy_ev: np.ndarray,
    peak_energy_ev: float = 500,
    peak_yield: float = 1.5,
) -> np.ndarray:
    """
    Calculates the Sternglass secondary electron yield.

    Args:
        impact_energy_ev (np.ndarray): The impact energy at the surface in eV.
        peak_energy_ev (float): The energy where the yield peaks in eV.
        peak_yield (float): The maximum number of secondaries per incident primary.

    Returns:
        np.ndarray: The Sternglass secondary electron yield.
    """
    out = (
        7.4
        * peak_yield
        * (impact_energy_ev / peak_energy_ev)
        * np.exp(-2.0 * np.sqrt(np.maximum(impact_energy_ev, 0.0) / peak_energy_ev))
    )
    out[impact_energy_ev <= 0.0] = 0.0
    return out


def calculate_shaded_currents(
    spacecraft_potential: float,
    uncorrected_fit: FitResults,
    energy_grid: np.ndarray,
    sey_E_m: float,
    sey_delta_m: float,
) -> tuple[float, float, float]:
    """
    Current components in shade at a trial potential U.

    Computes ambient κ temperature from the uncorrected fit and U, then
      evaluates:
        - Je: collected electron current density [A m^-2]
        - Jsee: secondary-electron emission current density [A m^-2]
        - Ji: collected ion current density [A m^-2]

    Args:
        - spacecraft_potential: Trial U [V], expected negative in shade.
        - uncorrected_fit: κ-fit on the unshifted spectrum.
        - energy_grid: Energy grid [eV] for the integrals.
        - sey_E_m: SEE peak energy [eV] for Sternglass yield.
        - sey_delta_m: SEE peak yield (dimensionless) for Sternglass yield.

    Returns:
        - Je: collected electron current density [A m^-2]
        - Jsee: secondary-electron emission current density [A m^-2]
        - Ji: collected ion current density [A m^-2]
    """

    density_magnitude, kappa, theta_uncorrected_m_per_s = (
        uncorrected_fit.params.to_tuple()
    )
    temperature_uncorrected_ev = theta_to_temperature_ev(
        theta_uncorrected_m_per_s, kappa
    )

    # boltzmann constant k = 1.0 when temperature is in eV
    temperature_corrected_ev = temperature_uncorrected_ev + spacecraft_potential / (
        kappa - 1.5
    )
    # Robustness: if the κ mapping yields a nonphysical ambient temperature,
    # bias the balance toward less-negative U by returning a large positive Ji.
    if temperature_corrected_ev <= 0.0:
        large_positive_current = 1e12
        return 0.0, 0.0, large_positive_current

    # Calculating Je
    theta_corrected_m_per_s = temperature_ev_to_theta(temperature_corrected_ev, kappa)
    omnidirectional_flux = omnidirectional_flux_magnitude(
        density_mag=density_magnitude,  # unitless for normalization,
        kappa=kappa,
        theta_mag=theta_corrected_m_per_s,
        energy_mag=energy_grid,
    )
    # Isotropic 4π omni → plane: multiply by 1/4; then cm^-2 → m^-2.
    flux_to_spacecraft = 0.25 * omnidirectional_flux * CM2_TO_M2_FACTOR

    energy_above_barrier = energy_grid >= abs(spacecraft_potential)
    Je = config.ELECTRON_CHARGE_MAGNITUDE * np.trapezoid(
        flux_to_spacecraft[energy_above_barrier],
        energy_grid[energy_above_barrier],
    )

    # Calculating Jsee
    impact_energy_ev = energy_grid[energy_above_barrier] - abs(spacecraft_potential)
    Jsee = config.ELECTRON_CHARGE_MAGNITUDE * np.trapezoid(
        sternglass_secondary_yield(
            impact_energy_ev=impact_energy_ev,
            peak_energy_ev=sey_E_m,
            peak_yield=sey_delta_m,
        )
        * flux_to_spacecraft[energy_above_barrier],
        energy_grid[energy_above_barrier],
    )

    # Calculating Ji
    # Assuming ion temperature is same as electron temperature
    ion_temperature = temperature_corrected_ev + config.EPS  # upcoming denominator
    vth_i = np.sqrt(
        config.ELECTRON_CHARGE_MAGNITUDE
        * ion_temperature
        / (2.0 * np.pi * config.PROTON_MASS_MAGNITUDE)
    )
    ion_current_density_reference = (
        config.ELECTRON_CHARGE_MAGNITUDE * density_magnitude * vth_i
    )
    Ji = ion_current_density_reference * np.sqrt(
        max(0.0, 1.0 - spacecraft_potential / ion_temperature)
    )

    return Je, Jsee, Ji


def current_balance(
    spacecraft_potential: float,
    uncorrected_fit: FitResults,
    energy_grid: np.ndarray,
    sey_E_m: float,
    sey_delta_m: float,
) -> float:
    """
    Signed current balance F(U) = Ji(U) + Jsee(U) − Je(U) [A m^-2].

    Root at F(U)=0 defines the floating potential in shade. Positive F favors
    less-negative U; negative F favors more-negative U.
    """
    Je, Jsee, Ji = calculate_shaded_currents(
        spacecraft_potential=spacecraft_potential,
        uncorrected_fit=uncorrected_fit,
        energy_grid=energy_grid,
        sey_E_m=sey_E_m,
        sey_delta_m=sey_delta_m,
    )
    # Currents in the balance are signed: electron collection removes charge from
    # the spacecraft (negative current), while ion collection and SEE are
    # positive contributions.  Expressed in magnitudes, the net current is
    # Ji + Jsee − Je.
    return Ji + Jsee - Je


def calculate_potential(
    er_data: ERData,
    spec_no: int,
    E_min_ev: float = 1.0,
    E_max_ev: float = 2.0e4,
    n_steps: int = 500,
    spacecraft_potential_low: float = -1500.0,
    spacecraft_potential_high: float = 0.0,
    sey_E_m: float = 500.0,
    sey_delta_m: float = 1.5,
) -> tuple[KappaParams, VoltageType] | None:
    """
    Estimate spacecraft potential for a given spectrum.

    Branches on illumination using Sun–Moon geometry:
        - Daylight: compute U from JU curve, pre-shift energies by −U (in eV), refit,
        and return corrected κ and U (>0 V).
        - Shade: build F(U)=Je+Ji−Jsee and solve with Brent for U<0 V; return ambient
        κ (with θ mapped at the solved U) and U.

    Args:
        - er_data: Electron spectrometer dataset containing time and energy columns.
        - spec_no: Spectrum index to evaluate.
        - E_min_ev/E_max_ev/n_steps: Energy grid [eV] and resolution for integrals.
        - spacecraft_potential_low/high: Initial bracket for nightside U [V].
        - sey_E_m/sey_delta_m: SEE calibration parameters (Sternglass model).

    Returns:
        - (KappaParams, U[volt]) on success; None on failure (e.g., poor κ fit or
        unbracketed root in shade).

    Notes:
        - Day branch mutates `er_data.data[ENERGY_COLUMN]` during pre-shift to refit;
        consider copying/restoring upstream if this side effect is undesirable.
        - Uses a private `_prepare_data()` on the fitter; a public helper may be
        preferable if the interface evolves.
    """
    fitter = Kappa(er_data, spec_no)

    # Decide if measurement occurs exposed to sunlight or in shadow using
    # the timestamp for this spectrum number.
    rows = er_data.data[er_data.data[config.SPEC_NO_COLUMN] == spec_no]
    if rows.empty:
        return None
    # Prefer UTC column; fall back to TIME if necessary
    utc_val = rows.iloc[0].get(config.UTC_COLUMN, rows.iloc[0].get(config.TIME_COLUMN))
    try:
        et = spice.str2et(str(utc_val))
    except Exception:
        return None
    lp_position_wrt_moon = get_lp_position_wrt_moon(et)
    lp_vector_to_sun = get_lp_vector_to_sun_in_lunar_frame(et)

    intersection = get_intersection_or_none(
        lp_position_wrt_moon, lp_vector_to_sun, config.LUNAR_RADIUS
    )
    is_day = intersection is None

    # Perform initial fit
    initial_fit_result = fitter.fit()

    if is_day:
        if not initial_fit_result:
            return None  # Place holder for daytime potential calculation

        initial_fit_params = initial_fit_result.params

        initial_spacecraft_current_density = electron_current_density_magnitude(
            *initial_fit_params.to_tuple(), E_min=1e1, E_max=2e4, n_steps=10
        )
        # The spacecraft charges to low 1 or 2 digit positive voltages
        initial_spacecraft_potential = U_from_J(
            J_target=initial_spacecraft_current_density, U_min=0.0, U_max=150.0
        )

        # Applying energy correction
        fitter.is_data_valid = False
        corrected_energy_centers = (
            fitter.energy_centers_mag - initial_spacecraft_potential
        )
        # Clip to prevent negative energies (can happen if spacecraft potential
        # exceeds low-energy bin values)
        corrected_energy_centers = np.clip(corrected_energy_centers, config.EPS, None)
        fitter.er_data.data[config.ENERGY_COLUMN] = corrected_energy_centers
        fitter._prepare_data()
        fitter.density_estimate = fitter._get_density_estimate()
        fitter.density_estimate_mag = fitter.density_estimate.to(
            ureg.particle / ureg.meter**3
        ).magnitude

        refit_result = fitter.fit()

        if not refit_result:
            return None  # Place holder for daytime potential calculation

        corrected_fit_params = refit_result.params
        corrected_spacecraft_current_density = electron_current_density_magnitude(
            *corrected_fit_params.to_tuple(), E_min=1e1, E_max=2e4, n_steps=100
        )
        corrected_spacecraft_potential = U_from_J(
            J_target=corrected_spacecraft_current_density, U_min=0.0, U_max=150.0
        )

        return corrected_fit_params, corrected_spacecraft_potential * ureg.volt

    else:
        if not initial_fit_result:
            return None

        # Unpack initial fit parameters
        initial_fit_params = initial_fit_result.params
        density_magnitude, kappa, theta_uncorrected_m_per_s = (
            initial_fit_params.to_tuple()
        )
        temperature_uncorrected_ev = theta_to_temperature_ev(
            theta_uncorrected_m_per_s, kappa
        )

        energy_grid = np.geomspace(max(E_min_ev, 0.5), E_max_ev, n_steps)

        balance_low, balance_high = (
            current_balance(
                spacecraft_potential_low,
                initial_fit_result,
                energy_grid,
                sey_E_m,
                sey_delta_m,
            ),
            current_balance(
                spacecraft_potential_high,
                initial_fit_result,
                energy_grid,
                sey_E_m,
                sey_delta_m,
            ),
        )
        bracket_expansions = 0
        while np.sign(balance_low) == np.sign(balance_high) and bracket_expansions < 10:
            spacecraft_potential_low *= 1.5
            balance_low = current_balance(
                spacecraft_potential_low,
                initial_fit_result,
                energy_grid,
                sey_E_m,
                sey_delta_m,
            )

            bracket_expansions += 1

        if (
            np.isnan(balance_low)
            or np.isnan(balance_high)
            or np.sign(balance_low) == np.sign(balance_high)
        ):
            return None

        spacecraft_potential = float(
            brentq(
                current_balance,
                spacecraft_potential_low,
                spacecraft_potential_high,
                args=(initial_fit_result, energy_grid, sey_E_m, sey_delta_m),
                maxiter=200,
                xtol=1e-3,
            )
        )

        temperature_corrected_ev = temperature_uncorrected_ev + spacecraft_potential / (
            kappa - 1.5
        )
        theta_corrected_m_per_s = temperature_ev_to_theta(
            temperature_corrected_ev, kappa
        )

        corrected_fit_params = KappaParams(
            density=density_magnitude * ureg.particle / ureg.meter**3,
            kappa=kappa,
            theta=theta_corrected_m_per_s * ureg.meter / ureg.second,
        )

        return corrected_fit_params, spacecraft_potential * ureg.volt
