import logging
import math
from dataclasses import dataclass

import numpy as np
import spiceypy as spice
from numba import jit
from pint import Quantity
from scipy.optimize import brentq, minimize
from scipy.stats import qmc

from src import config
from src.flux import ERData, PitchAngle
from src.physics.charging import electron_current_density_magnitude
from src.physics.jucurve import U_from_J
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux_fast,
    omnidirectional_flux_magnitude,
    velocity_from_energy,
)
from src.utils.geometry import get_intersection_or_none
from src.utils.spice_ops import (
    get_lp_position_wrt_moon,
    get_lp_vector_to_sun_in_lunar_frame,
)
from src.utils.units import (
    EnergyType,
    FluxType,
    NumberDensityType,
    ureg,
)


@dataclass
class FitResults:
    """
    Represents the results of a kappa distribution fit.

    Attributes:
        params (KappaParams): The best-fit parameters.
        params_uncertainty (KappaParams): The 1-sigma uncertainties of the fit
            parameters.
        error (float): The final chi-squared error of the fit.
        is_good_fit (bool): A flag indicating if the fit is considered
            reliable.
    """

    params: KappaParams
    params_uncertainty: KappaParams
    error: float
    is_good_fit: bool


class Kappa:
    """
    Class for handling kappa distribution fitting and evaluation.

    This class prepares the data for fitting, estimates the density, and
    provides methods to fit the kappa distribution parameters (kappa and
    theta) to the data.
    """

    DEFAULT_BOUNDS = [(2.5, 6.0), (6, 8)]  # kappa  # theta in log m/s

    def __init__(self, er_data: ERData, spec_no: int) -> None:
        """
        Initialize the Kappa class with ERData and specification number.

        Args:
            er_data (ERData): The ERData object containing the electron flux data.
            spec_no (int): The specification number to filter the data.
        """

        self.er_data = er_data
        self.spec_no = spec_no
        self.is_data_valid = False

        self.solid_angles = (
            np.loadtxt(
                config.DATA_DIR / config.SOLID_ANGLES_FILE,
                dtype=np.float64,
            ).reshape(-1, config.CHANNELS)
            * ureg.steradian
        )

        self.omnidirectional_differential_particle_flux: FluxType | None = None
        self.omnidirectional_differential_particle_flux_mag: np.ndarray | None = None
        self.omnidirectional_count: np.ndarray | None = None
        self.sigma_omnidirectional_count: np.ndarray | None = None
        self.sigma_flux_mag: np.ndarray | None = None
        self.sigma_log_flux: np.ndarray | None = None
        self.energy_centers: EnergyType | None = None
        self.energy_centers_mag: np.ndarray | None = None
        self.energy_windows: EnergyType | None = None

        self._prepare_data()
        self.density_estimate: NumberDensityType = self._get_density_estimate()
        self.density_estimate_mag = self.density_estimate.to(
            ureg.particle / ureg.meter**3
        ).magnitude

        if __debug__:
            if not isinstance(
                self.omnidirectional_differential_particle_flux, Quantity
            ) or not self.omnidirectional_differential_particle_flux.is_compatible_with(
                ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
            ):
                raise TypeError(
                    (
                        "omnidirectional_differential_particle_flux must be a "
                        "pint Quantity (OmnidirectionalFlux)"
                    )
                )
            if not isinstance(
                self.energy_centers, Quantity
            ) or not self.energy_centers.is_compatible_with(ureg.electron_volt):
                raise TypeError("energy_centers must be a pint Quantity (Energy)")
            if not isinstance(
                self.energy_windows, Quantity
            ) or not self.energy_windows.is_compatible_with(ureg.electron_volt):
                raise TypeError("energy_windows must be a pint Quantity (Energy)")
            if not isinstance(
                self.density_estimate, Quantity
            ) or not self.density_estimate.is_compatible_with(
                ureg.particle / ureg.meter**3
            ):
                raise TypeError(
                    "density_estimate must be a pint Quantity (NumberDensity)"
                )

    def _prepare_data(self) -> None:
        """
        Prepare the data for fitting.

        This method extracts the relevant data for the specified `spec_no` from
        the `ERData` object, calculates the sum of the electron flux over valid
        pitch angles, and sets the energy windows.
        """
        if self.spec_no not in self.er_data.data[config.SPEC_NO_COLUMN].values:
            logging.warning(f"Spec no {self.spec_no} not found in ERData.")
            return

        spec_dataframe = self.er_data.data[
            self.er_data.data[config.SPEC_NO_COLUMN] == self.spec_no
        ].copy()
        spec_dataframe.reset_index(drop=True, inplace=True)

        spec_er_data = self.er_data.__class__.from_dataframe(
            spec_dataframe, self.er_data.er_data_file
        )
        self.er_data = spec_er_data

        spec_pitch_angle = PitchAngle(
            spec_er_data, str(config.DATA_DIR / config.THETA_FILE)
        )

        pitch_angles = spec_pitch_angle.pitch_angles
        pitch_angles_mask = pitch_angles < 90  # shape (Energy Bins, Pitch Angles)
        if not np.any(pitch_angles_mask):
            logging.warning("No valid pitch angles found for the specified spec_no.")
            return

        directional_flux_units = ureg.particle / (
            ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt
        )
        electron_flux = (
            spec_er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)
            * directional_flux_units
        )  # shape (Energy Bins, Pitch Angles)

        masked_electron_flux = np.where(
            pitch_angles_mask, electron_flux, 0 * electron_flux.units
        )
        # Integrate directional flux over the incident hemisphere (pitch < 90°)
        # and scale by 2 to approximate the full 4π omnidirectional flux under
        # an isotropy assumption. Without this factor the derived density is
        # biased low by ~2× because only half-space is included.
        self.omnidirectional_differential_particle_flux = 2 * np.sum(
            masked_electron_flux * self.solid_angles, axis=1
        )  # shape (Energy Bins,)
        self.omnidirectional_differential_particle_flux_mag = (
            self.omnidirectional_differential_particle_flux.magnitude
        )
        count_sigma_count = spec_er_data.data[config.COUNT_COLS].to_numpy(
            dtype=np.float64
        )
        self.omnidirectional_count = count_sigma_count[:, 0]  # shape (Energy Bins,)
        self.sigma_omnidirectional_count = np.sqrt(
            self.omnidirectional_count
            + (config.E_GAIN * self.omnidirectional_count) ** 2
            + (config.E_G * self.omnidirectional_count) ** 2
            + config.N_BG
        )  # shape (Energy Bins,)

        self.sigma_flux_mag = (
            self.sigma_omnidirectional_count / (self.omnidirectional_count + config.EPS)
        ) * self.omnidirectional_differential_particle_flux_mag
        self.sigma_log_flux = (
            self.sigma_flux_mag
            / (self.omnidirectional_differential_particle_flux_mag + config.EPS)
            + config.EPS
        )  # shape (Energy Bins,)

        energies = (
            spec_er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
        ) * ureg.electron_volt  # shape (Energy Bins,)

        self.energy_centers = energies
        self.energy_centers_mag = energies.to(ureg.electron_volt).magnitude
        from src.utils.energy import make_relative_energy_bounds

        self.energy_windows = make_relative_energy_bounds(
            energies, rel_width=config.ENERGY_WINDOW_WIDTH_RELATIVE
        )  # shape (Energy Bins, 2)

        self.is_data_valid = True

    def _objective_function(
        self, kappa_theta: np.ndarray, use_weights: bool, use_convolution: bool = True
    ) -> float:
        """
        Original objective function for optimization.

        Computes the squared difference between the log of the model
        differential flux and the log of the measured flux.

        Args:
            kappa_theta (np.ndarray): Array containing kappa and log10(theta).
            use_weights (bool): Whether to use weights in the calculation.

        Returns:
            float: The chi-squared value representing the difference.
        """
        density = self.density_estimate
        kappa = kappa_theta[0]
        theta = 10 ** kappa_theta[1] * ureg.meter / ureg.second

        params = KappaParams(density, kappa, theta)
        W = (
            self.build_log_energy_response_matrix(
                self.energy_centers_mag,
                energy_window_width_relative=config.ENERGY_WINDOW_WIDTH_RELATIVE,
            )
            if use_convolution
            else np.eye(len(self.energy_centers_mag))
        )
        model_differential_flux = W @ omnidirectional_flux_fast(
            params, self.energy_centers
        )

        omnidirectional_flux_units = ureg.particle / (
            ureg.centimeter**2 * ureg.second * ureg.electron_volt
        )
        if __debug__:
            if not isinstance(
                model_differential_flux, Quantity
            ) or not model_differential_flux.is_compatible_with(
                omnidirectional_flux_units
            ):
                raise TypeError(
                    (
                        "model_differential_flux must be a pint Quantity "
                        "(OmnidirectionalFlux)"
                    )
                )
            if not isinstance(
                self.omnidirectional_differential_particle_flux, Quantity
            ) or not self.omnidirectional_differential_particle_flux.is_compatible_with(
                omnidirectional_flux_units
            ):
                raise TypeError(
                    (
                        "omnidirectional_differential_particle_flux must be a "
                        "pint Quantity (OmnidirectionalFlux)"
                    )
                )

        log_model_differential_flux = np.log(
            model_differential_flux.to(omnidirectional_flux_units).magnitude
            + config.EPS
        )
        log_data_flux = np.log(
            self.omnidirectional_differential_particle_flux.to(
                omnidirectional_flux_units
            ).magnitude
            + config.EPS
        )

        weights = (
            (1 / (self.sigma_log_flux + config.EPS))
            if use_weights
            else np.ones_like(self.omnidirectional_count)
        )

        chi2 = np.sum(((log_model_differential_flux - log_data_flux) * weights) ** 2)
        return chi2

    @staticmethod
    @jit(nopython=True, cache=True)
    def _compute_chi2_numba(
        model_flux_mag: np.ndarray, measured_flux_mag: np.ndarray, weights: np.ndarray
    ) -> float:
        """
        Compute the squared difference of the logarithm of model and measured fluxes.

        This function is optimized for performance using Numba.

        Args:
            model_flux_mag (np.ndarray): Magnitudes of the model flux.
            measured_flux_mag (np.ndarray): Magnitudes of the measured flux.
        """

        log_model = np.log(model_flux_mag)
        log_data = np.log(measured_flux_mag)
        diff = (log_model - log_data) * weights
        chi2 = np.sum(diff * diff)
        return chi2

    @staticmethod
    @jit(nopython=True, cache=True)
    def build_log_energy_response_matrix(
        energy_centers: np.ndarray,
        energy_window_width_relative: float = config.ENERGY_WINDOW_WIDTH_RELATIVE,
    ) -> np.ndarray:
        """
        Build a log-energy response matrix for instrumental energy resolution effects.

        Args:
            energy_centers: Central energy values for each energy bin
            energy_window_width_relative: Relative energy resolution (FWHM/E),
                                        e.g., 0.5 means 50% energy resolution

        Returns:
            Response matrix W where W[i,j] represents the fraction of flux
            from true energy j that appears in measured energy bin i
        """
        ln_energy_centers = np.log(energy_centers)

        # Convert relative width to Gaussian sigma in log-energy space
        # The asinh transform helps convert relative width to log space
        s = math.asinh(0.5 * energy_window_width_relative) / math.sqrt(
            2.0 * math.log(2.0)
        )

        W = np.exp(
            -0.5 * ((ln_energy_centers[:, None] - ln_energy_centers[None, :]) / s) ** 2
        )
        W /= np.sum(W, axis=1).reshape(-1, 1)
        return W

    def _objective_function_fast(
        self, kappa_theta: np.ndarray, use_weights: bool, use_convolution: bool = True
    ) -> float:
        """
        Fast objective for optimization using the fast
        omnidirectional flux calculation.

        Args:
            kappa_theta (np.ndarray): Array containing kappa and log10(theta).

        Returns:
            float: The chi-squared value representing the difference.
        """
        density_mag = self.density_estimate.magnitude
        kappa = kappa_theta[0]
        theta_mag = 10 ** kappa_theta[1]

        W = (
            self.build_log_energy_response_matrix(
                self.energy_centers_mag,
                energy_window_width_relative=config.ENERGY_WINDOW_WIDTH_RELATIVE,
            )
            if use_convolution
            else np.eye(len(self.energy_centers_mag))
        )

        model_flux_magnitudes = W @ omnidirectional_flux_magnitude(
            density_mag, kappa, theta_mag, self.energy_centers_mag
        )
        weights = (
            (1 / (self.sigma_log_flux + config.EPS))
            if use_weights
            else np.ones_like(self.omnidirectional_count)
        )

        return self._compute_chi2_numba(
            model_flux_magnitudes + config.EPS,
            self.omnidirectional_differential_particle_flux_mag + config.EPS,
            weights,
        )

    def _get_density_estimate(self) -> NumberDensityType:
        """
        Estimate the density based on the sum of the flux and the energy windows.

        Returns:
            NumberDensity: Estimated density in m^-3.
        """
        if (
            self.omnidirectional_differential_particle_flux is None
            or self.energy_centers is None
            or self.energy_windows is None
        ):
            raise ValueError("Data not prepared. Call _prepare_data() first.")

        if __debug__:
            if not isinstance(
                self.omnidirectional_differential_particle_flux, Quantity
            ) or not self.omnidirectional_differential_particle_flux.is_compatible_with(
                ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
            ):
                raise TypeError(
                    (
                        "omnidirectional_differential_particle_flux must be a "
                        "pint Quantity (OmnidirectionalFlux)"
                    )
                )
            if not isinstance(
                self.energy_centers, Quantity
            ) or not self.energy_centers.is_compatible_with(ureg.electron_volt):
                raise TypeError("energy_centers must be a pint Quantity (Energy)")
            if not isinstance(
                self.energy_windows, Quantity
            ) or not self.energy_windows.is_compatible_with(ureg.electron_volt):
                raise TypeError("energy_windows must be a pint Quantity (Energy)")

        velocities = velocity_from_energy(self.energy_centers)  # shape (Energy Bins,)
        delta_energy = (
            self.energy_windows[:, 1] - self.energy_windows[:, 0]
        )  # shape (Energy Bins,)

        integrand = (
            self.omnidirectional_differential_particle_flux / velocities
        )  # shape (Energy Bins,)

        density_estimate = np.sum(integrand * delta_energy)

        return density_estimate.to(ureg.particle / ureg.meter**3)

    def fit(
        self,
        n_starts: int = 10,
        use_fast: bool = True,
        use_weights: bool = True,
        use_convolution: bool = True,
    ) -> FitResults | None:
        """
        Fit the kappa distribution parameters (kappa and theta) to the data.

        Args:
            n_starts (int): Number of random starts for the optimization.
            use_fast (bool): Whether to use the fast objective function.

        Returns:
            FitResults | None: The fit results, or None if the fit fails.
        """
        if not self.is_data_valid:
            raise ValueError(
                "Data is not valid. Ensure that the data has been prepared."
            )

        sampler = qmc.LatinHypercube(d=len(self.DEFAULT_BOUNDS), seed=42)
        samples = sampler.random(n_starts)
        scaled_samples = qmc.scale(
            samples,
            [b[0] for b in self.DEFAULT_BOUNDS],
            [b[1] for b in self.DEFAULT_BOUNDS],
        )

        objective_func = (
            self._objective_function_fast if use_fast else self._objective_function
        )
        best_result = None

        for i, x0 in enumerate(scaled_samples):
            logging.debug(f"Running optimization for sample {i + 1}/{n_starts}")
            result = minimize(
                objective_func,
                x0,
                args=(use_weights, use_convolution),
                bounds=self.DEFAULT_BOUNDS,
                method="L-BFGS-B",
                options={"maxiter": 1000},
            )
            if best_result is None or (result.success and result.fun < best_result.fun):
                best_result = result

            if best_result and best_result.fun < 0.01:
                logging.debug(
                    f"Early stopping at sample {i + 1} with chi2={best_result.fun:.4f}"
                )

        if best_result is None:
            logging.warning("No valid optimization result found.")
            return None

        # Calculate uncertainties from the inverse Hessian matrix
        sigma_kappa, sigma_log_theta = 0.0, 0.0
        if best_result.success and hasattr(best_result, "hess_inv"):
            try:
                # The inverse Hessian is the covariance matrix
                covariance_matrix = best_result.hess_inv.todense()
                # The sqrt of the diagonal elements are the 1-sigma uncertainties
                uncertainties = np.sqrt(np.diag(covariance_matrix))
                sigma_kappa = uncertainties[0]
                sigma_log_theta = uncertainties[1]
            except Exception as e:
                logging.warning(f"Failed to compute uncertainties: {e}")

        # Create the results object
        fitted_params = KappaParams(
            density=self.density_estimate,
            kappa=best_result.x[0],
            theta=10 ** best_result.x[1] * ureg.meter / ureg.second,
        )

        params_uncertainty = KappaParams(
            density=0
            * ureg.particle
            / ureg.meter**3,  # Density is not a fitted parameter
            kappa=sigma_kappa,
            theta=sigma_log_theta
            * ureg.meter
            / ureg.second,  # This is uncertainty in log(theta)
        )

        is_good_fit = best_result.fun < config.FIT_ERROR_THRESHOLD

        return FitResults(
            params=fitted_params,
            params_uncertainty=params_uncertainty,
            error=best_result.fun,
            is_good_fit=is_good_fit,
        )

    def corrected_fit(
        self,
        n_starts: int = 10,
        use_fast: bool = True,
        use_weights: bool = True,
        use_convolution: bool = True,
    ) -> tuple[FitResults, float] | tuple[None, None]:
        """
        Fit parameters and estimate spacecraft potential U following
        the illumination-dependent logic used in spacecraft_potential:

        - Sunlight (dayside):
            1) Fit κ on unshifted data
            2) Compute electron current density Je from the fit
            3) Invert J–U to get U in [0, 150] V
            4) Shift energies by −U, recompute density, and refit κ
            5) Recompute Je and U from corrected fit and return (fit, U)

        - Shade (nightside):
            1) Fit κ on unshifted data
            2) Solve Je(U) + Ji(U) − Jsee(U) = 0 for U via brentq
            3) Map κ temperature to ambient at U, build corrected params and return

        Returns:
            (FitResults, float) | (None, None): Best-fit (possibly corrected)
            parameters and spacecraft potential U [V], or (None, None) if
            fitting fails or geometry unavailable.
        """
        if not self.is_data_valid:
            raise ValueError(
                "Data is not valid. Ensure that the data has been prepared."
            )

        # Determine illumination for this spectrum number
        rows = self.er_data.data[
            self.er_data.data[config.SPEC_NO_COLUMN] == self.spec_no
        ]
        if rows.empty:
            return None, None
        utc_val = rows.iloc[0].get(
            config.UTC_COLUMN, rows.iloc[0].get(config.TIME_COLUMN)
        )
        try:
            et = spice.str2et(str(utc_val))
            lp_position_wrt_moon = get_lp_position_wrt_moon(et)
            lp_vector_to_sun = get_lp_vector_to_sun_in_lunar_frame(et)
        except Exception as e:
            logging.warning(f"SPICE time/geometry failed for spec {self.spec_no}: {e}")
            return None, None

        intersection = get_intersection_or_none(
            lp_position_wrt_moon, lp_vector_to_sun, config.LUNAR_RADIUS
        )
        is_day = intersection is None

        # Initial uncorrected fit
        original_fit = self.fit(
            n_starts=n_starts,
            use_fast=use_fast,
            use_weights=use_weights,
            use_convolution=use_convolution,
        )
        if not original_fit or not original_fit.is_good_fit:
            return None, None

        # Daylight branch: JU inversion + energy pre-shift + refit
        # (mirrors spacecraft_potential)
        if is_day:
            original_fit_params = original_fit.params
            Je = electron_current_density_magnitude(
                *original_fit_params.to_tuple(), E_min=1e1, E_max=2e4, n_steps=100
            )
            U = U_from_J(J_target=Je, U_min=0.0, U_max=150.0)

            # Apply pre-shift and refit
            corrected_energy_centers = self.energy_centers_mag - U
            self.er_data.data[config.ENERGY_COLUMN] = corrected_energy_centers
            self.is_data_valid = False
            self._prepare_data()
            self.density_estimate = self._get_density_estimate()
            self.density_estimate_mag = self.density_estimate.to(
                ureg.particle / ureg.meter**3
            ).magnitude

            corrected_fit = self.fit(
                n_starts=n_starts,
                use_fast=use_fast,
                use_weights=use_weights,
                use_convolution=use_convolution,
            )
            if corrected_fit and corrected_fit.is_good_fit:
                params = corrected_fit.params
                Je_c = electron_current_density_magnitude(
                    *params.to_tuple(), E_min=1e1, E_max=2e4, n_steps=100
                )
                U_c = U_from_J(J_target=Je_c, U_min=0.0, U_max=150.0)
                return corrected_fit, U_c
            # Fall back to original if corrected fit degrades
            return original_fit, U

        # Nightside branch: root solve Je(U) + Ji(U) − Jsee(U) = 0
        density_mag, kappa, theta_uc = original_fit.params.to_tuple()

        def theta_to_temperature_ev(theta: float, kappa: float) -> float:
            prefactor = 0.5 * kappa / (kappa - 1.5)
            return (
                prefactor
                * theta
                * theta
                * config.ELECTRON_MASS_MAGNITUDE
                / config.ELECTRON_CHARGE_MAGNITUDE
            )

        def temperature_ev_to_theta(temperature_ev: float, kappa: float) -> float:
            prefactor = 2.0 * (kappa - 1.5) / kappa
            return math.sqrt(
                prefactor
                * config.ELECTRON_CHARGE_MAGNITUDE
                * temperature_ev
                / config.ELECTRON_MASS_MAGNITUDE
            )

        def sternglass_secondary_yield(
            E_imp: np.ndarray, E_m: float, delta_m: float
        ) -> np.ndarray:
            out = (
                7.4
                * delta_m
                * (E_imp / E_m)
                * np.exp(-2.0 * np.sqrt(np.maximum(E_imp, 0.0) / E_m))
            )
            out[E_imp <= 0.0] = 0.0
            return out

        temperature_uc = theta_to_temperature_ev(theta_uc, kappa)
        E_min_ev, E_max_ev, n_steps = 1.0, 2e4, 401
        energy_grid = np.geomspace(max(E_min_ev, 0.5), E_max_ev, n_steps)

        CM2_TO_M2 = 1.0e4

        def calc_currents(
            U: float, sey_E_m: float, sey_delta_m: float
        ) -> tuple[float, float, float]:
            T_c = temperature_uc + U / (kappa - 1.5)
            if T_c <= 0.0:
                BIG = 1e12
                return 0.0, 0.0, BIG
            theta_c = temperature_ev_to_theta(T_c, kappa)
            omni = omnidirectional_flux_magnitude(
                density_mag=density_mag,
                kappa=kappa,
                theta_mag=theta_c,
                energy_mag=energy_grid,
            )
            flux_to_sc = 0.25 * omni * CM2_TO_M2
            mask = energy_grid >= abs(U)
            Je = config.ELECTRON_CHARGE_MAGNITUDE * np.trapezoid(
                flux_to_sc[mask], energy_grid[mask]
            )
            Eimp = energy_grid[mask] - abs(U)
            Jsee = config.ELECTRON_CHARGE_MAGNITUDE * np.trapezoid(
                sternglass_secondary_yield(Eimp, E_m=sey_E_m, delta_m=sey_delta_m)
                * flux_to_sc[mask],
                energy_grid[mask],
            )
            T_i = T_c + config.EPS
            vth_i = np.sqrt(
                config.ELECTRON_CHARGE_MAGNITUDE
                * T_i
                / (2.0 * np.pi * config.PROTON_MASS_MAGNITUDE)
            )
            Ji0 = config.ELECTRON_CHARGE_MAGNITUDE * density_mag * vth_i
            Ji = Ji0 * np.sqrt(max(0.0, 1.0 - U / T_i))
            return Je, Jsee, Ji

        def balance(U: float, E_m: float, delta_m: float) -> float:
            Je, Jsee, Ji = calc_currents(U, E_m, delta_m)
            return Je + Ji - Jsee

        sey_E_m, sey_delta_m = 500.0, 1.5
        U_low, U_high = -10.0, 10.0
        f_low, f_high = balance(U_low, sey_E_m, sey_delta_m), balance(
            U_high, sey_E_m, sey_delta_m
        )
        tries = 0
        while np.sign(f_low) == np.sign(f_high) and tries < 10:
            U_low *= 1.5
            f_low = balance(U_low, sey_E_m, sey_delta_m)
            tries += 1
        if np.isnan(f_low) or np.isnan(f_high) or np.sign(f_low) == np.sign(f_high):
            return None, None

        U_star = float(
            brentq(
                balance,
                U_low,
                U_high,
                args=(sey_E_m, sey_delta_m),
                maxiter=200,
                xtol=1e-3,
            )
        )
        T_c = temperature_uc + U_star / (kappa - 1.5)
        theta_c = temperature_ev_to_theta(T_c, kappa)
        corrected_params = KappaParams(
            density=density_mag * ureg.particle / ureg.meter**3,
            kappa=kappa,
            theta=theta_c * ureg.meter / ureg.second,
        )
        # Wrap into FitResults with original error/uncertainty since we did not refit
        return (
            FitResults(
                params=corrected_params,
                params_uncertainty=original_fit.params_uncertainty,
                error=original_fit.error,
                is_good_fit=original_fit.is_good_fit,
            ),
            U_star,
        )
