import logging
from dataclasses import dataclass
import math


import numpy as np
from numba import jit
from pint import Quantity
from scipy.optimize import minimize
from scipy.stats import qmc

from src import config
from src.flux import ERData, PitchAngle
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux_fast,
    omnidirectional_flux_magnitude,
    velocity_from_energy,
)
from src.potential_mapper import DataLoader
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
        params_uncertainty (KappaParams): The 1-sigma uncertainties of the fit parameters.
        error (float): The final chi-squared error of the fit.
        is_good_fit (bool): A flag indicating if the fit is considered reliable.
    """
    params: KappaParams
    params_uncertainty: KappaParams
    error: float
    is_good_fit: bool


class Kappa:
    """
    Class for handling kappa distribution fitting and evaluation.

    This class prepares the data for fitting, estimates the density, and provides methods
    to fit the kappa distribution parameters (kappa and theta) to the data.
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
                    "omnidirectional_differential_particle_flux must be a pint Quantity (OmnidirectionalFlux)"
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

        This method extracts the relevant data for the specified `spec_no` from the `ERData` object,
        calculates the sum of the electron flux over valid pitch angles, and sets the energy windows.
        """
        if self.spec_no not in self.er_data.data["spec_no"].values:
            logging.warning(f"Spec no {self.spec_no} not found in ERData.")
            return

        spec_dataframe = self.er_data.data[
            self.er_data.data["spec_no"] == self.spec_no
        ].copy()
        spec_dataframe.reset_index(drop=True, inplace=True)

        spec_er_data = self.er_data.__class__.from_dataframe(
            spec_dataframe, self.er_data.er_data_file
        )

        spec_pitch_angle = PitchAngle(
            spec_er_data, DataLoader.get_theta_file(config.DATA_DIR)
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
        self.omnidirectional_differential_particle_flux = np.sum(
            masked_electron_flux * self.solid_angles, axis=1
        )  # shape (Energy Bins,)
        self.omnidirectional_differential_particle_flux_mag = (
            self.omnidirectional_differential_particle_flux.magnitude
        )
        count_sigma_count = spec_er_data.data[config.COUNT_COLS].to_numpy(dtype=np.float64)
        self.omnidirectional_count = count_sigma_count[:, 0]  # shape (Energy Bins,)
        self.sigma_omnidirectional_count = np.sqrt(
            self.omnidirectional_count
            + (config.E_GAIN * self.omnidirectional_count) ** 2
            + (config.E_G * self.omnidirectional_count) ** 2
            + config.N_BG
        ) # shape (Energy Bins,)

        self.sigma_flux_mag = (
            self.sigma_omnidirectional_count / (self.omnidirectional_count + config.EPS)
        ) * self.omnidirectional_differential_particle_flux_mag
        self.sigma_log_flux = self.sigma_flux_mag / (
            self.omnidirectional_differential_particle_flux_mag + config.EPS
        ) + config.EPS  # shape (Energy Bins,)

        energies = (
            spec_er_data.data["energy"].to_numpy(dtype=np.float64)
        ) * ureg.electron_volt  # shape (Energy Bins,)

        self.energy_centers = energies
        self.energy_centers_mag = energies.to(ureg.electron_volt).magnitude
        self.energy_windows = np.column_stack(
            [0.75 * energies, 1.25 * energies]
        )  # shape (Energy Bins, 2)

        self.is_data_valid = True

    def _objective_function(self, 
                            kappa_theta: np.ndarray, 
                            use_weights: bool,
                            use_convolution: bool = True
        ) -> float:
        """
        Original objective function for optimization.

        This function computes the squared difference between the log of the model differential flux
        and the log of the measured flux.

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
        W = self.build_log_energy_response_matrix(
            self.energy_centers_mag,
            energy_window_width_relative=config.ENERGY_WINDOW_WIDTH_RELATIVE,
        ) if use_convolution else np.eye(len(self.energy_centers_mag))
        model_differential_flux = W @ omnidirectional_flux_fast(params, self.energy_centers)

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
                    "model_differential_flux must be a pint Quantity (OmnidirectionalFlux)"
                )
            if not isinstance(
                self.omnidirectional_differential_particle_flux, Quantity
            ) or not self.omnidirectional_differential_particle_flux.is_compatible_with(
                omnidirectional_flux_units
            ):
                raise TypeError(
                    "omnidirectional_differential_particle_flux must be a pint Quantity (OmnidirectionalFlux)"
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

        weights = (1 / (self.sigma_log_flux + config.EPS)) if use_weights else np.ones_like(self.omnidirectional_count)

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
        energy_window_width_relative: float = 0.5,
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
        # The asinh transformation helps handle the conversion from relative to log space
        s = math.asinh(0.5 * energy_window_width_relative) / math.sqrt(2.0 * math.log(2.0))

        W = np.exp(-0.5 * ((ln_energy_centers[:, None] - ln_energy_centers[None, :]) / s) ** 2)
        W /= np.sum(W, axis=1).reshape(-1, 1)
        return W

    def _objective_function_fast(self, 
                                 kappa_theta: np.ndarray, 
                                 use_weights: bool,
                                 use_convolution: bool = True

        ) -> float:
        """
        Fast Objective function for optimization using fast omnidirectional flux calculation.

        Args:
            kappa_theta (np.ndarray): Array containing kappa and log10(theta).

        Returns:
            float: The chi-squared value representing the difference.
        """
        density_mag = self.density_estimate.magnitude
        kappa = kappa_theta[0]
        theta_mag = 10 ** kappa_theta[1]

        W = self.build_log_energy_response_matrix(
            self.energy_centers_mag,
            energy_window_width_relative=config.ENERGY_WINDOW_WIDTH_RELATIVE,
        ) if use_convolution else np.eye(len(self.energy_centers_mag))

        model_flux_magnitudes = W @ omnidirectional_flux_magnitude(
            density_mag, kappa, theta_mag, self.energy_centers_mag
        )
        weights = (1 / (self.sigma_log_flux + config.EPS)) if use_weights else np.ones_like(self.omnidirectional_count)

        return self._compute_chi2_numba(
            model_flux_magnitudes + config.EPS,
            self.omnidirectional_differential_particle_flux_mag + config.EPS,
            weights
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
                    "omnidirectional_differential_particle_flux must be a pint Quantity (OmnidirectionalFlux)"
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
        self, n_starts: int = 50, use_fast: bool = True, use_weights: bool = True, use_convolution: bool = True
    ) -> FitResults | None:
        """
        Fit the kappa distribution parameters (kappa and theta) to the data.

        Args:
            n_starts (int): Number of random starts for the optimization.
            use_fast (bool): Whether to use the fast objective function.

        Returns:
            FitResults | None: An object containing the fit results, or None if the fit fails.
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

            if best_result and best_result.fun < 1e-3:
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
            density=0 * ureg.particle / ureg.meter**3,  # Density is not a fitted parameter
            kappa=sigma_kappa,
            theta=sigma_log_theta * ureg.meter / ureg.second, # This is uncertainty in log(theta)
        )

        is_good_fit = best_result.fun < config.FIT_ERROR_THRESHOLD

        return FitResults(
            params=fitted_params,
            params_uncertainty=params_uncertainty,
            error=best_result.fun,
            is_good_fit=is_good_fit,
        )
