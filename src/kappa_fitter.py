import logging
from dataclasses import dataclass

import numpy as np
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import qmc

from . import config
from .flux import ERData, PitchAngle
from .potential_mapper import DataLoader


@dataclass
class KappaParams:
    """Represents the parameters of a kappa distribution."""

    density: float  # number density n in m⁻³
    kappa: float  # kappa > 1.5
    theta: float  # effective thermal speed in m s⁻¹

    def to_tuple(self) -> tuple[float, float, float]:
        return (self.density, self.kappa, self.theta)


@dataclass
class KappaFitResult:
    """Result of the kappa fitting process."""

    params: KappaParams
    chi2: float
    success: bool
    message: str
    sigma: KappaParams | None = None


def vec_from_params(params: KappaParams) -> np.ndarray:
    """
    Converts KappaParams to a numpy array for optimization.
    """
    return np.array([np.log(params.density), params.kappa, np.log(params.theta)])


def params_from_vec(vec: np.ndarray) -> KappaParams:
    """
    Converts a numpy array back to KappaParams.
    """
    return KappaParams(density=np.exp(vec[0]), kappa=vec[1], theta=np.exp(vec[2]))


class KappaFitter:
    """
    Fits a kappa distribution to measured electron flux data.

    This class encapsulates the entire workflow:
    1. Preparing the measured data for a specific instrument spectrum.
    2. Defining the physical model (kappa distribution).
    3. Running the optimization to find the best-fit parameters.
    """

    # Default bounds for the optimization parameters
    DEFAULT_BOUNDS = [
        (9.2, 18.4),  # log density n in m⁻³
        (2.0, 6.0),  # kappa
        (5.7, 8.8),
    ]  # log theta in m s⁻¹

    def __init__(self, er_data: ERData, spec_no: int):
        """
        Initializes the fitter with data for a specific spectrum number.
        """
        self.er_data = er_data
        self.spec_no = spec_no
        self.is_data_valid = False

        self.sum_flux: np.ndarray | None = None
        self.energy_windows: np.ndarray | None = None

        self._prepare_data()

    def _prepare_data(self):
        """
        Filters and processes the ERData for the specified spectrum number.
        """
        if self.spec_no not in self.er_data.data["spec_no"].values:
            logging.warning(f"Spec no {self.spec_no} not found in ER data.")
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
        pitch_angles_mask = pitch_angles < 90  # shape (nE, nPitch)
        if not np.any(pitch_angles_mask):
            logging.warning(f"No valid pitch angles found for spec no {self.spec_no}.")
            return

        electron_flux = spec_er_data.data[config.FLUX_COLS].to_numpy(
            dtype=np.float64
        )  # shape (nE, nPitch)

        masked_electron_flux = electron_flux * pitch_angles_mask
        self.sum_flux = np.sum(masked_electron_flux, axis=1)  # shape (nE,)

        energies = (
            spec_er_data.data["energy"].to_numpy(dtype=np.float64).reshape(-1)
        )  # shape (nE,)

        # Based on delta E / E ~ 0.5, the energy window for each channel is
        # approximately [E_center * (1 - 0.25), E_center * (1 + 0.25)]
        self.energy_windows = np.column_stack(
            [
                0.75 * energies,
                1.25 * energies,
            ]
        )  # shape (nE, 2)

        self.is_data_valid = True

    @staticmethod
    def pdf_kappa(params: KappaParams, velocity: np.ndarray) -> np.ndarray:
        """
        Isotropic κ distribution fκ(v).
        """
        assert params.kappa > 1.5, "κ must exceed 3/2 for finite temperature"

        prefac = gamma(params.kappa + 1) / (
            np.power(np.pi * params.kappa, 1.5) * gamma(params.kappa - 0.5)
        )
        core = params.density / params.theta**3
        tail_ln = -(params.kappa + 1) * np.log1p(
            (velocity / params.theta) ** 2 / params.kappa
        )  # log-safe

        return prefac * core * np.exp(tail_ln)

    @staticmethod
    def omnidirectional_flux(params: KappaParams, energy: np.ndarray) -> np.ndarray:
        """
        Omnidirectional electron flux from κ distribution.
        """
        velocity = np.sqrt(
            2 * config.ELECTRON_CHARGE_C * energy / config.ELECTRON_MASS_KG
        )  # m/s
        pdf_values = KappaFitter.pdf_kappa(params, velocity)
        return (
            4
            * np.pi
            * config.ELECTRON_CHARGE_C
            / config.ELECTRON_MASS_KG
            * pdf_values
            * velocity**2
        )

    @staticmethod
    def omnidirectional_flux_integral(
        params: KappaParams, energy_bounds: np.ndarray, n_samples: int = 101
    ) -> np.ndarray:
        """
        Integrate omnidirectional flux over energy windows.
        """
        assert n_samples % 2 == 1, "n_samples must be odd for Simpson's rule"

        energy_grid = np.geomspace(
            energy_bounds[:, 0], energy_bounds[:, 1], num=n_samples, axis=1
        )
        flux_values = KappaFitter.omnidirectional_flux(params, energy_grid)
        return simpson(flux_values, x=energy_grid, axis=1)

    def _chi_squared_difference(self, model_flux: np.ndarray) -> float:
        """
        Calculate the chi-squared difference between log of measured and model fluxes.
        """
        log_measured = np.log(self.sum_flux + config.EPS)
        log_model = np.log(model_flux + config.EPS)
        diff = log_measured - log_model
        return np.sum(diff**2)

    def _objective_function(self, vec: np.ndarray) -> float:
        """
        Objective function for the scipy optimizer.
        """
        params = params_from_vec(vec)
        model_flux = self.omnidirectional_flux_integral(params, self.energy_windows)
        return self._chi_squared_difference(model_flux)

    def fit(self, n_starts: int = 50) -> KappaFitResult | None:
        """
        Runs the fitting process using a multi-start optimization strategy.
        """
        if not self.is_data_valid:
            logging.error("Cannot run fit, data is not valid.")
            return None

        # --- Multi-start optimization using Latin Hypercube Sampling ---
        # 1. Set up the sampler
        sampler = qmc.LatinHypercube(d=len(self.DEFAULT_BOUNDS))
        samples = sampler.random(n=n_starts)
        scaled_samples = qmc.scale(
            samples,
            [b[0] for b in self.DEFAULT_BOUNDS],
            [b[1] for b in self.DEFAULT_BOUNDS],
        )

        best_res = None

        # 2. Run minimization for each starting point
        for i, x0 in enumerate(scaled_samples):
            logging.debug(f"Running optimization start {i+1}/{n_starts}...")
            res = minimize(
                self._objective_function,
                x0=x0,
                args=(),
                bounds=self.DEFAULT_BOUNDS,
                method="L-BFGS-B",
                options={"maxiter": 300, "ftol": 1e-10, "disp": False},
            )
            if best_res is None or (res.success and res.fun < best_res.fun):
                best_res = res

        if best_res is None:
            logging.error("All optimization runs failed.")
            return None

        # 3. Calculate final parameters and uncertainties from the best result
        sigma = None
        if best_res.success and hasattr(best_res, "hess_inv"):
            try:
                cov = best_res.hess_inv.todense()
                errs = np.sqrt(np.diag(cov))
                sigma = params_from_vec(errs)
            except Exception as e:
                logging.warning(f"Failed to compute parameter uncertainties: {e}")

        return KappaFitResult(
            params=params_from_vec(best_res.x),
            chi2=best_res.fun,
            success=best_res.success,
            message=best_res.message,
            sigma=sigma,
        )


def run_test_case(n_starts: int = 50):
    """
    Runs a self-contained test case to verify the fitting procedure.
    It generates synthetic data from a known kappa distribution and then
    runs the fitter to check if it can recover the original parameters.
    """
    print("--- Running Self-Contained Test Case ---")
    rng = np.random.default_rng(0)
    true_params = KappaParams(density=3e5, kappa=3.0, theta=800.0)

    Ecent = np.logspace(1, 4, 32)
    windows = np.column_stack([0.75 * Ecent, 1.25 * Ecent])
    window_widths = Ecent * 0.5  # 50% width for each energy window
    print(f"window_widths.shape: {window_widths.shape}, windows.shape: {windows.shape}")

    # Generate true flux and add 5% log-normal noise
    Ftrue = KappaFitter.omnidirectional_flux_integral(true_params, windows)
    Fnoisy = Ftrue * rng.lognormal(0, 0.05, size=Ftrue.size)

    print(f"FNoisy.shape: {Fnoisy.shape}, FTrue.shape: {Ftrue.shape}")

    # Density estimate
    density = np.sum(
        Fnoisy
        * window_widths
        * np.sqrt(config.ELECTRON_MASS_KG / (2 * config.ELECTRON_CHARGE_C * Ecent))
    )

    print(f"Estimated Density: {density:.2e} m⁻³")

    exit(0)  # Exit early to avoid running the test case in production

    # Create a dummy objective function for the test case
    def test_objective(param_vec: np.ndarray, energy_windows, measured_flux) -> float:
        params = params_from_vec(param_vec)
        model_flux = KappaFitter.omnidirectional_flux_integral(params, energy_windows)
        log_measured = np.log(measured_flux + config.EPS)
        log_model = np.log(model_flux + config.EPS)
        return np.sum((log_measured - log_model) ** 2)

    # --- Multi-start optimization using Latin Hypercube Sampling ---
    # 1. Set up the sampler
    sampler = qmc.LatinHypercube(d=len(KappaFitter.DEFAULT_BOUNDS))
    samples = sampler.random(n=n_starts)
    scaled_samples = qmc.scale(
        samples,
        [b[0] for b in KappaFitter.DEFAULT_BOUNDS],
        [b[1] for b in KappaFitter.DEFAULT_BOUNDS],
    )

    best_fit_result = None

    # 2. Run minimization for each starting point
    for i, x0 in enumerate(scaled_samples):
        print(f"Running test optimization start {i+1}/{n_starts}...")
        fit_result = minimize(
            test_objective,
            x0=x0,
            args=(windows, Fnoisy),
            bounds=KappaFitter.DEFAULT_BOUNDS,
            method="L-BFGS-B",
        )
        if best_fit_result is None or (
            fit_result.success and fit_result.fun < best_fit_result.fun
        ):
            best_fit_result = fit_result

    if best_fit_result is None:
        print("All test optimization runs failed.")
        return

    print(f"True Parameters: {true_params}")
    print(f"""Fitted Parameters: {params_from_vec(best_fit_result.x)}""")
    print(f"Success: {best_fit_result.success}")

    # --- Format and print the covariance matrix ---
    cov_matrix = "N/A"
    if hasattr(best_fit_result, "hess_inv"):
        # The L-BFGS-B optimizer returns an approximation of the inverse Hessian.
        # To get the dense matrix, we can multiply it by the identity matrix.
        hess_inv = best_fit_result.hess_inv
        cov_matrix = hess_inv.dot(np.identity(len(best_fit_result.x)))

    print(f"Covariance Matrix (approx):\n{cov_matrix}")
    print("--- Test Case Finished ---")


if __name__ == "__main__":
    # This block allows the script to be used as a library or run directly.
    # When run directly, it executes the test case.
    run_test_case(200)
