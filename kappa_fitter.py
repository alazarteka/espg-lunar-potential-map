import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass

import config
from flux import ERData, PitchAngle
from potential_mapper import DataLoader

from scipy.integrate import simpson
from scipy.special import gamma
from scipy.optimize import minimize

@dataclass
class KappaParams:
    """Represents the parameters of a kappa distribution."""
    density: float  # number density n in m⁻³
    kappa: float    # kappa > 1.5
    theta: float    # effective thermal speed in m s⁻¹

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
    return KappaParams(
        density=np.exp(vec[0]),
        kappa=vec[1],
        theta=np.exp(vec[2])
    )

class KappaFitter:
    """
    Fits a kappa distribution to measured electron flux data.

    This class encapsulates the entire workflow:
    1. Preparing the measured data for a specific instrument spectrum.
    2. Defining the physical model (kappa distribution).
    3. Running the optimization to find the best-fit parameters.
    """
    # Default bounds for the optimization parameters    
    DEFAULT_BOUNDS = [(np.log(1e4), np.log(1e9)),  # log density n in m⁻³
                      (1.5001, 10.0),          # kappa
                      (np.log(50.0), np.log(5.0e5))]  # log theta in m s⁻¹

    # Default initial guess for the optimization
    DEFAULT_X0 = vec_from_params(KappaParams(
        density=1e5,  # initial guess for density n in m⁻³
        kappa=3.0,   # initial guess for kappa
        theta=100.0  # initial guess for theta in m s⁻¹
    ))

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
        if self.spec_no not in self.er_data.data['spec_no'].values:
            logging.warning(f"Spec no {self.spec_no} not found in ER data.")
            return

        spec_dataframe = self.er_data.data[self.er_data.data['spec_no'] == self.spec_no].copy()
        spec_dataframe.reset_index(drop=True, inplace=True)

        spec_er_data = self.er_data.__class__.from_dataframe(
            spec_dataframe,
            self.er_data.er_data_file
        )

        spec_pitch_angle = PitchAngle(spec_er_data, DataLoader.get_theta_file(config.DATA_DIR))

        pitch_angles = spec_pitch_angle.pitch_angles
        pitch_angles_mask = pitch_angles < 90  # shape (nE, nPitch)
        if not np.any(pitch_angles_mask):
            logging.warning(f"No valid pitch angles found for spec no {self.spec_no}.")
            return

        electron_flux = spec_er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)  # shape (nE, nPitch)

        masked_electron_flux = electron_flux * pitch_angles_mask
        self.sum_flux = np.sum(masked_electron_flux, axis=1)  # shape (nE,)

        energies = spec_er_data.data['energy'].to_numpy(dtype=np.float64).reshape(-1)  # shape (nE,)

        # Based on delta E / E ~ 0.5, the energy window for each channel is
        # approximately [E_center * (1 - 0.25), E_center * (1 + 0.25)]
        self.energy_windows = np.column_stack([
            0.75 * energies,
            1.25 * energies,
        ])  # shape (nE, 2)

        self.is_data_valid = True

    def _gradient(self, params: tuple[float, float, float]) -> np.ndarray:
        """
        Computes the gradient of the objective function for optimization.

        TODO: Decide if this is needed or we can remove it. If needed, implement  the dJ_dtheta and dJ_dkappa methods.
        """
        kappa_params = KappaParams(*params)
        model_flux = self.omnidirectional_flux_integral(kappa_params, self.energy_windows)
        log_measured = np.log(self.sum_flux + config.EPS)
        log_model = np.log(model_flux + config.EPS)
        diff = log_measured - log_model

        dJ_dn = model_flux / params[0]
        dJ_dkappa = self._dJ_dkappa(params)
        dJ_dtheta = self._dJ_dtheta(params)

        grad_n = 2*np.sum(diff / model_flux * dJ_dn)
        grad_kappa = 2*np.sum(diff / model_flux * dJ_dkappa)
        grad_theta = 2*np.sum(diff / model_flux * dJ_dtheta)

        return np.array([grad_n, grad_kappa, grad_theta])

    @staticmethod
    def pdf_kappa(params: KappaParams, velocity: np.ndarray) -> np.ndarray:
        """
        Isotropic κ distribution fκ(v).
        """
        assert params.kappa > 1.5, "κ must exceed 3/2 for finite temperature"

        prefac = gamma(params.kappa + 1) / (np.power(np.pi * params.kappa, 1.5) * gamma(params.kappa - 0.5))
        core = params.density / params.theta**3
        tail_ln = -(params.kappa + 1) * np.log1p((velocity / params.theta)**2 / params.kappa)  # log-safe

        return prefac * core * np.exp(tail_ln)

    @staticmethod
    def omnidirectional_flux(params: KappaParams, energy: np.ndarray) -> np.ndarray:
        """
        Omnidirectional electron flux from κ distribution.
        """
        velocity = np.sqrt(2 * config.ELECTRON_CHARGE_C * energy / config.ELECTRON_MASS_KG)  # m/s
        pdf_values = KappaFitter.pdf_kappa(params, velocity)
        return 4 * np.pi * config.ELECTRON_CHARGE_C / config.ELECTRON_MASS_KG * velocity**2 * pdf_values

    @staticmethod
    def omnidirectional_flux_integral(params: KappaParams, energy_bounds: np.ndarray, n_samples: int = 101) -> np.ndarray:
        """
        Integrate omnidirectional flux over energy windows.
        """
        assert n_samples % 2 == 1, "n_samples must be odd for Simpson's rule"

        energy_grid = np.linspace(energy_bounds[:, 0], energy_bounds[:, 1], num=n_samples, axis=1)
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

    def fit(self) -> KappaFitResult | None:
        """
        Runs the fitting process.
        """
        if not self.is_data_valid:
            logging.error("Cannot run fit, data is not valid.")
            return None

        res = minimize(
            self._objective_function,
            x0=self.DEFAULT_X0.to_tuple(),
            args=(),
            bounds=self.DEFAULT_BOUNDS,
            method='L-BFGS-B',
            options={'maxiter': 300, 'ftol': 1e-10, 'disp': True}
        )
        sigma = None

        if res.success and hasattr(res, 'hess_inv'):
            try:
                cov = res.hess_inv.todense()
                errs = np.sqrt(np.diag(cov))
                sigma = KappaParams(*errs)
            except Exception as e:
                logging.warning(f"Failed to compute parameter uncertainties: {e}")
        
        return KappaFitResult(
            params  = KappaParams(*res.x),
            chi2    = res.fun,
            success = res.success,
            message = res.message,
            sigma   = sigma
        )

def run_test_case():
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

    # Generate true flux and add 5% log-normal noise
    Ftrue = KappaFitter.omnidirectional_flux_integral(true_params, windows)
    Fnoisy = Ftrue * rng.lognormal(0, 0.05, size=Ftrue.size)

    # Create a dummy objective function for the test case
    def test_objective(param_vec: np.ndarray, energy_windows, measured_flux) -> float:
        params = params_from_vec(param_vec)
        model_flux = KappaFitter.omnidirectional_flux_integral(params, energy_windows)
        log_measured = np.log(measured_flux + config.EPS)
        log_model = np.log(model_flux + config.EPS)
        return np.sum((log_measured - log_model)**2)

    # Run the minimization
    fit_result = minimize(
        test_objective,
        x0=KappaFitter.DEFAULT_X0.to_tuple(),
        args=(windows, Fnoisy),
        bounds=KappaFitter.DEFAULT_BOUNDS,
        method='L-BFGS-B'
    )

    print(f"True Parameters: {true_params}")
    print(f"Fitted Parameters: {KappaParams(*fit_result.x)}")
    print(f"Success: {fit_result.success}")
    print("--- Test Case Finished ---")


if __name__ == '__main__':
    # This block allows the script to be used as a library or run directly.
    # When run directly, it executes the test case.
    run_test_case()
