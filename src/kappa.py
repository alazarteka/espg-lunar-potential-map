import logging
from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc

from src import config
from src.flux import ERData, PitchAngle
from src.physics.kappa import (
    KappaParams,
    omnidirectional_flux,
    velocity_from_energy,
)
from src.potential_mapper import DataLoader
from src.utils.units import (
    Energy,
    Flux,
    NumberDensity,
    ureg,
)


@dataclass
class KappaFitResult:
    """Represents the result of a kappa distribution fit."""

    params: KappaParams
    chi2: float
    success: bool
    message: str
    sigma: KappaParams | None = None


class Kappa:
    """Class for handling kappa distribution fitting and evaluation."""

    DEFAULT_BOUNDS = [(2.5, 6.0), (100, 10000000)]  # kappa  # theta in m/s

    def __init__(self, er_data: ERData, spec_no: int):

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

        self.omnidirectional_differential_particle_flux: Flux | None = None
        self.energy_centers: Energy | None = None
        self.energy_windows: Energy | None = None

        self._prepare_data()
        self.density_estimate: NumberDensity = self._get_density_estimate()

    def _prepare_data(self):
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

        # Strictly speaking, this should be scaled by the steradian of each pitch angle bin
        # TODO: Implement proper scaling by pitch angle bin size
        electron_flux = spec_er_data.data[config.FLUX_COLS].to_numpy(
            dtype=np.float64
        ) * (
            ureg.particle
            / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)
        )  # shape (Energy Bins, Pitch Angles)

        masked_electron_flux = np.where(
            pitch_angles_mask, electron_flux, 0 * electron_flux.units
        )
        self.omnidirectional_differential_particle_flux = np.sum(
            masked_electron_flux * self.solid_angles, axis=1
        )  # shape (Energy Bins,)

        energies = (
            spec_er_data.data["energy"].to_numpy(dtype=np.float64)
        ) * ureg.electron_volt  # shape (Energy Bins,)

        self.energy_centers = energies
        self.energy_windows = np.column_stack(
            [0.75 * energies, 1.25 * energies]
        )  # shape (Energy Bins, 2)

        self.is_data_valid = True

    def _objective_function(self, kappa_theta: np.ndarray) -> float:
        params = KappaParams(
            density=self.density_estimate,
            kappa=kappa_theta[0],
            theta=kappa_theta[1] * ureg.meter / ureg.second,
        )

        # TODO: Decide whether to use average flux or flux at energy centers
        # model_differential_flux = omnidirectional_flux_integrated(
        #     params, self.energy_windows
        # ) / self.energy_centers
        model_differential_flux = omnidirectional_flux(params, self.energy_centers)

        log_model_differential_flux = np.log(
            model_differential_flux.to(
                ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
            ).magnitude
        )
        log_data_flux = np.log(
            self.omnidirectional_differential_particle_flux.to(
                ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.electron_volt)
            ).magnitude
        )
        chi2 = np.sum((log_model_differential_flux - log_data_flux) ** 2)
        return chi2

    def _get_density_estimate(self) -> NumberDensity:
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

        velocities = velocity_from_energy(self.energy_centers)  # shape (Energy Bins,)
        delta_energy = (
            self.energy_windows[:, 1] - self.energy_windows[:, 0]
        )  # shape (Energy Bins,)

        integrand = (
            self.omnidirectional_differential_particle_flux / velocities
        )  # shape (Energy Bins,)

        density_estimate = np.sum(integrand * delta_energy)

        return density_estimate

    def fit(self, n_starts: int = 50):
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

        best_result = None

        for i, x0 in enumerate(scaled_samples):
            logging.debug(f"Running optimization for sample {i + 1}/{n_starts}")
            result = minimize(
                self._objective_function,
                x0,
                args=(),
                bounds=self.DEFAULT_BOUNDS,
                method="Nelder-Mead",
                options={"maxiter": 1000, "disp": False},
            )
            if best_result is None or (result.success and result.fun < best_result.fun):
                best_result = result

        if best_result is None:
            logging.warning("No valid optimization result found.")
            return None

        sigma = None
        if best_result.success and hasattr(best_result, "hess_inv"):
            try:
                sigma = np.sqrt(np.diag(best_result.hess_inv.todense()))
                sigma = KappaParams(
                    density=0
                    * ureg.particle
                    / ureg.meter
                    ** 3,  # Placeholder, density is not estimated from Hessian
                    kappa=sigma[0],
                    theta=sigma[1] * ureg.meter / ureg.second,
                )
            except Exception as e:
                logging.warning(f"Failed to compute sigma: {e}")

        return KappaParams(
            density=self.density_estimate,
            kappa=best_result.x[0],
            theta=best_result.x[1] * ureg.meter / ureg.second,
        )
