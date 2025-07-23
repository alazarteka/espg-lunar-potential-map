import logging
from dataclasses import dataclass, field
from typing import Annotated

import numpy as np
import pint
from pint import Quantity
from scipy.integrate import simpson
from scipy.optimize import minimize
from scipy.special import gamma
from scipy.stats import qmc

from src import config
from src.flux import ERData, PitchAngle
from src.potential_mapper import DataLoader
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
    
    def __init__(
            self,
            er_data: ERData,
            spec_no: int):
        
        self.er_data = er_data
        self.spec_no = spec_no
        self.is_data_valid = False

        self.sum_flux: Flux | None = None
        self.energy_windows: Energy | None = None

        self._prepare_data()
    
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
        pitch_angles_mask = pitch_angles < 90 # shape (Energy Bins, Pitch Angles)
        if not np.any(pitch_angles_mask):
            logging.warning("No valid pitch angles found for the specified spec_no.")
            return
        
        # Strictly speaking, this should be scaled by the steradian of each pitch angle bin
        # TODO: Implement proper scaling by pitch angle bin size
        electron_flux = spec_er_data.data[config.FLUX_COLS].to_numpy(
            dtype=np.float64
        ) * (ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)) # shape (Energy Bins, Pitch Angles)

        masked_electron_flux = np.where(
            pitch_angles_mask, electron_flux, 0 * electron_flux.units
        )
        self.sum_flux = np.sum(masked_electron_flux, axis=1)

        energies = (
            spec_er_data.data["energy"].to_numpy(dtype=np.float64) 
        ) * ureg.electron_volt  # shape (Energy Bins,)

        self.energy_windows = np.column_stack(
            [0.75 * energies, 1.25 * energies]
        )  # shape (Energy Bins, 2)

        self.is_data_valid = True

    @staticmethod
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

    @staticmethod
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

        distribution = Kappa.kappa_distribution(
            parameters, velocity
        )
        return (distribution * velocity**2 / config.ELECTRON_MASS).to(ureg.particle / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt))

    @staticmethod
    def omnidirectional_flux(
        parameters: KappaParams,
        energy: Energy
    ):
        return (4 * np.pi * ureg.steradian) * Kappa.directional_flux(parameters, energy)
    
    

    