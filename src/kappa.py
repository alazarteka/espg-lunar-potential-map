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