import numpy as np
import pandas as pd
import spiceypy as spice
import os
from functools import partial
from scipy.optimize import minimize, Bounds
from scipy.stats.qmc import LatinHypercube, scale

from model import synth_losscone
import config

class FluxData:
    def __init__(self, er_data_file: str, thetas: str):
        """
        Initialize the FluxData class with the path to the ER data file.

        Note:
        The thetas parameter is expected to be a numpy array of shape (88,) with theta values in radians.
        """

        self.misc_cols = ["UTC", "time", "energy", "spec_no", "mag_x", "mag_y", "mag_z"]
        self.flux_cols = [f"ele_flux_{i}" for i in range(88)]
        self.phi_cols = [f"dist_phi_{i}" for i in range(88)]
        self.all_cols = self.misc_cols + self.flux_cols + self.phi_cols


        self.er_data_file = er_data_file
        self.data = None
        self.thetas = np.loadtxt(thetas, dtype=np.float64)

        self.cartesian_coords = None
        self.pitch_angles = None
        self.load_data()
        self.process_data()
        

    def load_data(self):
        """
        Load the ER data from the specified file.
        """
        # Read the data file
        try:
            self.data = pd.read_csv(self.er_data_file, sep=r"\s+", engine="python", header=None, names=self.all_cols)
        except FileNotFoundError:
            print(f"Error: The file {self.er_data_file} was not found.")
            self.data = None
            return
        except pd.errors.ParserError:
            print(f"Error: The file {self.er_data_file} could not be parsed. Please check the file format.")
            self.data = None
            return
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            self.data = None
            return

        # Analyze magnetic field
        magnetic_field = self.data[["mag_x", "mag_y", "mag_z"]].to_numpy(dtype=np.float64)
        magnetic_field_magnitude = np.linalg.norm(magnetic_field, axis=1, keepdims=True)

        # Find non-zero magnetic field indices
        mask = (magnetic_field_magnitude > 1e-9) & (magnetic_field_magnitude < 1e3)
        good_idx = np.where(mask.flatten())[0]
        unit_magnetic_field = magnetic_field[good_idx] / magnetic_field_magnitude[good_idx]
        tiled_unit_magnetic_field = np.tile(unit_magnetic_field[:, None, :], (1, 88, 1))
        self.tiled_unit_magnetic_field = tiled_unit_magnetic_field


        # Reset the data to only include non-zero magnetic field indices
        self.data = self.data.iloc[good_idx].reset_index(drop=True)

    def process_data(self):
        """
        Process the loaded data to calculate
        """

        # Check if data is loaded
        assert self.data is not None, "Data not loaded. Please load the data first."

        # Convert spherical coordinates to cartesian coordinates
        phis = np.deg2rad(self.data[self.phi_cols].to_numpy(dtype=np.float64))
        thetas = self.thetas

        X = np.cos(phis) * np.cos(thetas)
        Y = np.sin(phis) * np.cos(thetas)
        z_base = np.sin(thetas)
        # Z = np.tile(z_base, (X.shape[0], 1))
        Z = np.broadcast_to(z_base, X.shape)


        self.cartesian_coords = np.stack((X, Y, Z), axis=-1)

        # Calculate the pitch angles
        dot_product = -np.einsum('ijk,ijk->ij', self.tiled_unit_magnetic_field, self.cartesian_coords)
        dot_product = np.clip(dot_product, -1, 1)

        pitch_angles = np.arccos(dot_product)
        pitch_angles = np.rad2deg(pitch_angles)

        self.pitch_angles = pitch_angles
        self.electron_flux = self.data[self.flux_cols].to_numpy(dtype=np.float64)
        self.energy_bins = self.data["energy"].to_numpy(dtype=np.float64)
        self.chunks = len(self.data) // config.SWEEP_ROWS

    def get_normalized_flux(self, energy_bin: int, measurement_chunk: int) -> np.ndarray:
        """
        Get the normalized flux for a specific energy bin and measurement chunk.

        Args:
            energy_bin (int): The index of the energy bin.
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            np.ndarray: The normalized flux for the specified energy bin and measurement chunk.
        """
        # Check if data is loaded
        assert self.data is not None, "Data not loaded. Please load the data first."

        # Get the electron flux for the specified energy bin and measurement chunk
        index = measurement_chunk * config.SWEEP_ROWS + energy_bin
        electron_flux = self.electron_flux[index]
        angles = self.pitch_angles[index]
        incident_mask = angles < 90
        reflected_mask = ~incident_mask

        # Get the angles and fluxes for the incident and reflected regions
        incident_flux = np.mean(electron_flux[incident_mask]) + 1e-6
        normalized_flux = electron_flux[reflected_mask] / incident_flux

        # Combine the incident and reflected fluxes
        combined_flux = np.zeros_like(electron_flux)
        combined_flux[incident_mask] = electron_flux[incident_mask]
        combined_flux[reflected_mask] = normalized_flux
        return combined_flux

    def build_norm2d(self, measurement_chunk: int):
        """
        Build a 2D normalized flux distribution for a specific measurement chunk.

        Args:
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            np.ndarray: The 2D normalized flux distribution for the specified measurement chunk.
        """
        # Check if data is loaded
        assert self.data is not None, "Data not loaded. Please load the data first."

        norm2d = np.vstack([
            self.get_normalized_flux(energy_bin, measurement_chunk)
            for energy_bin in range(config.SWEEP_ROWS)
        ])

        return norm2d
    
    def _fit_surface_potential(self, measurement_chunk: int):
        """
        Fit surface potential (ΔU) and B_s/B_m for one 15-row measurement chunk
        using χ² minimisation with scipy.optimize.minimize.

        Returns
        -------
        delta_U   : best-fit surface potential in volts
        bs_over_bm: best-fit B_s/B_m ratio
        chi2      : final χ² value
        """
        assert self.data is not None, "Data not loaded."

        # --- prepare data for the chunk -------------------------------------------------
        eps        = 1e-6
        norm2d     = self.build_norm2d(measurement_chunk)
        s, e       = measurement_chunk * config.SWEEP_ROWS, (measurement_chunk + 1) * config.SWEEP_ROWS
        energies   = self.energy_bins[s:e]
        pitches    = self.pitch_angles[s:e]

        # objective ---------------------------------------------------------------------
        def chi2(params):
            delta_U, bs_over_bm = params
            model = synth_losscone(energies, pitches, delta_U, bs_over_bm)

            if not np.all(np.isfinite(model)) or (model <= 0).all():
                return 1e30  # big penalty

            diff = np.log(norm2d + eps) - np.log(model + eps)
            return np.sum(diff * diff)

        # -------------------------------------------------------------------------------
        # 1) Latin-hypercube global scan (20×20 ≈ 400 evaluations)
        # -------------------------------------------------------------------------------
        bounds = np.array([[-1000.0,  0.1],   # lower
                        [ 1000.0, 10.0]])  # upper

        sampler = LatinHypercube(d=2, seed=42)
        lhs     = sampler.random(n=400)            # 400 points
        sample  = scale(lhs, bounds[0], bounds[1]) # map to bounds

        chi2_vals = np.apply_along_axis(chi2, 1, sample)
        best_idx  = np.argmin(chi2_vals)
        x0        = sample[best_idx]               # ΔU, Bₛ/Bₘ for local start

        # -------------------------------------------------------------------------------
        # 2) Local Nelder–Mead refinement
        # -------------------------------------------------------------------------------
        result = minimize(
            chi2, x0,
            method="Nelder-Mead",
            options=dict(maxiter=1000, xatol=1e-4, fatol=1e-4)
        )

        if not result.success:
            raise RuntimeError(f"Optimisation failed: {result.message}")

        delta_U, bs_over_bm = result.x
        return float(delta_U), float(bs_over_bm), float(result.fun)
    
    def fit_surface_potential(self):
        """
        Fit surface potential (ΔU) and B_s/B_m for all 15-row measurement chunks
        using χ² minimisation with scipy.optimize.minimize.

        Returns
        -------
        delta_U   : best-fit surface potential in volts
        bs_over_bm: best-fit B_s/B_m ratio
        chi2      : final χ² value
        """
        assert self.data is not None, "Data not loaded."

        # Fit for each chunk
        results = np.zeros((self.chunks, 4))
        for i in range(self.chunks):
            delta_U, bs_over_bm, chi2 = self._fit_surface_potential(i)
            results[i] = [delta_U, bs_over_bm, chi2, i]

        return results

    