import numpy as np
import pandas as pd
import spiceypy as spice
import os

from model import synth_losscone

# SPICE kernels
# Local path: data/spice_kernels
spice_kernels = [
    '../lpephemu.bsp', # Lunar and Planetary Ephemeris
    "../naif0012.tls", # NAIF Leap Second Kernel
    "lp_ask_990401-990730.bsp", # Lunar and Planetary Ephemeris
    "../pck00010.tpc" # Planetary Constants Kernel
]

# ER Data
# Local path: data/er_data
er_data_file = "3D990429.TAB" # 3D data file

# Spacecraft Thetas
# Local path: data/ 
theta_file = "../theta.tab" # theta file

class FluxData:
    def __init__(self, er_data_file: str, thetas: np.ndarray):
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
        self.thetas = thetas

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
            self.data = pd.read_csv(self.er_data_file, delim_whitespace=True, header=None, names=self.all_cols)
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
        dot_product = -np.einsum('ijk,ik->ij', self.tiled_unit_magnetic_field, self.cartesian_coords)
        dot_product = np.clip(dot_product, -1, 1)

        pitch_angles = np.arccos(dot_product)
        pitch_angles = np.rad2deg(pitch_angles)

        self.pitch_angles = pitch_angles
        self.electron_flux = self.data[self.flux_cols].to_numpy(dtype=np.float64)
        self.energy_bins = self.data["energy"].to_numpy(dtype=np.float64)

    def get_normalized_flux(self, energy_bin: int, measurement_chunk: int):
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
        index = measurement_chunk * 15 + energy_bin
        electron_flux = self.electron_flux[index]
        angles = self.pitch_angles[index]
        incident_mask = np.where(angles < 90.0)[0]
        reflected_mask = ~incident_mask

        # Get the angles and fluxes for the incident and reflected regions
        incident_flux = np.mean(electron_flux[incident_mask])
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
            for energy_bin in range(15)
        ])

        return norm2d
    
    def fit_surface_potential(self, measurement_chunk: int):
        """
        Fit the surface potential for a specific measurement chunk.

        Args:
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            float: The fitted surface potential for the specified measurement chunk.
        """
        # Check if data is loaded
        assert self.data is not None, "Data not loaded. Please load the data first."

        # Get the 2D normalized flux distribution
        eps = 1e-6
        norm2d = self.build_norm2d(measurement_chunk)

        
        start = measurement_chunk * 15
        end = start + 15
        energies = self.energy_bins[start:end]
        pitch_angles = self.pitch_angles[start:end]

        best = {"chi2": np.inf}
        for delta_U in np.linspace(-1000, 1000, 100):
            for bs_over_bm in np.linspace(0.1, 10, 100):
                model = synth_losscone(
                    energies,
                    pitch_angles,
                    delta_U=delta_U,
                    bs_over_bm=bs_over_bm,
                    beam_width_eV=0.0,
                    beam_amp=0.0
                )
                chi2 = ((np.log(norm2d + eps) - np.log(model + eps)) ** 2).sum()
                if chi2 < best["chi2"]:
                    best.update({
                        "chi2": chi2,
                        "delta_U": delta_U,
                        "bs_over_bm": bs_over_bm
                    })

        return best["delta_U"], best["bs_over_bm"], best["chi2"]
    
