"""Pitch-angle geometry derived from ER data and magnetic field vectors."""

import numpy as np

from src import config
from src.losscone.er_data import ERData
from src.utils import thetas as thetas_module

__all__ = ["PitchAngle"]


class PitchAngle:
    """
    Initialize the PitchAngle class with the ER data and theta values.

    Data rows with invalid B-field are retained; all such rows are flagged via
    valid_mask and their derived quantities are NaN. Down-stream algorithms
    must honor this mask.

    Attributes:
        er_data: The ER data object.
        thetas: The theta values in degrees.
        cartesian_coords: The Cartesian coordinates of the data points.
        pitch_angles: The pitch angles in degrees.
        unit_magnetic_field: The unit magnetic field vectors.
        polarity: Optional per-row polarity sign (+1/-1/0).
        valid_mask: A mask indicating valid data points.
    """

    def __init__(
        self,
        er_data: ERData,
        polarity: np.ndarray | None = None,
    ):
        """
        Initialize the PitchAngle class with the ER data and theta values.

        Args:
            er_data (ERData): The ER data object.
            polarity (np.ndarray | None): Optional per-row polarity sign (+1/-1/0).
                Use +1 for Moonward along +B, -1 for Moonward along -B, and 0 to
                mark rows as disconnected (pitch angles set to NaN).

        Notes:
            Theta values are loaded from config via get_thetas(); custom theta
            files require updating config or extending this API.
        """
        self.er_data = er_data
        self.thetas = thetas_module.get_thetas()
        self.polarity = polarity
        self.cartesian_coords = np.array([])
        self.pitch_angles = np.array([])
        self.unit_magnetic_field = np.array([])
        self.valid_mask = np.array([])

        self._process_data()

    def _get_cartesian_coords(self, phis: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates (phi, theta) to Cartesian coordinates (X, Y, Z).

        Args:
            phis (np.ndarray): The phi values in radians.
            thetas (np.ndarray): The theta values in radians.

        Returns:
            np.ndarray: The Cartesian coordinates (X, Y, Z).
        """
        X = np.cos(phis) * np.cos(thetas)
        Y = np.sin(phis) * np.cos(thetas)
        z_base = np.sin(thetas)
        Z = np.broadcast_to(z_base, X.shape)
        return np.stack((X, Y, Z), axis=-1)

    def _process_data(self) -> None:
        """
        Process the ER data to calculate the Cartesian coordinates and prepare
        the unit magnetic field vectors for pitch angle calculation.

        This function performs data validation and transformation from spherical
        to Cartesian coordinates. It also normalizes the magnetic field vectors
        and stores indices of valid and invalid data points.
        """

        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )
        assert len(self.thetas) == config.CHANNELS, (
            f"Theta values must match the number of channels {config.CHANNELS}."
        )

        # Convert spherical coordinates (phi, theta) to Cartesian coordinates (X, Y, Z)
        phis = np.deg2rad(self.er_data.data[config.PHI_COLS].to_numpy(dtype=np.float64))
        thetas = np.deg2rad(self.thetas)
        self.cartesian_coords = self._get_cartesian_coords(phis, thetas)

        magnetic_field = self.er_data.data[config.MAG_COLS].to_numpy(dtype=np.float64)
        magnetic_field_magnitude = np.linalg.norm(magnetic_field, axis=1, keepdims=True)
        # ER convention points +B roughly sunward; loss-cone tracing expects
        # Moonward orientation. If polarity is provided, use it as the sign.
        if self.polarity is None:
            unit_magnetic_field = -magnetic_field / magnetic_field_magnitude
        else:
            polarity = np.asarray(self.polarity)
            if polarity.shape[0] != magnetic_field.shape[0]:
                raise ValueError("polarity must match the number of ER rows")
            unit_magnetic_field = (
                magnetic_field / magnetic_field_magnitude
            ) * polarity[:, None]
            # Rows with polarity=0 are treated as disconnected; downstream masks
            # should ignore the resulting NaNs.
            invalid_polarity = ~np.isfinite(polarity) | (polarity == 0)
            if np.any(invalid_polarity):
                unit_magnetic_field[invalid_polarity] = np.nan
        unit_magnetic_field = np.tile(
            unit_magnetic_field[:, None, :], (1, config.CHANNELS, 1)
        )
        self.unit_magnetic_field = unit_magnetic_field

        self.calculate_pitch_angles()

    def calculate_pitch_angles(self) -> None:
        """
        Calculate the pitch angles based on the loaded ER data and theta values.

        The pitch angle is the angle between the magnetic field line and the
        radial direction. It is calculated as the arccosine of the dot product
        between the unit magnetic field vector and the radial direction vector.
        """
        # Check if data is loaded
        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )

        dot_product = np.einsum(
            "ijk,ijk->ij", self.unit_magnetic_field, self.cartesian_coords
        ).clip(-1, 1)
        pitch_angles = np.arccos(dot_product)
        pitch_angles = np.rad2deg(pitch_angles)

        self.pitch_angles = pitch_angles
