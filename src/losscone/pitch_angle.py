"""Pitch-angle geometry derived from ER data and magnetic field vectors."""

from dataclasses import dataclass

import numpy as np

from src import config
from src.losscone.er_data import ERData
from src.utils import thetas as thetas_module

__all__ = ["PitchAngle", "PitchAngleGeometry", "compute_pitch_angle_geometry"]


@dataclass(frozen=True)
class PitchAngleGeometry:
    """Pure-array pitch-angle calculation result."""

    pitch_angles: np.ndarray
    cartesian_coords: np.ndarray
    unit_magnetic_field: np.ndarray
    valid_rows: np.ndarray


def compute_pitch_angle_geometry(
    phis_deg: np.ndarray,
    thetas_deg: np.ndarray,
    magnetic_field: np.ndarray,
    polarity: np.ndarray | None = None,
) -> PitchAngleGeometry:
    """Calculate pitch angles without constructing :class:`ERData`.

    Args:
        phis_deg: Archived SCD azimuths, shape ``(n_rows, n_channels)``.
        thetas_deg: Fixed SCD elevations, shape ``(n_channels,)``.
        magnetic_field: SCD magnetic-field vectors, shape ``(n_rows, 3)``.
        polarity: Optional Moonward orientation sign per row. With no supplied
            polarity, preserve the legacy convention of orienting along ``-B``.

    Returns:
        Derived pitch geometry. Invalid magnetic-field or polarity rows are
        retained and filled with NaN pitch angles.
    """
    phis = np.asarray(phis_deg, dtype=np.float64)
    thetas = np.asarray(thetas_deg, dtype=np.float64)
    magnetic = np.asarray(magnetic_field, dtype=np.float64)

    if phis.ndim != 2:
        raise ValueError("phis_deg must have shape (n_rows, n_channels)")
    if thetas.ndim != 1 or thetas.shape[0] != phis.shape[1]:
        raise ValueError("thetas_deg must match the channel dimension of phis_deg")
    if magnetic.shape != (phis.shape[0], 3):
        raise ValueError("magnetic_field must have shape (n_rows, 3)")

    phis_rad = np.deg2rad(phis)
    thetas_rad = np.deg2rad(thetas)
    x = np.cos(phis_rad) * np.cos(thetas_rad)
    y = np.sin(phis_rad) * np.cos(thetas_rad)
    z = np.broadcast_to(np.sin(thetas_rad), x.shape)
    cartesian = np.stack((x, y, z), axis=-1)

    magnitude = np.linalg.norm(magnetic, axis=1)
    valid = (
        np.isfinite(magnetic).all(axis=1)
        & np.isfinite(magnitude)
        & (magnitude > 1.0e-9)
        & (magnitude < 1.0e3)
    )

    if polarity is None:
        orientation = np.full(phis.shape[0], -1.0, dtype=np.float64)
    else:
        orientation = np.asarray(polarity, dtype=np.float64)
        if orientation.shape != (phis.shape[0],):
            raise ValueError("polarity must match the number of rows")
        valid &= np.isfinite(orientation) & (orientation != 0)

    unit = np.full_like(magnetic, np.nan, dtype=np.float64)
    unit[valid] = (magnetic[valid] / magnitude[valid, None]) * orientation[valid, None]
    unit_channels = np.broadcast_to(unit[:, None, :], (*phis.shape, 3)).copy()

    dot = np.einsum("ijk,ijk->ij", unit_channels, cartesian)
    pitch = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    pitch[~valid] = np.nan
    return PitchAngleGeometry(
        pitch_angles=pitch,
        cartesian_coords=cartesian,
        unit_magnetic_field=unit_channels,
        valid_rows=valid,
    )


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

        phis = self.er_data.data[config.PHI_COLS].to_numpy(dtype=np.float64)
        magnetic_field = self.er_data.data[config.MAG_COLS].to_numpy(dtype=np.float64)
        geometry = compute_pitch_angle_geometry(
            phis,
            self.thetas,
            magnetic_field,
            polarity=self.polarity,
        )
        self.cartesian_coords = geometry.cartesian_coords
        self.unit_magnetic_field = geometry.unit_magnetic_field
        self.pitch_angles = geometry.pitch_angles
        self.valid_mask = geometry.valid_rows

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
