from typing import Union

import numpy as np


def ra_dec_to_unit(
    ra_deg: Union[float, np.ndarray], dec_deg: Union[float, np.ndarray]
) -> np.ndarray:
    """
    Convert right ascension and declination to unit vectors.

    Args:
        ra_deg: Right ascension in degrees
        dec_deg: Declination in degrees

    Returns:
        Unit vectors in Cartesian coordinates (shape: (..., 3))
    """
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.stack([x, y, z], axis=-1)


def cartesian_to_lat_lon(coordinate: np.ndarray) -> tuple[float, float]:
    """
    Convert Cartesian coordinates to latitude and longitude.

    Args:
        coordinate: 3D Cartesian coordinate vector [x, y, z]

    Returns:
        Tuple of (latitude, longitude) in degrees
    """
    x, y, z = coordinate
    lat = np.rad2deg(np.arcsin(z / np.linalg.norm(coordinate)))
    lon = np.rad2deg(np.arctan2(y, x))
    return lat, lon


def lat_lon_to_cartesian(lat_long: np.ndarray) -> np.ndarray:
    """
    Convert latitude and longitude to Cartesian coordinates.

    Args:
        lat_long: Array of shape (N, 2) with [latitude, longitude] in degrees

    Returns:
        Array of shape (N, 3) with Cartesian coordinates [x, y, z]
    """
    lat_rad = np.deg2rad(lat_long[:, 0])
    lon_rad = np.deg2rad(lat_long[:, 1])

    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)

    return np.column_stack((x, y, z))


def build_scd_to_j2000(spin_vecs: np.ndarray, sun_vecs: np.ndarray) -> np.ndarray:
    """
    Build rotation matrices from spacecraft (SCD) frame to J2000 frame.

    The SCD frame is defined as:
    - X-axis: projection of sun vector onto plane perpendicular to spin axis
    - Y-axis: Z × X (right-handed system)
    - Z-axis: spin axis

    Args:
        spin_vecs: Array of shape (N, 3) with spin vectors (unit, in J2000)
        sun_vecs: Array of shape (N, 3) with sun direction vectors (unit, in J2000)

    Returns:
        Array of shape (N, 3, 3) with rotation matrices (SCD → J2000)
    """
    mats = []

    for z_hat, sun in zip(spin_vecs, sun_vecs, strict=False):
        z_hat = z_hat / np.linalg.norm(z_hat)

        # Project sun onto plane orthogonal to spin
        sun_proj = sun - np.dot(sun, z_hat) * z_hat
        norm = np.linalg.norm(sun_proj)

        if norm < 1e-8:
            # Sun too aligned with spin, can't define X
            mats.append(np.full((3, 3), np.nan))
            continue

        x_hat = sun_proj / norm
        y_hat = np.cross(z_hat, x_hat)

        # Rotation matrix with columns = SCD axes in J2000
        R = np.stack([x_hat, y_hat, z_hat], axis=1)
        mats.append(R)

    return np.array(mats)
