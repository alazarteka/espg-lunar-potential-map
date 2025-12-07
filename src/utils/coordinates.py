
import numpy as np


def ra_dec_to_unit(
    ra_deg: float | np.ndarray, dec_deg: float | np.ndarray
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
    # Ensure inputs are arrays
    spin_vecs = np.asarray(spin_vecs)
    sun_vecs = np.asarray(sun_vecs)

    # Normalize spin vectors
    z_norms = np.linalg.norm(spin_vecs, axis=1, keepdims=True)
    # Avoid division by zero if any spin vector is zero (unlikely but safe)
    z_hat = np.divide(
        spin_vecs, z_norms, out=np.zeros_like(spin_vecs), where=z_norms != 0
    )

    # Project sun onto plane orthogonal to spin
    # dot product (N,3) . (N,3) -> (N,1)
    dots = np.einsum("ij,ij->i", sun_vecs, z_hat)[:, None]
    sun_proj = sun_vecs - dots * z_hat

    proj_norms = np.linalg.norm(sun_proj, axis=1, keepdims=True)

    # Mask for valid projection
    valid_mask = proj_norms.flatten() >= 1e-8

    x_hat = np.zeros_like(sun_proj)
    np.divide(sun_proj, proj_norms, out=x_hat, where=valid_mask[:, None])

    y_hat = np.cross(z_hat, x_hat)

    # Stack columns [x_hat, y_hat, z_hat] to form (N, 3, 3) matrices
    # x_hat is (N, 3), y_hat is (N, 3), z_hat is (N, 3)
    # Result should be (N, 3, 3) where mat[i] = [x_hat[i], y_hat[i], z_hat[i]] (columns)
    # So we stack along last axis? No, stack along axis 2.
    # np.stack([x, y, z], axis=1) creates (N, 3, 3) where axis 1 is the column index?
    # Let's verify: mat[i, :, 0] should be x_hat[i].
    # np.stack([x, y, z], axis=2) -> shape (N, 3, 3).
    # mat[i, :, 0] -> x[i, :] Correct.

    mats = np.stack([x_hat, y_hat, z_hat], axis=2)

    # Set invalid matrices to NaN
    if not np.all(valid_mask):
        mats[~valid_mask] = np.nan

    return mats
