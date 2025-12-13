import logging

import numpy as np
import spiceypy as spice

# SPICE object codes
LP = "-25"
MOON = "301"
SUN = "10"


def get_lp_position_wrt_moon(time: float) -> np.ndarray | None:
    """
    Get the position of the Lunar Prospector at a given time with respect to the Moon.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        Position vector in IAU_MOON frame (km), or None if error
    """
    try:
        pos, _ = spice.spkpos(LP, time, "J2000", "NONE", MOON)
        mat = spice.pxform("J2000", "IAU_MOON", time)
        pos = spice.mxv(mat, pos)
        return pos
    except Exception as e:
        logging.error(f"Error getting LP position: {e}")
        return None


def get_lp_vector_to_sun_in_lunar_frame(time: float) -> np.ndarray | None:
    """
    Get the vector from Lunar Prospector to the Sun in lunar frame.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        Vector from LP to Sun in IAU_MOON frame (km), or None if error
    """
    try:
        pos, _ = spice.spkpos(SUN, time, "J2000", "NONE", LP)
        mat = spice.pxform("J2000", "IAU_MOON", time)
        pos = spice.mxv(mat, pos)
        return pos
    except Exception as e:
        logging.error(f"Error getting LP-to-Sun vector: {e}")
        return None


def get_sun_vector_wrt_moon(time: float) -> np.ndarray | None:
    """
    Get the position vector of the Sun with respect to the Moon.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        Vector from Moon to Sun in IAU_MOON frame (km), or None if error
    """
    try:
        pos, _ = spice.spkpos(SUN, time, "J2000", "NONE", MOON)
        mat = spice.pxform("J2000", "IAU_MOON", time)
        pos = spice.mxv(mat, pos)
        return pos
    except Exception as e:
        logging.error(f"Error getting Sun position: {e}")
        return None


def get_j2000_iau_moon_transform_matrix(time: float) -> np.ndarray:
    """
    Get the transformation matrix from J2000 to IAU_MOON frame.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        3x3 transformation matrix, or NaN matrix if error
    """
    try:
        mat = spice.pxform("J2000", "IAU_MOON", time)
        return mat
    except Exception as e:
        logging.error(f"Error getting transformation matrix: {e}")
        return np.full((3, 3), np.nan)


def get_lp_position_wrt_moon_batch(times: np.ndarray) -> np.ndarray:
    """
    Get the position of the Lunar Prospector at multiple times with respect to the Moon.

    Args:
        times: Array of ephemeris times (seconds past J2000).

    Returns:
        np.ndarray: Position vectors in IAU_MOON frame (km) with shape (N, 3).
                    Rows with errors contain NaNs.
    """
    n = len(times)
    positions = np.full((n, 3), np.nan)

    # Pre-fetch common frames if possible, but spkpos depends on time.
    # We loop here to avoid overhead in the main pipeline.
    # spiceypy doesn't support vectorized spkpos yet.
    for i, t in enumerate(times):
        try:
            pos, _ = spice.spkpos(LP, t, "J2000", "NONE", MOON)
            mat = spice.pxform("J2000", "IAU_MOON", t)
            positions[i] = spice.mxv(mat, pos)
        except Exception:
            pass  # Leave as NaN

    return positions


def get_lp_vector_to_sun_in_lunar_frame_batch(times: np.ndarray) -> np.ndarray:
    """
    Get vectors from Lunar Prospector to the Sun in lunar frame for multiple times.

    Args:
        times: Array of ephemeris times (seconds past J2000).

    Returns:
        np.ndarray: Vectors from LP to Sun in IAU_MOON frame (km) with shape (N, 3).
                    Rows with errors contain NaNs.
    """
    n = len(times)
    vectors = np.full((n, 3), np.nan)

    for i, t in enumerate(times):
        try:
            pos, _ = spice.spkpos(SUN, t, "J2000", "NONE", LP)
            mat = spice.pxform("J2000", "IAU_MOON", t)
            vectors[i] = spice.mxv(mat, pos)
        except Exception:
            pass

    return vectors


def get_sun_vector_wrt_moon_batch(times: np.ndarray) -> np.ndarray:
    """
    Get position vectors of the Sun with respect to the Moon for multiple times.

    Args:
        times: Array of ephemeris times (seconds past J2000).

    Returns:
        np.ndarray: Vectors from Moon to Sun in IAU_MOON frame (km) with shape (N, 3).
                    Rows with errors contain NaNs.
    """
    n = len(times)
    vectors = np.full((n, 3), np.nan)

    for i, t in enumerate(times):
        try:
            pos, _ = spice.spkpos(SUN, t, "J2000", "NONE", MOON)
            mat = spice.pxform("J2000", "IAU_MOON", t)
            vectors[i] = spice.mxv(mat, pos)
        except Exception:
            pass

    return vectors


def get_j2000_iau_moon_transform_matrix_batch(times: np.ndarray) -> np.ndarray:
    """
    Get transformation matrices from J2000 to IAU_MOON frame for multiple times.

    Args:
        times: Array of ephemeris times (seconds past J2000).

    Returns:
        np.ndarray: Transformation matrices with shape (N, 3, 3).
                    Rows with errors contain NaNs.
    """
    n = len(times)
    mats = np.full((n, 3, 3), np.nan)

    for i, t in enumerate(times):
        try:
            mats[i] = spice.pxform("J2000", "IAU_MOON", t)
        except Exception:
            pass

    return mats
