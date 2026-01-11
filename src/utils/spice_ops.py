import logging
from collections.abc import Callable

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
    pos = get_lp_position_wrt_moon_batch(np.array([time]))[0]
    if np.isnan(pos).any():
        logging.error("Error getting LP position")
        return None
    return pos


def get_lp_vector_to_sun_in_lunar_frame(time: float) -> np.ndarray | None:
    """
    Get the vector from Lunar Prospector to the Sun in lunar frame.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        Vector from LP to Sun in IAU_MOON frame (km), or None if error
    """
    pos = get_lp_vector_to_sun_in_lunar_frame_batch(np.array([time]))[0]
    if np.isnan(pos).any():
        logging.error("Error getting LP-to-Sun vector")
        return None
    return pos


def get_sun_vector_wrt_moon(time: float) -> np.ndarray | None:
    """
    Get the position vector of the Sun with respect to the Moon.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        Vector from Moon to Sun in IAU_MOON frame (km), or None if error
    """
    pos = get_sun_vector_wrt_moon_batch(np.array([time]))[0]
    if np.isnan(pos).any():
        logging.error("Error getting Sun position")
        return None
    return pos


def get_j2000_iau_moon_transform_matrix(time: float) -> np.ndarray:
    """
    Get the transformation matrix from J2000 to IAU_MOON frame.

    Args:
        time: Ephemeris time (seconds past J2000)

    Returns:
        3x3 transformation matrix, or NaN matrix if error
    """
    mat = get_j2000_iau_moon_transform_matrix_batch(np.array([time]))[0]
    if np.isnan(mat).any():
        logging.error("Error getting transformation matrix")
    return mat


def _batch_spice_vector(
    times: np.ndarray,
    compute_fn: Callable[[float], np.ndarray | None],
    output_shape: tuple[int, ...] = (3,),
) -> np.ndarray:
    """
    Batch helper for SPICE vector-like outputs.
    """
    n = len(times)
    values = np.full((n, *output_shape), np.nan)

    for i, t in enumerate(times):
        try:
            result = compute_fn(t)
        except Exception:
            continue
        if result is None:
            continue
        values[i] = result

    return values


def get_lp_position_wrt_moon_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_lp_position_wrt_moon.
    Returns (N, 3) array. Rows with errors are NaN.
    """

    def compute(t: float) -> np.ndarray | None:
        try:
            pos, _ = spice.spkpos(LP, t, "J2000", "NONE", MOON)
            mat = spice.pxform("J2000", "IAU_MOON", t)
            return spice.mxv(mat, pos)
        except Exception:
            return None

    return _batch_spice_vector(times, compute)


def get_lp_vector_to_sun_in_lunar_frame_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_lp_vector_to_sun_in_lunar_frame.
    Returns (N, 3) array. Rows with errors are NaN.
    """

    def compute(t: float) -> np.ndarray | None:
        try:
            pos, _ = spice.spkpos(SUN, t, "J2000", "NONE", LP)
            mat = spice.pxform("J2000", "IAU_MOON", t)
            return spice.mxv(mat, pos)
        except Exception:
            return None

    return _batch_spice_vector(times, compute)


def get_sun_vector_wrt_moon_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_sun_vector_wrt_moon.
    Returns (N, 3) array. Rows with errors are NaN.
    """

    def compute(t: float) -> np.ndarray | None:
        try:
            pos, _ = spice.spkpos(SUN, t, "J2000", "NONE", MOON)
            mat = spice.pxform("J2000", "IAU_MOON", t)
            return spice.mxv(mat, pos)
        except Exception:
            return None

    return _batch_spice_vector(times, compute)


def get_j2000_iau_moon_transform_matrix_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_j2000_iau_moon_transform_matrix.
    Returns (N, 3, 3) array. Rows with errors are NaN.
    """

    def compute(t: float) -> np.ndarray | None:
        try:
            return spice.pxform("J2000", "IAU_MOON", t)
        except Exception:
            return None

    return _batch_spice_vector(times, compute, output_shape=(3, 3))
