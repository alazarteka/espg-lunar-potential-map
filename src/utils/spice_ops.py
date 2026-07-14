"""SPICE lookups of LP/Moon/Sun geometry in explicitly named frames."""

import logging
from collections.abc import Callable

import numpy as np
import spiceypy as spice

# SPICE object codes
LP = "-25"
MOON = "301"
SUN = "10"
ECLIPJ2000 = "ECLIPJ2000"
IAU_MOON = "IAU_MOON"
J2000 = "J2000"


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


def _get_position_batch(
    times: np.ndarray,
    *,
    target: str,
    observer: str,
    reference_frame: str,
) -> np.ndarray:
    """Return the target position relative to the observer in the named frame."""

    def compute(t: float) -> np.ndarray | None:
        try:
            position, _ = spice.spkpos(target, t, reference_frame, "NONE", observer)
            return position
        except Exception:
            return None

    return _batch_spice_vector(times, compute)


def _get_frame_transform_batch(
    times: np.ndarray, *, from_frame: str, to_frame: str
) -> np.ndarray:
    """Return position transforms between two SPICE frames."""

    def compute(t: float) -> np.ndarray | None:
        try:
            return spice.pxform(from_frame, to_frame, t)
        except Exception:
            return None

    return _batch_spice_vector(times, compute, output_shape=(3, 3))


def get_lp_position_wrt_moon_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_lp_position_wrt_moon.
    Returns (N, 3) array. Rows with errors are NaN.
    """

    return _get_position_batch(
        times, target=LP, observer=MOON, reference_frame=IAU_MOON
    )


def get_lp_vector_to_sun_in_lunar_frame_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_lp_vector_to_sun_in_lunar_frame.
    Returns (N, 3) array. Rows with errors are NaN.
    """

    return _get_position_batch(times, target=SUN, observer=LP, reference_frame=IAU_MOON)


def get_lp_vector_to_sun_in_eclipj2000_batch(times: np.ndarray) -> np.ndarray:
    """Return LP-to-Sun vectors in the attitude product's ECLIPJ2000 frame."""
    return _get_position_batch(
        times, target=SUN, observer=LP, reference_frame=ECLIPJ2000
    )


def get_sun_vector_wrt_moon_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_sun_vector_wrt_moon.
    Returns (N, 3) array. Rows with errors are NaN.
    """

    return _get_position_batch(
        times, target=SUN, observer=MOON, reference_frame=IAU_MOON
    )


def get_j2000_iau_moon_transform_matrix_batch(times: np.ndarray) -> np.ndarray:
    """
    Batch version of get_j2000_iau_moon_transform_matrix.
    Returns (N, 3, 3) array. Rows with errors are NaN.
    """

    return _get_frame_transform_batch(times, from_frame=J2000, to_frame=IAU_MOON)


def get_eclipj2000_iau_moon_transform_matrix_batch(
    times: np.ndarray,
) -> np.ndarray:
    """Return ECLIPJ2000-to-IAU_MOON position transforms."""
    return _get_frame_transform_batch(times, from_frame=ECLIPJ2000, to_frame=IAU_MOON)
