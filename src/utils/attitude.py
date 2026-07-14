"""Load and interpolate LP spin-axis attitude in the ECLIPJ2000 frame."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import spiceypy as spice


def load_attitude_data(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[None, None, None]:
    """
    Load attitude data from the specified file.

    Args:
        path: Path to the attitude data file

    Returns:
        Tuple of ephemeris time, ECLIPJ2000 right ascension, and ECLIPJ2000
        declination arrays, or (None, None, None) on error. Callers must check
        for the None case (they already do).
    """
    try:
        attitude_data = pd.read_csv(
            path, names=["UTC", "RPM", "RA", "DEC"], engine="python", header=None
        )
    except FileNotFoundError:
        logging.error(f"Error: The file {path} was not found.")
        return None, None, None
    except Exception as e:
        logging.error(f"Error loading attitude data: {e}")
        return None, None, None

    et_spin = np.array([spice.str2et(t) for t in attitude_data["UTC"]])
    ra_vals = np.array(attitude_data["RA"])
    dec_vals = np.array(attitude_data["DEC"])

    return et_spin, ra_vals, dec_vals


def get_current_ra_dec(
    time: float, et_spin: np.ndarray, ra_vals: np.ndarray, dec_vals: np.ndarray
) -> tuple[float | None, float | None]:
    """
    Get the current right ascension and declination at a given time.

    Args:
        time: Ephemeris time to query
        et_spin: Array of ephemeris times for attitude data
        ra_vals: Array of ECLIPJ2000 right ascension values
        dec_vals: Array of ECLIPJ2000 declination values

    Returns:
        Tuple of (ra, dec) values, or (None, None) if error
    """
    ra, dec = get_current_ra_dec_batch(
        np.asarray([time], dtype=np.float64), et_spin, ra_vals, dec_vals
    )
    if not np.isfinite(ra[0]) or not np.isfinite(dec[0]):
        logging.error("Index out of bounds for attitude data.")
        return None, None
    return float(ra[0]), float(dec[0])


def get_current_ra_dec_batch(
    times: np.ndarray, et_spin: np.ndarray, ra_vals: np.ndarray, dec_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Interpolate attitude at each query time.

    Args:
        times: Array of ephemeris times to query
        et_spin: Array of ephemeris times for attitude data (must be sorted)
        ra_vals: Array of ECLIPJ2000 right ascension values
        dec_vals: Array of ECLIPJ2000 declination values

    Returns:
        Tuple of (ra, dec) arrays. Values are NaN where indices are out of bounds.
    """
    query = np.asarray(times, dtype=np.float64)
    attitude_times = np.asarray(et_spin, dtype=np.float64)
    ra = np.asarray(ra_vals, dtype=np.float64)
    dec = np.asarray(dec_vals, dtype=np.float64)
    if not (attitude_times.ndim == ra.ndim == dec.ndim == 1):
        raise ValueError("Attitude arrays must be one-dimensional")
    if not (len(attitude_times) == len(ra) == len(dec)):
        raise ValueError("Attitude arrays must have equal lengths")
    if len(attitude_times) == 0:
        return np.full_like(query, np.nan), np.full_like(query, np.nan)
    if np.any(np.diff(attitude_times) <= 0.0):
        raise ValueError("Attitude times must be strictly increasing")

    ra_out = np.full_like(query, np.nan)
    dec_out = np.full_like(query, np.nan)
    finite = np.isfinite(query)
    covered = finite & (query >= attitude_times[0]) & (query <= attitude_times[-1])
    if not np.any(covered):
        return ra_out, dec_out

    if len(attitude_times) == 1:
        exact = covered & (query == attitude_times[0])
        ra_out[exact] = np.mod(ra[0], 360.0)
        dec_out[exact] = dec[0]
        return ra_out, dec_out

    covered_times = query[covered]
    right = np.searchsorted(attitude_times, covered_times, side="right")
    right = np.clip(right, 1, len(attitude_times) - 1)
    left = right - 1
    interval = attitude_times[right] - attitude_times[left]
    fraction = (covered_times - attitude_times[left]) / interval

    # Interpolate RA along the shortest circular arc and declination linearly.
    ra_delta = (ra[right] - ra[left] + 180.0) % 360.0 - 180.0
    ra_out[covered] = np.mod(ra[left] + fraction * ra_delta, 360.0)
    dec_out[covered] = dec[left] + fraction * (dec[right] - dec[left])

    return ra_out, dec_out
