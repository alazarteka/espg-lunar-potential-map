"""Load LP attitude tables (UTC, RPM, RA, DEC) and look up spin-axis
right ascension/declination at query ephemeris times."""

import logging
from bisect import bisect_right
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
        Tuple of (et_spin, ra_vals, dec_vals) arrays, or (None, None, None) on
        error. Callers must check for the None case (they already do).
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
        ra_vals: Array of right ascension values
        dec_vals: Array of declination values

    Returns:
        Tuple of (ra, dec) values, or (None, None) if error
    """
    idx = bisect_right(et_spin, time)

    if idx <= 0 or idx >= len(ra_vals):
        logging.error("Index out of bounds for attitude data.")
        return None, None

    return ra_vals[idx], dec_vals[idx]


def get_current_ra_dec_batch(
    times: np.ndarray, et_spin: np.ndarray, ra_vals: np.ndarray, dec_vals: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Batch version of get_current_ra_dec using np.searchsorted.

    Args:
        times: Array of ephemeris times to query
        et_spin: Array of ephemeris times for attitude data (must be sorted)
        ra_vals: Array of right ascension values
        dec_vals: Array of declination values

    Returns:
        Tuple of (ra, dec) arrays. Values are NaN where indices are out of bounds.
    """
    # searchsorted returns indices where elements should be inserted to maintain order.
    # bisect_right is equivalent to searchsorted(side='right')
    idxs = np.searchsorted(et_spin, times, side="right")

    # Mask invalid indices: valid insertion points are 0 < idx < len(ra_vals).
    # idx == 0 means the time precedes the first spin sample and idx == len means
    # it follows the last; both are out of range and yield NaN below.

    n = len(ra_vals)
    valid_mask = (idxs > 0) & (idxs < n)

    ra_out = np.full_like(times, np.nan)
    dec_out = np.full_like(times, np.nan)

    # Use valid indices to fetch values
    # We can just use boolean indexing
    valid_idxs = idxs[valid_mask]

    if len(valid_idxs) > 0:
        ra_out[valid_mask] = ra_vals[valid_idxs]
        dec_out[valid_mask] = dec_vals[valid_idxs]

    return ra_out, dec_out
