import logging
import warnings
from bisect import bisect_right
from typing import Any

import numpy as np
import pandas as pd
import spiceypy as spice


def load_attitude_data(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load attitude data from the specified file.

    Args:
        path: Path to the attitude data file

    Returns:
        Tuple of (et_spin, ra_vals, dec_vals) arrays, or (None, None, None) if error
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

    # Mask invalid indices.
    # Valid indices for bisect_right logic in get_current_ra_dec are > 0 and
    # < len(ra_vals). Original code checks idx <= 0 or idx >= len(ra_vals),
    # so valid is 0 < idx < len(ra_vals).
    # Wait, original code: if idx <= 0 or idx >= len(ra_vals): return None
    # So valid range is 1 to len(ra_vals)-1?
    # Let's check: ra_vals[idx] is accessed, so idx must be < len(ra_vals).
    # And idx > 0 means we don't accept time < et_spin[0]
    # (where insertion point is 0).

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


def get_time_range(flux_data: Any) -> tuple[str, str]:
    """
    Get the time range from the flux data.

    Args:
        flux_data: FluxData object with loaded data

    Returns:
        Tuple of (start_time, end_time) as strings, or ("", "") if error
    """
    warnings.warn(
        "get_time_range is deprecated and will be removed in a future version",
        DeprecationWarning,
        stacklevel=2,
    )
    if flux_data.data is None:
        logging.error("Flux data is not loaded.")
        return "", ""

    start_time = flux_data.data["UTC"].iloc[0]
    end_time = flux_data.data["UTC"].iloc[-1]

    return start_time, end_time
