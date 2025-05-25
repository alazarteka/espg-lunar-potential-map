import logging
import numpy as np
import pandas as pd
import spiceypy as spice
from bisect import bisect_right
from typing import Tuple, Union, Any


def load_attitude_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load attitude data from the specified file.
    
    Args:
        path: Path to the attitude data file
        
    Returns:
        Tuple of (et_spin, ra_vals, dec_vals) arrays, or (None, None, None) if error
    """
    try:
        attitude_data = pd.read_csv(
            path, 
            names=["UTC", "RPM", "RA", "DEC"], 
            engine="python", 
            header=None
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
    time: float, 
    et_spin: np.ndarray, 
    ra_vals: np.ndarray, 
    dec_vals: np.ndarray
) -> Tuple[Union[float, None], Union[float, None]]:
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


def get_time_range(flux_data: Any) -> Tuple[str, str]:
    """
    Get the time range from the flux data.
    
    Args:
        flux_data: FluxData object with loaded data
        
    Returns:
        Tuple of (start_time, end_time) as strings, or ("", "") if error
    """
    if flux_data.data is None:
        logging.error("Flux data is not loaded.")
        return "", ""

    start_time = flux_data.data['UTC'].iloc[0]
    end_time = flux_data.data['UTC'].iloc[-1]
    
    return start_time, end_time