import logging
import os
from flux import FluxData
from typing import Tuple
import numpy as np
import spiceypy as spice
import pandas as pd

import config

LP = "-25"
MOON = "301"
SUN = "10"

def list_files(directory: str) -> list[str]:    
    try:
        files = os.listdir(directory)
        return [f for f in files if os.path.isfile(os.path.join(directory, f))]
    except FileNotFoundError:
        logging.error(f"Directory {directory} not found.")
        return []
    
def list_folder_files(directory: str) -> list[str]:
    try:
        files = os.listdir(directory)
        return [f for f in files if os.path.isdir(os.path.join(directory, f))]
    except FileNotFoundError:
        logging.error(f"Directory {directory} not found.")
        return []

def get_time_range(flux_data: FluxData) -> Tuple[float, float]:
    """
    Get the time range from the flux data.
    """
    if flux_data.data is None:
        logging.error("Flux data is not loaded.")
        return 0.0, 0.0

    start_time = flux_data.data['UTC'].iloc[0]
    end_time = flux_data.data['UTC'].iloc[-1]
    
    return start_time, end_time

def get_lp_position_wrt_moon(time: float) -> np.ndarray:
    """
    Get the position of the LP at a given time with respect to the Moon.
    """
    pos, _ = spice.spkpos(LP, time, "J2000", "NONE", MOON)

    try:
        mat = spice.pxform("J2000", "IAU_MOON", time)
        pos = spice.mxv(mat, pos)
    except Exception as e:
        logging.error(f"Error transforming position: {e}")
        return None
    return pos

def get_lp_vector_to_sun_in_lunar_frame(time: float) -> np.ndarray:
    """
    Get the position of the LP at a given time with respect to the Sun.
    """
    pos, _ = spice.spkpos(SUN, time, "J2000", "NONE", LP)
    try:
        mat = spice.pxform("J2000", "IAU_MOON", time)
        pos = spice.mxv(mat, pos)
    except Exception as e:
        logging.error(f"Error transforming position: {e}")
        return None
    return pos

def get_sun_vector_wrt_moon(time: float) -> np.ndarray:
    """
    Get the position of the Sun at a given time with respect to the Moon.
    """
    pos, _ = spice.spkpos(SUN, time, "J2000", "NONE", MOON)
    try:
        mat = spice.pxform("J2000", "IAU_MOON", time)
        pos = spice.mxv(mat, pos)
    except Exception as e:
        logging.error(f"Error transforming position: {e}")
        return None
    
    return pos

def get_j2000_iau_moon_transform_matrix(time: float) -> np.ndarray:
    try:
        mat = spice.pxform("J2000", "IAU_MOON", time)
    except Exception as e:
        logging.error(f"Error transforming position: {e}")
        return np.full((3, 3), np.nan)
    
    return mat
    

        

def bisect_right(a, x, lo=0, hi=None, *, key=None):
    """Return the index where to insert item x in list a, assuming a is sorted.

    The return value i is such that all e in a[:i] have e <= x, and all e in
    a[i:] have e > x.  So if x already appears in the list, a.insert(i, x) will
    insert just after the rightmost x already there.

    Optional args lo (default 0) and hi (default len(a)) bound the
    slice of a to be searched.
    """

    if lo < 0:
        raise ValueError('lo must be non-negative')
    if hi is None:
        hi = len(a)
    # Note, the comparison uses "<" to match the
    # __lt__() logic in list.sort() and in heapq.
    if key is None:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < a[mid]:
                hi = mid
            else:
                lo = mid + 1
    else:
        while lo < hi:
            mid = (lo + hi) // 2
            if x < key(a[mid]):
                hi = mid
            else:
                lo = mid + 1
    return lo if lo < len(a) else len(a) - 1 # return the last index if lo is out of bounds

def load_attitude_data(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the attitude data from the specified file.
    """
    try:
        attitude_data = pd.read_csv(path, names=["UTC", "RPM", "RA", "DEC"], engine="python", header=None)

    except FileNotFoundError:
        logging.error(f"Error: The file {path} was not found.")
        return None, None, None
    
    et_spin = np.array([spice.str2et(t) for t in attitude_data["UTC"]])
    ra_vals = np.array(attitude_data["RA"])
    dec_vals = np.array(attitude_data["DEC"])

    return et_spin, ra_vals, dec_vals

def get_current_ra_dec(time: float, et_spin: np.ndarray, ra_vals: np.ndarray, dec_vals: np.ndarray) -> Tuple[float, float]:
    idx = bisect_right(et_spin, time)
    
    if idx < 0:
        logging.error("Index out of bounds for attitude data.")
        return None, None
    return ra_vals[idx], dec_vals[idx]

def ra_dec_to_unit(ra_deg, dec_deg):
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)
    x = np.cos(dec_rad) * np.cos(ra_rad)
    y = np.cos(dec_rad) * np.sin(ra_rad)
    z = np.sin(dec_rad)
    return np.stack([x, y, z], axis=-1)  # shape (N, 3)

def build_scd_to_j2000(spin_vecs, sun_vecs):
    """
    spin_vecs: Nx3 array of spin vectors (unit, in J2000)
    sun_vecs: Nx3 array of sun direction vectors (unit, in J2000)

    returns: list of 3x3 rotation matrices (SCD → J2000)
    """
    mats = []

    for z_hat, sun in zip(spin_vecs, sun_vecs):
        z_hat = z_hat / np.linalg.norm(z_hat)

        # project sun onto plane orthogonal to spin
        sun_proj = sun - np.dot(sun, z_hat) * z_hat
        norm = np.linalg.norm(sun_proj)

        if norm < 1e-8:
            # sun too aligned with spin, can't define X
            mats.append(np.full((3, 3), np.nan))
            continue

        x_hat = sun_proj / norm
        y_hat = np.cross(z_hat, x_hat)

        R = np.stack([x_hat, y_hat, z_hat], axis=1)  # columns = SCD axes in J2000
        mats.append(R)

    return np.array(mats)



def get_intersection_or_none(pos, direction, radius=config.LUNAR_RADIUS_KM):
    # normalize direction just in case
    v = direction / np.linalg.norm(direction)
    p = pos

    a = np.dot(v, v)
    b = 2 * np.dot(p, v)
    c = np.dot(p, p) - radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return None  # no intersection

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # we want the closest *positive* t
    t_candidates = [t for t in (t1, t2) if t > 0]
    if not t_candidates:
        return None

    t = min(t_candidates)
    return p + t * v  # intersection point

def cartesian_to_lat_lon(coordinate: np.ndarray) -> Tuple[float, float]:
    """
    Convert Cartesian coordinates to latitude and longitude.
    """
    x, y, z = coordinate
    lat = np.rad2deg(np.arcsin(z / np.linalg.norm(coordinate)))
    lon = np.rad2deg(np.arctan2(y, x))
    return lat, lon

def lat_lon_to_cartesian(lat_long: np.ndarray) -> np.ndarray:
    """
    Convert latitude and longitude to Cartesian coordinates.
    """
    lat_rad = np.deg2rad(lat_long[:, 0])
    lon_rad = np.deg2rad(lat_long[:, 1])
    
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    
    return np.column_stack((x, y, z))