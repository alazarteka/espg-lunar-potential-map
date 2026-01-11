"""Data loading utilities for visualization."""

from datetime import date
from pathlib import Path

import numpy as np
import spiceypy as spice

from src.potential_mapper.spice import load_spice_files
from src.utils import spice_ops

from .utils import date_range


def _discover_date_files(cache_dir: Path, start_day: date, end_day: date) -> list[Path]:
    """Find NPZ cache files matching the date range.

    Args:
        cache_dir: Directory to search.
        start_day: Start date (inclusive).
        end_day: End date (inclusive).

    Returns:
        List of matching file paths, sorted by date.

    Raises:
        FileNotFoundError: If no matching files found.
    """
    days = date_range(start_day, end_day)
    pattern_list = [f"3D{day.strftime('%y%m%d')}.npz" for day in days]

    files = []
    for pattern in pattern_list:
        matches = list(cache_dir.rglob(pattern))
        if matches:
            files.append(matches[0])

    if not files:
        raise FileNotFoundError(
            f"No cache files found for date range {start_day} to {end_day} in {cache_dir}"
        )

    return files


def load_measurements(
    cache_dir: Path, start_day: date, end_day: date
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached potential data for date range.

    Returns:
        lats: Latitude array
        lons: Longitude array
        potentials: Surface potential array
        in_sun: Boolean array (True if sunlit)
    """
    files = _discover_date_files(cache_dir, start_day, end_day)

    all_lats = []
    all_lons = []
    all_pots = []
    all_sun = []

    for npz_file in files:
        with np.load(npz_file) as data:
            lats = data["rows_projection_latitude"]
            lons = data["rows_projection_longitude"]
            pots = data["rows_projected_potential"]
            in_sun = data["rows_projection_in_sun"]

            # Filter finite values
            valid = np.isfinite(pots) & np.isfinite(lats) & np.isfinite(lons)

            all_lats.append(lats[valid])
            all_lons.append(lons[valid])
            all_pots.append(pots[valid])

            # in_sun might be integer or bool in older files, cast to bool
            all_sun.append(in_sun[valid].astype(bool))

    if not all_lats:
        return (np.array([]), np.array([]), np.array([]), np.array([]))

    return (
        np.concatenate(all_lats),
        np.concatenate(all_lons),
        np.concatenate(all_pots),
        np.concatenate(all_sun),
    )


def load_date_range_data_with_sza(
    cache_dir: Path, start_day: date, end_day: date
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached potential data for date range and compute SZA.

    Returns:
        sza: Solar zenith angles (degrees)
        potentials: Surface potentials (V)
        in_sun: Boolean array for illumination
    """
    files = _discover_date_files(cache_dir, start_day, end_day)

    # Load SPICE kernels
    load_spice_files()

    all_sza = []
    all_potentials = []
    all_in_sun = []

    for npz_file in files:
        with np.load(npz_file) as data:
            lats = data["rows_projection_latitude"]
            lons = data["rows_projection_longitude"]
            pots = data["rows_projected_potential"]
            utcs = data["rows_utc"]
            in_sun = data["rows_projection_in_sun"]

            # Compute SZA for each measurement
            # Vectorized approach would be better but keeping original loop for fidelity first
            # Or wait, the original loop was slow?
            # Original code in plot_terminator_profile_paper.py:
            # for lat, lon, pot, utc_str, sun_flag in zip(...)

            # I'll stick to the original implementation logic to ensure correctness,
            # but maybe try to be slightly more robust with try/except

            for lat, lon, pot, utc_str, sun_flag in zip(
                lats, lons, pots, utcs, in_sun, strict=False
            ):
                if not np.isfinite(pot):
                    continue

                try:
                    et = spice.utc2et(utc_str)
                    sun_vec = spice_ops.get_sun_vector_wrt_moon(et)

                    # Convert lat/lon to cartesian
                    lat_rad = np.radians(lat)
                    lon_rad = np.radians(lon)
                    point_vec = np.array(
                        [
                            np.cos(lat_rad) * np.cos(lon_rad),
                            np.cos(lat_rad) * np.sin(lon_rad),
                            np.sin(lat_rad),
                        ]
                    )

                    # Compute SZA
                    norm_point = np.linalg.norm(point_vec)
                    norm_sun = np.linalg.norm(sun_vec)
                    if norm_point == 0 or norm_sun == 0:
                        continue

                    cos_sza = np.dot(point_vec, sun_vec) / (norm_point * norm_sun)
                    cos_sza = np.clip(cos_sza, -1.0, 1.0)
                    sza = np.degrees(np.arccos(cos_sza))

                    all_sza.append(sza)
                    all_potentials.append(pot)
                    all_in_sun.append(sun_flag)

                except Exception:
                    continue

    return np.array(all_sza), np.array(all_potentials), np.array(all_in_sun)
