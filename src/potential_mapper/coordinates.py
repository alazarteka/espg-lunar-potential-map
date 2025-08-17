import logging
from dataclasses import dataclass

import numpy as np
import spiceypy as spice

from src.flux import FluxData
import src.config as config
from src.utils.attitude import get_current_ra_dec
from src.utils.coordinates import build_scd_to_j2000, ra_dec_to_unit
from src.utils.spice_ops import (
    get_lp_position_wrt_moon,
    get_j2000_iau_moon_transform_matrix,
    get_lp_vector_to_sun_in_lunar_frame,
    get_sun_vector_wrt_moon
)
from src.utils.geometry import get_intersections_or_none_batch

@dataclass
class CoordinateArrays:
    """
    Container for per-row geometry and transform arrays in IAU_MOON/J2000 frames.

    Shapes:
    - lp_positions: (N, 3) LP position in IAU_MOON (km)
    - lp_vectors_to_sun: (N, 3) LP→Sun vector in IAU_MOON (km)
    - ra_dec_cartesian: (N, 3) spin axis unit vectors in J2000
    - moon_vectors_to_sun: (N, 3) Moon→Sun vector in IAU_MOON (km)
    - j2000_to_iau_moon_mats: (N, 3, 3) rotation matrices
    - scd_to_iau_moon_mats: (N, 3, 3) rotation matrices (SCD→IAU_MOON)
    """

    lp_positions: np.ndarray
    lp_vectors_to_sun: np.ndarray
    ra_dec_cartesian: np.ndarray
    moon_vectors_to_sun: np.ndarray
    j2000_to_iau_moon_mats: np.ndarray
    scd_to_iau_moon_mats: np.ndarray

class CoordinateCalculator:
    """
    Compute coordinate transformations between spacecraft (SCD), J2000, and IAU_MOON.
    """

    def __init__(self, et_spin: np.ndarray, right_ascension: np.ndarray, declination: np.ndarray):
        """
        Initialize with spacecraft spin rate and celestial coordinates.

        Args:
            et_spin: Ephemeris times for attitude data (shape: (N,))
            right_ascension: Right ascension in degrees (shape: (N,))
            declination: Declination in degrees (shape: (N,))
        """
        self.et_spin = et_spin
        self.right_ascension = right_ascension
        self.declination = declination

    def calculate_coordinate_transformation(self, flux_data: FluxData) -> CoordinateArrays:
        """
        Build all geometry arrays for each row in `flux_data`.

        Returns a CoordinateArrays with per-row positions, Sun vectors, and
        transformation matrices sufficient to project vectors between frames.
        """
        n_points = len(flux_data.data)

        lp_positions = np.zeros((n_points, 3)) # Lunar frame
        lp_vectors_to_sun = np.zeros((n_points, 3)) # Lunar frame
        ra_dec_cartesian = np.zeros((n_points, 3))
        moon_vectors_to_sun = np.zeros((n_points, 3))
        j2000_to_iau_moon_mats = np.zeros((n_points, 3, 3))

        for t, utc_time in enumerate(flux_data.data[config.UTC_COLUMN]):
            time = spice.str2et(utc_time)

            lp_position_t = get_lp_position_wrt_moon(time)
            lp_vector_to_sun_in_lunar_frame_t = get_lp_vector_to_sun_in_lunar_frame(time)
            right_ascension_t, declination_t = get_current_ra_dec(
                time, self.et_spin, self.right_ascension, self.declination
            )
            moon_vector_to_sun_in_lunar_frame_t = get_sun_vector_wrt_moon(time)
            j2000_to_iau_moon_transformation_mats_t = get_j2000_iau_moon_transform_matrix(time)

            if any(x is None for x in [lp_position_t, lp_vector_to_sun_in_lunar_frame_t, right_ascension_t, declination_t]):
                logging.warning(f"Invalid data at time {utc_time}, skipping...")

                lp_position_t = np.array([np.nan, np.nan, np.nan])
                lp_vector_to_sun_in_lunar_frame_t = np.array([np.nan, np.nan, np.nan])
                right_ascension_t = np.nan
                declination_t = np.nan

            lp_positions[t] = lp_position_t
            lp_vectors_to_sun[t] = lp_vector_to_sun_in_lunar_frame_t
            ra_dec_cartesian[t] = ra_dec_to_unit(right_ascension_t, declination_t) # TODO: Fix the type issue
            moon_vectors_to_sun[t] = moon_vector_to_sun_in_lunar_frame_t
            j2000_to_iau_moon_mats[t] = j2000_to_iau_moon_transformation_mats_t

        unit_lp_vectors_to_sun = lp_vectors_to_sun / np.linalg.norm(lp_vectors_to_sun, axis=1, keepdims=True)
        scd_to_j2000_transformation_mats = build_scd_to_j2000(ra_dec_cartesian, unit_lp_vectors_to_sun)
        scd_to_iau_moon_transformation_mats = np.einsum(
            'nij,njk->nik', j2000_to_iau_moon_mats, scd_to_j2000_transformation_mats
        )

        return CoordinateArrays(
            lp_positions=lp_positions,
            lp_vectors_to_sun=lp_vectors_to_sun,
            ra_dec_cartesian=ra_dec_cartesian,
            moon_vectors_to_sun=moon_vectors_to_sun,
            j2000_to_iau_moon_mats=j2000_to_iau_moon_mats,
            scd_to_iau_moon_mats=scd_to_iau_moon_transformation_mats,
        )

def project_magnetic_fields(
    flux_data: FluxData,
    coordinate_arrays: CoordinateArrays
) -> np.ndarray:
    """
    Project ER-frame magnetic field vectors into IAU_MOON frame as unit vectors.

    Returns an array of shape (N, 3) in IAU_MOON coordinates.
    """
    magnetic_field = flux_data.data[config.MAG_COLS].to_numpy()
    unit_magnetic_field = magnetic_field / np.linalg.norm(magnetic_field, axis=1, keepdims=True)
    projected_magnetic_field = np.einsum(
        "nij,nj->ni", coordinate_arrays.scd_to_iau_moon_mats, unit_magnetic_field
    )
    return projected_magnetic_field

def find_surface_intersection(
    coordinate_arrays: CoordinateArrays,
    projected_magnetic_field: np.ndarray
):
    """
    Compute ray–sphere intersections for LP positions along projected B-field.

    Returns (points, mask) from get_intersections_or_none_batch.
    """
    intersections = get_intersections_or_none_batch(
        pos=coordinate_arrays.lp_positions,
        direction=projected_magnetic_field,
        radius=config.LUNAR_RADIUS,
    )

    return intersections
