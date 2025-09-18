import logging
from dataclasses import dataclass

import numpy as np
import spiceypy as spice

import src.config as config
from src.flux import FluxData
from src.utils.attitude import get_current_ra_dec
from src.utils.coordinates import build_scd_to_j2000, ra_dec_to_unit
from src.utils.geometry import get_intersections_or_none_batch
from src.utils.spice_ops import (
    get_j2000_iau_moon_transform_matrix,
    get_lp_position_wrt_moon,
    get_lp_vector_to_sun_in_lunar_frame,
    get_sun_vector_wrt_moon,
)


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

    def __init__(
        self, et_spin: np.ndarray, right_ascension: np.ndarray, declination: np.ndarray
    ):
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

    def calculate_coordinate_transformation(
        self, flux_data: FluxData
    ) -> CoordinateArrays:
        """
        Build all geometry arrays for each row in `flux_data`.

        Returns a CoordinateArrays with per-row positions, Sun vectors, and
        transformation matrices sufficient to project vectors between frames.
        """
        n_points = len(flux_data.data)

        lp_positions = np.full((n_points, 3), np.nan)
        lp_vectors_to_sun = np.full((n_points, 3), np.nan)
        ra_dec_cartesian = np.full((n_points, 3), np.nan)
        moon_vectors_to_sun = np.full((n_points, 3), np.nan)
        j2000_to_iau_moon_mats = np.full((n_points, 3, 3), np.nan)

        for t, utc_time in enumerate(flux_data.data[config.UTC_COLUMN]):
            try:
                time = spice.str2et(utc_time)
            except Exception as exc:
                logging.warning(
                    "Failed to convert UTC %s to ET (%s); marking row invalid.",
                    utc_time,
                    exc,
                )
                continue

            lp_position_t = get_lp_position_wrt_moon(time)
            lp_vector_to_sun_in_lunar_frame_t = get_lp_vector_to_sun_in_lunar_frame(
                time
            )
            moon_vector_to_sun_in_lunar_frame_t = get_sun_vector_wrt_moon(time)
            ra_dec_pair = get_current_ra_dec(
                time, self.et_spin, self.right_ascension, self.declination
            )
            if ra_dec_pair is None:
                ra_dec_pair = (None, None)
            right_ascension_t, declination_t = ra_dec_pair

            if any(
                x is None
                for x in [
                    lp_position_t,
                    lp_vector_to_sun_in_lunar_frame_t,
                    moon_vector_to_sun_in_lunar_frame_t,
                    right_ascension_t,
                    declination_t,
                ]
            ):
                logging.warning("Incomplete geometry at %s; skipping row.", utc_time)
                continue

            if not np.isfinite([right_ascension_t, declination_t]).all():
                logging.warning("Non-finite attitude values at %s; skipping row.", utc_time)
                continue

            vectors = (
                ("lp_position", lp_position_t),
                ("lp_to_sun", lp_vector_to_sun_in_lunar_frame_t),
                ("moon_to_sun", moon_vector_to_sun_in_lunar_frame_t),
            )
            invalid_vector = False
            for name, vec in vectors:
                if not np.isfinite(vec).all() or np.linalg.norm(vec) <= 0.0:
                    logging.warning("Invalid %s vector at %s; skipping row.", name, utc_time)
                    invalid_vector = True
                    break
            if invalid_vector:
                continue

            j2m = get_j2000_iau_moon_transform_matrix(time)
            if not np.isfinite(j2m).all():
                logging.warning("Invalid J2000->IAU_MOON matrix at %s; skipping row.", utc_time)
                continue

            lp_positions[t] = lp_position_t
            lp_vectors_to_sun[t] = lp_vector_to_sun_in_lunar_frame_t
            ra_dec_cartesian[t] = ra_dec_to_unit(right_ascension_t, declination_t)
            moon_vectors_to_sun[t] = moon_vector_to_sun_in_lunar_frame_t
            j2000_to_iau_moon_mats[t] = j2m

        norms = np.linalg.norm(lp_vectors_to_sun, axis=1)
        safe_norms = np.where(norms > 0.0, norms, np.nan)
        unit_lp_vectors_to_sun = lp_vectors_to_sun / safe_norms[:, None]
        scd_to_j2000_transformation_mats = build_scd_to_j2000(
            ra_dec_cartesian, unit_lp_vectors_to_sun
        )
        scd_to_iau_moon_transformation_mats = np.einsum(
            "nij,njk->nik", j2000_to_iau_moon_mats, scd_to_j2000_transformation_mats
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
    flux_data: FluxData, coordinate_arrays: CoordinateArrays
) -> np.ndarray:
    """
    Project ER-frame magnetic field vectors into IAU_MOON frame as unit vectors.

    Returns an array of shape (N, 3) in IAU_MOON coordinates.
    """
    magnetic_field = flux_data.data[config.MAG_COLS].to_numpy(dtype=float)
    finite_mask = np.isfinite(magnetic_field).all(axis=1)
    norms = np.linalg.norm(magnetic_field, axis=1)
    valid = finite_mask & (norms > 0.0)

    if np.any(~valid):
        logging.warning(
            "Magnetic field invalid for %d of %d rows; projection set to NaN.",
            np.count_nonzero(~valid),
            magnetic_field.shape[0],
        )

    unit_magnetic_field = np.full_like(magnetic_field, np.nan)
    unit_magnetic_field[valid] = magnetic_field[valid] / norms[valid, None]

    projected_magnetic_field = np.einsum(
        "nij,nj->ni", coordinate_arrays.scd_to_iau_moon_mats, unit_magnetic_field
    )
    return projected_magnetic_field


def find_surface_intersection(
    coordinate_arrays: CoordinateArrays, projected_magnetic_field: np.ndarray
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
