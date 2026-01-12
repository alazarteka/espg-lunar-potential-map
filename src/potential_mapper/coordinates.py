import logging
from dataclasses import dataclass

import numpy as np
import spiceypy as spice

import src.config as config
from src.flux import FluxData
from src.utils.attitude import get_current_ra_dec_batch
from src.utils.coordinates import build_scd_to_j2000, ra_dec_to_unit
from src.utils.geometry import (
    get_connections_and_polarity_batch,
    get_intersections_or_none_batch,
)
from src.utils.spice_ops import (
    get_j2000_iau_moon_transform_matrix_batch,
    get_lp_position_wrt_moon_batch,
    get_lp_vector_to_sun_in_lunar_frame_batch,
    get_sun_vector_wrt_moon_batch,
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
        # 1. Convert UTC to ET
        # spice.str2et is scalar, so we loop.
        # We can use a list comprehension which is reasonably fast for string parsing.
        utc_times = flux_data.data[config.UTC_COLUMN].to_numpy()
        n_points = len(utc_times)

        et_times = np.full(n_points, np.nan)
        for i, utc in enumerate(utc_times):
            try:
                et_times[i] = spice.str2et(utc)
            except Exception as exc:
                logging.warning(
                    "Failed to convert UTC %s to ET (%s); marking row invalid.",
                    utc,
                    exc,
                )
                # Leave as NaN

        # 2. Batch SPICE calls
        # These functions handle NaNs in et_times gracefully (returning NaNs)
        lp_positions = get_lp_position_wrt_moon_batch(et_times)
        lp_vectors_to_sun = get_lp_vector_to_sun_in_lunar_frame_batch(et_times)
        moon_vectors_to_sun = get_sun_vector_wrt_moon_batch(et_times)
        j2000_to_iau_moon_mats = get_j2000_iau_moon_transform_matrix_batch(et_times)

        # 3. Attitude lookup
        ra_vals, dec_vals = get_current_ra_dec_batch(
            et_times, self.et_spin, self.right_ascension, self.declination
        )

        # 4. Validate and filter
        # We need to identify rows where any component is invalid (NaN)
        # Check finiteness
        valid_mask = (
            np.isfinite(et_times)
            & np.all(np.isfinite(lp_positions), axis=1)
            & np.all(np.isfinite(lp_vectors_to_sun), axis=1)
            & np.all(np.isfinite(moon_vectors_to_sun), axis=1)
            & np.all(np.isfinite(j2000_to_iau_moon_mats.reshape(n_points, -1)), axis=1)
            & np.isfinite(ra_vals)
            & np.isfinite(dec_vals)
        )

        # Also check vector norms > 0
        lp_sun_norms = np.linalg.norm(lp_vectors_to_sun, axis=1)
        valid_mask &= lp_sun_norms > 0

        # Apply mask to invalidate bad rows (set to NaN)
        # Note: The original code skipped rows, effectively leaving them as NaNs
        # (initialized to np.full(..., np.nan)).
        # So we just need to ensure invalid rows remain NaNs.
        # The batch functions already return NaNs on error, but we might have partial failures.
        # We explicitly set invalid rows to NaN to be safe and consistent.

        if not np.all(valid_mask):
            lp_positions[~valid_mask] = np.nan
            lp_vectors_to_sun[~valid_mask] = np.nan
            moon_vectors_to_sun[~valid_mask] = np.nan
            j2000_to_iau_moon_mats[~valid_mask] = np.nan
            ra_vals[~valid_mask] = np.nan
            dec_vals[~valid_mask] = np.nan

        # 5. Derived quantities
        ra_dec_cartesian = np.full((n_points, 3), np.nan)
        # Only compute for valid rows to avoid warnings/errors
        if np.any(valid_mask):
            ra_dec_cartesian[valid_mask] = ra_dec_to_unit(
                ra_vals[valid_mask], dec_vals[valid_mask]
            )

        # Unit vectors to sun
        unit_lp_vectors_to_sun = np.full_like(lp_vectors_to_sun, np.nan)
        # Safe division
        safe_norms = np.where(lp_sun_norms > 0, lp_sun_norms, 1.0)
        np.divide(
            lp_vectors_to_sun,
            safe_norms[:, None],
            out=unit_lp_vectors_to_sun,
            where=valid_mask[:, None],
        )

        # SCD to J2000
        scd_to_j2000_transformation_mats = np.full((n_points, 3, 3), np.nan)
        if np.any(valid_mask):
            scd_to_j2000_transformation_mats[valid_mask] = build_scd_to_j2000(
                ra_dec_cartesian[valid_mask], unit_lp_vectors_to_sun[valid_mask]
            )

        # SCD to IAU_MOON
        # Matrix multiplication: M_iau_j2000 @ M_j2000_scd
        # j2000_to_iau_moon_mats is (N, 3, 3)
        # scd_to_j2000_transformation_mats is (N, 3, 3)
        # Result (N, 3, 3)
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


def find_surface_intersection_with_polarity(
    coordinate_arrays: CoordinateArrays, projected_magnetic_field: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute +/- ray–sphere intersections for LP positions along projected B-field.

    Returns (points, mask, polarity) where polarity is +1 for +B, -1 for -B.
    """
    points, mask, polarity = get_connections_and_polarity_batch(
        pos=coordinate_arrays.lp_positions,
        direction=projected_magnetic_field,
        radius=config.LUNAR_RADIUS,
    )
    return points, mask, polarity
