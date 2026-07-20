import logging

import numpy as np
import pandas as pd
import pytest
import spiceypy as spice

from src import config
from src.potential_mapper.coordinates import (
    CoordinateCalculator,
    find_surface_intersection_with_polarity,
    project_magnetic_fields,
)


class _DummyFluxData:
    def __init__(self, utc_rows: list[str]):
        self.data = pd.DataFrame({config.UTC_COLUMN: utc_rows})


def test_coordinate_calculator_masks_invalid_rows(monkeypatch, caplog):
    utc_rows = [
        "1998-01-01T00:00:00",  # valid
        "bad-et",  # spice.str2et failure
        "1998-01-01T00:00:02",  # zero-length LP->Sun vector
        "1998-01-01T00:00:03",  # invalid transform matrix
    ]
    times = {
        "1998-01-01T00:00:00": 0.0,
        "1998-01-01T00:00:02": 2.0,
        "1998-01-01T00:00:03": 3.0,
    }

    def fake_str2et(utc: str) -> float:
        if utc == "bad-et":
            raise ValueError("bad et")
        return times[utc]

    valid_position = np.array([1.0, 0.0, 0.0])
    valid_sun_vec = np.array([0.0, 1.0, 0.0])

    def fake_lp_position_batch(times_arr: np.ndarray) -> np.ndarray:
        # Return (N, 3)
        n = len(times_arr)
        return np.tile(valid_position, (n, 1))

    def fake_lp_vector_to_sun_eclipj2000_batch(
        times_arr: np.ndarray,
    ) -> np.ndarray:
        n = len(times_arr)
        res = np.tile(valid_sun_vec, (n, 1))
        # Order is preserved; row 2 exercises the zero-vector mask.
        if n > 2:
            res[2] = np.zeros(3)
        return res

    def fake_lp_vector_to_sun_lunar_batch(times_arr: np.ndarray) -> np.ndarray:
        return np.tile(valid_sun_vec, (len(times_arr), 1))

    def fake_moon_vector_to_sun_batch(times_arr: np.ndarray) -> np.ndarray:
        n = len(times_arr)
        return np.tile(valid_sun_vec, (n, 1))

    def fake_get_current_ra_dec_batch(times_arr, *args):
        n = len(times_arr)
        ra = np.full(n, 10.0)
        dec = np.full(n, 20.0)
        return ra, dec

    def fake_transform_matrix_batch(times_arr: np.ndarray) -> np.ndarray:
        n = len(times_arr)
        res = np.tile(np.eye(3), (n, 1, 1))
        # Row 3 (index 3) corresponds to time=3.0 -> invalid matrix
        if n > 3:
            res[3] = np.full((3, 3), np.nan)
        return res

    monkeypatch.setattr("src.potential_mapper.coordinates.spice.str2et", fake_str2et)
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_position_wrt_moon_batch",
        fake_lp_position_batch,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_vector_to_sun_in_lunar_frame_batch",
        fake_lp_vector_to_sun_lunar_batch,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_vector_to_sun_in_eclipj2000_batch",
        fake_lp_vector_to_sun_eclipj2000_batch,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_sun_vector_wrt_moon_batch",
        fake_moon_vector_to_sun_batch,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_current_ra_dec_batch",
        fake_get_current_ra_dec_batch,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates."
        "get_eclipj2000_iau_moon_transform_matrix_batch",
        fake_transform_matrix_batch,
    )

    flux_data = _DummyFluxData(utc_rows)
    calculator = CoordinateCalculator(
        et_spin=np.array([0.0, 1.0]),
        right_ascension=np.array([10.0, 20.0]),
        declination=np.array([30.0, 40.0]),
    )

    caplog.set_level(logging.WARNING)
    arrays = calculator.calculate_coordinate_transformation(flux_data)

    assert np.all(np.isfinite(arrays.lp_positions[0]))
    assert np.all(np.isnan(arrays.lp_positions[1:]))
    assert np.all(np.isfinite(arrays.eclipj2000_to_iau_moon_mats[0]))
    assert np.all(np.isnan(arrays.eclipj2000_to_iau_moon_mats[1:]))
    assert np.all(np.isfinite(arrays.spin_vectors_eclipj2000[0]))
    assert np.all(np.isnan(arrays.spin_vectors_eclipj2000[1:]))

    messages = "\n".join(caplog.messages)
    assert "Failed to convert UTC bad-et to ET" in messages


@pytest.mark.skip_ci
def test_corrected_geometry_matches_published_pds_footprint() -> None:
    """Cross-check one LP ER footprint against the official Level-2 product."""
    from src.potential_mapper.spice import load_spice_files
    from src.utils.attitude import load_attitude_data

    if not (config.DATA_DIR / config.ATTITUDE_FILE).exists():
        pytest.skip("Local LP attitude table is unavailable")
    if not config.SPICE_KERNELS_DIR.exists():
        pytest.skip("Local SPICE kernels are unavailable")

    spice.kclear()
    load_spice_files()
    attitude = load_attitude_data(config.DATA_DIR / config.ATTITUDE_FILE)
    if any(value is None for value in attitude):
        pytest.skip("Local LP attitude table could not be loaded")
    et_spin, ra, dec = attitude
    assert et_spin is not None and ra is not None and dec is not None

    # The PDS Level-2 220 eV record at 1998-06-08/00:27:41 reports
    # longitude 133.120 E, latitude -23.970. Its nearest 3-D product MAG
    # vector is the 1998-06-08T00:27:40 record.
    flux_data = _DummyFluxData(["1998-06-08T00:27:41"])
    flux_data.data[config.MAG_COLS] = np.array([[9.724349, -1.277785, 1.788874]])
    calculator = CoordinateCalculator(et_spin, ra, dec)
    arrays = calculator.calculate_coordinate_transformation(flux_data)
    projected = project_magnetic_fields(flux_data, arrays)
    points, connected, _ = find_surface_intersection_with_polarity(arrays, projected)

    assert connected.tolist() == [True]
    point = points[0] / np.linalg.norm(points[0])
    actual_lat = np.rad2deg(np.arcsin(point[2]))
    actual_lon = np.rad2deg(np.arctan2(point[1], point[0])) % 360.0
    expected_lat = -23.970
    expected_lon = 133.120
    cosine = np.sin(np.deg2rad(actual_lat)) * np.sin(np.deg2rad(expected_lat)) + np.cos(
        np.deg2rad(actual_lat)
    ) * np.cos(np.deg2rad(expected_lat)) * np.cos(np.deg2rad(actual_lon - expected_lon))
    separation = np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))

    assert separation < 1.0
