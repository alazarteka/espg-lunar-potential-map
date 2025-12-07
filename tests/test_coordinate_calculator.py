import logging

import numpy as np
import pandas as pd

from src import config
from src.potential_mapper.coordinates import CoordinateCalculator


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

    def fake_lp_vector_to_sun_batch(times_arr: np.ndarray) -> np.ndarray:
        n = len(times_arr)
        res = np.tile(valid_sun_vec, (n, 1))
        # Row 2 (index 2) corresponds to time=2.0 -> zero vector
        # But wait, row 1 is bad-et, so time is NaN.
        # Row 2 is time=2.0.
        # We need to find which index corresponds to time=2.0
        # times_arr will have NaNs for bad-et.

        # Logic:
        # Row 0: time=0.0
        # Row 1: time=NaN
        # Row 2: time=2.0
        # Row 3: time=3.0

        # We can just use the index if we assume order is preserved (it is)
        if n > 2:
             res[2] = np.zeros(3)
        return res

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

    monkeypatch.setattr(
        "src.potential_mapper.coordinates.spice.str2et", fake_str2et
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_position_wrt_moon_batch",
        fake_lp_position_batch,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_vector_to_sun_in_lunar_frame_batch",
        fake_lp_vector_to_sun_batch,
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
        "src.potential_mapper.coordinates.get_j2000_iau_moon_transform_matrix_batch",
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
    assert np.all(np.isfinite(arrays.j2000_to_iau_moon_mats[0]))
    assert np.all(np.isnan(arrays.j2000_to_iau_moon_mats[1:]))

    messages = "\n".join(caplog.messages)
    assert "Failed to convert UTC bad-et to ET" in messages
    # We no longer log per-row warnings for performance reasons
    # assert "Invalid lp_to_sun vector" in messages
    # assert "Invalid J2000->IAU_MOON matrix" in messages
