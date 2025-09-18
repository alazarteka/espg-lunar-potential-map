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

    def fake_lp_position(time: float) -> np.ndarray:
        return valid_position

    def fake_lp_vector_to_sun(time: float) -> np.ndarray:
        if time == times["1998-01-01T00:00:02"]:
            return np.zeros(3)
        return valid_sun_vec

    def fake_moon_vector_to_sun(time: float) -> np.ndarray:
        return valid_sun_vec

    def fake_get_current_ra_dec(*_args, **_kwargs):
        return 10.0, 20.0

    def fake_transform_matrix(time: float) -> np.ndarray:
        if time == times["1998-01-01T00:00:03"]:
            return np.full((3, 3), np.nan)
        return np.eye(3)

    monkeypatch.setattr(
        "src.potential_mapper.coordinates.spice.str2et", fake_str2et
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_position_wrt_moon",
        fake_lp_position,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_lp_vector_to_sun_in_lunar_frame",
        fake_lp_vector_to_sun,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_sun_vector_wrt_moon",
        fake_moon_vector_to_sun,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_current_ra_dec",
        fake_get_current_ra_dec,
    )
    monkeypatch.setattr(
        "src.potential_mapper.coordinates.get_j2000_iau_moon_transform_matrix",
        fake_transform_matrix,
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
    assert "Invalid lp_to_sun vector" in messages
    assert "Invalid J2000->IAU_MOON matrix" in messages
