"""Tests for src.potential_mapper.results.PlasmaEnvironment classification."""

import numpy as np

from src.potential_mapper.results import PlasmaEnvironment as PE


def test_shadow_is_wake_regardless_of_temperature() -> None:
    f = PE.from_temperature_and_illumination
    assert f(200.0, projection_in_sun=False) == PE.WAKE
    assert f(10.0, projection_in_sun=False) == PE.WAKE
    assert f(0.0, projection_in_sun=False) == PE.WAKE
    assert f(np.nan, projection_in_sun=False) == PE.WAKE


def test_invalid_temperature_is_unknown() -> None:
    f = PE.from_temperature_and_illumination
    assert f(0.0, projection_in_sun=True) == PE.UNKNOWN
    assert f(-5.0, projection_in_sun=True) == PE.UNKNOWN
    assert f(np.nan, projection_in_sun=True) == PE.UNKNOWN


def test_sunlit_temperature_bands() -> None:
    f = PE.from_temperature_and_illumination
    assert f(10.0, projection_in_sun=True) == PE.SOLAR_WIND
    assert f(50.0, projection_in_sun=True) == PE.MAGNETOSHEATH
    assert f(100.0, projection_in_sun=True) == PE.TAIL_LOBES
    assert f(300.0, projection_in_sun=True) == PE.PLASMA_SHEET


def test_band_boundaries_are_half_open() -> None:
    """Bands are [.,30) SW, [30,80) MS, [80,150) TL, [150,inf) PS."""
    f = PE.from_temperature_and_illumination
    assert f(29.999, projection_in_sun=True) == PE.SOLAR_WIND
    assert f(30.0, projection_in_sun=True) == PE.MAGNETOSHEATH
    assert f(80.0, projection_in_sun=True) == PE.TAIL_LOBES
    assert f(150.0, projection_in_sun=True) == PE.PLASMA_SHEET
