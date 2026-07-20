from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.visualization.loaders import _discover_date_files, load_measurements
from src.visualization.utils import date_range


def test_date_range_valid():
    start = date(2023, 1, 1)
    end = date(2023, 1, 3)
    days = date_range(start, end)
    assert len(days) == 3
    assert days[0] == start
    assert days[-1] == end


def test_date_range_single_day():
    start = date(2023, 1, 1)
    days = date_range(start, start)
    assert len(days) == 1
    assert days[0] == start


def test_date_range_invalid():
    start = date(2023, 1, 2)
    end = date(2023, 1, 1)
    with pytest.raises(ValueError):
        date_range(start, end)


def _write_potential_cache(path: Path) -> None:
    np.savez(
        path,
        rows_projection_latitude=np.array([10.0, np.nan]),
        rows_projection_longitude=np.array([20.0, 30.0]),
        rows_projected_potential=np.array([-5.0, -6.0]),
        rows_projection_in_sun=np.array([1, 0]),
    )


def test_load_measurements_discovers_canonical_daily_batch_file(tmp_path: Path) -> None:
    cache_file = tmp_path / "nested" / "potential_batch_1998_04_29.npz"
    cache_file.parent.mkdir()
    _write_potential_cache(cache_file)

    lats, lons, potentials, in_sun = load_measurements(
        tmp_path, date(1998, 4, 29), date(1998, 4, 29)
    )

    np.testing.assert_allclose(lats, [10.0])
    np.testing.assert_allclose(lons, [20.0])
    np.testing.assert_allclose(potentials, [-5.0])
    assert in_sun.tolist() == [True]


def test_discover_date_files_rejects_ambiguous_daily_candidates(tmp_path: Path) -> None:
    day = date(1998, 4, 29)
    legacy = tmp_path / "legacy" / "3D980429.npz"
    canonical = tmp_path / "batch" / "potential_batch_1998_04_29.npz"
    legacy.parent.mkdir()
    canonical.parent.mkdir()
    _write_potential_cache(legacy)
    _write_potential_cache(canonical)

    with pytest.raises(
        ValueError, match="Ambiguous potential cache files for 1998-04-29"
    ):
        _discover_date_files(tmp_path, day, day)
