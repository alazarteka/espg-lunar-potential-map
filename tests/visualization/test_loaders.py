"""Tests for visualization data loading utilities."""

from datetime import date
from pathlib import Path

import numpy as np
import pytest

from src.visualization.loaders import (
    _date_range,
    load_measurements,
)


class TestDateRange:
    """Tests for date range generation."""

    def test_single_day_range(self):
        """Test range with start == end returns single day."""
        start = date(2020, 1, 1)
        end = date(2020, 1, 1)

        result = _date_range(start, end)

        assert len(result) == 1
        assert result[0] == start

    def test_multi_day_range(self):
        """Test range with multiple days."""
        start = date(2020, 1, 1)
        end = date(2020, 1, 5)

        result = _date_range(start, end)

        assert len(result) == 5
        assert result[0] == date(2020, 1, 1)
        assert result[1] == date(2020, 1, 2)
        assert result[4] == date(2020, 1, 5)

    def test_month_boundary(self):
        """Test range crossing month boundary."""
        start = date(2020, 1, 30)
        end = date(2020, 2, 2)

        result = _date_range(start, end)

        assert len(result) == 4
        assert result[0] == date(2020, 1, 30)
        assert result[1] == date(2020, 1, 31)
        assert result[2] == date(2020, 2, 1)
        assert result[3] == date(2020, 2, 2)

    def test_year_boundary(self):
        """Test range crossing year boundary."""
        start = date(2019, 12, 30)
        end = date(2020, 1, 2)

        result = _date_range(start, end)

        assert len(result) == 4
        assert result[0] == date(2019, 12, 30)
        assert result[-1] == date(2020, 1, 2)

    def test_end_before_start_raises_error(self):
        """Test that end < start raises ValueError."""
        start = date(2020, 1, 5)
        end = date(2020, 1, 1)

        with pytest.raises(ValueError, match="end must be >= start"):
            _date_range(start, end)

    def test_leap_year_february(self):
        """Test range in leap year February."""
        start = date(2020, 2, 28)  # 2020 is leap year
        end = date(2020, 3, 1)

        result = _date_range(start, end)

        assert len(result) == 3
        assert date(2020, 2, 29) in result


class TestLoadMeasurements:
    """Tests for loading measurement data from cache files."""

    @pytest.fixture
    def cache_dir(self, tmp_path):
        """Create a temporary cache directory."""
        cache = tmp_path / "cache"
        cache.mkdir()
        return cache

    @pytest.fixture
    def sample_npz_file(self, cache_dir):
        """Create a sample NPZ file with measurement data."""
        n_points = 100
        data = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.random.randint(0, 2, n_points),
        }

        # Create file for Jan 1, 1998
        filepath = cache_dir / "3D980101.npz"
        np.savez(filepath, **data)
        return filepath, data

    def test_load_single_file(self, cache_dir, sample_npz_file):
        """Test loading measurements from a single file."""
        filepath, original_data = sample_npz_file

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 1),
        )

        # All data should be valid (no NaNs in sample)
        assert len(lats) == 100
        assert len(lons) == 100
        assert len(pots) == 100
        assert len(in_sun) == 100

        # Verify data matches original
        np.testing.assert_array_almost_equal(lats, original_data["rows_projection_latitude"])
        np.testing.assert_array_almost_equal(lons, original_data["rows_projection_longitude"])
        np.testing.assert_array_almost_equal(pots, original_data["rows_projected_potential"])

    def test_load_multiple_files(self, cache_dir):
        """Test loading and concatenating data from multiple files."""
        # Create two files
        n_points = 50

        data1 = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.ones(n_points, dtype=int),
        }
        np.savez(cache_dir / "3D980101.npz", **data1)

        data2 = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.zeros(n_points, dtype=int),
        }
        np.savez(cache_dir / "3D980102.npz", **data2)

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 2),
        )

        # Should have data from both files
        assert len(lats) == 100
        assert len(lons) == 100
        assert len(pots) == 100
        assert len(in_sun) == 100

    def test_load_filters_invalid_values(self, cache_dir):
        """Test that NaN and inf values are filtered out."""
        n_points = 100
        data = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.ones(n_points, dtype=int),
        }

        # Inject some invalid values
        data["rows_projection_latitude"][10] = np.nan
        data["rows_projected_potential"][20] = np.inf
        data["rows_projection_longitude"][30] = -np.inf

        np.savez(cache_dir / "3D980101.npz", **data)

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 1),
        )

        # Should filter out invalid rows
        assert len(lats) < 100
        # Verify no NaN or inf values remain
        assert np.all(np.isfinite(lats))
        assert np.all(np.isfinite(lons))
        assert np.all(np.isfinite(pots))

    def test_load_handles_bool_in_sun_values(self, cache_dir):
        """Test that in_sun can be loaded as bool or int."""
        n_points = 50
        data = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.array([True, False] * 25),  # Boolean
        }

        np.savez(cache_dir / "3D980101.npz", **data)

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 1),
        )

        assert in_sun.dtype == bool
        assert len(in_sun) == 50

    def test_load_no_matching_files_raises_error(self, cache_dir):
        """Test that missing files raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="No cache files found"):
            load_measurements(
                cache_dir,
                start_day=date(1999, 1, 1),
                end_day=date(1999, 1, 1),
            )

    def test_load_partial_date_range(self, cache_dir):
        """Test loading when only some files exist in date range."""
        n_points = 50
        data = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.ones(n_points, dtype=int),
        }

        # Only create file for Jan 1, request Jan 1-3
        np.savez(cache_dir / "3D980101.npz", **data)

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 3),  # Files for Jan 2-3 don't exist
        )

        # Should load available file
        assert len(lats) == 50

    def test_load_empty_after_filtering(self, cache_dir):
        """Test loading when all data is filtered out."""
        n_points = 10
        data = {
            "rows_projection_latitude": np.full(n_points, np.nan),
            "rows_projection_longitude": np.full(n_points, np.nan),
            "rows_projected_potential": np.full(n_points, np.nan),
            "rows_projection_in_sun": np.ones(n_points, dtype=int),
        }

        np.savez(cache_dir / "3D980101.npz", **data)

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 1),
        )

        # Should return empty arrays
        assert len(lats) == 0
        assert len(lons) == 0
        assert len(pots) == 0
        assert len(in_sun) == 0

    def test_load_with_nested_directory_structure(self, tmp_path):
        """Test that rglob finds files in nested directories."""
        # Create nested directory structure
        nested_dir = tmp_path / "cache" / "subdir" / "nested"
        nested_dir.mkdir(parents=True)

        n_points = 30
        data = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points),
            "rows_projection_in_sun": np.ones(n_points, dtype=int),
        }

        # Save in nested directory
        np.savez(nested_dir / "3D980101.npz", **data)

        lats, lons, pots, in_sun = load_measurements(
            tmp_path / "cache",
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 1),
        )

        # Should find file in nested directory
        assert len(lats) == 30

    def test_load_date_formatting(self, cache_dir):
        """Test that date formatting matches expected pattern."""
        # Test various dates
        dates_and_patterns = [
            (date(1998, 1, 1), "3D980101.npz"),
            (date(2000, 12, 31), "3D001231.npz"),
            (date(1999, 7, 15), "3D990715.npz"),
        ]

        n_points = 20
        for test_date, expected_pattern in dates_and_patterns:
            data = {
                "rows_projection_latitude": np.random.uniform(-90, 90, n_points),
                "rows_projection_longitude": np.random.uniform(-180, 180, n_points),
                "rows_projected_potential": np.random.uniform(-5, 5, n_points),
                "rows_projection_in_sun": np.ones(n_points, dtype=int),
            }

            filepath = cache_dir / expected_pattern
            np.savez(filepath, **data)

            lats, lons, pots, in_sun = load_measurements(
                cache_dir,
                start_day=test_date,
                end_day=test_date,
            )

            assert len(lats) == n_points

            # Clean up for next iteration
            filepath.unlink()

    def test_load_preserves_data_types(self, cache_dir):
        """Test that loaded data has correct dtypes."""
        n_points = 50
        data = {
            "rows_projection_latitude": np.random.uniform(-90, 90, n_points).astype(np.float64),
            "rows_projection_longitude": np.random.uniform(-180, 180, n_points).astype(np.float64),
            "rows_projected_potential": np.random.uniform(-5, 5, n_points).astype(np.float32),
            "rows_projection_in_sun": np.random.randint(0, 2, n_points, dtype=np.int32),
        }

        np.savez(cache_dir / "3D980101.npz", **data)

        lats, lons, pots, in_sun = load_measurements(
            cache_dir,
            start_day=date(1998, 1, 1),
            end_day=date(1998, 1, 1),
        )

        # Verify dtypes
        assert lats.dtype in (np.float32, np.float64)
        assert lons.dtype in (np.float32, np.float64)
        assert pots.dtype in (np.float32, np.float64)
        assert in_sun.dtype == bool  # Should be converted to bool
