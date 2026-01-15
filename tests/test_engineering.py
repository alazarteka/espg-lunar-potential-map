"""Tests for engineering analysis module."""

import numpy as np
import pytest

from src.engineering.analysis import (
    DEFAULT_CURRENT_DENSITY,
    GlobalStats,
    SiteStats,
    compute_global_stats,
    extract_site_stats,
)
from src.engineering.sites import Site
from src.temporal.dataset import TemporalDataset


@pytest.fixture
def mock_dataset():
    """Create a synthetic dataset with known coefficients."""
    # lmax=1
    # U(t) = a_00 Y_00 + ...
    # Y_00 = 1/sqrt(4pi) approx 0.282
    # Let's set a_00 = 100.0 (constant potential)
    # Then U = 100 * 0.282 = 28.2 V

    lmax = 1
    n_coeffs = 4
    n_times = 10

    # Constant 100V everywhere (if only Y00)
    # But wait, Y00 is constant.
    # coeffs shape: (n_times, n_coeffs)
    coeffs = np.zeros((n_times, n_coeffs), dtype=np.complex128)

    # a_00 = 100.0 for all times
    coeffs[:, 0] = 100.0 / 0.28209479177387814  # so result is ~100

    # Correctly create datetime array
    times = np.arange("2020-01-01", "2020-01-11", dtype="datetime64[D]")

    return TemporalDataset(
        times=times,
        lmax=lmax,
        coeffs=coeffs,
    )


def test_compute_global_stats(mock_dataset):
    """Test global stats computation with constant field."""
    stats = compute_global_stats(
        mock_dataset,
        lat_steps=10,
        lon_steps=20,
    )

    assert isinstance(stats, GlobalStats)
    assert stats.latitudes.shape == (10,)
    assert stats.longitudes.shape == (20,)

    # Check mean potential is approx 100
    # Allow some numerical slack
    assert np.allclose(stats.mean_potential, 100.0, atol=1e-5)

    # Check power
    expected_power = 100.0 * DEFAULT_CURRENT_DENSITY
    assert np.allclose(stats.mean_power, expected_power, atol=1e-5)

    # Check thresholds
    # 100 < 500, so fractions should be 0
    assert np.all(stats.frac_500V == 0.0)


def test_extract_site_stats(mock_dataset):
    """Test site stats extraction."""
    site = Site(name="Test", lat=0.0, lon=0.0, description="Test Site")

    stats = extract_site_stats(mock_dataset, site)

    assert isinstance(stats, SiteStats)
    assert stats.site == site
    assert np.isclose(stats.mean_potential, 100.0)
    assert stats.frac_500V == 0.0
    assert stats.risk_assessment == "Low Resource / Low Risk"  # < 1mW/m2 power


def test_high_potential_case():
    """Test case with high potential to trigger thresholds."""
    lmax = 0
    n_times = 5
    coeffs = np.zeros((n_times, 1), dtype=np.complex128)
    # Make potential 2500 V
    # 2500 / Y00
    coeffs[:, 0] = 2500.0 / 0.28209479177387814

    times = np.arange("2020-01-01", "2020-01-06", dtype="datetime64[D]")

    dataset = TemporalDataset(
        times=times,
        lmax=lmax,
        coeffs=coeffs,
    )

    site = Site(name="High", lat=0, lon=0, description="High Voltage")
    stats = extract_site_stats(dataset, site)

    assert stats.mean_potential > 2000.0
    assert stats.frac_2kV == 1.0
    assert stats.frac_1kV == 1.0
    assert (
        "High Resource" in stats.risk_assessment or "High Risk" in stats.risk_assessment
    )
