"""Tests for src/utils/attitude.py"""

import numpy as np
import pytest

from src.utils.attitude import get_current_ra_dec, get_current_ra_dec_batch


def test_get_current_ra_dec_batch_matches_scalar():
    """Test that batch version produces same results as scalar calls."""

    # Setup test data
    et_spin = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
    ra_vals = np.array([10.0, 15.0, 20.0, 25.0, 30.0])
    dec_vals = np.array([5.0, 10.0, 15.0, 20.0, 25.0])

    # Query times between attitude records.
    query_times = np.array([5.0, 15.0, 25.0, 35.0])

    # Batch call
    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    # Scalar calls
    ra_scalar = []
    dec_scalar = []
    for t in query_times:
        ra, dec = get_current_ra_dec(t, et_spin, ra_vals, dec_vals)
        ra_scalar.append(ra if ra is not None else np.nan)
        dec_scalar.append(dec if dec is not None else np.nan)

    ra_scalar = np.array(ra_scalar)
    dec_scalar = np.array(dec_scalar)

    np.testing.assert_allclose(ra_batch, [12.5, 17.5, 22.5, 27.5])
    np.testing.assert_allclose(dec_batch, [7.5, 12.5, 17.5, 22.5])
    np.testing.assert_allclose(ra_batch, ra_scalar)
    np.testing.assert_allclose(dec_batch, dec_scalar)


def test_get_current_ra_dec_batch_handles_out_of_bounds():
    """Test that batch version returns NaN for out-of-bounds queries."""

    et_spin = np.array([10.0, 20.0, 30.0])
    ra_vals = np.array([100.0, 200.0, 300.0])
    dec_vals = np.array([10.0, 20.0, 30.0])

    # Query times outside the covered interpolation interval are invalid.
    query_times = np.array(
        [
            5.0,  # Before first -> idx=0 -> invalid
            15.0,  # Between 0,1 -> idx=1 -> valid
            25.0,  # Between 1,2 -> idx=2 -> valid
            35.0,  # After last -> idx=3 -> invalid
        ]
    )

    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    # First should be NaN
    assert np.isnan(ra_batch[0])
    assert np.isnan(dec_batch[0])

    # Second and third should be valid
    assert np.isfinite(ra_batch[1])
    assert np.isfinite(dec_batch[1])
    assert np.isfinite(ra_batch[2])
    assert np.isfinite(dec_batch[2])

    # Fourth should be NaN
    assert np.isnan(ra_batch[3])
    assert np.isnan(dec_batch[3])


def test_get_current_ra_dec_batch_edge_cases():
    """Test edge cases for batch RA/DEC lookup."""

    et_spin = np.array([0.0, 10.0])
    ra_vals = np.array([50.0, 60.0])
    dec_vals = np.array([5.0, 6.0])

    # Edge case: exactly at boundary
    query_times = np.array([10.0])
    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    assert ra_batch[0] == 60.0
    assert dec_batch[0] == 6.0

    # Just before boundary should be valid
    query_times = np.array([9.999])
    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    assert np.isfinite(ra_batch[0])
    assert ra_batch[0] == pytest.approx(59.999)
    assert dec_batch[0] == pytest.approx(5.9999)


def test_get_current_ra_dec_batch_matches_linear_interpolation():
    """Verify interpolation against NumPy for non-wrapping RA."""

    # Large dataset
    n_points = 100
    et_spin = np.linspace(0, 1000, n_points)
    ra_vals = np.linspace(0, 360, n_points)
    dec_vals = np.linspace(-90, 90, n_points)

    # Random query times
    np.random.seed(42)
    query_times = np.random.uniform(-100, 1100, 50)

    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    valid_mask = (query_times >= et_spin[0]) & (query_times <= et_spin[-1])
    expected_ra = np.interp(query_times[valid_mask], et_spin, ra_vals)
    expected_dec = np.interp(query_times[valid_mask], et_spin, dec_vals)

    valid_index = 0
    for i, _time in enumerate(query_times):
        if valid_mask[i]:
            assert ra_batch[i] == pytest.approx(expected_ra[valid_index])
            assert dec_batch[i] == pytest.approx(expected_dec[valid_index])
            valid_index += 1
        else:
            assert np.isnan(ra_batch[i])
            assert np.isnan(dec_batch[i])


def test_get_current_ra_dec_batch_interpolates_ra_across_zero() -> None:
    et_spin = np.array([0.0, 10.0])
    ra_vals = np.array([350.0, 10.0])
    dec_vals = np.array([-5.0, 5.0])

    ra, dec = get_current_ra_dec_batch(
        np.array([0.0, 5.0, 10.0]), et_spin, ra_vals, dec_vals
    )

    np.testing.assert_allclose(ra, [350.0, 0.0, 10.0])
    np.testing.assert_allclose(dec, [-5.0, 0.0, 5.0])


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_get_current_ra_dec_batch_randomized(seed):
    """Randomized testing of batch vs scalar equivalence."""

    np.random.seed(seed)

    # Random attitude data
    n_attitude = np.random.randint(10, 50)
    et_spin = np.sort(np.random.uniform(0, 1000, n_attitude))
    ra_vals = np.random.uniform(0, 360, n_attitude)
    dec_vals = np.random.uniform(-90, 90, n_attitude)

    # Random queries
    n_queries = np.random.randint(5, 20)
    query_times = np.random.uniform(-100, 1100, n_queries)

    # Batch
    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    # Scalar
    ra_scalar = []
    dec_scalar = []
    for t in query_times:
        ra, dec = get_current_ra_dec(t, et_spin, ra_vals, dec_vals)
        ra_scalar.append(ra if ra is not None else np.nan)
        dec_scalar.append(dec if dec is not None else np.nan)

    ra_scalar = np.array(ra_scalar)
    dec_scalar = np.array(dec_scalar)

    np.testing.assert_allclose(
        ra_batch, ra_scalar, err_msg=f"RA mismatch for seed {seed}"
    )
    np.testing.assert_allclose(
        dec_batch, dec_scalar, err_msg=f"DEC mismatch for seed {seed}"
    )
