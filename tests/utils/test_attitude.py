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

    # Query times that should return valid results
    # bisect_right logic: idx must be > 0 and < len(ra_vals)
    # et_spin[0]=0, so time=5 -> idx=1 -> valid
    # et_spin[2]=20, so time=25 -> idx=3 -> valid
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

    # Compare - use nan-safe comparison
    np.testing.assert_array_equal(ra_batch, ra_scalar)
    np.testing.assert_array_equal(dec_batch, dec_scalar)


def test_get_current_ra_dec_batch_handles_out_of_bounds():
    """Test that batch version returns NaN for out-of-bounds queries."""

    et_spin = np.array([10.0, 20.0, 30.0])
    ra_vals = np.array([100.0, 200.0, 300.0])
    dec_vals = np.array([10.0, 20.0, 30.0])

    # Query times outside valid range
    # time < et_spin[0] -> idx=0 -> invalid (idx <= 0)
    # time > et_spin[-1] -> idx=len(et_spin) -> invalid (idx >= len(ra_vals))
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

    # bisect_right(10.0) in [0, 10] -> idx=2 -> invalid (>= len(ra_vals)=2)
    assert np.isnan(ra_batch[0])
    assert np.isnan(dec_batch[0])

    # Just before boundary should be valid
    query_times = np.array([9.999])
    ra_batch, dec_batch = get_current_ra_dec_batch(
        query_times, et_spin, ra_vals, dec_vals
    )

    # bisect_right(9.999) -> idx=1 -> valid
    assert np.isfinite(ra_batch[0])
    assert ra_batch[0] == ra_vals[1]
    assert dec_batch[0] == dec_vals[1]


def test_get_current_ra_dec_batch_consistency_with_searchsorted():
    """Verify that batch implementation uses searchsorted correctly."""

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

    # Verify with manual searchsorted
    idxs = np.searchsorted(et_spin, query_times, side="right")
    valid_mask = (idxs > 0) & (idxs < len(ra_vals))

    for i, t in enumerate(query_times):
        if valid_mask[i]:
            assert np.isfinite(ra_batch[i])
            assert ra_batch[i] == ra_vals[idxs[i]]
            assert dec_batch[i] == dec_vals[idxs[i]]
        else:
            assert np.isnan(ra_batch[i])
            assert np.isnan(dec_batch[i])


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

    # Should match exactly (no floating point ops, just indexing)
    np.testing.assert_array_equal(
        ra_batch, ra_scalar, err_msg=f"RA mismatch for seed {seed}"
    )
    np.testing.assert_array_equal(
        dec_batch, dec_scalar, err_msg=f"DEC mismatch for seed {seed}"
    )
