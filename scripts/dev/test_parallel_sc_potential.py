#!/usr/bin/env python3
"""
Test that parallel spacecraft potential produces identical results to sequential.

Usage:
    uv run python scripts/dev/test_parallel_sc_potential.py [--year YYYY] [--month MM] [--day DD]
"""

import argparse
import logging
import sys

import numpy as np

sys.path.insert(0, ".")

from src.potential_mapper import pipeline
from src.potential_mapper.spice import load_spice_files


def compare_results(seq: np.ndarray, par: np.ndarray, name: str = "SC Potential") -> bool:
    """Compare two arrays with NaN-aware comparison."""

    if seq.shape != par.shape:
        print(f"❌ {name}: Shape mismatch {seq.shape} vs {par.shape}")
        return False

    # Check NaN masks match
    seq_nan_mask = np.isnan(seq)
    par_nan_mask = np.isnan(par)

    if not np.array_equal(seq_nan_mask, par_nan_mask):
        nan_diff = np.sum(seq_nan_mask != par_nan_mask)
        print(f"❌ {name}: NaN mask mismatch ({nan_diff} differences)")
        print(f"   Sequential NaNs: {np.sum(seq_nan_mask)}, Parallel NaNs: {np.sum(par_nan_mask)}")
        return False

    # Compare non-NaN values
    valid_mask = ~seq_nan_mask
    if not np.allclose(seq[valid_mask], par[valid_mask], rtol=1e-9, atol=1e-12):
        diff = np.abs(seq[valid_mask] - par[valid_mask])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)

        mismatch_indices = np.where(~np.isclose(seq, par, rtol=1e-9, atol=1e-12, equal_nan=True))[0]
        print(f"❌ {name}: Value mismatch")
        print(f"   Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        print(f"   Mismatches: {len(mismatch_indices)} / {np.sum(valid_mask)}")
        print("   First 5 mismatches:")
        for i in mismatch_indices[:5]:
            print(f"     [{i}] seq={seq[i]:.6f}, par={par[i]:.6f}, diff={abs(seq[i]-par[i]):.2e}")
        return False

    print(f"✅ {name}: Match ({np.sum(valid_mask)} valid values)")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test parallel SC potential correctness")
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day", type=int, default=16)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    print(f"Testing parallel SC potential for {args.year}-{args.month:02d}-{args.day:02d}")
    print("=" * 70)

    # Load SPICE
    logging.info("Loading SPICE kernels...")
    load_spice_files()

    # Discover and load data
    logging.info("Discovering files...")
    files = pipeline.DataLoader.discover_flux_files(args.year, args.month, args.day)

    if not files:
        print("❌ No files found")
        return 1

    print(f"Found {len(files)} file(s)")

    # Load and merge data
    logging.info("Loading data...")
    er_data = pipeline.load_all_data(files)

    if er_data.data.empty:
        print("❌ Merged dataset is empty")
        return 1

    print(f"Loaded {len(er_data.data)} rows")
    print()

    # Run sequential
    print("Running SEQUENTIAL processing...")
    try:
        seq_results = pipeline.process_merged_data(er_data, use_parallel=False)
    except Exception as e:
        print(f"❌ Sequential processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("✅ Sequential complete")
    print()

    # Run parallel
    print("Running PARALLEL processing...")
    try:
        par_results = pipeline.process_merged_data(er_data, use_parallel=True)
    except Exception as e:
        print(f"❌ Parallel processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("✅ Parallel complete")
    print()

    # Compare all fields
    print("Comparing results...")
    print("-" * 70)

    all_match = True

    all_match &= compare_results(
        seq_results.spacecraft_latitude,
        par_results.spacecraft_latitude,
        "Spacecraft Latitude"
    )

    all_match &= compare_results(
        seq_results.spacecraft_longitude,
        par_results.spacecraft_longitude,
        "Spacecraft Longitude"
    )

    all_match &= compare_results(
        seq_results.projection_latitude,
        par_results.projection_latitude,
        "Projection Latitude"
    )

    all_match &= compare_results(
        seq_results.projection_longitude,
        par_results.projection_longitude,
        "Projection Longitude"
    )

    all_match &= compare_results(
        seq_results.spacecraft_potential,
        par_results.spacecraft_potential,
        "Spacecraft Potential"
    )

    all_match &= compare_results(
        seq_results.projected_potential,
        par_results.projected_potential,
        "Projected Potential"
    )

    # Boolean arrays
    if not np.array_equal(seq_results.spacecraft_in_sun, par_results.spacecraft_in_sun):
        diff_count = np.sum(seq_results.spacecraft_in_sun != par_results.spacecraft_in_sun)
        print(f"❌ Spacecraft In Sun: {diff_count} differences")
        all_match = False
    else:
        print("✅ Spacecraft In Sun: Match")

    if not np.array_equal(seq_results.projection_in_sun, par_results.projection_in_sun):
        diff_count = np.sum(seq_results.projection_in_sun != par_results.projection_in_sun)
        print(f"❌ Projection In Sun: {diff_count} differences")
        all_match = False
    else:
        print("✅ Projection In Sun: Match")

    print("-" * 70)

    if all_match:
        print()
        print("🎉 SUCCESS: Parallel and sequential produce identical results!")
        return 0
    else:
        print()
        print("⚠️  WARNING: Parallel and sequential differ!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
