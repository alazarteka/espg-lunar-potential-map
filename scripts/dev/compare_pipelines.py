
import argparse
import logging
import sys

import numpy as np

# Add src to path
sys.path.append(".")

from src.potential_mapper import pipeline as pipeline_new
from src.potential_mapper import pipeline_seq as pipeline_old
from src.potential_mapper.results import PotentialResults


def compare_results(res1: PotentialResults, res2: PotentialResults):
    """Compare two PotentialResults objects."""

    attrs = [
        "spacecraft_latitude",
        "spacecraft_longitude",
        "projection_latitude",
        "projection_longitude",
        "spacecraft_potential",
        "projected_potential",
        "spacecraft_in_sun",
        "projection_in_sun",
    ]

    all_match = True

    for attr in attrs:
        val1 = getattr(res1, attr)
        val2 = getattr(res2, attr)

        # Handle nan mismatch
        # If both are nan, they are equal
        # If one is nan, they are not

        # Check shapes
        if val1.shape != val2.shape:
            print(f"❌ {attr}: Shape mismatch {val1.shape} vs {val2.shape}")
            all_match = False
            continue

        if np.issubdtype(val1.dtype, np.floating):
            # Use allclose with nan handling
            mask1 = np.isnan(val1)
            mask2 = np.isnan(val2)

            if not np.array_equal(mask1, mask2):
                print(f"❌ {attr}: NaN mask mismatch")
                all_match = False
                continue

            # Compare non-nan values
            valid = ~mask1
            if not np.allclose(val1[valid], val2[valid], equal_nan=True):
                diff = np.abs(val1[valid] - val2[valid])
                max_diff = np.max(diff)
                mismatch_indices = np.where(~np.isclose(val1, val2, equal_nan=True))[0]
                print(f"❌ {attr}: Value mismatch (max diff: {max_diff})")
                print(f"   Count: {len(mismatch_indices)} / {len(val1)}")
                print(f"   Indices: {mismatch_indices[:10]} ...")
                print(f"   Val1: {val1[mismatch_indices[:5]]}")
                print(f"   Val2: {val2[mismatch_indices[:5]]}")
                all_match = False
            else:
                print(f"✅ {attr}: Match")
        else:
            # Boolean or integer
            if not np.array_equal(val1, val2):
                mismatch_count = np.sum(val1 != val2)
                print(f"❌ {attr}: Mismatch count: {mismatch_count}")
                all_match = False
            else:
                print(f"✅ {attr}: Match")

    return all_match

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day", type=int, default=16)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    from src.potential_mapper.spice import load_spice_files
    load_spice_files()

    print(f"Running OLD pipeline for {args.year}-{args.month}-{args.day}...")
    # Mock args for pipeline run
    class Args:
        year = args.year
        month = args.month
        day = args.day
        output = None
        display = False
        illumination = None

    # We need to capture the return value of run, but run() returns int exit code.
    # We need to modify/import the internal processing functions or monkeypatch.
    # pipeline_old.run calls process_lp_file and aggregates.
    # Let's just call process_lp_file directly for the discovered files.

    files = pipeline_old.DataLoader.discover_flux_files(args.year, args.month, args.day)
    if not files:
        print("No files found.")
        return

    # Run OLD
    results_old = []
    for f in files:
        print(f"Old: Processing {f.name}")
        results_old.append(pipeline_old.process_lp_file(f))
    agg_old = pipeline_old._concat_results(results_old)

    print(f"Running NEW pipeline for {args.year}-{args.month}-{args.day}...")
    # Run NEW
    # pipeline_new.process_merged_data takes an ERData object.
    # We can use pipeline_new.load_all_data
    er_data = pipeline_new.load_all_data(files)
    agg_new = pipeline_new.process_merged_data(er_data)

    print("\nComparing results...")
    match = compare_results(agg_old, agg_new)

    if match:
        print("\n✅ Pipelines produce identical results!")
    else:
        print("\n❌ Pipelines differ!")

if __name__ == "__main__":
    main()
