#!/usr/bin/env python3
"""
Benchmark sequential vs parallel spacecraft potential calculation.

Measures the time spent on SC potential calculation only.
"""

import argparse
import logging
import sys
import time

sys.path.insert(0, ".")

from src.potential_mapper import pipeline
from src.potential_mapper.spice import load_spice_files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=1998)
    parser.add_argument("--month", type=int, default=4)
    parser.add_argument("--day", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)  # Suppress most logs

    print(f"Benchmarking SC potential for {args.year}-{args.month:02d}")
    if args.day:
        print(f"  Day: {args.day}")
    print("=" * 70)

    # Load SPICE
    load_spice_files()

    # Discover and load data
    files = pipeline.DataLoader.discover_flux_files(args.year, args.month, args.day)
    print(f"Files: {len(files)}")

    er_data = pipeline.load_all_data(files)
    n_rows = len(er_data.data)
    print(f"Rows: {n_rows:,}")

    # Extract unique spec count
    import src.config as config
    unique_specs = er_data.data[config.SPEC_NO_COLUMN].nunique()
    print(f"Unique spectra: {unique_specs:,}")
    print()

    # Sequential benchmark
    print("Running SEQUENTIAL...")
    start = time.time()
    sc_potential_seq = pipeline._spacecraft_potential_per_row(er_data, n_rows)
    seq_time = time.time() - start
    print(f"✅ Sequential: {seq_time:.1f}s ({seq_time/60:.1f}m)")
    print()

    # Parallel benchmark
    print("Running PARALLEL...")
    start = time.time()
    sc_potential_par = pipeline._spacecraft_potential_per_row_parallel(er_data, n_rows)
    par_time = time.time() - start
    print(f"✅ Parallel: {par_time:.1f}s ({par_time/60:.1f}m)")
    print()

    # Results
    print("=" * 70)
    print(f"Sequential: {seq_time:.1f}s")
    print(f"Parallel:   {par_time:.1f}s")
    print(f"Speedup:    {seq_time/par_time:.2f}x")
    print(f"Saved:      {seq_time - par_time:.1f}s ({(seq_time - par_time)/60:.1f}m)")
    print()

    # Quick correctness check
    import numpy as np

    seq_nan = np.isnan(sc_potential_seq)
    par_nan = np.isnan(sc_potential_par)

    if not np.array_equal(seq_nan, par_nan):
        print("⚠️  WARNING: NaN masks differ!")
        return 1

    valid = ~seq_nan
    if not np.allclose(sc_potential_seq[valid], sc_potential_par[valid], rtol=1e-9):
        print("⚠️  WARNING: Values differ!")
        return 1

    print("✅ Correctness check passed (identical results)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
