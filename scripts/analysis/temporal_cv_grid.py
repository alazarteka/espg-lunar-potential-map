#!/usr/bin/env python
"""Grid search for temporal basis configurations.

Tests multiple combinations of synodic/sidereal harmonics to find optimal basis.
Outputs results to a CSV file.

Example:
    uv run python scripts/analysis/temporal_cv_grid.py \
        --start 1998-04-01 --end 1998-04-30 \
        --output artifacts/paper/cv_results/april_1998_grid.csv
"""

import argparse
import csv
import logging
from pathlib import Path

import numpy as np

from scripts.analysis.temporal_cv import run_cv
from src.temporal.coefficients import (
    DEFAULT_CACHE_DIR,
    _discover_npz,
    _load_all_data,
)

# Basis configurations to test
BASIS_CONFIGS = [
    "constant",
    "constant,synodic",
    "constant,sidereal",
    "constant,synodic,sidereal",
    "constant,synodic,synodic2",
    "constant,synodic,synodic2,synodic3",
    "constant,synodic,synodic2,synodic3,synodic4",
    "constant,synodic,sidereal,synodic2",
    "constant,synodic,sidereal,synodic2,sidereal2",
    "constant,synodic,synodic2,sidereal,sidereal2",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Grid search for temporal basis CV")
    parser.add_argument("--start", type=np.datetime64, required=True, help="Start date")
    parser.add_argument("--end", type=np.datetime64, required=True, help="End date")
    parser.add_argument("--lmax", type=int, default=10, help="Max spherical harmonic degree")
    parser.add_argument("--l2-penalty", type=float, default=100.0, help="Ridge penalty")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output", type=Path, required=True, help="Output CSV file")
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load data once
    files = _discover_npz(args.cache_dir)
    start_ts = args.start.astype("datetime64[s]")
    end_ts = (args.end + np.timedelta64(1, "D")).astype("datetime64[s]")
    utc, lat, lon, potential = _load_all_data(files, start_ts, end_ts)
    logging.info("Loaded %d measurements", len(potential))

    results = []

    for basis_spec in BASIS_CONFIGS:
        logging.info("Testing basis: %s", basis_spec)
        try:
            cv_results = run_cv(
                utc, lat, lon, potential,
                lmax=args.lmax,
                basis_spec=basis_spec,
                l2_penalty=args.l2_penalty,
                test_fraction=args.test_fraction,
                seed=args.seed,
            )

            n_bases = len(basis_spec.split(","))
            results.append({
                "basis": basis_spec,
                "n_bases": n_bases,
                "random_r2": cv_results["random"]["r_squared"],
                "random_skill": cv_results["random"]["skill"],
                "random_rms": cv_results["random"]["test_rms"],
                "random_overfit": cv_results["random"]["overfit_ratio"],
                "temporal_r2": cv_results["temporal"]["r_squared"],
                "temporal_skill": cv_results["temporal"]["skill"],
                "temporal_rms": cv_results["temporal"]["test_rms"],
                "temporal_overfit": cv_results["temporal"]["overfit_ratio"],
            })
        except Exception as e:
            logging.error("Failed for %s: %s", basis_spec, e)

    # Write CSV
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    logging.info("Results written to %s", args.output)

    # Print summary
    print("\n" + "=" * 80)
    print("GRID SEARCH RESULTS")
    print("=" * 80)
    print(f"{'Basis':<45} {'R²':>8} {'Skill':>8} {'Overfit':>8}")
    print("-" * 80)
    for r in sorted(results, key=lambda x: -x["random_r2"]):
        print(f"{r['basis']:<45} {r['random_r2']:>8.3f} {r['random_skill']:>+7.1%} {r['random_overfit']:>8.3f}")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
