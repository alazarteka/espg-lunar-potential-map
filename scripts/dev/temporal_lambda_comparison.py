#!/usr/bin/env python3
"""
Compare temporal fitter performance with different temporal_lambda values
and co_rotate settings.
"""

import logging
import subprocess
from pathlib import Path
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# Test parameters
TEMPORAL_LAMBDAS = [0.0, 0.1, 1.0, 10.0, 100.0, 1000.0]
CO_ROTATE_OPTIONS = [False, True]

# Fixed parameters
YEAR = 1998
MONTH = 4  # April 1998
START_DATE = "1998-04-01"
END_DATE = "1998-04-30"
CACHE_DIR = Path("data/potential_cache")
OUTPUT_ROOT = Path("data/temporal_coeffs/lambda_comparison")
LMAX = 15
WINDOW_HOURS = 24.0
REGULARIZE_L2 = 10.0
MIN_SAMPLES = 100
MIN_COVERAGE = 0.1

OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

results_summary = []

for co_rotate in CO_ROTATE_OPTIONS:
    for temporal_lambda in TEMPORAL_LAMBDAS:
        # Build output filename
        rotate_str = "corotate" if co_rotate else "fixed"
        lambda_str = f"{temporal_lambda:.1e}".replace(".", "p").replace("+", "").replace("-", "m")
        output_file = OUTPUT_ROOT / f"1998-04_{rotate_str}_lambda_{lambda_str}.npz"

        logging.info(
            "Running: co_rotate=%s, temporal_lambda=%.2e → %s",
            co_rotate,
            temporal_lambda,
            output_file.name,
        )

        # Build command
        cmd = [
            "uv", "run", "python", "-m", "src.temporal.coefficients",
            "--start", START_DATE,
            "--end", END_DATE,
            "--cache-dir", str(CACHE_DIR),
            "--output", str(output_file),
            "--lmax", str(LMAX),
            "--window-hours", str(WINDOW_HOURS),
            "--regularize-l2", str(REGULARIZE_L2),
            "--temporal-lambda", str(temporal_lambda),
            "--min-samples", str(MIN_SAMPLES),
            "--min-coverage", str(MIN_COVERAGE),
            "--log-level", "WARNING",  # Suppress verbose output
        ]

        if co_rotate:
            cmd.extend(["--co-rotate", "--rotation-period-days", "29.530588"])

        # Run the fit
        try:
            start_time = datetime.now()
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            elapsed = (datetime.now() - start_time).total_seconds()

            # Load results and extract summary statistics
            with np.load(output_file) as data:
                n_windows = data["times"].size
                rms_residuals = data["rms_residuals"]
                spatial_coverage = data["spatial_coverage"]

                summary = {
                    "co_rotate": co_rotate,
                    "temporal_lambda": temporal_lambda,
                    "n_windows": n_windows,
                    "median_rms": float(np.median(rms_residuals)),
                    "mean_rms": float(np.mean(rms_residuals)),
                    "std_rms": float(np.std(rms_residuals)),
                    "median_coverage": float(np.median(spatial_coverage)),
                    "elapsed_s": elapsed,
                    "output_file": str(output_file),
                }
                results_summary.append(summary)

                logging.info(
                    "  ✓ n_windows=%d, median_rms=%.2f V, coverage=%.1f%%, elapsed=%.1fs",
                    n_windows,
                    summary["median_rms"],
                    summary["median_coverage"] * 100,
                    elapsed,
                )

        except subprocess.CalledProcessError as exc:
            logging.error("  ✗ Failed: %s", exc)
            logging.error("STDERR: %s", exc.stderr)
            continue

# Print summary table
print("\n" + "="*100)
print("TEMPORAL LAMBDA COMPARISON SUMMARY")
print("="*100)
print(f"{'Co-Rotate':<12} {'Lambda':<12} {'Windows':<10} {'Median RMS':<12} {'Mean RMS':<12} {'Std RMS':<12} {'Coverage':<10}")
print("-"*100)

for s in results_summary:
    rotate_label = "YES" if s["co_rotate"] else "NO"
    print(
        f"{rotate_label:<12} {s['temporal_lambda']:<12.1e} {s['n_windows']:<10} "
        f"{s['mean_rms']:<12.2f} {s['mean_rms']:<12.2f} {s['std_rms']:<12.2f} {s['median_coverage']*100:<10.1f}"
    )

print("="*100)
print(f"\nResults saved to: {OUTPUT_ROOT}")
print("="*100)
