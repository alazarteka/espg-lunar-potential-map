#!/usr/bin/env python
"""Compute regime-conditioned surface potential statistics for Table 1.

Loads cached potential results and computes per-regime statistics:
- Median U_M
- Interquartile range (IQR)
- Count N

Usage:
    uv run python scripts/analysis/compute_regime_stats.py
    uv run python scripts/analysis/compute_regime_stats.py --output table1.txt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.potential_mapper.results import PlasmaEnvironment


def load_batch_data(cache_path: Path) -> dict:
    """Load cached potential batch data."""
    data = np.load(cache_path, allow_pickle=True)
    return {key: data[key] for key in data.keys()}


def classify_environments_5class(
    te_ev: np.ndarray, projection_in_sun: np.ndarray
) -> np.ndarray:
    """Classify plasma environment using 5-class system.

    Args:
        te_ev: Electron temperature array in eV
        projection_in_sun: Boolean array, True if footprint is sunlit

    Returns:
        Array of PlasmaEnvironment enum values
    """
    n = len(te_ev)
    env_class = np.zeros(n, dtype=np.int8)

    for i in range(n):
        env_class[i] = PlasmaEnvironment.from_temperature_and_illumination(
            te_ev[i], bool(projection_in_sun[i])
        )

    return env_class


def compute_regime_stats(
    potential: np.ndarray,
    env_class: np.ndarray,
    chi2: np.ndarray | None = None,
    chi2_threshold: float = 1000.0,
    exclude_boundary: bool = True,
    boundary_value: float = -2000.0,
) -> dict[PlasmaEnvironment, dict]:
    """Compute statistics for each plasma regime.

    Args:
        potential: Surface potential array (V)
        env_class: Environment classification array
        chi2: Optional chi-squared array for quality filtering
        chi2_threshold: Maximum chi2 for valid fits
        exclude_boundary: If True, exclude values at the boundary
        boundary_value: The boundary value to exclude

    Returns:
        Dict mapping PlasmaEnvironment to stats dict with keys:
        - median: Median potential (V)
        - q25, q75: 25th and 75th percentiles
        - iqr: Interquartile range
        - n: Sample count
        - n_valid: Count of finite values
    """
    stats = {}

    # Base quality mask
    quality_mask = np.isfinite(potential)
    if exclude_boundary:
        quality_mask &= potential > boundary_value
    if chi2 is not None:
        quality_mask &= np.isfinite(chi2) & (chi2 < chi2_threshold)

    for regime in PlasmaEnvironment:
        mask = (env_class == regime) & quality_mask
        n_total = np.sum(env_class == regime)
        n_valid = np.sum(mask)

        if n_valid == 0:
            stats[regime] = {
                "median": np.nan,
                "q25": np.nan,
                "q75": np.nan,
                "iqr": np.nan,
                "n": int(n_total),
                "n_valid": 0,
            }
            continue

        subset = potential[mask]
        q25, median, q75 = np.percentile(subset, [25, 50, 75])

        stats[regime] = {
            "median": float(median),
            "q25": float(q25),
            "q75": float(q75),
            "iqr": float(q75 - q25),
            "n": int(n_total),
            "n_valid": int(n_valid),
        }

    return stats


def format_latex_table(stats: dict[PlasmaEnvironment, dict]) -> str:
    """Format statistics as LaTeX table rows."""
    lines = []

    # Order: Solar wind, Magnetosheath, Tail lobes, Plasma sheet, Wake
    regime_order = [
        PlasmaEnvironment.SOLAR_WIND,
        PlasmaEnvironment.MAGNETOSHEATH,
        PlasmaEnvironment.TAIL_LOBES,
        PlasmaEnvironment.PLASMA_SHEET,
        PlasmaEnvironment.WAKE,
    ]

    for regime in regime_order:
        s = stats[regime]
        if s["n_valid"] == 0:
            lines.append(f"        {regime.name.replace('_', ' ').title()} & -- & -- & -- \\\\")
        else:
            lines.append(
                f"        {regime.name.replace('_', ' ').title()} & "
                f"{s['median']:.0f} & {s['iqr']:.0f} & {s['n_valid']:,} \\\\"
            )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compute regime-conditioned surface potential statistics"
    )
    parser.add_argument(
        "--cache-path",
        type=Path,
        default=Path("artifacts/potential_cache/potential_batch_1998_04.npz"),
        help="Path to cached potential batch NPZ file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file for results (stdout if not specified)",
    )
    args = parser.parse_args()

    # Load data
    print(f"Loading data from {args.cache_path}...")
    data = load_batch_data(args.cache_path)

    # Extract relevant arrays
    potential = data["rows_projected_potential"]
    te_ev = data["rows_electron_temperature"]
    projection_in_sun = data["rows_projection_in_sun"]
    chi2 = data["rows_fit_chi2"]

    print(f"Loaded {len(potential):,} measurements")

    # Reclassify with 5-class system
    print("Classifying with 5-class regime system...")
    env_class = classify_environments_5class(te_ev, projection_in_sun)

    # Compute statistics with quality filtering
    print("Computing statistics (excluding boundary values and high chi2)...")
    stats = compute_regime_stats(potential, env_class, chi2=chi2)

    # Print summary
    print("\n" + "=" * 70)
    print("REGIME-CONDITIONED SURFACE POTENTIAL STATISTICS")
    print("=" * 70)
    print(f"{'Regime':<20} {'Median (V)':>12} {'IQR (V)':>12} {'N':>12}")
    print("-" * 70)

    total_valid = 0
    for regime in PlasmaEnvironment:
        s = stats[regime]
        total_valid += s["n_valid"]
        if s["n_valid"] == 0:
            print(f"{regime.name:<20} {'--':>12} {'--':>12} {s['n']:>12,}")
        else:
            print(
                f"{regime.name:<20} {s['median']:>12.1f} {s['iqr']:>12.1f} {s['n_valid']:>12,}"
            )

    print("-" * 70)
    print(f"{'TOTAL (valid fits)':<20} {'':>12} {'':>12} {total_valid:>12,}")
    print("=" * 70)

    # Print LaTeX table
    print("\nLaTeX table rows for paper:")
    print("-" * 70)
    print(format_latex_table(stats))
    print("-" * 70)

    # Save to file if requested
    if args.output:
        with open(args.output, "w") as f:
            f.write("REGIME-CONDITIONED SURFACE POTENTIAL STATISTICS\n")
            f.write(f"Source: {args.cache_path}\n\n")
            f.write(f"{'Regime':<20} {'Median (V)':>12} {'IQR (V)':>12} {'N':>12}\n")
            f.write("-" * 60 + "\n")
            for regime in PlasmaEnvironment:
                s = stats[regime]
                if s["n_valid"] == 0:
                    f.write(f"{regime.name:<20} {'--':>12} {'--':>12} {s['n']:>12,}\n")
                else:
                    f.write(
                        f"{regime.name:<20} {s['median']:>12.1f} {s['iqr']:>12.1f} {s['n_valid']:>12,}\n"
                    )
            f.write("\nLaTeX:\n")
            f.write(format_latex_table(stats))
        print(f"\nSaved to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
