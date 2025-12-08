#!/usr/bin/env python
"""Cross-validation for temporal basis fitting.

Tests whether the temporal basis model is genuinely predictive by:
1. Random holdout: Train on 75%, test on 25% randomly selected
2. Temporal split: Train on first 75% of time, test on last 25%

Usage:
    uv run python scripts/analysis/temporal_cv.py \
        --start 1998-07-01 --end 1998-07-31 \
        --lmax 10 --l2-penalty 100 \
        --temporal-basis constant,synodic
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.temporal.coefficients import (
    DEFAULT_CACHE_DIR,
    _build_harmonic_design,
    _discover_npz,
    _load_all_data,
)
from src.temporal.basis import (
    _get_basis_func_by_name,
    _harmonic_coefficient_count,
    build_temporal_design,
    fit_temporal_basis,
    parse_basis_spec,
)


def predict_with_result(
    result,
    utc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    reference_time: np.datetime64,
) -> np.ndarray:
    """Predict potential using fitted basis coefficients."""
    t_hours = (utc - reference_time).astype("timedelta64[s]").astype(np.float64) / 3600.0
    
    # Use the expanded basis names directly
    K = len(result.basis_names)
    T = np.column_stack([
        _get_basis_func_by_name(name)(t_hours) for name in result.basis_names
    ])
    
    Y = _build_harmonic_design(lat, lon, result.lmax)

    n_coeffs = _harmonic_coefficient_count(result.lmax)
    design = np.empty((len(utc), K * n_coeffs), dtype=np.complex128)
    for k in range(K):
        design[:, k * n_coeffs : (k + 1) * n_coeffs] = Y * T[:, k : k + 1]

    b_flat = result.basis_coeffs.flatten()
    return np.real(design @ b_flat)


def run_cv(
    utc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    potential: np.ndarray,
    lmax: int,
    basis_spec: str,
    l2_penalty: float,
    test_fraction: float = 0.25,
    seed: int = 42,
) -> dict:
    """Run cross-validation and return metrics."""
    np.random.seed(seed)
    n = len(potential)
    n_test = int(test_fraction * n)
    n_train = n - n_test

    results = {}

    # --- Random holdout ---
    perm = np.random.permutation(n)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    train_result = fit_temporal_basis(
        utc[train_idx], lat[train_idx], lon[train_idx], potential[train_idx],
        lmax=lmax, basis_spec=basis_spec, l2_penalty=l2_penalty,
    )

    ref_time = utc[train_idx].min()
    pred_test = predict_with_result(train_result, utc[test_idx], lat[test_idx], lon[test_idx], ref_time)
    residual_test = potential[test_idx] - pred_test
    rms_test = float(np.sqrt(np.mean(residual_test**2)))

    # Naive baseline (mean prediction)
    naive_pred = np.mean(potential[train_idx])
    rms_naive = float(np.sqrt(np.mean((potential[test_idx] - naive_pred)**2)))

    results["random"] = {
        "train_rms": train_result.rms_residual,
        "test_rms": rms_test,
        "overfit_ratio": rms_test / train_result.rms_residual,
        "naive_rms": rms_naive,
        "skill": 1 - rms_test / rms_naive,
        "r_squared": 1 - (rms_test**2 / rms_naive**2),
        "n_train": n_train,
        "n_test": n_test,
    }

    # --- Temporal split ---
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n)

    train_result = fit_temporal_basis(
        utc[train_idx], lat[train_idx], lon[train_idx], potential[train_idx],
        lmax=lmax, basis_spec=basis_spec, l2_penalty=l2_penalty,
    )

    ref_time = utc[train_idx].min()
    pred_test = predict_with_result(train_result, utc[test_idx], lat[test_idx], lon[test_idx], ref_time)
    residual_test = potential[test_idx] - pred_test
    rms_test = float(np.sqrt(np.mean(residual_test**2)))

    naive_pred = np.mean(potential[train_idx])
    rms_naive = float(np.sqrt(np.mean((potential[test_idx] - naive_pred)**2)))

    results["temporal"] = {
        "train_rms": train_result.rms_residual,
        "test_rms": rms_test,
        "overfit_ratio": rms_test / train_result.rms_residual,
        "naive_rms": rms_naive,
        "skill": 1 - rms_test / rms_naive,
        "r_squared": 1 - (rms_test**2 / rms_naive**2),
        "n_train": n_train,
        "n_test": n_test,
    }

    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Cross-validation for temporal basis fitting")
    parser.add_argument("--start", type=np.datetime64, required=True, help="Start date")
    parser.add_argument("--end", type=np.datetime64, required=True, help="End date")
    parser.add_argument("--lmax", type=int, default=10, help="Max spherical harmonic degree")
    parser.add_argument("--l2-penalty", type=float, default=100.0, help="Ridge penalty")
    parser.add_argument("--temporal-basis", default="constant,synodic", help="Basis specification")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--test-fraction", type=float, default=0.25, help="Fraction for testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Load data
    files = _discover_npz(args.cache_dir)
    start_ts = args.start.astype("datetime64[s]")
    end_ts = (args.end + np.timedelta64(1, "D")).astype("datetime64[s]")
    utc, lat, lon, potential = _load_all_data(files, start_ts, end_ts)

    logging.info("Loaded %d measurements", len(potential))

    # Run CV
    results = run_cv(
        utc, lat, lon, potential,
        lmax=args.lmax,
        basis_spec=args.temporal_basis,
        l2_penalty=args.l2_penalty,
        test_fraction=args.test_fraction,
        seed=args.seed,
    )

    # Print results
    print("\n" + "=" * 70)
    print("Temporal Basis Cross-Validation Results")
    print("=" * 70)
    print(f"Date range     : {args.start} → {args.end}")
    print(f"Basis          : {args.temporal_basis}")
    print(f"lmax           : {args.lmax}")
    print(f"L2 penalty     : {args.l2_penalty}")
    print(f"Test fraction  : {args.test_fraction:.0%}")
    print("-" * 70)

    for name, r in results.items():
        print(f"\n{name.upper()} HOLDOUT:")
        print(f"  Train/Test split : {r['n_train']:,} / {r['n_test']:,}")
        print(f"  Train RMS        : {r['train_rms']:.2f} V")
        print(f"  Test RMS         : {r['test_rms']:.2f} V")
        print(f"  Overfit ratio    : {r['overfit_ratio']:.3f}")
        print(f"  Naive RMS        : {r['naive_rms']:.2f} V")
        print(f"  Skill (vs naive) : {r['skill']:+.1%}")
        print(f"  R²               : {r['r_squared']:.3f}")

    print("\n" + "=" * 70)
    print("INTERPRETATION:")
    if results["random"]["overfit_ratio"] < 1.1:
        print("  ✓ Random holdout: Good generalization (overfit ratio < 1.1)")
    else:
        print("  ✗ Random holdout: Overfitting detected")

    if results["temporal"]["skill"] > 0:
        print("  ✓ Temporal split: Predicts future better than naive")
    else:
        print("  ⚠ Temporal split: Cannot predict unseen time/longitude regions")
        print("    (This is expected due to LP's 13°/day longitude drift)")

    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
