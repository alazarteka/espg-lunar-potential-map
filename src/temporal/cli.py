"""Command-line interface for temporal harmonic coefficient computation."""

import argparse
import logging
from datetime import datetime
from pathlib import Path

import numpy as np

from .coefficients import (
    DEFAULT_CACHE_DIR,
    DEFAULT_SYNODIC_PERIOD_DAYS,
    _harmonic_coefficient_count,
    compute_temporal_harmonics,
    save_temporal_coefficients,
)


def _parse_iso_date(value: str) -> np.datetime64:
    """Parse YYYY-MM-DD into numpy datetime64."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
        return np.datetime64(value, "D")
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Dates must be YYYY-MM-DD") from exc


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for temporal coefficient computation."""
    parser = argparse.ArgumentParser(
        prog="python -m src.temporal",
        description="Compute time-dependent spherical harmonic coefficients a_lm(t)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        required=True,
        type=_parse_iso_date,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=_parse_iso_date,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Root directory with potential_cache NPZ files",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=5,
        help="Maximum spherical harmonic degree",
    )
    parser.add_argument(
        "--window-hours",
        type=float,
        default=24.0,
        help="Temporal window duration in hours",
    )
    parser.add_argument(
        "--window-stride",
        type=float,
        default=None,
        help="Temporal stride in hours (default: same as --window-hours for non-overlapping)",
    )
    parser.add_argument(
        "--l2-penalty",
        type=float,
        default=0.0,
        dest="regularize_l2",
        help="Ridge penalty for spatial coefficient fitting",
    )
    parser.add_argument(
        "--temporal-lambda",
        type=float,
        default=0.0,
        help="Temporal continuity regularization strength (0 = independent windows)",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum measurements required per window",
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.1,
        help="Minimum spatial coverage fraction (0-1)",
    )
    parser.add_argument(
        "--co-rotate",
        action="store_true",
        help="Rotate temporal derivative into a solar co-rotating frame",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=1,
        help="Max temporal lag for multi-scale regularization (1=adjacent, 5=±5 windows)",
    )
    parser.add_argument(
        "--decay-factor",
        type=float,
        default=0.5,
        help="Weight decay per lag step (0.5 = half weight per additional lag)",
    )
    parser.add_argument(
        "--spatial-weight-exponent",
        type=float,
        default=None,
        help="Exponent for degree-weighted spatial damping (None disables weighting)",
    )
    parser.add_argument(
        "--rotation-period-days",
        type=float,
        default=DEFAULT_SYNODIC_PERIOD_DAYS,
        help="Rotation period (days) used when --co-rotate is set (sign controls direction)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output NPZ file for temporal coefficients",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point for temporal harmonic computation."""
    args = parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if args.end < args.start:
        logging.error("--end must be >= --start")
        return 1

    if args.lmax < 0:
        logging.error("--lmax must be non-negative")
        return 1

    if args.window_hours <= 0:
        logging.error("--window-hours must be positive")
        return 1

    if args.co_rotate and args.rotation_period_days == 0.0:
        logging.error("--rotation-period-days must be non-zero when --co-rotate is set")
        return 1

    if args.spatial_weight_exponent is not None and args.spatial_weight_exponent < 0.0:
        logging.error("--spatial-weight-exponent must be non-negative")
        return 1

    try:
        results = compute_temporal_harmonics(
            cache_dir=args.cache_dir,
            start_date=args.start,
            end_date=args.end,
            lmax=args.lmax,
            window_hours=args.window_hours,
            stride_hours=args.window_stride,
            l2_penalty=args.regularize_l2,
            temporal_lambda=args.temporal_lambda,
            min_samples=args.min_samples,
            min_coverage=args.min_coverage,
            co_rotate=args.co_rotate,
            rotation_period_days=args.rotation_period_days,
            spatial_weight_exponent=args.spatial_weight_exponent,
            max_lag=args.max_lag,
            decay_factor=args.decay_factor,
        )
    except Exception as exc:
        logging.exception("Failed to compute temporal harmonics: %s", exc)
        return 1

    if not results:
        logging.warning("No valid time windows produced; nothing to save.")
        return 1

    try:
        save_temporal_coefficients(results, args.output)
    except Exception as exc:
        logging.exception("Failed to save results: %s", exc)
        return 1

    # Print summary
    coverages = np.array([r.spatial_coverage for r in results])
    rms_vals = np.array([r.rms_residual for r in results])

    print("\n" + "=" * 60)
    print("Time-Dependent Spherical Harmonic Summary")
    print("=" * 60)
    print(f"Date range      : {args.start} → {args.end}")
    print(f"Time windows    : {len(results)}")
    print(f"Window duration : {args.window_hours:.1f} hours")
    print(f"Max degree      : lmax = {args.lmax}")
    if args.co_rotate:
        direction = "forward" if args.rotation_period_days > 0 else "reverse"
        print(
            f"Temporal frame  : solar co-rotating ({abs(args.rotation_period_days):.3f} d, {direction})"
        )
    else:
        print("Temporal frame  : moon-fixed")
    if args.spatial_weight_exponent is None:
        print("Spatial weighting: uniform")
    else:
        print(f"Spatial weighting: [l(l+1)]^{args.spatial_weight_exponent:.2f} (l>0)")
    print(f"Coefficients    : {_harmonic_coefficient_count(args.lmax)} per window")
    print(
        f"Coverage range  : {coverages.min() * 100:.1f}% - {coverages.max() * 100:.1f}%"
    )
    print(f"RMS residuals   : {rms_vals.min():.2f} - {rms_vals.max():.2f} V")
    print(f"Median RMS      : {np.median(rms_vals):.2f} V")
    print(f"\nSaved to: {args.output}")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
