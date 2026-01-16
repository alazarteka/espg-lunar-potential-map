#!/usr/bin/env python3
"""
Survey beam detection rates across the dataset.

Samples files across the date range and runs the peak scan on each,
aggregating results by month to show temporal trends in data quality.

Usage:
  uv run python scripts/diagnostics/beam_detection_survey.py
  uv run python scripts/diagnostics/beam_detection_survey.py --samples-per-month 3
  uv run python scripts/diagnostics/beam_detection_survey.py --all
"""

from __future__ import annotations

import argparse
import logging
import re
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from src import config
from src.diagnostics import LossConeSession
from src.flux import ERData


def _build_energy_profile(
    energies: np.ndarray,
    pitches: np.ndarray,
    norm2d: np.ndarray,
    pitch_min: float,
    pitch_max: float,
    min_band_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    band_mask = (pitches >= pitch_min) & (pitches <= pitch_max)
    values = np.full(energies.shape, np.nan, dtype=float)

    for idx in range(len(energies)):
        row_mask = band_mask[idx] & np.isfinite(norm2d[idx])
        if min_band_points > 0 and np.count_nonzero(row_mask) < min_band_points:
            continue
        if np.any(row_mask):
            values[idx] = float(np.nanmean(norm2d[idx][row_mask]))

    order = np.argsort(energies)
    return energies[order], values[order]


def _has_peak(
    profile: np.ndarray,
    *,
    contrast: float = 1.2,
    min_peak: float = 2.0,
    neighbor_window: int = 1,
    edge_skip: int = 1,
    min_neighbor: float = 1.5,
) -> bool:
    if profile.size == 0:
        return False

    window = max(1, neighbor_window)
    start = max(edge_skip, window)
    end = profile.size - max(edge_skip, window)
    if end <= start:
        return False

    for idx in range(start, end):
        value = profile[idx]
        if not np.isfinite(value) or value < min_peak:
            continue
        left = profile[idx - window : idx]
        right = profile[idx + 1 : idx + 1 + window]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            left_max = np.nanmax(left) if left.size else np.nan
            right_max = np.nanmax(right) if right.size else np.nan
        if not np.isfinite(left_max) or not np.isfinite(right_max):
            continue
        if not (value >= left_max * contrast and value >= right_max * contrast):
            continue
        if left_max < min_neighbor and right_max < min_neighbor:
            continue
        return True
    return False


def scan_file(er_file: Path) -> dict:
    """
    Scan a single ER file and return statistics.

    Returns dict with:
        - file: filename
        - total: total spec_nos
        - valid: polarity-valid spec_nos
        - peaks: detected peaks
        - error: error message if failed
    """
    try:
        er_data = ERData(str(er_file))
        if er_data.data.empty:
            return {"file": er_file.name, "error": "empty"}

        total_spec_nos = np.unique(er_data.data[config.SPEC_NO_COLUMN].to_numpy())

        session = LossConeSession(
            er_file=er_file,
            theta_file=config.DATA_DIR / config.THETA_FILE,
            normalization_mode="ratio2",
            incident_flux_stat="mean",
            use_torch=False,
            use_polarity=True,
        )
        valid_spec_nos = np.unique(
            session.er_data.data[config.SPEC_NO_COLUMN].to_numpy()
        )

        peak_count = 0
        for chunk_idx in range(session.chunk_count()):
            chunk = session.get_chunk_data(chunk_idx)
            norm2d = session.get_norm2d(chunk_idx)
            _, profile = _build_energy_profile(
                energies=chunk.energies,
                pitches=chunk.pitches,
                norm2d=norm2d,
                pitch_min=150.0,
                pitch_max=180.0,
                min_band_points=5,
            )
            if _has_peak(profile):
                peak_count += 1

        return {
            "file": er_file.name,
            "total": len(total_spec_nos),
            "valid": len(valid_spec_nos),
            "peaks": peak_count,
            "error": None,
        }
    except Exception as e:
        return {"file": er_file.name, "error": str(e)}


def parse_date_from_filename(filename: str) -> tuple[int, int] | None:
    """Extract (year, month) from filename like 3D980116.TAB."""
    match = re.match(r"3D(\d{2})(\d{2})(\d{2})\.TAB", filename)
    if match:
        yy, mm, dd = match.groups()
        year = 1900 + int(yy) if int(yy) > 50 else 2000 + int(yy)
        return year, int(mm)
    return None


def get_files_by_month(data_dir: Path) -> dict[tuple[int, int], list[Path]]:
    """Group all 3D*.TAB files by (year, month)."""
    files_by_month: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for f in sorted(data_dir.rglob("3D*.TAB")):
        date = parse_date_from_filename(f.name)
        if date:
            files_by_month[date].append(f)
    return files_by_month


def sample_files(
    files_by_month: dict[tuple[int, int], list[Path]],
    samples_per_month: int,
) -> list[Path]:
    """Sample N files per month, spread evenly."""
    sampled = []
    for month_key in sorted(files_by_month.keys()):
        files = files_by_month[month_key]
        if len(files) <= samples_per_month:
            sampled.extend(files)
        else:
            # Evenly spaced indices
            indices = np.linspace(0, len(files) - 1, samples_per_month, dtype=int)
            sampled.extend([files[i] for i in indices])
    return sampled


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey beam detection rates across the dataset."
    )
    parser.add_argument(
        "--samples-per-month",
        type=int,
        default=2,
        help="Number of files to sample per month (default: 2)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all files (slow, ~500 files)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING)  # Suppress INFO logs

    data_dir = config.DATA_DIR
    files_by_month = get_files_by_month(data_dir)

    if args.all:
        files_to_process = [f for files in files_by_month.values() for f in files]
        print(f"Processing ALL {len(files_to_process)} files...")
    else:
        files_to_process = sample_files(files_by_month, args.samples_per_month)
        print(
            f"Sampling {args.samples_per_month} files/month: "
            f"{len(files_to_process)} files total"
        )

    # Process files in parallel
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(scan_file, f): f for f in files_to_process}
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if i % 10 == 0 or i == len(files_to_process):
                print(f"  Processed {i}/{len(files_to_process)} files...", end="\r")

    print()  # Clear progress line

    # Aggregate by month
    monthly_stats: dict[tuple[int, int], dict] = defaultdict(
        lambda: {"total": 0, "valid": 0, "peaks": 0, "files": 0, "errors": 0}
    )

    for r in results:
        date = parse_date_from_filename(r["file"])
        if not date:
            continue
        stats = monthly_stats[date]
        stats["files"] += 1
        if r["error"]:
            stats["errors"] += 1
        else:
            stats["total"] += r["total"]
            stats["valid"] += r["valid"]
            stats["peaks"] += r["peaks"]

    # Print results
    print()
    print("=" * 70)
    print("Beam Detection Survey Results")
    print("=" * 70)
    print()
    print(
        f"{'Month':<12} {'Files':>6} {'Total':>8} {'Valid':>8} "
        f"{'Peaks':>8} {'Peak%':>8} {'Errors':>7}"
    )
    print("-" * 70)

    totals = {"files": 0, "total": 0, "valid": 0, "peaks": 0, "errors": 0}
    month_names = [
        "",
        "Jan",
        "Feb",
        "Mar",
        "Apr",
        "May",
        "Jun",
        "Jul",
        "Aug",
        "Sep",
        "Oct",
        "Nov",
        "Dec",
    ]

    for month_key in sorted(monthly_stats.keys()):
        year, month = month_key
        stats = monthly_stats[month_key]
        peak_pct = (
            (stats["peaks"] / stats["valid"] * 100) if stats["valid"] > 0 else 0.0
        )
        month_str = f"{month_names[month]} {year}"
        print(
            f"{month_str:<12} {stats['files']:>6} {stats['total']:>8} "
            f"{stats['valid']:>8} {stats['peaks']:>8} {peak_pct:>7.1f}% "
            f"{stats['errors']:>7}"
        )
        for k in totals:
            totals[k] += stats[k]

    print("-" * 70)
    total_pct = (totals["peaks"] / totals["valid"] * 100) if totals["valid"] > 0 else 0
    print(
        f"{'TOTAL':<12} {totals['files']:>6} {totals['total']:>8} "
        f"{totals['valid']:>8} {totals['peaks']:>8} {total_pct:>7.1f}% "
        f"{totals['errors']:>7}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
