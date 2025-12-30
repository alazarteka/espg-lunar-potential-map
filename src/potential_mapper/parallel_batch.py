"""
Parallel batch processing for lunar potential mapping.

Processes multiple days in parallel using one process per CPU core.
Each day is processed independently and saved to its own file.

Example:
    # Process all of April 1998 using 8 cores
    uv run python -m src.potential_mapper.parallel_batch --year 1998 --month 4

    # Process with torch acceleration
    uv run python -m src.potential_mapper.parallel_batch --year 1998 --month 4 --fast
"""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

from src import config
from src.potential_mapper.date_utils import MONTH_INT_TO_ABBREV, parse_3d_filename
from src.potential_mapper.logging_utils import setup_logging


def discover_day_files(year: int, month: int | None = None) -> list[Path]:
    """
    Discover all ER data files for a given year/month.

    Args:
        year: Year to process
        month: Optional month filter (1-12)

    Returns:
        List of Path objects for each day file
    """
    data_dir = config.DATA_DIR / str(year)
    if not data_dir.exists():
        return []

    day_files = []
    for subdir in sorted(data_dir.iterdir()):
        if not subdir.is_dir():
            continue

        # Filter by month if specified
        if month is not None:
            # Subdirs are like "091_120APR" - need to check month
            if MONTH_INT_TO_ABBREV.get(month, "") not in subdir.name:
                continue

        # Find all .TAB files in this subdir
        for tab_file in sorted(subdir.glob("3D*.TAB")):
            day_files.append(tab_file)

    return day_files


def process_single_day(
    day_file: Path,
    output_dir: Path,
    use_torch: bool = False,
) -> tuple[Path, float, str]:
    """
    Process a single day's ER data file.

    This function runs in a separate process, so it must import
    everything it needs inside the function.

    Args:
        day_file: Path to the .TAB file for this day
        output_dir: Directory to save output
        use_torch: Use PyTorch-accelerated loss-cone fitter

    Returns:
        Tuple of (output_path, elapsed_seconds, status)
    """
    import time

    start_time = time.time()

    try:
        # Import inside function to avoid multiprocessing issues
        from src import config
        from src.flux import ERData
        from src.potential_mapper.pipeline import process_merged_data
        from src.potential_mapper.spice import load_spice_files

        # Load SPICE kernels (each process needs its own)
        load_spice_files()

        # Parse date from filename (e.g., 3D980415.TAB -> 1998-04-15)
        year, mm, dd = parse_3d_filename(day_file.name)

        # Load data for this day
        er_data = ERData(str(day_file))
        n_rows = len(er_data.data)

        if n_rows == 0:
            return (day_file, time.time() - start_time, "empty")

        # Process the data
        results = process_merged_data(
            er_data,
            use_parallel=False,  # No sub-parallelism within day
            use_torch=use_torch,
        )

        # Extract UTC timestamps from source data (needed for temporal reconstruction)
        utc_col = (
            config.UTC_COLUMN
            if config.UTC_COLUMN in er_data.data.columns
            else config.TIME_COLUMN
        )
        rows_utc = er_data.data[utc_col].to_numpy(dtype=str)

        # Save results with rows_* prefix to match temporal loader expectations
        output_file = output_dir / f"potential_{year:04d}_{mm:02d}_{dd:02d}.npz"
        output_dir.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_file,
            # Row-level arrays with rows_* prefix for temporal/coefficients.py
            rows_utc=rows_utc,
            rows_spacecraft_latitude=results.spacecraft_latitude,
            rows_spacecraft_longitude=results.spacecraft_longitude,
            rows_projection_latitude=results.projection_latitude,
            rows_projection_longitude=results.projection_longitude,
            rows_spacecraft_potential=results.spacecraft_potential,
            rows_projected_potential=results.projected_potential,
            rows_spacecraft_in_sun=results.spacecraft_in_sun,
            rows_projection_in_sun=results.projection_in_sun,
        )

        elapsed = time.time() - start_time
        return (output_file, elapsed, "success")

    except Exception as e:
        elapsed = time.time() - start_time
        return (day_file, elapsed, f"error: {e}")


def run_parallel_batch(
    year: int,
    month: int | None = None,
    output_dir: Path | None = None,
    use_torch: bool = False,
    max_workers: int | None = None,
    overwrite: bool = False,
) -> int:
    """
    Run batch processing with parallel day-level execution.

    Args:
        year: Year to process
        month: Optional month filter
        output_dir: Output directory (default: artifacts/potential_cache/daily/)
        use_torch: Use PyTorch-accelerated fitter
        max_workers: Number of parallel workers (default: CPU count)
        overwrite: Overwrite existing files

    Returns:
        Exit code (0 for success)
    """
    start_time = datetime.now()

    if output_dir is None:
        output_dir = (
            Path(__file__).parent.parent.parent
            / "artifacts"
            / "potential_cache"
            / "daily"
        )

    if max_workers is None:
        max_workers = os.cpu_count() or 4

    # Discover day files
    day_files = discover_day_files(year, month)
    if not day_files:
        logging.error(
            f"No day files found for {year}" + (f"-{month:02d}" if month else "")
        )
        return 1

    logging.info(f"Found {len(day_files)} day files to process")
    logging.info(f"Using {max_workers} parallel workers")
    logging.info(f"PyTorch acceleration: {use_torch}")

    # Filter out already-processed files if not overwriting
    if not overwrite:
        to_process = []
        for day_file in day_files:
            yr, mm, dd = parse_3d_filename(day_file.name)
            output_file = output_dir / f"potential_{yr:04d}_{mm:02d}_{dd:02d}.npz"
            if output_file.exists():
                logging.info(f"Skipping {day_file.name} (already processed)")
            else:
                to_process.append(day_file)
        day_files = to_process

    if not day_files:
        logging.info("All files already processed. Use --overwrite to reprocess.")
        return 0

    logging.info(f"Processing {len(day_files)} files...")

    # Process in parallel
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_single_day, f, output_dir, use_torch): f
            for f in day_files
        }

        for future in as_completed(futures):
            day_file = futures[future]
            try:
                output_path, elapsed, status = future.result()
                results.append((day_file, output_path, elapsed, status))

                if status == "success":
                    logging.info(
                        f"✓ {day_file.name} -> {output_path.name} ({elapsed:.1f}s)"
                    )
                else:
                    logging.warning(f"✗ {day_file.name}: {status}")
            except Exception as e:
                logging.error(f"✗ {day_file.name}: {e}")
                results.append((day_file, None, 0, f"exception: {e}"))

    # Summary
    elapsed_total = (datetime.now() - start_time).total_seconds()
    successful = sum(1 for r in results if r[3] == "success")
    failed = len(results) - successful

    logging.info("")
    logging.info("=" * 60)
    logging.info(f"Completed: {successful}/{len(results)} days successful")
    logging.info(f"Total time: {elapsed_total:.1f}s")
    if successful > 0:
        avg_time = sum(r[2] for r in results if r[3] == "success") / successful
        logging.info(f"Average time per day: {avg_time:.1f}s")
    logging.info(f"Output directory: {output_dir}")
    logging.info("=" * 60)

    return 0 if failed == 0 else 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parallel batch processing for lunar potential mapping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Year to process",
    )
    parser.add_argument(
        "--month",
        type=int,
        default=None,
        help="Optional month filter (1-12)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for daily .npz files",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use PyTorch-accelerated loss-cone fitter (~5x faster)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    setup_logging(args.verbose)

    return run_parallel_batch(
        year=args.year,
        month=args.month,
        output_dir=args.output_dir,
        use_torch=args.fast,
        max_workers=args.workers,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    raise SystemExit(main())
