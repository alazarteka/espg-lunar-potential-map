"""Batch processing with merged data loading and parallel fitting."""

from __future__ import annotations

# Disable BLAS/LAPACK multi-threading to ensure deterministic results
# when using multiprocessing. Must be set before numpy is imported.
import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

import src.config as config
from src.losscone.types import parse_fit_method
from src.potential_mapper.cli_args import add_common_batch_args, validate_date_filters
from src.potential_mapper.logging_utils import setup_logging
from src.potential_mapper.npz_io import write_npz_atomic
from src.potential_mapper.pipeline import DataLoader, load_all_data, process_merged_data
from src.potential_mapper.spice import load_spice_files

if TYPE_CHECKING:
    from src.losscone.types import FitMethod
    from src.potential_mapper.results import PotentialResults


def _to_unicode(arr) -> np.ndarray:
    """Convert array to Unicode string array."""
    if arr is None:
        return np.array([], dtype="U1")
    return np.asarray(arr, dtype="U64")


def _prepare_payload(
    er_data,
    results: PotentialResults,
    *,
    u_width_identifiable_max_v: float | None = None,
) -> dict[str, np.ndarray]:
    """
    Prepare NPZ payload from merged ERData and PotentialResults.

    Creates both row-level and spectrum-level aggregated data.
    Stores all fit parameters and plasma environment classification for analysis.
    """
    df = er_data.data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows != len(results.spacecraft_latitude):
        raise ValueError("Row count mismatch between ERData and PotentialResults")

    emit_u_width_qc = u_width_identifiable_max_v is not None

    spec_no = df[config.SPEC_NO_COLUMN].to_numpy(dtype=np.int64)
    utc_vals = _to_unicode(df.get(config.UTC_COLUMN))
    time_vals = _to_unicode(df.get(config.TIME_COLUMN))

    # Core row-level data
    payload: dict[str, np.ndarray] = {
        # Identifiers and time
        "rows_spec_no": spec_no,
        "rows_utc": utc_vals,
        "rows_time": time_vals,
        # Coordinates
        "rows_spacecraft_latitude": results.spacecraft_latitude.astype(np.float64),
        "rows_spacecraft_longitude": results.spacecraft_longitude.astype(np.float64),
        "rows_projection_latitude": results.projection_latitude.astype(np.float64),
        "rows_projection_longitude": results.projection_longitude.astype(np.float64),
        # Potentials
        "rows_spacecraft_potential": results.spacecraft_potential.astype(np.float64),
        "rows_projected_potential": results.projected_potential.astype(np.float64),
        # Illumination
        "rows_spacecraft_in_sun": results.spacecraft_in_sun.astype(bool),
        "rows_projection_in_sun": results.projection_in_sun.astype(bool),
        "rows_projection_polarity": results.projection_polarity.astype(np.int8),
        # Loss-cone fit parameters
        "rows_bs_over_bm": results.bs_over_bm.astype(np.float64),
        "rows_beam_amp": results.beam_amp.astype(np.float64),
        "rows_fit_chi2": results.fit_chi2.astype(np.float64),
        # Kappa/plasma parameters
        "rows_electron_temperature": results.electron_temperature.astype(np.float64),
        "rows_electron_density": results.electron_density.astype(np.float64),
        "rows_kappa_value": results.kappa_value.astype(np.float64),
        # Environment classification (0=unknown, 1=SW, 2=lobe, 3=PS)
        "rows_environment_class": results.environment_class.astype(np.int8),
    }

    if emit_u_width_qc:
        payload["rows_u_width_lhs_dchi2red_0p001"] = (
            results.u_width_lhs_dchi2red_0p001.astype(np.float64)
        )
        payload["rows_u_is_identifiable_lhs_dchi2red_0p001"] = (
            results.u_is_identifiable_lhs_dchi2red_0p001.astype(bool)
        )

    # Aggregate by spec_no
    uniq_specs, start_indices, counts = np.unique(
        spec_no, return_index=True, return_counts=True
    )
    # The (start, count) slicing below assumes each spec_no occupies one
    # contiguous block. That holds for per-file ER data (15 ordered rows per
    # sweep); guard against silently mixing spectra if unsorted/merged data
    # ever reaches here.
    n_runs = 1 + int(np.count_nonzero(np.diff(spec_no) != 0)) if spec_no.size else 0
    if n_runs != len(uniq_specs):
        raise ValueError(
            f"spec_no rows are not contiguous per spectrum ({n_runs} runs vs "
            f"{len(uniq_specs)} unique); refusing to aggregate."
        )
    spec_time_start = []
    spec_time_end = []
    spec_valid = []
    spec_u_surface = []
    spec_bs_over_bm = []
    spec_chi2 = []
    spec_te = []
    spec_ne = []
    spec_env = []
    spec_u_width_lhs: list[float] = []
    spec_u_is_identifiable_lhs: list[bool] = []

    for idx, count in zip(start_indices, counts, strict=False):
        row_slice = slice(idx, idx + count)
        spec_time_start.append(time_vals[row_slice.start])
        spec_time_end.append(time_vals[row_slice.stop - 1])

        # Get first valid value from the spectrum rows (all rows have same fit)
        u_chunk = results.projected_potential[row_slice]
        finite_u = u_chunk[np.isfinite(u_chunk)]
        spec_valid.append(bool(finite_u.size))
        spec_u_surface.append(float(finite_u[0]) if finite_u.size else np.nan)
        spec_bs_over_bm.append(results.bs_over_bm[idx])
        spec_chi2.append(results.fit_chi2[idx])
        if emit_u_width_qc:
            spec_u_width_lhs.append(results.u_width_lhs_dchi2red_0p001[idx])
            spec_u_is_identifiable_lhs.append(
                bool(results.u_is_identifiable_lhs_dchi2red_0p001[idx])
            )
        spec_te.append(results.electron_temperature[idx])
        spec_ne.append(results.electron_density[idx])
        spec_env.append(results.environment_class[idx])

    payload.update(
        {
            "spec_spec_no": uniq_specs.astype(np.int64),
            "spec_time_start": _to_unicode(spec_time_start),
            "spec_time_end": _to_unicode(spec_time_end),
            "spec_has_fit": np.array(spec_valid, dtype=bool),
            "spec_row_count": counts.astype(np.int64),
            # Spectrum-level fit results (one per spectrum)
            "spec_u_surface": np.array(spec_u_surface, dtype=np.float64),
            "spec_bs_over_bm": np.array(spec_bs_over_bm, dtype=np.float64),
            "spec_fit_chi2": np.array(spec_chi2, dtype=np.float64),
            "spec_electron_temperature": np.array(spec_te, dtype=np.float64),
            "spec_electron_density": np.array(spec_ne, dtype=np.float64),
            "spec_environment_class": np.array(spec_env, dtype=np.int8),
        }
    )

    if emit_u_width_qc:
        payload["spec_u_width_lhs_dchi2red_0p001"] = np.array(
            spec_u_width_lhs, dtype=np.float64
        )
        payload["spec_u_is_identifiable_lhs_dchi2red_0p001"] = np.array(
            spec_u_is_identifiable_lhs, dtype=bool
        )
        payload["u_width_lhs_delta_reduced"] = np.array(0.001, dtype=np.float64)
        payload["u_width_lhs_identifiable_max_v"] = np.array(
            float(u_width_identifiable_max_v), dtype=np.float64
        )

    return payload


def _build_output_filename(year: int | None, month: int | None, day: int | None) -> str:
    """Build output filename based on date filters."""
    parts = []
    if year is not None:
        parts.append(f"{year}")
    if month is not None:
        parts.append(f"{month:02d}")
    if day is not None:
        parts.append(f"{day:02d}")

    if parts:
        return f"potential_batch_{'_'.join(parts)}.npz"
    else:
        return "potential_batch_all.npz"


def run_batch(
    output_dir: Path,
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
    overwrite: bool = False,
    use_torch: bool = False,
    fit_method: str | FitMethod | None = None,
    spacecraft_potential_override: float | None = None,
    emit_u_width_qc: bool = False,
    u_width_identifiable_max_v: float = 200.0,
) -> int:
    """
    Run batch processing with merged data loading.

    Args:
        output_dir: Directory to save output NPZ file
        year: Optional year filter
        month: Optional month filter (1-12)
        day: Optional day filter (1-31)
        overwrite: Whether to overwrite existing output file
        use_torch: Use PyTorch-accelerated fitter (~5x faster)
        fit_method: Loss-cone fit method ("halekas" or "lillis")
        spacecraft_potential_override: Optional constant spacecraft potential [V]

    Returns:
        Exit code (0 for success, 1 for failure)
    """
    date_error = validate_date_filters(year, month, day)
    if date_error is not None:
        logging.error("Invalid date filters: %s", date_error)
        return 1

    start = datetime.now()

    # Load SPICE kernels
    logging.info("Loading SPICE kernels...")
    load_spice_files()

    # Discover files
    logging.info("Discovering flux files...")
    flux_files = DataLoader.discover_flux_files(year=year, month=month, day=day)

    if not flux_files:
        logging.info("No flux files found; exiting.")
        return 0

    logging.info(f"Found {len(flux_files)} file(s) to process")

    # Build output path
    output_filename = _build_output_filename(year, month, day)
    output_path = (output_dir / output_filename).resolve()

    if output_path.exists() and not overwrite:
        logging.info(f"Output file already exists: {output_path}")
        logging.info("Use --overwrite to recompute")
        return 0

    # Load and merge all data
    logging.info("Loading and merging data...")
    er_data = load_all_data(flux_files)

    if er_data.data.empty:
        logging.warning("Merged dataset is empty; exiting.")
        return 1

    logging.info(f"Loaded {len(er_data.data)} total rows")

    # Process merged data
    try:
        logging.info(f"Processing merged data (torch={use_torch})...")
        fit_method_parsed = (
            parse_fit_method(fit_method) if fit_method is not None else None
        )
        results = process_merged_data(
            er_data,
            use_torch=use_torch,
            fit_method=fit_method_parsed,
            spacecraft_potential_override=spacecraft_potential_override,
            emit_u_width_qc=bool(emit_u_width_qc),
            u_width_identifiable_max_v=float(u_width_identifiable_max_v),
        )
    except Exception as e:
        logging.exception(f"Failed to process merged data: {e}")
        return 1

    # Prepare and write output
    logging.info("Preparing output payload...")
    payload = _prepare_payload(
        er_data,
        results,
        u_width_identifiable_max_v=float(u_width_identifiable_max_v)
        if bool(emit_u_width_qc)
        else None,
    )

    logging.info(f"Writing to {output_path}...")
    write_npz_atomic(output_path, payload)

    duration = datetime.now() - start
    logging.info(
        f"Done in {duration.total_seconds():.1f}s "
        f"({duration.total_seconds() / 60:.1f}m)"
    )
    logging.info(f"Output saved to: {output_path}")

    return 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="python -m src.potential_mapper.batch",
        description="Batch process potential mapper with merged data loading.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/potential_cache"),
        help=(
            "Directory where output NPZ file is stored "
            "(default: artifacts/potential_cache)"
        ),
    )
    add_common_batch_args(parser)
    parser.add_argument(
        "--fast",
        action="store_true",
        help=(
            "Use PyTorch-accelerated fitter (~5x faster). Requires: uv sync --extra gpu"
        ),
    )
    parser.add_argument(
        "--losscone-fit-method",
        choices=["halekas", "lillis"],
        default=None,
        help="Loss-cone fitting method (defaults to config)",
    )
    parser.add_argument(
        "--u-spacecraft",
        type=float,
        default=None,
        help=(
            "Override spacecraft potential [V] (constant for all rows; skips "
            "spacecraft potential estimation)"
        ),
    )
    parser.add_argument(
        "--emit-u-width-qc",
        action="store_true",
        help=(
            "Emit an LHS-based U identifiability proxy into the batch NPZ "
            "(adds rows/spec u_width + identifiable flag)"
        ),
    )
    parser.add_argument(
        "--u-width-identifiable-max-v",
        type=float,
        default=200.0,
        help="Max U-width [V] considered identifiable (only with --emit-u-width-qc)",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = _parse_args()

    # Configure logging
    setup_logging(args.verbose)

    return run_batch(
        output_dir=args.output_dir,
        year=args.year,
        month=args.month,
        day=args.day,
        overwrite=args.overwrite,
        use_torch=args.fast,
        fit_method=args.losscone_fit_method,
        spacecraft_potential_override=args.u_spacecraft,
        emit_u_width_qc=args.emit_u_width_qc,
        u_width_identifiable_max_v=args.u_width_identifiable_max_v,
    )


if __name__ == "__main__":
    raise SystemExit(main())
