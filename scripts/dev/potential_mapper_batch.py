"""Batch runner for potential mapper with per-file durable outputs."""

from __future__ import annotations

import argparse
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import numpy as np

import src.config as config
from src.flux import ERData
from src.potential_mapper.pipeline import DataLoader, process_lp_file
from src.potential_mapper.results import PotentialResults
from src.potential_mapper.spice import load_spice_files


_SPICE_LOADED = False


def _ensure_spice_loaded() -> None:
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        load_spice_files()
        _SPICE_LOADED = True


def _relative_output_path(file_path: Path, output_root: Path) -> Path:
    try:
        rel = file_path.relative_to(config.DATA_DIR)
    except ValueError:
        rel = file_path.name
    if isinstance(rel, Path):
        rel_path = rel.with_suffix(".npz")
    else:
        rel_path = Path(rel).with_suffix(".npz")
    return (output_root / rel_path).resolve()


def _to_unicode(arr) -> np.ndarray:
    if arr is None:
        return np.array([], dtype="U1")
    return np.asarray(arr, dtype="U64")


def _prepare_payload(er_data: ERData, results: PotentialResults) -> dict[str, np.ndarray]:
    df = er_data.data.reset_index(drop=True)
    n_rows = len(df)
    if n_rows != len(results.spacecraft_latitude):
        raise ValueError("Row count mismatch between ERData and PotentialResults")

    spec_no = df[config.SPEC_NO_COLUMN].to_numpy(dtype=np.int64)
    utc_vals = _to_unicode(df.get(config.UTC_COLUMN))
    time_vals = _to_unicode(df.get(config.TIME_COLUMN))

    payload: dict[str, np.ndarray] = {
        "rows_spec_no": spec_no,
        "rows_utc": utc_vals,
        "rows_time": time_vals,
        "rows_spacecraft_latitude": results.spacecraft_latitude.astype(np.float64),
        "rows_spacecraft_longitude": results.spacecraft_longitude.astype(np.float64),
        "rows_projection_latitude": results.projection_latitude.astype(np.float64),
        "rows_projection_longitude": results.projection_longitude.astype(np.float64),
        "rows_spacecraft_potential": results.spacecraft_potential.astype(np.float64),
        "rows_projected_potential": results.projected_potential.astype(np.float64),
        "rows_spacecraft_in_sun": results.spacecraft_in_sun.astype(bool),
        "rows_projection_in_sun": results.projection_in_sun.astype(bool),
    }

    uniq_specs, start_indices, counts = np.unique(spec_no, return_index=True, return_counts=True)
    spec_time_start = []
    spec_time_end = []
    spec_valid = []
    for idx, count in zip(start_indices, counts):
        row_slice = slice(idx, idx + count)
        spec_time_start.append(time_vals[row_slice.start])
        spec_time_end.append(time_vals[row_slice.stop - 1])
        spec_chunk = results.projected_potential[row_slice]
        spec_valid.append(bool(np.isfinite(spec_chunk).any()))

    payload.update(
        {
            "spec_spec_no": uniq_specs.astype(np.int64),
            "spec_time_start": _to_unicode(spec_time_start),
            "spec_time_end": _to_unicode(spec_time_end),
            "spec_has_fit": np.array(spec_valid, dtype=bool),
            "spec_row_count": counts.astype(np.int64),
        }
    )

    return payload


def _write_npz_atomic(out_path: Path, payload: dict[str, np.ndarray]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=out_path.parent, suffix=".tmp", delete=False) as tmp:
        np.savez_compressed(tmp, **payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, out_path)


def _process_file(file_path: Path, output_root: Path, overwrite: bool) -> tuple[str, str]:
    _ensure_spice_loaded()
    out_path = _relative_output_path(file_path, output_root)
    if out_path.exists() and not overwrite:
        return (str(file_path), "skipped")

    results = process_lp_file(file_path)
    er_data = ERData(str(file_path))
    payload = _prepare_payload(er_data, results)
    _write_npz_atomic(out_path, payload)
    return (str(file_path), "written")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch compute potential mapper outputs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/potential_cache"),
        help="Directory where per-file NPZ results are stored.",
    )
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1, help="Parallel workers")
    parser.add_argument("--overwrite", action="store_true", help="Recompute even if output exists")
    parser.add_argument("--year", type=int, default=None, help="Optional year filter")
    parser.add_argument("--month", type=int, default=None, help="Optional month filter (1-12)")
    parser.add_argument("--day", type=int, default=None, help="Optional day filter (1-31)")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most this many files (debugging aid)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    start = datetime.now()
    flux_files = DataLoader.discover_flux_files(year=args.year, month=args.month, day=args.day)
    if args.limit is not None:
        flux_files = flux_files[: args.limit]

    if not flux_files:
        print("No flux files found; exiting.")
        return 0

    total_files = len(flux_files)
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"Processing {total_files} file{'s' if total_files != 1 else ''} with {args.workers} worker{'s' if args.workers != 1 else ''}...")
    print(f"Output directory: {output_root}")
    print()

    summary: dict[str, int] = {"written": 0, "skipped": 0, "failed": 0}
    completed = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(_process_file, file_path, output_root, args.overwrite): file_path
            for file_path in flux_files
        }
        for future in as_completed(futures):
            file_path = futures[future]
            try:
                _file, status = future.result()
                summary[status] = summary.get(status, 0) + 1
                completed += 1

                # Progress indicator
                pct = 100 * completed / total_files
                elapsed = (datetime.now() - start).total_seconds()
                rate = completed / elapsed if elapsed > 0 else 0
                eta = (total_files - completed) / rate if rate > 0 else 0

                print(f"[{completed}/{total_files} {pct:5.1f}%] [{status:7s}] {file_path.name} "
                      f"(eta: {eta/60:.1f}m, {rate:.2f} files/s)")
            except Exception as exc:
                summary["failed"] += 1
                completed += 1
                pct = 100 * completed / total_files
                print(f"[{completed}/{total_files} {pct:5.1f}%] [failed ] {file_path.name}: {exc}")

    duration = datetime.now() - start
    report = ", ".join(f"{k}={v}" for k, v in summary.items())
    print()
    print(f"Done in {duration.total_seconds():.1f}s ({duration.total_seconds()/60:.1f}m) - {report}")
    print(f"Average: {total_files/duration.total_seconds():.2f} files/s")
    return 0 if summary["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
