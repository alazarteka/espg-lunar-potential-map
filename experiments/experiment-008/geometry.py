#!/usr/bin/env python3
"""Build the geometry-only sweep cache for Experiment 008.

The loader reads only UTC, sweep metadata, energy, and MAG columns from raw
Level-1 files. It does not discover or read fitted-potential archives.

Examples:

    # Outcome-blind one-day implementation check
    uv run python experiments/experiment-008/geometry.py extract --limit 1

    # Full four-worker extraction and deterministic merge
    uv run python experiments/experiment-008/geometry.py extract --workers 4
    uv run python experiments/experiment-008/geometry.py merge
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import spiceypy as spice
from replay_context import ARTIFACT_DIR, CENSUS_PATH

from src import config
from src.potential_mapper.spice import load_spice_files
from src.utils.attitude import get_current_ra_dec_batch, load_attitude_data
from src.utils.coordinates import build_scd_to_reference, ra_dec_to_unit
from src.utils.geometry import get_connections_and_polarity_batch

if TYPE_CHECKING:
    from collections.abc import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIGURATION = EXPERIMENT_DIR / "configuration.json"
DEFAULT_MANIFEST = EXPERIMENT_DIR / "input-manifest.csv"
DEFAULT_ARTIFACT_DIR = ARTIFACT_DIR
RAW_COLUMNS = ("utc", "time", "energy", "spec_no", "mag_x", "mag_y", "mag_z")
NAT_INT64 = np.datetime64("NaT", "ns").astype(np.int64)

_ATTITUDE_ET: np.ndarray | None = None
_ATTITUDE_RA: np.ndarray | None = None
_ATTITUDE_DEC: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class FileCensus:
    """Per-file extraction counts and timing."""

    file_index: int
    path: str
    raw_rows: int
    contiguous_groups: int
    exact_15_row_sweeps: int
    malformed_sweeps: int
    structurally_valid_sweeps: int
    corrected_connected_sweeps: int
    corrected_complete_sweeps: int
    legacy_connected_sweeps: int
    legacy_complete_sweeps: int
    equatorial_connected_sweeps: int
    equatorial_complete_sweeps: int
    frame_mixed_connected_sweeps: int
    frame_mixed_complete_sweeps: int
    submitted_legacy_connected_sweeps: int
    submitted_legacy_complete_sweeps: int
    elapsed_seconds: float
    shard_path: str
    error: str = ""


def load_configuration(path: Path = DEFAULT_CONFIGURATION) -> dict[str, Any]:
    """Load the frozen Experiment 008 configuration."""
    return json.loads(path.read_text(encoding="ascii"))


def raw_paths_from_manifest(
    path: Path = DEFAULT_MANIFEST, project_root: Path = PROJECT_ROOT
) -> list[Path]:
    """Read the frozen raw-file list without discovering other data products."""
    with path.open(encoding="ascii", newline="") as stream:
        return [
            project_root / row["path"]
            for row in csv.DictReader(stream)
            if row["kind"] == "raw_er_mag"
        ]


def contiguous_group_slices(values: np.ndarray) -> list[slice]:
    """Return slices for contiguous equal-valued groups."""
    if values.ndim != 1:
        raise ValueError("group values must be one-dimensional")
    if values.size == 0:
        return []
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    stops = np.r_[starts[1:], values.size]
    return [
        slice(int(start), int(stop)) for start, stop in zip(starts, stops, strict=True)
    ]


def legacy_next_attitude(
    query_et: np.ndarray,
    attitude_et: np.ndarray,
    ra: np.ndarray,
    dec: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the pre-ac1f03c next-sample attitude lookup."""
    query = np.asarray(query_et, dtype=np.float64)
    indices = np.searchsorted(attitude_et, query, side="right")
    valid = (indices > 0) & (indices < len(attitude_et)) & np.isfinite(query)
    ra_out = np.full(query.shape, np.nan, dtype=np.float64)
    dec_out = np.full(query.shape, np.nan, dtype=np.float64)
    ra_out[valid] = ra[indices[valid]]
    dec_out[valid] = dec[indices[valid]]
    return ra_out, dec_out


def points_to_lat_lon(
    points: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert valid Cartesian surface points to degrees."""
    lat = np.full(points.shape[0], np.nan, dtype=np.float64)
    lon = np.full(points.shape[0], np.nan, dtype=np.float64)
    norms = np.linalg.norm(points, axis=1)
    keep = valid & np.isfinite(norms) & (norms > 0.0)
    lat[keep] = np.rad2deg(np.arcsin(points[keep, 2] / norms[keep]))
    lon[keep] = np.rad2deg(np.arctan2(points[keep, 1], points[keep, 0]))
    return lat, lon


def display_path(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Use a repository-relative path when possible, else an absolute path."""
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _load_raw_geometry_columns(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        engine="c",
        header=None,
        usecols=range(len(RAW_COLUMNS)),
        names=RAW_COLUMNS,
    )


def robust_spkpos(
    target: str,
    et: np.ndarray,
    reference_frame: str,
    observer: str,
) -> np.ndarray:
    """Vectorize SPK lookup while isolating uncovered epochs as NaN rows."""
    epochs = np.asarray(et, dtype=np.float64)
    output = np.full((len(epochs), 3), np.nan, dtype=np.float64)

    def fill(start: int, stop: int) -> None:
        if start >= stop:
            return
        try:
            positions, _ = spice.spkpos(
                target,
                epochs[start:stop],
                reference_frame,
                "NONE",
                observer,
            )
            output[start:stop] = np.asarray(positions).reshape(stop - start, 3)
        except Exception:
            if stop - start == 1:
                return
            midpoint = start + (stop - start) // 2
            fill(start, midpoint)
            fill(midpoint, stop)

    fill(0, len(epochs))
    return output


def _worker_initialize() -> None:
    """Load kernels and attitude once in each extraction process."""
    global _ATTITUDE_ET, _ATTITUDE_RA, _ATTITUDE_DEC
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    spice.kclear()
    load_spice_files()
    attitude = load_attitude_data(config.DATA_DIR / config.ATTITUDE_FILE)
    if any(value is None for value in attitude):
        raise RuntimeError("Attitude data unavailable")
    _ATTITUDE_ET, _ATTITUDE_RA, _ATTITUDE_DEC = attitude


def _project_rows(
    et: np.ndarray,
    magnetic_field: np.ndarray,
    *,
    legacy_attitude: bool,
    attitude_frame: str,
    frame_mixed_sun: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project rows, explicitly naming the frame of the attitude components."""
    if _ATTITUDE_ET is None or _ATTITUDE_RA is None or _ATTITUDE_DEC is None:
        raise RuntimeError("Geometry worker was not initialized")
    if legacy_attitude:
        ra, dec = legacy_next_attitude(et, _ATTITUDE_ET, _ATTITUDE_RA, _ATTITUDE_DEC)
    else:
        ra, dec = get_current_ra_dec_batch(
            et, _ATTITUDE_ET, _ATTITUDE_RA, _ATTITUDE_DEC
        )

    lp_positions = robust_spkpos("-25", et, "IAU_MOON", "301")
    sun_reference = robust_spkpos("10", et, attitude_frame, "-25")
    sun_norms = np.linalg.norm(sun_reference, axis=1)
    sun_unit = np.divide(
        sun_reference,
        sun_norms[:, None],
        out=np.full_like(sun_reference, np.nan),
        where=sun_norms[:, None] > 0.0,
    )
    spin_unit = ra_dec_to_unit(ra, dec)
    reference_to_moon = np.stack(
        [spice.pxform(attitude_frame, "IAU_MOON", float(value)) for value in et]
    )
    sun_for_builder = sun_unit
    if frame_mixed_sun:
        # Reproduce the submitted production mismatch: the Sun vector is in
        # IAU_MOON while attitude components are interpreted as J2000.
        sun_for_builder = np.einsum("nij,nj->ni", reference_to_moon, sun_unit)
    scd_to_reference = build_scd_to_reference(spin_unit, sun_for_builder)
    mag_norms = np.linalg.norm(magnetic_field, axis=1)
    mag_unit = magnetic_field / mag_norms[:, None]
    mag_reference = np.einsum("nij,nj->ni", scd_to_reference, mag_unit)

    mag_moon = np.einsum("nij,nj->ni", reference_to_moon, mag_reference)
    points, connected, polarity = get_connections_and_polarity_batch(
        np.asarray(lp_positions), mag_moon, config.LUNAR_RADIUS
    )
    lat, lon = points_to_lat_lon(points, connected)
    return lat, lon, polarity


def _empty_shard_arrays(sweep_count: int, rows_per_sweep: int) -> dict[str, np.ndarray]:
    shape = (sweep_count, rows_per_sweep)
    return {
        "row_utc_ns": np.full(shape, NAT_INT64, dtype=np.int64),
        "corrected_latitude": np.full(shape, np.nan, dtype=np.float64),
        "corrected_longitude": np.full(shape, np.nan, dtype=np.float64),
        "corrected_polarity": np.zeros(shape, dtype=np.int8),
        "legacy_latitude": np.full(shape, np.nan, dtype=np.float64),
        "legacy_longitude": np.full(shape, np.nan, dtype=np.float64),
        "legacy_polarity": np.zeros(shape, dtype=np.int8),
        "equatorial_latitude": np.full(shape, np.nan, dtype=np.float64),
        "equatorial_longitude": np.full(shape, np.nan, dtype=np.float64),
        "equatorial_polarity": np.zeros(shape, dtype=np.int8),
        "frame_mixed_latitude": np.full(shape, np.nan, dtype=np.float64),
        "frame_mixed_longitude": np.full(shape, np.nan, dtype=np.float64),
        "frame_mixed_polarity": np.zeros(shape, dtype=np.int8),
        "submitted_legacy_latitude": np.full(shape, np.nan, dtype=np.float64),
        "submitted_legacy_longitude": np.full(shape, np.nan, dtype=np.float64),
        "submitted_legacy_polarity": np.zeros(shape, dtype=np.int8),
    }


def extract_file(
    file_index: int,
    path: Path,
    shard_dir: Path,
    rows_per_sweep: int,
) -> FileCensus:
    """Extract one raw file into a deterministic geometry shard."""
    started = perf_counter()
    relative_path = path.relative_to(PROJECT_ROOT).as_posix()
    shard_path = shard_dir / f"{file_index:04d}-{path.stem}.npz"
    try:
        frame = _load_raw_geometry_columns(path)
        spec_no = frame["spec_no"].to_numpy(dtype=np.int64)
        groups = contiguous_group_slices(spec_no)
        exact = [
            group for group in groups if group.stop - group.start == rows_per_sweep
        ]
        arrays = _empty_shard_arrays(len(exact), rows_per_sweep)
        sweep_spec_no = np.empty(len(exact), dtype=np.int64)
        structural_valid = np.zeros(len(exact), dtype=bool)

        utc_text = frame["utc"].astype(str).to_numpy()
        utc_parsed = pd.to_datetime(utc_text, errors="coerce", utc=True)
        utc_ns = utc_parsed.to_numpy(dtype="datetime64[ns]").astype(np.int64)
        mag = frame[["mag_x", "mag_y", "mag_z"]].to_numpy(dtype=np.float64)
        mag_norm = np.linalg.norm(mag, axis=1)

        source_indices: list[int] = []
        destination_indices: list[int] = []
        for sweep_index, group in enumerate(exact):
            source = np.arange(group.start, group.stop, dtype=np.int64)
            sweep_spec_no[sweep_index] = spec_no[group.start]
            arrays["row_utc_ns"][sweep_index] = utc_ns[source]
            valid = (
                np.all(utc_ns[source] != NAT_INT64)
                and np.all(np.isfinite(mag[source]))
                and np.all((mag_norm[source] > 1e-9) & (mag_norm[source] < 1e3))
            )
            structural_valid[sweep_index] = bool(valid)
            if valid:
                source_indices.extend(source.tolist())
                destination_indices.extend(
                    range(
                        sweep_index * rows_per_sweep,
                        (sweep_index + 1) * rows_per_sweep,
                    )
                )

        if source_indices:
            source = np.asarray(source_indices, dtype=np.int64)
            destination = np.asarray(destination_indices, dtype=np.int64)
            et = np.asarray(spice.str2et(utc_text[source]), dtype=np.float64)
            corrected = _project_rows(
                et,
                mag[source],
                legacy_attitude=False,
                attitude_frame="ECLIPJ2000",
                frame_mixed_sun=False,
            )
            legacy = _project_rows(
                et,
                mag[source],
                legacy_attitude=True,
                attitude_frame="ECLIPJ2000",
                frame_mixed_sun=False,
            )
            equatorial = _project_rows(
                et,
                mag[source],
                legacy_attitude=False,
                attitude_frame="J2000",
                frame_mixed_sun=False,
            )
            frame_mixed = _project_rows(
                et,
                mag[source],
                legacy_attitude=False,
                attitude_frame="J2000",
                frame_mixed_sun=True,
            )
            submitted_legacy = _project_rows(
                et,
                mag[source],
                legacy_attitude=True,
                attitude_frame="J2000",
                frame_mixed_sun=True,
            )
            for key, values in zip(
                (
                    "corrected_latitude",
                    "corrected_longitude",
                    "corrected_polarity",
                ),
                corrected,
                strict=True,
            ):
                arrays[key].reshape(-1)[destination] = values
            for key, values in zip(
                ("legacy_latitude", "legacy_longitude", "legacy_polarity"),
                legacy,
                strict=True,
            ):
                arrays[key].reshape(-1)[destination] = values
            for key, values in zip(
                (
                    "equatorial_latitude",
                    "equatorial_longitude",
                    "equatorial_polarity",
                ),
                equatorial,
                strict=True,
            ):
                arrays[key].reshape(-1)[destination] = values
            for key, values in zip(
                (
                    "frame_mixed_latitude",
                    "frame_mixed_longitude",
                    "frame_mixed_polarity",
                ),
                frame_mixed,
                strict=True,
            ):
                arrays[key].reshape(-1)[destination] = values
            for key, values in zip(
                (
                    "submitted_legacy_latitude",
                    "submitted_legacy_longitude",
                    "submitted_legacy_polarity",
                ),
                submitted_legacy,
                strict=True,
            ):
                arrays[key].reshape(-1)[destination] = values

        corrected_count = np.sum(np.isfinite(arrays["corrected_latitude"]), axis=1)
        legacy_count = np.sum(np.isfinite(arrays["legacy_latitude"]), axis=1)
        equatorial_count = np.sum(np.isfinite(arrays["equatorial_latitude"]), axis=1)
        frame_mixed_count = np.sum(np.isfinite(arrays["frame_mixed_latitude"]), axis=1)
        submitted_legacy_count = np.sum(
            np.isfinite(arrays["submitted_legacy_latitude"]), axis=1
        )
        shard_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            shard_path,
            file_index=np.array(file_index, dtype=np.int64),
            source_path=np.array(relative_path),
            sweep_spec_no=sweep_spec_no,
            structural_valid=structural_valid,
            **arrays,
        )
        return FileCensus(
            file_index=file_index,
            path=relative_path,
            raw_rows=len(frame),
            contiguous_groups=len(groups),
            exact_15_row_sweeps=len(exact),
            malformed_sweeps=len(groups) - len(exact),
            structurally_valid_sweeps=int(np.sum(structural_valid)),
            corrected_connected_sweeps=int(np.sum(corrected_count >= 1)),
            corrected_complete_sweeps=int(np.sum(corrected_count == rows_per_sweep)),
            legacy_connected_sweeps=int(np.sum(legacy_count >= 1)),
            legacy_complete_sweeps=int(np.sum(legacy_count == rows_per_sweep)),
            equatorial_connected_sweeps=int(np.sum(equatorial_count >= 1)),
            equatorial_complete_sweeps=int(np.sum(equatorial_count == rows_per_sweep)),
            frame_mixed_connected_sweeps=int(np.sum(frame_mixed_count >= 1)),
            frame_mixed_complete_sweeps=int(
                np.sum(frame_mixed_count == rows_per_sweep)
            ),
            submitted_legacy_connected_sweeps=int(np.sum(submitted_legacy_count >= 1)),
            submitted_legacy_complete_sweeps=int(
                np.sum(submitted_legacy_count == rows_per_sweep)
            ),
            elapsed_seconds=perf_counter() - started,
            shard_path=display_path(shard_path),
        )
    except Exception as exc:
        return FileCensus(
            file_index=file_index,
            path=relative_path,
            raw_rows=0,
            contiguous_groups=0,
            exact_15_row_sweeps=0,
            malformed_sweeps=0,
            structurally_valid_sweeps=0,
            corrected_connected_sweeps=0,
            corrected_complete_sweeps=0,
            legacy_connected_sweeps=0,
            legacy_complete_sweeps=0,
            equatorial_connected_sweeps=0,
            equatorial_complete_sweeps=0,
            frame_mixed_connected_sweeps=0,
            frame_mixed_complete_sweeps=0,
            submitted_legacy_connected_sweeps=0,
            submitted_legacy_complete_sweeps=0,
            elapsed_seconds=perf_counter() - started,
            shard_path=display_path(shard_path),
            error=f"{type(exc).__name__}: {exc}",
        )


def _write_census(rows: Iterable[FileCensus], path: Path) -> None:
    records = [asdict(row) for row in sorted(rows, key=lambda row: row.file_index)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=records[0].keys(), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def run_extract(args: argparse.Namespace) -> int:
    """Extract frozen raw files into daily geometry shards."""
    configuration = load_configuration(args.configuration)
    rows_per_sweep = int(configuration["inputs"]["expected_rows_per_sweep"])
    maximum_workers = int(configuration["resources"]["maximum_cpu_workers"])
    workers = min(args.workers, maximum_workers)
    paths = list(enumerate(raw_paths_from_manifest(args.manifest)))
    paths = paths[args.start_index :]
    if args.limit is not None:
        paths = paths[: args.limit]
    args.shard_dir.mkdir(parents=True, exist_ok=True)

    rows: list[FileCensus] = []
    if workers == 1:
        _worker_initialize()
        for completed, (file_index, path) in enumerate(paths, start=1):
            row = extract_file(file_index, path, args.shard_dir, rows_per_sweep)
            rows.append(row)
            error_suffix = f" ERROR {row.error}" if row.error else ""
            print(
                f"[{completed}/{len(paths)}] {path.name}: "
                f"{row.corrected_connected_sweeps} connected, "
                f"{row.elapsed_seconds:.2f}s{error_suffix}"
            )
    else:
        with ProcessPoolExecutor(
            max_workers=workers, initializer=_worker_initialize
        ) as executor:
            futures = {
                executor.submit(
                    extract_file, file_index, path, args.shard_dir, rows_per_sweep
                ): file_index
                for file_index, path in paths
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                row = future.result()
                rows.append(row)
                error_suffix = f" ERROR {row.error}" if row.error else ""
                print(
                    f"[{completed}/{len(paths)}] {Path(row.path).name}: "
                    f"{row.corrected_connected_sweeps} connected, "
                    f"{row.elapsed_seconds:.2f}s{error_suffix}"
                )

    _write_census(rows, args.census)
    failures = sum(bool(row.error) for row in rows)
    print(f"Wrote {len(rows)} census rows to {args.census}; failures={failures}")
    return 1 if failures else 0


def run_merge(args: argparse.Namespace) -> int:
    """Merge deterministic daily shards into the geometry-only cache."""
    shards = sorted(args.shard_dir.glob("*.npz"))
    if not shards:
        raise FileNotFoundError(f"No geometry shards found in {args.shard_dir}")
    arrays: dict[str, list[np.ndarray]] = {
        "source_file_index": [],
        "sweep_spec_no": [],
        "structural_valid": [],
        "row_utc_ns": [],
        "corrected_latitude": [],
        "corrected_longitude": [],
        "corrected_polarity": [],
        "legacy_latitude": [],
        "legacy_longitude": [],
        "legacy_polarity": [],
        "equatorial_latitude": [],
        "equatorial_longitude": [],
        "equatorial_polarity": [],
        "frame_mixed_latitude": [],
        "frame_mixed_longitude": [],
        "frame_mixed_polarity": [],
        "submitted_legacy_latitude": [],
        "submitted_legacy_longitude": [],
        "submitted_legacy_polarity": [],
    }
    source_paths: list[str] = []
    for shard in shards:
        with np.load(shard, allow_pickle=False) as data:
            sweep_count = len(data["sweep_spec_no"])
            file_index = int(data["file_index"])
            source_paths.append(str(data["source_path"]))
            arrays["source_file_index"].append(
                np.full(sweep_count, file_index, dtype=np.int64)
            )
            for key in arrays:
                if key != "source_file_index":
                    arrays[key].append(np.array(data[key]))
    merged = {key: np.concatenate(parts) for key, parts in arrays.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        source_paths=np.asarray(source_paths),
        **merged,
    )
    print(f"Merged {len(shards)} shards and {len(merged['sweep_spec_no'])} sweeps")
    print(f"Wrote {args.output}")
    return 0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    extract = subparsers.add_parser("extract", help="extract daily geometry shards")
    extract.add_argument("--configuration", type=Path, default=DEFAULT_CONFIGURATION)
    extract.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    extract.add_argument(
        "--shard-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "geometry-shards"
    )
    extract.add_argument("--census", type=Path, default=CENSUS_PATH)
    extract.add_argument("--workers", type=int, default=4)
    extract.add_argument("--start-index", type=int, default=0)
    extract.add_argument("--limit", type=int)
    extract.set_defaults(handler=run_extract)

    merge = subparsers.add_parser("merge", help="merge daily geometry shards")
    merge.add_argument(
        "--shard-dir", type=Path, default=DEFAULT_ARTIFACT_DIR / "geometry-shards"
    )
    merge.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "geometry.npz"
    )
    merge.set_defaults(handler=run_merge)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    if getattr(args, "workers", 1) < 1:
        raise ValueError("workers must be positive")
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
