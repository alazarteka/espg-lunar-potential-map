"""Archive census and oracle-pattern audit for hard-bin identification.

The empirical workflow uses calibrated files only for their recorded energy,
time, magnetic-field, and look-direction geometry. It does not inspect flux
values or classify observed loss cones.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from src import config
from src.losscone.identifiability import (
    DELTA_V_DOMAIN,
    HardBinDesign,
    boundary_brackets_from_pattern,
    generate_hard_pattern,
    identify_delta_v_interval,
    standard_nuisance_scenarios,
)
from src.losscone.pitch_angle import compute_pitch_angle_geometry

MAX_WORKERS = 8
MAX_TIME_LIMIT_MINUTES = 55.0
DEFAULT_OUTPUT = Path("artifacts/diagnostics/hardbin-identifiability")
TRUTH_DELTA_V = (
    -2000.0,
    -1000.0,
    -500.0,
    -200.0,
    -100.0,
    -50.0,
    -20.0,
    0.0,
    20.0,
    200.0,
)
TRUTH_SPACECRAFT_V = (-100.0, 0.0, 10.0)
TRUTH_FIELD_RATIO = (0.3, 0.5, 0.8, 1.1)
SCENARIO_ORDER = (
    "oracle",
    "common_r",
    "sc_pm5",
    "sc_pm20",
    "affine_drift_20",
    "broad_sc",
)
SPACECRAFT_SAFETY_THRESHOLDS_V = (
    -100.0,
    -95.0,
    -80.0,
    -70.0,
    0.0,
    5.0,
    10.0,
    15.0,
    20.0,
    30.0,
    40.0,
    150.0,
)

_GEOMETRY_COLUMNS = [
    config.UTC_COLUMN,
    config.TIME_COLUMN,
    config.ENERGY_COLUMN,
    config.SPEC_NO_COLUMN,
    *config.MAG_COLS,
    *config.PHI_COLS,
]


def validate_workers(workers: int) -> int:
    """Validate the project-wide audit worker cap."""
    workers = int(workers)
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"workers must be between 1 and {MAX_WORKERS}")
    return workers


def validate_time_limit(minutes: float) -> float:
    """Validate the sub-hour command budget."""
    minutes = float(minutes)
    if not 0 < minutes <= MAX_TIME_LIMIT_MINUTES:
        raise ValueError(f"time limit must be in (0, {MAX_TIME_LIMIT_MINUTES}] minutes")
    return minutes


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Hash a source or result artifact without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _implementation_sha256(project_root: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/losscone/identifiability.py",
        "src/losscone/identifiability_audit.py",
        "src/losscone/pitch_angle.py",
    ):
        digest.update(relative.encode())
        digest.update((project_root / relative).read_bytes())
    return digest.hexdigest()


def _atomic_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.gz")
    frame.to_csv(
        temporary,
        index=False,
        compression={"method": "gzip", "mtime": 0},
    )
    temporary.replace(path)


def _atomic_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _read_geometry_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=" ",
        engine="c",
        skipinitialspace=True,
        header=None,
        names=config.ALL_COLS,
        usecols=_GEOMETRY_COLUMNS,
    )


def _geometry_shard_map(root: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    for path in sorted(root.glob("*.npz")):
        stem = path.stem.split("-", maxsplit=1)[-1]
        if stem in mapping:
            raise ValueError(f"duplicate geometry shard for {stem}")
        mapping[stem] = path
    return mapping


def _source_relative(path: Path, project_root: Path) -> str:
    try:
        return str(path.resolve().relative_to(project_root.resolve()))
    except ValueError:
        return str(path.resolve())


def _load_polarity_shard(path: Path) -> tuple[dict[int, int], dict[str, np.ndarray]]:
    with np.load(path) as payload:
        required = {"sweep_spec_no", "structural_valid", "corrected_polarity"}
        missing = required.difference(payload.files)
        if missing:
            raise ValueError(f"geometry shard {path} lacks {sorted(missing)}")
        arrays = {name: np.asarray(payload[name]).copy() for name in required}
    lookup = {int(value): i for i, value in enumerate(arrays["sweep_spec_no"])}
    return lookup, arrays


def _row_times_seconds(group: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    timestamps = pd.to_datetime(group[config.UTC_COLUMN], utc=True, errors="coerce")
    valid = ~pd.isna(timestamps)
    seconds = np.full(len(group), np.nan, dtype=np.float64)
    if np.any(valid):
        seconds[valid] = timestamps[valid].astype("int64") / 1.0e9
    return seconds, pd.DatetimeIndex(timestamps)


def _energy_signature(energies: np.ndarray) -> str:
    values = np.round(np.asarray(energies, dtype=np.float64), decimals=6)
    return hashlib.sha256(values.tobytes()).hexdigest()[:16]


def _threshold_metrics(pitches: np.ndarray) -> dict[str, float | int]:
    row_max_gaps: list[float] = []
    all_gaps: list[float] = []
    reflected_cells = 0
    reflected_rows = 0
    for row in pitches:
        reflected = row[np.isfinite(row) & (row > 90.0)]
        reflected_cells += int(reflected.size)
        if reflected.size == 0:
            continue
        reflected_rows += 1
        thresholds = np.unique(np.sin(np.deg2rad(reflected)) ** 2)
        edges = np.concatenate(([0.0], np.sort(thresholds), [1.0]))
        gaps = np.diff(edges)
        row_max_gaps.append(float(np.max(gaps)))
        all_gaps.extend(float(value) for value in gaps)
    return {
        "reflected_cells": reflected_cells,
        "reflected_rows": reflected_rows,
        "threshold_gap_median": float(np.median(all_gaps)) if all_gaps else math.nan,
        "row_max_gap_median": (
            float(np.median(row_max_gaps)) if row_max_gaps else math.nan
        ),
        "row_max_gap_max": max(row_max_gaps, default=math.nan),
    }


def _energy_leverage(
    energies: np.ndarray,
    spacecraft_v: float,
    valid_rows: np.ndarray | None = None,
) -> float:
    if valid_rows is None:
        valid = np.ones(np.asarray(energies).shape, dtype=bool)
    else:
        valid = np.asarray(valid_rows, dtype=bool)
    safe = valid & np.isfinite(energies) & (energies > float(spacecraft_v))
    if np.count_nonzero(safe) < 2:
        return math.nan
    z = 1.0 / (energies[safe] - float(spacecraft_v))
    return float(np.max(z) - np.min(z))


def _invalid_reasons(
    group: pd.DataFrame,
    *,
    shard_index: int | None,
    shard_arrays: dict[str, np.ndarray],
    row_times_s: np.ndarray,
) -> list[str]:
    reasons: list[str] = []
    if len(group) != config.SWEEP_ROWS:
        reasons.append("row_count")
    if shard_index is None:
        reasons.append("missing_polarity_shard_entry")
    elif not bool(shard_arrays["structural_valid"][shard_index]):
        reasons.append("geometry_structural_invalid")
    finite_times = row_times_s[np.isfinite(row_times_s)]
    if finite_times.size < 2 or np.ptp(finite_times) <= 0:
        reasons.append("invalid_row_times")
    return reasons


def _row_invalidity_summary(
    group: pd.DataFrame,
    *,
    shard_index: int | None,
    shard_arrays: dict[str, np.ndarray],
    row_times_s: np.ndarray,
) -> str:
    energies = group[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
    invalid_energy = ~np.isfinite(energies) | (energies <= 0)
    magnetic = group[config.MAG_COLS].to_numpy(dtype=np.float64)
    magnitude = np.linalg.norm(magnetic, axis=1)
    invalid_magnetic = (
        ~np.isfinite(magnetic).all(axis=1)
        | ~np.isfinite(magnitude)
        | (magnitude <= 1.0e-9)
        | (magnitude >= 1.0e3)
    )
    invalid_time = ~np.isfinite(row_times_s)
    disconnected = np.ones(len(group), dtype=bool)
    if shard_index is not None:
        polarity = np.asarray(shard_arrays["corrected_polarity"][shard_index])
        if polarity.shape == (len(group),):
            disconnected = ~np.isfinite(polarity) | (polarity == 0)
    counts = {
        "invalid_energy_rows": int(np.count_nonzero(invalid_energy)),
        "invalid_magnetic_rows": int(np.count_nonzero(invalid_magnetic)),
        "invalid_time_rows": int(np.count_nonzero(invalid_time)),
        "disconnected_polarity_rows": int(np.count_nonzero(disconnected)),
    }
    return ";".join(f"{name}={count}" for name, count in counts.items() if count)


def _build_design(
    group: pd.DataFrame,
    *,
    shard_index: int,
    shard_arrays: dict[str, np.ndarray],
    thetas: np.ndarray,
) -> HardBinDesign:
    polarity = np.asarray(shard_arrays["corrected_polarity"][shard_index])
    if polarity.shape != (len(group),):
        raise ValueError("corrected polarity does not match the sweep row count")
    phis = group[config.PHI_COLS].to_numpy(dtype=np.float64)
    magnetic = group[config.MAG_COLS].to_numpy(dtype=np.float64)
    geometry = compute_pitch_angle_geometry(phis, thetas, magnetic, polarity)
    row_times_s, _ = _row_times_seconds(group)
    return HardBinDesign(
        energies_ev=group[config.ENERGY_COLUMN].to_numpy(dtype=np.float64),
        pitches_deg=geometry.pitch_angles,
        row_times_s=row_times_s,
        valid_rows=geometry.valid_rows,
    )


def _census_one_day(task: tuple[str, str, str, str, bool]) -> dict[str, Any]:
    data_name, shard_name, theta_name, output_name, overwrite = task
    data_path = Path(data_name)
    shard_path = Path(shard_name)
    theta_path = Path(theta_name)
    output_path = Path(output_name)
    manifest_path = output_path.with_suffix("").with_suffix(".json")
    if output_path.exists() and manifest_path.exists() and not overwrite:
        return json.loads(manifest_path.read_text())

    started = time.monotonic()
    table = _read_geometry_table(data_path)
    thetas = np.loadtxt(theta_path, dtype=np.float64)
    lookup, shard_arrays = _load_polarity_shard(shard_path)
    project_root = Path(__file__).resolve().parents[2]
    records: list[dict[str, Any]] = []

    for raw_spec, group in table.groupby(config.SPEC_NO_COLUMN, sort=False):
        spec_no = int(raw_spec)
        group = group.reset_index(drop=True)
        row_times_s, timestamps = _row_times_seconds(group)
        shard_index = lookup.get(spec_no)
        reasons = _invalid_reasons(
            group,
            shard_index=shard_index,
            shard_arrays=shard_arrays,
            row_times_s=row_times_s,
        )
        row_invalidity = _row_invalidity_summary(
            group,
            shard_index=shard_index,
            shard_arrays=shard_arrays,
            row_times_s=row_times_s,
        )
        design: HardBinDesign | None = None
        if not reasons and shard_index is not None:
            try:
                design = _build_design(
                    group,
                    shard_index=shard_index,
                    shard_arrays=shard_arrays,
                    thetas=thetas,
                )
            except (ValueError, FloatingPointError):
                reasons.append("pitch_geometry_failure")
        if design is not None and not np.any(design.valid_rows):
            reasons.append("no_valid_geometry_rows")
            design = None

        energies = group[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
        fixed_valid = (
            np.asarray(design.valid_rows, dtype=bool)
            if design is not None
            else np.zeros(len(group), dtype=bool)
        )
        first_timestamp = timestamps[0] if len(timestamps) else pd.NaT
        record: dict[str, Any] = {
            "source_path": _source_relative(data_path, project_root),
            "geometry_shard": _source_relative(shard_path, project_root),
            "source_stem": data_path.stem,
            "spec_no": spec_no,
            "utc_start": (
                first_timestamp.isoformat() if not pd.isna(first_timestamp) else ""
            ),
            "month": (
                first_timestamp.strftime("%Y-%m")
                if not pd.isna(first_timestamp)
                else "unknown"
            ),
            "structural_valid": not reasons,
            "invalid_reason": ";".join(reasons),
            "row_invalidity": row_invalidity,
            "row_count": len(group),
            "valid_geometry_rows": int(np.count_nonzero(fixed_valid)),
            "energy_signature": _energy_signature(energies),
            "energy_min_ev": float(np.nanmin(energies)),
            "energy_max_ev": float(np.nanmax(energies)),
            "energy_leverage_sc_m100": _energy_leverage(energies, -100.0, fixed_valid),
            "energy_leverage_sc_0": _energy_leverage(energies, 0.0, fixed_valid),
            "energy_leverage_sc_10": _energy_leverage(energies, 10.0, fixed_valid),
        }
        record.update(
            {
                f"safe_rows_sc_{str(threshold).replace('-', 'm').replace('.', 'p')}": (
                    int(np.count_nonzero(fixed_valid & (energies > threshold)))
                )
                for threshold in SPACECRAFT_SAFETY_THRESHOLDS_V
            }
        )
        if design is None:
            record.update(
                {
                    "reflected_cells": 0,
                    "reflected_rows": 0,
                    "threshold_gap_median": math.nan,
                    "row_max_gap_median": math.nan,
                    "row_max_gap_max": math.nan,
                }
            )
        else:
            record.update(_threshold_metrics(design.pitches_deg))
        records.append(record)

    frame = pd.DataFrame.from_records(records)
    _atomic_dataframe_csv(frame, output_path)
    manifest = {
        "complete": True,
        "data_path": _source_relative(data_path, project_root),
        "data_sha256": sha256_file(data_path),
        "geometry_path": _source_relative(shard_path, project_root),
        "geometry_sha256": sha256_file(shard_path),
        "output_path": _source_relative(output_path, project_root),
        "output_sha256": sha256_file(output_path),
        "sweeps": len(frame),
        "valid_sweeps": int(frame["structural_valid"].sum()),
        "elapsed_seconds": time.monotonic() - started,
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def _bounded_parallel_map[T, R](
    function: Callable[[T], R],
    tasks: list[T],
    *,
    workers: int,
    time_limit_minutes: float,
) -> tuple[list[R], int]:
    """Run bounded tasks without ever having more than eight active workers."""
    workers = validate_workers(workers)
    limit = validate_time_limit(time_limit_minutes) * 60.0
    deadline = time.monotonic() + limit
    results: list[R] = []
    next_task = 0

    with ProcessPoolExecutor(max_workers=workers) as executor:
        active = set()
        while next_task < len(tasks) and len(active) < workers:
            active.add(executor.submit(function, tasks[next_task]))
            next_task += 1
        while active:
            done, active = wait(active, return_when=FIRST_COMPLETED)
            results.extend(future.result() for future in done)
            if time.monotonic() >= deadline:
                break
            while next_task < len(tasks) and len(active) < workers:
                active.add(executor.submit(function, tasks[next_task]))
                next_task += 1
        results.extend(future.result() for future in active)
    return results, next_task


def run_census(args: argparse.Namespace) -> int:
    data_root = Path(args.data_root)
    geometry_root = Path(args.geometry_root)
    output_root = Path(args.output_dir)
    shards_root = output_root / "census_shards"
    geometry = _geometry_shard_map(geometry_root)
    data_files = sorted(data_root.rglob("3D*.TAB"))
    if args.max_files is not None:
        data_files = data_files[: int(args.max_files)]
    tasks: list[tuple[str, str, str, str, bool]] = []
    missing: list[str] = []
    for data_path in data_files:
        shard = geometry.get(data_path.stem)
        if shard is None:
            missing.append(data_path.stem)
            continue
        output = shards_root / f"{data_path.stem}.csv.gz"
        tasks.append(
            (
                str(data_path),
                str(shard),
                str(args.theta_file),
                str(output),
                bool(args.overwrite),
            )
        )
    if missing:
        raise FileNotFoundError(f"missing geometry shards for {missing[:10]}")
    manifests, submitted = _bounded_parallel_map(
        _census_one_day,
        tasks,
        workers=args.workers,
        time_limit_minutes=args.time_limit_minutes,
    )
    shard_paths = sorted(shards_root.glob("3D*.csv.gz"))
    census = pd.concat((pd.read_csv(path) for path in shard_paths), ignore_index=True)
    census_path = output_root / "geometry_census.csv.gz"
    _atomic_dataframe_csv(census, census_path)
    payload = {
        "complete": submitted == len(tasks),
        "requested_files": len(tasks),
        "submitted_files": submitted,
        "completed_manifests": len(manifests),
        "sweeps": len(census),
        "valid_sweeps": int(census["structural_valid"].sum()),
        "census_sha256": sha256_file(census_path),
        "worker_limit": validate_workers(args.workers),
        "gpu_count": 0,
    }
    _atomic_json(payload, output_root / "census_manifest.json")
    return 0 if payload["complete"] else 3


def _evenly_spaced_indices(size: int, count: int) -> np.ndarray:
    if size <= count:
        return np.arange(size, dtype=int)
    return np.unique(np.linspace(0, size - 1, count, dtype=int))


def deterministic_stratified_sample(
    census: pd.DataFrame,
    *,
    per_month: int = 64,
) -> pd.DataFrame:
    """Select geometry-only monthly strata reproducibly."""
    if per_month < 1:
        raise ValueError("per_month must be positive")
    valid = census[census["structural_valid"].astype(bool)].copy()
    valid = valid[np.isfinite(valid["row_max_gap_median"])].copy()
    if valid.empty:
        return valid

    forced_indices: set[int] = set()
    reference = valid[(valid["source_stem"] == "3D980116") & (valid["spec_no"] == 1)]
    forced_indices.update(int(index) for index in reference.index)
    extreme_columns = (
        "valid_geometry_rows",
        "energy_min_ev",
        "energy_max_ev",
        "energy_leverage_sc_m100",
        "energy_leverage_sc_0",
        "energy_leverage_sc_10",
        "threshold_gap_median",
        "row_max_gap_median",
        "row_max_gap_max",
        "safe_rows_sc_150p0",
        "reflected_cells",
        "reflected_rows",
        *(
            f"safe_rows_sc_{str(value).replace('-', 'm').replace('.', 'p')}"
            for value in SPACECRAFT_SAFETY_THRESHOLDS_V
        ),
    )
    for column in extreme_columns:
        if column not in valid.columns:
            continue
        finite = valid[np.isfinite(valid[column])]
        if finite.empty:
            continue
        forced_indices.add(int(finite[column].idxmin()))
        forced_indices.add(int(finite[column].idxmax()))

    selected_indices: set[int] = set()
    for _, monthly in valid.groupby("month", sort=True):
        monthly = monthly.sort_values(["utc_start", "source_path", "spec_no"])
        try:
            monthly["pitch_quantile"] = pd.qcut(
                monthly["row_max_gap_median"],
                q=4,
                labels=False,
                duplicates="drop",
            ).fillna(0)
        except ValueError:
            monthly["pitch_quantile"] = 0
        forced_month = set(monthly.index).intersection(forced_indices)
        if len(forced_month) >= per_month:
            ordered_forced = monthly.loc[sorted(forced_month)].sort_values(
                ["utc_start", "source_path", "spec_no"]
            )
            positions = _evenly_spaced_indices(len(ordered_forced), per_month)
            selected_indices.update(
                int(index) for index in ordered_forced.iloc[positions].index
            )
            continue

        available = monthly.drop(index=list(forced_month))
        remaining_count = min(per_month - len(forced_month), len(available))
        candidate_rows: list[tuple[int, int, str, int]] = []
        for (signature, pitch_quantile), stratum in monthly.groupby(
            ["energy_signature", "pitch_quantile"], sort=True
        ):
            if forced_month.intersection(stratum.index):
                continue
            ordered = stratum.sort_values(["utc_start", "source_path", "spec_no"])
            midpoint = (len(ordered) - 1) // 2
            candidate_rows.append(
                (
                    int(ordered.index[midpoint]),
                    len(ordered),
                    str(signature),
                    int(pitch_quantile),
                )
            )
        if len(candidate_rows) > remaining_count:
            # When the quota cannot cover every composite stratum, retain the
            # largest strata first and break ties by stable category labels.
            candidate_rows.sort(key=lambda item: (-item[1], item[2], item[3]))
            candidate_rows = candidate_rows[:remaining_count]
        candidate_frame = monthly.loc[
            sorted(item[0] for item in candidate_rows)
        ].sort_values(["utc_start", "source_path", "spec_no"])
        if len(candidate_frame) < remaining_count:
            remaining = available.drop(index=candidate_frame.index)
            positions = _evenly_spaced_indices(
                len(remaining), remaining_count - len(candidate_frame)
            )
            candidate_frame = pd.concat([candidate_frame, remaining.iloc[positions]])
        selected_indices.update(forced_month)
        selected_indices.update(int(index) for index in candidate_frame.index)

    sample = (
        valid.loc[sorted(selected_indices)]
        .copy()
        .sort_values(["month", "utc_start", "source_path", "spec_no"])
    )
    sample["forced_reference"] = (sample["source_stem"] == "3D980116") & (
        sample["spec_no"] == 1
    )
    sample["sample_rank"] = np.arange(len(sample), dtype=int)
    return sample


def run_sample(args: argparse.Namespace) -> int:
    census = pd.read_csv(args.census)
    sample = deterministic_stratified_sample(census, per_month=args.per_month)
    output = Path(args.output)
    _atomic_dataframe_csv(sample, output)
    manifest = {
        "complete": True,
        "census": str(args.census),
        "census_sha256": sha256_file(Path(args.census)),
        "per_month": int(args.per_month),
        "sample_rows": len(sample),
        "sample_sha256": sha256_file(output),
        "selection_uses_flux": False,
    }
    _atomic_json(manifest, output.with_suffix("").with_suffix(".json"))
    return 0


def _designs_for_source(
    data_path: Path,
    shard_path: Path,
    theta_path: Path,
    requested_specs: set[int],
) -> dict[int, HardBinDesign]:
    table = _read_geometry_table(data_path)
    lookup, shard_arrays = _load_polarity_shard(shard_path)
    thetas = np.loadtxt(theta_path, dtype=np.float64)
    designs: dict[int, HardBinDesign] = {}
    for raw_spec, group in table.groupby(config.SPEC_NO_COLUMN, sort=False):
        spec_no = int(raw_spec)
        if spec_no not in requested_specs:
            continue
        shard_index = lookup.get(spec_no)
        if shard_index is None or len(group) != config.SWEEP_ROWS:
            continue
        designs[spec_no] = _build_design(
            group.reset_index(drop=True),
            shard_index=shard_index,
            shard_arrays=shard_arrays,
            thetas=thetas,
        )
    return designs


def _audit_one_month(
    task: tuple[str, str, str, str, str, float, bool, str, str, str],
) -> dict[str, Any]:
    (
        month,
        sample_name,
        project_root_name,
        theta_name,
        output_name,
        time_limit,
        overwrite,
        sample_sha256,
        theta_sha256,
        implementation_sha256,
    ) = task
    sample = pd.read_csv(sample_name)
    monthly = sample[sample["month"] == month].copy()
    output = Path(output_name)
    manifest_path = output.with_suffix("").with_suffix(".json")
    if output.exists() and manifest_path.exists() and not overwrite:
        prior_manifest = json.loads(manifest_path.read_text())
        if (
            prior_manifest.get("complete", False)
            and prior_manifest.get("sample_sha256") == sample_sha256
            and prior_manifest.get("theta_sha256") == theta_sha256
            and prior_manifest.get("implementation_sha256") == implementation_sha256
        ):
            return prior_manifest
    started = time.monotonic()
    deadline = started + float(time_limit) * 60.0
    project_root = Path(project_root_name)
    theta_path = Path(theta_name)
    records: list[dict[str, Any]] = []
    complete = True
    missing_designs: list[str] = []

    for source_path, source_rows in monthly.groupby("source_path", sort=True):
        data_path = project_root / str(source_path)
        shard_path = project_root / str(source_rows.iloc[0]["geometry_shard"])
        requested = {int(value) for value in source_rows["spec_no"]}
        designs = _designs_for_source(data_path, shard_path, theta_path, requested)
        for sample_row in source_rows.itertuples(index=False):
            design = designs.get(int(sample_row.spec_no))
            if design is None:
                missing_designs.append(
                    f"{sample_row.source_stem}:spec={int(sample_row.spec_no)}"
                )
                continue
            for truth_delta in TRUTH_DELTA_V:
                for truth_spacecraft in TRUTH_SPACECRAFT_V:
                    for truth_r in TRUTH_FIELD_RATIO:
                        pattern = generate_hard_pattern(
                            design,
                            delta_v_midpoint=truth_delta,
                            spacecraft_midpoint_v=truth_spacecraft,
                            field_ratio=truth_r,
                        )
                        truth_valid_rows = (
                            design.valid_rows
                            & np.isfinite(design.tau)
                            & (design.energies_ev > truth_spacecraft + 1.0e-9)
                        )
                        brackets = boundary_brackets_from_pattern(
                            design,
                            pattern,
                            valid_rows=truth_valid_rows,
                        )
                        for scenario in standard_nuisance_scenarios(
                            truth_field_ratio=truth_r,
                            truth_spacecraft_potential_v=truth_spacecraft,
                        ):
                            result = identify_delta_v_interval(
                                design, brackets, scenario
                            )
                            row = {
                                "month": month,
                                "source_path": source_path,
                                "source_stem": sample_row.source_stem,
                                "spec_no": int(sample_row.spec_no),
                                "truth_delta_v": truth_delta,
                                "truth_spacecraft_v": truth_spacecraft,
                                "truth_field_ratio": truth_r,
                                "scenario_id": scenario.scenario_id,
                                **asdict(result),
                            }
                            row["status"] = result.status.value
                            row["delta_v_width"] = (
                                result.delta_v_upper - result.delta_v_lower
                            )
                            row["truth_in_closure"] = bool(
                                np.isfinite(result.delta_v_lower)
                                and result.delta_v_lower <= truth_delta
                                and truth_delta <= result.delta_v_upper
                            )
                            records.append(row)
                if time.monotonic() >= deadline:
                    complete = False
                    break
            if not complete:
                break
        if not complete:
            break

    frame = pd.DataFrame.from_records(records)
    expected_results = len(monthly) * len(TRUTH_DELTA_V) * len(TRUTH_SPACECRAFT_V)
    expected_results *= len(TRUTH_FIELD_RATIO) * 6
    complete &= not missing_designs and len(frame) == expected_results
    _atomic_dataframe_csv(frame, output)
    manifest = {
        "complete": complete,
        "month": month,
        "sample_sweeps": len(monthly),
        "sample_sha256": sample_sha256,
        "theta_sha256": theta_sha256,
        "implementation_sha256": implementation_sha256,
        "missing_designs": missing_designs,
        "expected_result_rows": expected_results,
        "result_rows": len(frame),
        "truth_grid": {
            "delta_v": TRUTH_DELTA_V,
            "spacecraft_v": TRUTH_SPACECRAFT_V,
            "field_ratio": TRUTH_FIELD_RATIO,
        },
        "delta_v_domain": DELTA_V_DOMAIN,
        "output_sha256": sha256_file(output),
        "elapsed_seconds": time.monotonic() - started,
        "worker_pid": os.getpid(),
    }
    _atomic_json(manifest, manifest_path)
    return manifest


def run_audit(args: argparse.Namespace) -> int:
    sample = pd.read_csv(args.sample)
    sample_sha256 = sha256_file(Path(args.sample))
    theta_sha256 = sha256_file(Path(args.theta_file))
    months = sorted(str(value) for value in sample["month"].unique())
    if args.month:
        months = [month for month in months if month == args.month]
    output_root = Path(args.output_dir) / "audit_shards"
    project_root = Path(__file__).resolve().parents[2]
    implementation_sha256 = _implementation_sha256(project_root)
    tasks = [
        (
            month,
            str(args.sample),
            str(project_root),
            str(args.theta_file),
            str(output_root / f"{month}.csv.gz"),
            float(args.time_limit_minutes),
            bool(args.overwrite),
            sample_sha256,
            theta_sha256,
            implementation_sha256,
        )
        for month in months
    ]
    manifests, submitted = _bounded_parallel_map(
        _audit_one_month,
        tasks,
        workers=args.workers,
        time_limit_minutes=args.time_limit_minutes,
    )
    complete = submitted == len(tasks) and all(item["complete"] for item in manifests)
    payload = {
        "complete": complete,
        "requested_months": months,
        "submitted_months": submitted,
        "completed_months": [item["month"] for item in manifests if item["complete"]],
        "sample_sha256": sample_sha256,
        "theta_sha256": theta_sha256,
        "implementation_sha256": implementation_sha256,
        "worker_limit": validate_workers(args.workers),
        "gpu_count": 0,
    }
    _atomic_json(payload, Path(args.output_dir) / "audit_manifest.json")
    return 0 if complete else 3


def _quantile(series: pd.Series, value: float) -> float:
    finite = series[np.isfinite(series)]
    return float(finite.quantile(value)) if len(finite) else math.nan


def _identified_set_summary_row(frame: pd.DataFrame) -> dict[str, float | int]:
    widths = frame["delta_v_width"]
    return {
        "results": len(frame),
        "truth_coverage_fraction": float(frame["truth_in_closure"].mean()),
        "sharp_fraction": float((frame["status"] == "sharp_interval").mean()),
        "full_domain_fraction": float((frame["status"] == "full_domain").mean()),
        "infeasible_fraction": float((frame["status"] == "infeasible_pattern").mean()),
        "invalid_design_fraction": float((frame["status"] == "invalid_design").mean()),
        "unresolved_fraction": float(
            (frame["status"] == "unresolved_numerical").mean()
        ),
        "width_p10_v": _quantile(widths, 0.1),
        "width_p50_v": _quantile(widths, 0.5),
        "width_p90_v": _quantile(widths, 0.9),
        "width_p99_v": _quantile(widths, 0.99),
        "width_le_20v_fraction": float((widths <= 20.0).mean()),
        "width_le_50v_fraction": float((widths <= 50.0).mean()),
        "width_le_100v_fraction": float((widths <= 100.0).mean()),
        "width_le_200v_fraction": float((widths <= 200.0).mean()),
        "informative_rows_p50": _quantile(frame["informative_rows"], 0.5),
        "used_rows_p50": _quantile(frame["used_rows"], 0.5),
    }


def run_summarize(args: argparse.Namespace) -> int:
    output_root = Path(args.output_dir)
    compact_root = Path(args.compact_output)
    census_manifest_path = Path(args.census).parent / "census_manifest.json"
    if census_manifest_path.exists():
        census_manifest = json.loads(census_manifest_path.read_text())
        if not census_manifest.get("complete", False):
            raise RuntimeError("cannot summarize an incomplete geometry census")
    audit_manifest_path = output_root / "audit_manifest.json"
    if not audit_manifest_path.exists():
        raise FileNotFoundError("audit manifest is missing")
    audit_manifest = json.loads(audit_manifest_path.read_text())
    if not audit_manifest.get("complete", False):
        raise RuntimeError("cannot summarize an incomplete audit")
    census = pd.read_csv(args.census)
    audit_paths = sorted((output_root / "audit_shards").glob("*.csv.gz"))
    if not audit_paths:
        raise FileNotFoundError("no audit shards found")
    incomplete_shards: list[str] = []
    for path in audit_paths:
        manifest_path = path.with_suffix("").with_suffix(".json")
        if not manifest_path.exists():
            incomplete_shards.append(f"{path.name}:missing manifest")
            continue
        manifest = json.loads(manifest_path.read_text())
        if not manifest.get("complete", False):
            incomplete_shards.append(f"{path.name}:incomplete")
    if incomplete_shards:
        raise RuntimeError(
            "cannot summarize incomplete audit shards: "
            + ", ".join(incomplete_shards[:10])
        )
    audit = pd.concat((pd.read_csv(path) for path in audit_paths), ignore_index=True)

    census_rows: list[dict[str, Any]] = []
    for month, frame in census.groupby("month", sort=True):
        valid = frame[frame["structural_valid"].astype(bool)]
        census_rows.append(
            {
                "month": month,
                "sweeps": len(frame),
                "valid_sweeps": len(valid),
                "valid_fraction": len(valid) / len(frame) if len(frame) else math.nan,
                "valid_geometry_rows_p10": _quantile(valid["valid_geometry_rows"], 0.1),
                "valid_geometry_rows_p50": _quantile(valid["valid_geometry_rows"], 0.5),
                "row_invalidity_fraction": float(
                    valid["row_invalidity"].fillna("").ne("").mean()
                ),
                "reflected_cells_p50": _quantile(valid["reflected_cells"], 0.5),
                "row_max_gap_median": _quantile(valid["row_max_gap_median"], 0.5),
                "row_max_gap_p90": _quantile(valid["row_max_gap_median"], 0.9),
                "energy_leverage_sc_0_p50": _quantile(
                    valid["energy_leverage_sc_0"], 0.5
                ),
                "safe_rows_sc_150_mean": float(valid["safe_rows_sc_150p0"].mean()),
            }
        )
    census_summary = pd.DataFrame(census_rows)

    scenario_rows: list[dict[str, Any]] = []
    monthly_rows: list[dict[str, Any]] = []
    truth_rows: list[dict[str, Any]] = []
    for scenario in SCENARIO_ORDER:
        frame = audit[audit["scenario_id"] == scenario]
        scenario_rows.append(
            {"scenario_id": scenario, **_identified_set_summary_row(frame)}
        )
    for month in sorted(audit["month"].unique()):
        for scenario in SCENARIO_ORDER:
            frame = audit[
                (audit["month"] == month) & (audit["scenario_id"] == scenario)
            ]
            monthly_rows.append(
                {
                    "month": month,
                    "scenario_id": scenario,
                    **_identified_set_summary_row(frame),
                }
            )
    for scenario in SCENARIO_ORDER:
        for truth_delta in TRUTH_DELTA_V:
            frame = audit[
                (audit["scenario_id"] == scenario)
                & (audit["truth_delta_v"] == truth_delta)
            ]
            truth_rows.append(
                {
                    "scenario_id": scenario,
                    "truth_delta_v": truth_delta,
                    "results": len(frame),
                    "truth_coverage_fraction": float(frame["truth_in_closure"].mean()),
                    "width_p50_v": _quantile(frame["delta_v_width"], 0.5),
                    "width_p90_v": _quantile(frame["delta_v_width"], 0.9),
                    "full_domain_fraction": float(
                        (frame["status"] == "full_domain").mean()
                    ),
                }
            )

    scenario_summary = pd.DataFrame(scenario_rows)
    monthly_summary = pd.DataFrame(monthly_rows)
    truth_summary = pd.DataFrame(truth_rows)
    compact_root.mkdir(parents=True, exist_ok=True)
    census_path = compact_root / "geometry_census_monthly.csv"
    scenario_path = compact_root / "identified_set_scenarios.csv"
    monthly_path = compact_root / "identified_set_monthly.csv"
    truth_path = compact_root / "identified_set_by_truth.csv"
    census_summary.to_csv(census_path, index=False)
    scenario_summary.to_csv(scenario_path, index=False)
    monthly_summary.to_csv(monthly_path, index=False)
    truth_summary.to_csv(truth_path, index=False)
    summary = {
        "estimand": "delta_v = V_surface - V_spacecraft",
        "census_sweeps": len(census),
        "census_valid_sweeps": int(census["structural_valid"].sum()),
        "audit_results": len(audit),
        "truth_coverage_fraction": float(audit["truth_in_closure"].mean()),
        "sample_sha256": audit_manifest["sample_sha256"],
        "theta_sha256": audit_manifest["theta_sha256"],
        "implementation_sha256": audit_manifest["implementation_sha256"],
        "truth_grid": {
            "delta_v": TRUTH_DELTA_V,
            "spacecraft_v": TRUTH_SPACECRAFT_V,
            "field_ratio": TRUTH_FIELD_RATIO,
        },
        "scenario_order": SCENARIO_ORDER,
        "resource_contract": {
            "gpu_count": 0,
            "maximum_workers": MAX_WORKERS,
            "maximum_time_limit_minutes": MAX_TIME_LIMIT_MINUTES,
            "numerical_library_threads_per_worker": 1,
        },
        "reproduce_commands": [
            "uv run python "
            "scripts/diagnostics/losscone_hardbin_identifiability.py "
            "census --workers 8",
            "uv run python "
            "scripts/diagnostics/losscone_hardbin_identifiability.py sample",
            "uv run python "
            "scripts/diagnostics/losscone_hardbin_identifiability.py "
            "audit --workers 8",
            "uv run python "
            "scripts/diagnostics/losscone_hardbin_identifiability.py summarize",
        ],
        "large_artifact_hashes": {
            str(Path(args.census)): sha256_file(Path(args.census)),
            **{str(path): sha256_file(path) for path in audit_paths},
        },
        "compact_hashes": {
            path.name: sha256_file(path)
            for path in (census_path, scenario_path, monthly_path, truth_path)
        },
    }
    _atomic_json(summary, compact_root / "summary.json")
    return 0


def _add_resource_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--time-limit-minutes", type=float, default=MAX_TIME_LIMIT_MINUTES
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Geometry-only hard-bin loss-cone identification audit."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    census = subparsers.add_parser("census", help="Census all archived sweep designs")
    census.add_argument("--data-root", type=Path, default=config.DATA_DIR)
    census.add_argument(
        "--geometry-root",
        type=Path,
        default=Path("artifacts/experiment-008/geometry-shards"),
    )
    census.add_argument(
        "--theta-file", type=Path, default=config.DATA_DIR / config.THETA_FILE
    )
    census.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    census.add_argument("--max-files", type=int)
    census.add_argument("--overwrite", action="store_true")
    _add_resource_arguments(census)

    sample = subparsers.add_parser("sample", help="Select monthly geometry strata")
    sample.add_argument(
        "--census", type=Path, default=DEFAULT_OUTPUT / "geometry_census.csv.gz"
    )
    sample.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT / "stratified_sample.csv.gz"
    )
    sample.add_argument("--per-month", type=int, default=64)

    audit = subparsers.add_parser("audit", help="Audit oracle patterns on the sample")
    audit.add_argument(
        "--sample", type=Path, default=DEFAULT_OUTPUT / "stratified_sample.csv.gz"
    )
    audit.add_argument(
        "--theta-file", type=Path, default=config.DATA_DIR / config.THETA_FILE
    )
    audit.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    audit.add_argument("--month")
    audit.add_argument("--overwrite", action="store_true")
    _add_resource_arguments(audit)

    summarize = subparsers.add_parser("summarize", help="Write compact result tables")
    summarize.add_argument(
        "--census", type=Path, default=DEFAULT_OUTPUT / "geometry_census.csv.gz"
    )
    summarize.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    summarize.add_argument(
        "--compact-output",
        type=Path,
        default=Path("docs/research/honest_ci/results"),
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(args, "workers"):
        validate_workers(args.workers)
        validate_time_limit(args.time_limit_minutes)
    commands: dict[str, Callable[[argparse.Namespace], int]] = {
        "census": run_census,
        "sample": run_sample,
        "audit": run_audit,
        "summarize": run_summarize,
    }
    return commands[args.command](args)


__all__ = [
    "MAX_TIME_LIMIT_MINUTES",
    "MAX_WORKERS",
    "TRUTH_DELTA_V",
    "TRUTH_FIELD_RATIO",
    "TRUTH_SPACECRAFT_V",
    "deterministic_stratified_sample",
    "main",
    "parse_args",
    "run_audit",
    "run_census",
    "run_sample",
    "run_summarize",
    "sha256_file",
    "validate_time_limit",
    "validate_workers",
]
