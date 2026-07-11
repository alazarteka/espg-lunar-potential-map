#!/usr/bin/env python3
"""Freeze outcome-blind Experiment 006 calibration/confirmation supports.

Run from the repository root:

    uv run python experiments/experiment-006/support_selection.py

The selector uses only timestamps, structural validity, SPICE-derived surface
connection and polarity, native energy support, pitch coverage, and magnetic
variation. It never computes a normalized distribution or beam fit.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import itertools
import json
import platform
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import spiceypy as spice

from src import config
from src.losscone.er_data import ERData
from src.losscone.pitch_angle import PitchAngle
from src.potential_mapper.coordinates import (
    CoordinateCalculator,
    find_surface_intersection_with_polarity,
    project_magnetic_fields,
)
from src.potential_mapper.spice import load_spice_files
from src.utils.attitude import load_attitude_data
from src.utils.spice_lock import verify_kernels_lock

SELECTION_FEATURES = (
    "field_elevation_deg",
    "spacecraft_altitude_km",
    "footprint_lat_deg",
    "footprint_lon_sin",
    "footprint_lon_cos",
    "axis_support_rows",
    "minimum_axis_sector_count",
    "median_axis_sector_count",
    "median_incident_sector_count",
    "median_reflected_sector_count",
    "median_pitch_span_deg",
    "polarity",
)


@dataclass(frozen=True)
class SupportMetric:
    """Outcome-blind structural and geometry metrics for one spectrum."""

    spec_no: int
    utc_start: str
    utc_end: str
    utc_midpoint: str
    temporal_stratum: int
    structural_rows: int
    connected_rows: int
    polarity: int
    guard_energy_rows: int
    usable_guard_rows: int
    axis_support_rows: int
    minimum_axis_sector_count: float
    median_axis_sector_count: float
    median_incident_sector_count: float
    median_reflected_sector_count: float
    median_pitch_span_deg: float
    field_elevation_deg: float
    spacecraft_altitude_km: float
    footprint_lat_deg: float
    footprint_lon_deg: float
    footprint_lon_sin: float
    footprint_lon_cos: float
    magnetic_variation_deg: float
    mean_anchor_log_distance: float
    eligible: bool
    exclusion_reason: str


def load_configuration(path: Path) -> dict[str, Any]:
    """Load the Experiment 006 machine-readable design."""
    return json.loads(path.read_text(encoding="ascii"))


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def provenance_name(path: Path) -> str:
    """Return a repository-relative provenance name when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(config.PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def temporal_stratum(
    midpoint: pd.Timestamp, start: pd.Timestamp, end: pd.Timestamp, count: int
) -> int:
    """Assign a timestamp to one of ``count`` fixed half-open time strata."""
    span_seconds = (end - start).total_seconds()
    if span_seconds <= 0:
        raise ValueError("Selection window must have positive duration")
    fraction = (midpoint - start).total_seconds() / span_seconds
    return int(np.clip(np.floor(fraction * count), 0, count - 1))


def magnetic_variation_deg(magnetic_field: np.ndarray) -> float:
    """Return the largest angular departure from the sweep-mean field."""
    norms = np.linalg.norm(magnetic_field, axis=1)
    valid = np.isfinite(magnetic_field).all(axis=1) & (norms > 0.0)
    if np.count_nonzero(valid) < 2:
        return float("nan")
    unit = magnetic_field[valid] / norms[valid, None]
    mean = np.mean(unit, axis=0)
    mean_norm = np.linalg.norm(mean)
    if not np.isfinite(mean_norm) or mean_norm <= 0.0:
        return float("nan")
    cosine = np.clip(unit @ (mean / mean_norm), -1.0, 1.0)
    return float(np.max(np.rad2deg(np.arccos(cosine))))


def anchor_log_distance(energies_ev: np.ndarray, anchors_ev: np.ndarray) -> float:
    """Measure native-channel distance from the frozen calibration anchors."""
    valid = energies_ev[np.isfinite(energies_ev) & (energies_ev > 0.0)]
    if valid.size == 0:
        return float("nan")
    distances = np.abs(np.log(anchors_ev[:, None]) - np.log(valid[None, :]))
    return float(np.mean(np.min(distances, axis=1)))


def build_metric(
    spec_no: int,
    row_indices: np.ndarray,
    er_data: ERData,
    pitches_deg: np.ndarray,
    surface_points: np.ndarray,
    spacecraft_positions: np.ndarray,
    projected_b: np.ndarray,
    connected: np.ndarray,
    polarity: np.ndarray,
    configuration: dict[str, Any],
) -> SupportMetric:
    """Compute one support record without normalized-flux morphology."""
    rows = er_data.data.iloc[row_indices]
    pitches = pitches_deg[row_indices]
    flux = rows[config.FLUX_COLS].to_numpy(dtype=np.float64)
    energies = rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
    magnetic_field = rows[config.MAG_COLS].to_numpy(dtype=np.float64)

    selection = configuration["input"]["support_selection"]
    geometry = configuration["geometry"]
    guard_low, guard_high = selection["guard_energy_ev"]
    guard = (energies >= guard_low) & (energies <= guard_high)
    finite_positive = np.isfinite(flux) & (flux > 0.0) & np.isfinite(pitches)
    incident_counts = np.count_nonzero(finite_positive & (pitches < 90.0), axis=1)
    reflected_counts = np.count_nonzero(finite_positive & (pitches >= 90.0), axis=1)
    axis_min = 180.0 - float(geometry["surface_axis_support_half_width_deg"])
    axis_counts = np.count_nonzero(finite_positive & (pitches >= axis_min), axis=1)

    usable_guard = (
        guard
        & (incident_counts >= int(selection["minimum_incident_sectors_per_row"]))
        & (reflected_counts >= int(selection["minimum_reflected_sectors_per_row"]))
    )
    axis_supported = usable_guard & (
        axis_counts >= int(geometry["minimum_surface_axis_sectors_per_row"])
    )

    pitch_spans = np.full(len(rows), np.nan, dtype=np.float64)
    for row_id in np.flatnonzero(usable_guard):
        valid_pitch = pitches[row_id][finite_positive[row_id]]
        if valid_pitch.size:
            pitch_spans[row_id] = float(np.max(valid_pitch) - np.min(valid_pitch))

    row_polarity = polarity[row_indices]
    connected_rows = int(np.count_nonzero(connected[row_indices]))
    unique_polarity = np.unique(row_polarity[row_polarity != 0])
    consistent_polarity = int(unique_polarity[0]) if unique_polarity.size == 1 else 0

    reasons: list[str] = []
    expected_rows = int(selection["required_structural_rows"])
    if len(rows) != expected_rows:
        reasons.append("structural_rows")
    if connected_rows != len(rows) or consistent_polarity == 0:
        reasons.append("surface_connection_or_polarity")
    usable_guard_count = int(np.count_nonzero(usable_guard))
    if usable_guard_count < int(selection["minimum_usable_guard_rows"]):
        reasons.append("usable_guard_rows")
    if int(np.count_nonzero(axis_supported)) < int(
        geometry["minimum_response_contributing_rows_with_axis_support"]
    ):
        reasons.append("surface_axis_support")

    reserved = set(configuration["input"]["excluded_experiment_005_supports"])
    reserved.update(configuration["input"]["known_positive_control_specs"])
    if spec_no in reserved:
        reasons.append("reserved_or_previously_used")

    utc_start = pd.to_datetime(rows.iloc[0][config.UTC_COLUMN], utc=True)
    utc_end = pd.to_datetime(rows.iloc[-1][config.UTC_COLUMN], utc=True)
    midpoint = utc_start + (utc_end - utc_start) / 2
    window_start = pd.to_datetime(configuration["input"]["window_start_utc"], utc=True)
    window_end = pd.to_datetime(configuration["input"]["window_end_utc"], utc=True)
    stratum = temporal_stratum(
        midpoint,
        window_start,
        window_end,
        int(selection["temporal_strata"]),
    )

    selected_rows = np.flatnonzero(usable_guard)

    def selected_median(values: np.ndarray) -> float:
        return (
            float(np.median(values[selected_rows]))
            if selected_rows.size
            else float("nan")
        )

    anchors = np.asarray(
        configuration["calibration"]["energy_anchors_ev"], dtype=np.float64
    )
    connected_indices = row_indices[connected[row_indices]]
    positions = spacecraft_positions[connected_indices]
    points = surface_points[connected_indices]
    moonward_b = projected_b[connected_indices] * polarity[connected_indices, None]
    position_norm = np.linalg.norm(positions, axis=1)
    direction_norm = np.linalg.norm(moonward_b, axis=1)
    valid_geometry = (
        np.isfinite(positions).all(axis=1)
        & np.isfinite(points).all(axis=1)
        & np.isfinite(moonward_b).all(axis=1)
        & (position_norm > 0.0)
        & (direction_norm > 0.0)
    )
    field_elevation = np.array([], dtype=np.float64)
    altitude = np.array([], dtype=np.float64)
    footprint_lat = np.array([], dtype=np.float64)
    footprint_lon = np.array([], dtype=np.float64)
    if np.any(valid_geometry):
        valid_positions = positions[valid_geometry]
        valid_points = points[valid_geometry]
        inward = -valid_positions / np.linalg.norm(valid_positions, axis=1)[:, None]
        direction = (
            moonward_b[valid_geometry]
            / np.linalg.norm(moonward_b[valid_geometry], axis=1)[:, None]
        )
        incidence = np.rad2deg(
            np.arccos(np.clip(np.einsum("ij,ij->i", direction, inward), -1.0, 1.0))
        )
        field_elevation = 90.0 - incidence
        lunar_radius_km = float(config.LUNAR_RADIUS.to("kilometer").magnitude)
        altitude = np.linalg.norm(valid_positions, axis=1) - lunar_radius_km
        point_norm = np.linalg.norm(valid_points, axis=1)
        footprint_lat = np.rad2deg(np.arcsin(valid_points[:, 2] / point_norm))
        footprint_lon = np.rad2deg(np.arctan2(valid_points[:, 1], valid_points[:, 0]))

    median_field_elevation = (
        float(np.median(field_elevation)) if field_elevation.size else float("nan")
    )
    if not np.isfinite(median_field_elevation) or median_field_elevation < float(
        selection["minimum_field_elevation_deg"]
    ):
        reasons.append("grazing_surface_connection")
    longitude_radians = np.deg2rad(footprint_lon)
    longitude_sin = (
        float(np.mean(np.sin(longitude_radians)))
        if longitude_radians.size
        else float("nan")
    )
    longitude_cos = (
        float(np.mean(np.cos(longitude_radians)))
        if longitude_radians.size
        else float("nan")
    )
    longitude_deg = (
        float(np.rad2deg(np.arctan2(longitude_sin, longitude_cos)))
        if np.isfinite(longitude_sin) and np.isfinite(longitude_cos)
        else float("nan")
    )
    return SupportMetric(
        spec_no=spec_no,
        utc_start=utc_start.isoformat().replace("+00:00", "Z"),
        utc_end=utc_end.isoformat().replace("+00:00", "Z"),
        utc_midpoint=midpoint.isoformat().replace("+00:00", "Z"),
        temporal_stratum=stratum,
        structural_rows=len(rows),
        connected_rows=connected_rows,
        polarity=consistent_polarity,
        guard_energy_rows=int(np.count_nonzero(guard)),
        usable_guard_rows=int(np.count_nonzero(usable_guard)),
        axis_support_rows=int(np.count_nonzero(axis_supported)),
        minimum_axis_sector_count=(
            float(np.min(axis_counts[selected_rows]))
            if selected_rows.size
            else float("nan")
        ),
        median_axis_sector_count=selected_median(axis_counts),
        median_incident_sector_count=selected_median(incident_counts),
        median_reflected_sector_count=selected_median(reflected_counts),
        median_pitch_span_deg=selected_median(pitch_spans),
        field_elevation_deg=median_field_elevation,
        spacecraft_altitude_km=(
            float(np.median(altitude)) if altitude.size else float("nan")
        ),
        footprint_lat_deg=(
            float(np.median(footprint_lat)) if footprint_lat.size else float("nan")
        ),
        footprint_lon_deg=longitude_deg,
        footprint_lon_sin=longitude_sin,
        footprint_lon_cos=longitude_cos,
        magnetic_variation_deg=magnetic_variation_deg(magnetic_field),
        mean_anchor_log_distance=anchor_log_distance(energies, anchors),
        eligible=not reasons,
        exclusion_reason=";".join(reasons),
    )


def feature_matrix(metrics: list[SupportMetric]) -> np.ndarray:
    """Return globally standardized selection features."""
    values = np.asarray(
        [
            [float(getattr(metric, name)) for name in SELECTION_FEATURES]
            for metric in metrics
        ],
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        raise ValueError("Eligible support features must all be finite")
    scale = np.std(values, axis=0)
    scale[scale < 1e-12] = 1.0
    return (values - np.mean(values, axis=0)) / scale


def select_groups(
    eligible: list[SupportMetric], temporal_strata: int, supports_per_stratum: int
) -> tuple[list[list[SupportMetric]], dict[int, float]]:
    """Select maximin-diverse supports inside every fixed time stratum."""
    standardized = feature_matrix(eligible)
    index_by_spec = {metric.spec_no: index for index, metric in enumerate(eligible)}
    groups: list[list[SupportMetric]] = []
    selection_distances: dict[int, float] = {}
    for stratum in range(temporal_strata):
        candidates = sorted(
            (metric for metric in eligible if metric.temporal_stratum == stratum),
            key=lambda metric: metric.spec_no,
        )
        if len(candidates) < supports_per_stratum:
            message = (
                f"Temporal stratum {stratum} has only {len(candidates)} "
                "eligible supports"
            )
            raise RuntimeError(message)
        best: tuple[float, int, int, SupportMetric, SupportMetric] | None = None
        for left, right in itertools.combinations(candidates, 2):
            distance = float(
                np.linalg.norm(
                    standardized[index_by_spec[left.spec_no]]
                    - standardized[index_by_spec[right.spec_no]]
                )
            )
            candidate = (distance, -left.spec_no, -right.spec_no, left, right)
            if best is None or candidate[:3] > best[:3]:
                best = candidate
        assert best is not None
        selected = [best[3], best[4]]
        selection_distances[best[3].spec_no] = best[0]
        selection_distances[best[4].spec_no] = best[0]
        while len(selected) < supports_per_stratum:
            remaining = [value for value in candidates if value not in selected]
            ranked: list[tuple[float, int, SupportMetric]] = []
            for candidate in remaining:
                candidate_vector = standardized[index_by_spec[candidate.spec_no]]
                minimum_distance = min(
                    float(
                        np.linalg.norm(
                            candidate_vector
                            - standardized[index_by_spec[value.spec_no]]
                        )
                    )
                    for value in selected
                )
                ranked.append((minimum_distance, -candidate.spec_no, candidate))
            distance, _tie_break, chosen = max(ranked, key=lambda value: value[:2])
            selected.append(chosen)
            selection_distances[chosen.spec_no] = distance
        groups.append(selected)
    return groups, selection_distances


def assign_groups(
    groups: list[list[SupportMetric]],
) -> tuple[set[int], set[int], dict[int, int], float]:
    """Assign two geometrically ordered supports per stratum to each split."""
    selected = [metric for group in groups for metric in group]
    standardized = feature_matrix(selected)
    index_by_spec = {metric.spec_no: index for index, metric in enumerate(selected)}
    calibration: set[int] = set()
    confirmation: set[int] = set()
    geometry_ranks: dict[int, int] = {}
    for stratum, group in enumerate(groups):
        ordered = sorted(
            group, key=lambda value: (value.field_elevation_deg, value.spec_no)
        )
        for rank, metric in enumerate(ordered, start=1):
            geometry_ranks[metric.spec_no] = rank
        calibration_ranks = {1, 4} if stratum % 2 == 0 else {2, 3}
        for metric in ordered:
            target = (
                calibration
                if geometry_ranks[metric.spec_no] in calibration_ranks
                else confirmation
            )
            target.add(metric.spec_no)
    calibration_mean = np.mean(
        [standardized[index_by_spec[value]] for value in calibration], axis=0
    )
    confirmation_mean = np.mean(
        [standardized[index_by_spec[value]] for value in confirmation], axis=0
    )
    imbalance = float(np.sum(np.square(calibration_mean - confirmation_mean)))
    return calibration, confirmation, geometry_ranks, imbalance


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    """Write deterministic CSV output."""
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def current_commit() -> str:
    """Return the repository commit used to generate the manifest."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--configuration",
        type=Path,
        default=experiment_dir / "configuration.json",
    )
    parser.add_argument("--output-dir", type=Path, default=experiment_dir)
    args = parser.parse_args()

    configuration = load_configuration(args.configuration)
    input_path = Path(configuration["input"]["data_path"])
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    lock_path = config.SPICE_KERNELS_DIR / "kernels.lock"
    lock_result = verify_kernels_lock(
        config.SPICE_KERNELS_DIR, lock_path, verify_hashes=True
    )
    if lock_result.missing or lock_result.extra or lock_result.mismatched:
        raise RuntimeError(f"SPICE kernel lock verification failed: {lock_result}")
    spice.kclear()
    load_spice_files()
    et_spin, ra_vals, dec_vals = load_attitude_data(
        config.DATA_DIR / config.ATTITUDE_FILE
    )
    if et_spin is None or ra_vals is None or dec_vals is None:
        raise RuntimeError("Attitude data unavailable")

    er_data = ERData(str(input_path))
    coordinate_calculator = CoordinateCalculator(et_spin, ra_vals, dec_vals)
    coordinate_arrays = coordinate_calculator.calculate_coordinate_transformation(
        er_data
    )
    projected_b = project_magnetic_fields(er_data, coordinate_arrays)
    points, connected, polarity = find_surface_intersection_with_polarity(
        coordinate_arrays, projected_b
    )
    pitches = PitchAngle(er_data, polarity=polarity).pitch_angles

    first_spec = int(configuration["input"]["first_spec_no"])
    last_spec = int(configuration["input"]["last_spec_no"])
    spec_values = er_data.data[config.SPEC_NO_COLUMN].to_numpy(dtype=np.int64)
    metrics: list[SupportMetric] = []
    missing_rows: list[dict[str, Any]] = []
    for spec_no in range(first_spec, last_spec + 1):
        indices = np.flatnonzero(spec_values == spec_no)
        if indices.size == 0:
            missing_rows.append(
                {
                    "spec_no": spec_no,
                    "eligible": False,
                    "exclusion_reason": "missing_or_structurally_invalid",
                }
            )
            continue
        metrics.append(
            build_metric(
                spec_no,
                indices,
                er_data,
                pitches,
                points,
                coordinate_arrays.lp_positions,
                projected_b,
                connected,
                polarity,
                configuration,
            )
        )

    eligible = [metric for metric in metrics if metric.eligible]
    strata = int(configuration["input"]["support_selection"]["temporal_strata"])
    supports_per_stratum = int(
        configuration["input"]["support_selection"]["supports_per_temporal_stratum"]
    )
    groups, selection_distances = select_groups(eligible, strata, supports_per_stratum)
    calibration, confirmation, geometry_ranks, imbalance = assign_groups(groups)
    selected_specs = calibration | confirmation

    metric_rows = [asdict(metric) for metric in metrics]
    all_fieldnames = list(asdict(metrics[0]))
    for row in metric_rows:
        spec_no = int(row["spec_no"])
        row["selected"] = spec_no in selected_specs
        row["split"] = (
            "calibration"
            if spec_no in calibration
            else "confirmation"
            if spec_no in confirmation
            else ""
        )
    all_fieldnames.extend(["selected", "split"])
    for row in missing_rows:
        complete = dict.fromkeys(all_fieldnames, "")
        complete.update(row)
        complete["selected"] = False
        complete["split"] = ""
        metric_rows.append(complete)
    metric_rows.sort(key=lambda row: int(row["spec_no"]))

    manifest_rows: list[dict[str, Any]] = []
    for metric in sorted(
        (metric for metric in metrics if metric.spec_no in selected_specs),
        key=lambda value: value.spec_no,
    ):
        row = asdict(metric)
        row.update(
            {
                "split": (
                    "calibration" if metric.spec_no in calibration else "confirmation"
                ),
                "geometry_rank": geometry_ranks[metric.spec_no],
                "selection_distance": f"{selection_distances[metric.spec_no]:.10g}",
            }
        )
        manifest_rows.append(row)
    manifest_fields = [
        *asdict(metrics[0]),
        "split",
        "geometry_rank",
        "selection_distance",
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "support_candidates.csv"
    manifest_path = args.output_dir / "support_manifest.csv"
    provenance_path = args.output_dir / "support_provenance.json"
    write_csv(candidate_path, metric_rows, all_fieldnames)
    write_csv(manifest_path, manifest_rows, manifest_fields)

    kernel_files = sorted(
        path
        for path in config.SPICE_KERNELS_DIR.iterdir()
        if path.is_file() and path.name != "kernels.lock"
    )
    energy_grid_hashes: dict[str, list[int]] = {}
    energy_grid_values: dict[str, list[float]] = {}
    for spec_no in range(first_spec, last_spec + 1):
        indices = np.flatnonzero(spec_values == spec_no)
        if indices.size == 0:
            continue
        grid = er_data.data.iloc[indices][config.ENERGY_COLUMN].to_numpy(dtype="<f8")
        grid_hash = hashlib.sha256(grid.tobytes()).hexdigest()
        energy_grid_hashes.setdefault(grid_hash, []).append(spec_no)
        energy_grid_values.setdefault(grid_hash, grid.tolist())
    provenance = {
        "generated_date": datetime.now(UTC).date().isoformat(),
        "repository_commit": current_commit(),
        "selection_is_outcome_blind": True,
        "normalization_or_beam_fit_evaluated": False,
        "selection_features": list(SELECTION_FEATURES),
        "candidate_specs": last_spec - first_spec + 1,
        "loaded_specs": len(metrics),
        "eligible_specs": len(eligible),
        "temporal_strata": strata,
        "calibration_specs": sorted(calibration),
        "confirmation_specs": sorted(confirmation),
        "standardized_split_imbalance": imbalance,
        "kernel_lock_verified": True,
        "native_energy_grid_audit": {
            "unique_grid_count": len(energy_grid_hashes),
            "grids": [
                {
                    "sha256": grid_hash,
                    "spec_count": len(energy_grid_hashes[grid_hash]),
                    "energies_ev": energy_grid_values[grid_hash],
                }
                for grid_hash in sorted(energy_grid_hashes)
            ],
        },
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "spiceypy": importlib.metadata.version("spiceypy"),
        },
        "files": {
            provenance_name(input_path): sha256_file(input_path),
            provenance_name(args.configuration): sha256_file(args.configuration),
            str(Path(__file__).resolve().relative_to(config.PROJECT_ROOT)): sha256_file(
                Path(__file__)
            ),
            provenance_name(config.DATA_DIR / config.ATTITUDE_FILE): sha256_file(
                config.DATA_DIR / config.ATTITUDE_FILE
            ),
            provenance_name(config.DATA_DIR / config.THETA_FILE): sha256_file(
                config.DATA_DIR / config.THETA_FILE
            ),
            str(lock_path.relative_to(config.PROJECT_ROOT)): sha256_file(lock_path),
            **{
                str(path.relative_to(config.PROJECT_ROOT)): sha256_file(path)
                for path in kernel_files
            },
            str(candidate_path.relative_to(config.PROJECT_ROOT)): sha256_file(
                candidate_path
            ),
            str(manifest_path.relative_to(config.PROJECT_ROOT)): sha256_file(
                manifest_path
            ),
        },
    }
    provenance_path.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    spice.kclear()

    print(f"Eligible supports: {len(eligible)} / {last_spec - first_spec + 1}")
    print(f"Calibration: {sorted(calibration)}")
    print(f"Confirmation: {sorted(confirmation)}")
    print(f"Split imbalance: {imbalance:.8f}")


if __name__ == "__main__":
    main()
