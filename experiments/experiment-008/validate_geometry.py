#!/usr/bin/env python3
"""Validate one Experiment 008 shard against the production geometry path.

This implementation check does not load fitted potentials or compute an
identifiability result.

Example:

    uv run python experiments/experiment-008/validate_geometry.py \
        --shard /tmp/experiment-008-one-day-v2/0000-3D980116.npz \
        --output experiments/experiment-008/implementation-check-one-day.json
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import spiceypy as spice

from src import config
from src.losscone.er_data import ERData
from src.potential_mapper.coordinates import (
    CoordinateCalculator,
    find_surface_intersection_with_polarity,
    project_magnetic_fields,
)
from src.potential_mapper.spice import load_spice_files
from src.utils.attitude import load_attitude_data

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_PATH = Path(__file__).with_name("geometry.py")
SPEC = importlib.util.spec_from_file_location("experiment_008_geometry", GEOMETRY_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Cannot load {GEOMETRY_PATH}")
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def angular_separation_degrees(
    lat_a: np.ndarray,
    lon_a: np.ndarray,
    lat_b: np.ndarray,
    lon_b: np.ndarray,
) -> np.ndarray:
    """Return great-circle separation in degrees."""
    lat_a_rad = np.deg2rad(lat_a)
    lat_b_rad = np.deg2rad(lat_b)
    delta_lon = np.deg2rad(lon_a - lon_b)
    cosine = np.sin(lat_a_rad) * np.sin(lat_b_rad) + np.cos(lat_a_rad) * np.cos(
        lat_b_rad
    ) * np.cos(delta_lon)
    return np.rad2deg(np.arccos(np.clip(cosine, -1.0, 1.0)))


def production_geometry(
    raw_path: Path,
) -> tuple[ERData, np.ndarray, np.ndarray, np.ndarray]:
    """Run the current production coordinate path without any fit."""
    spice.kclear()
    load_spice_files()
    attitude = load_attitude_data(config.DATA_DIR / config.ATTITUDE_FILE)
    if any(value is None for value in attitude):
        raise RuntimeError("Attitude data unavailable")
    et_spin, ra, dec = attitude
    er_data = ERData(str(raw_path))
    calculator = CoordinateCalculator(et_spin, ra, dec)
    coordinate_arrays = calculator.calculate_coordinate_transformation(er_data)
    projected_b = project_magnetic_fields(er_data, coordinate_arrays)
    points, connected, polarity = find_surface_intersection_with_polarity(
        coordinate_arrays, projected_b
    )
    lat, lon = geometry.points_to_lat_lon(points, connected)
    return er_data, lat, lon, polarity


def align_production_rows(
    er_data: ERData,
    sweep_spec_no: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    polarity: np.ndarray,
    rows_per_sweep: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align cleaned production rows to the raw-shard sweep order."""
    shape = (len(sweep_spec_no), rows_per_sweep)
    aligned_lat = np.full(shape, np.nan)
    aligned_lon = np.full(shape, np.nan)
    aligned_polarity = np.zeros(shape, dtype=np.int8)
    spec_rows = er_data.data[config.SPEC_NO_COLUMN].to_numpy(dtype=np.int64)
    lookup = {int(value): index for index, value in enumerate(sweep_spec_no)}
    for group in geometry.contiguous_group_slices(spec_rows):
        if group.stop - group.start != rows_per_sweep:
            continue
        spec_no = int(spec_rows[group.start])
        if spec_no not in lookup:
            continue
        destination = lookup[spec_no]
        aligned_lat[destination] = lat[group]
        aligned_lon[destination] = lon[group]
        aligned_polarity[destination] = polarity[group]
    return aligned_lat, aligned_lon, aligned_polarity


def shift_summary(
    canonical_lat: np.ndarray,
    canonical_lon: np.ndarray,
    variant_lat: np.ndarray,
    variant_lon: np.ndarray,
) -> dict[str, float | int]:
    """Summarize common-valid footprint shifts."""
    common = (
        np.isfinite(canonical_lat)
        & np.isfinite(canonical_lon)
        & np.isfinite(variant_lat)
        & np.isfinite(variant_lon)
    )
    separation = angular_separation_degrees(
        canonical_lat[common],
        canonical_lon[common],
        variant_lat[common],
        variant_lon[common],
    )
    radius_km = float(config.LUNAR_RADIUS.to("kilometer").magnitude)
    distance_km = np.deg2rad(separation) * radius_km
    return {
        "common_rows": int(np.sum(common)),
        "median_angular_shift_deg": float(np.median(separation)),
        "p95_angular_shift_deg": float(np.percentile(separation, 95.0)),
        "max_angular_shift_deg": float(np.max(separation)),
        "median_surface_shift_km": float(np.median(distance_km)),
        "p95_surface_shift_km": float(np.percentile(distance_km, 95.0)),
        "max_surface_shift_km": float(np.max(distance_km)),
    }


def validate(shard_path: Path) -> dict[str, Any]:
    """Validate a shard and return a JSON-serializable report."""
    with np.load(shard_path, allow_pickle=False) as data:
        shard = {key: np.array(data[key]) for key in data.files}
    raw_path = PROJECT_ROOT / str(shard["source_path"])
    er_data, prod_lat, prod_lon, prod_polarity = production_geometry(raw_path)
    rows_per_sweep = shard["row_utc_ns"].shape[1]
    aligned = align_production_rows(
        er_data,
        shard["sweep_spec_no"],
        prod_lat,
        prod_lon,
        prod_polarity,
        rows_per_sweep,
    )
    aligned_lat, aligned_lon, aligned_polarity = aligned
    canonical_lat = shard["corrected_latitude"]
    canonical_lon = shard["corrected_longitude"]
    canonical_polarity = shard["corrected_polarity"]
    reference_valid = np.isfinite(aligned_lat) & np.isfinite(aligned_lon)
    candidate_valid = np.isfinite(canonical_lat) & np.isfinite(canonical_lon)
    common = reference_valid & candidate_valid
    lon_difference = (
        canonical_lon[common] - aligned_lon[common] + 180.0
    ) % 360.0 - 180.0
    parity = {
        "reference_valid_rows": int(np.sum(reference_valid)),
        "candidate_valid_rows": int(np.sum(candidate_valid)),
        "valid_mask_mismatch_rows": int(np.sum(reference_valid != candidate_valid)),
        "polarity_mismatch_rows": int(
            np.sum(canonical_polarity[common] != aligned_polarity[common])
        ),
        "max_abs_latitude_difference_deg": float(
            np.max(np.abs(canonical_lat[common] - aligned_lat[common]))
        ),
        "max_abs_longitude_difference_deg": float(np.max(np.abs(lon_difference))),
    }
    parity["pass"] = bool(
        parity["valid_mask_mismatch_rows"] == 0
        and parity["polarity_mismatch_rows"] == 0
        and parity["max_abs_latitude_difference_deg"] <= 1e-9
        and parity["max_abs_longitude_difference_deg"] <= 1e-9
    )

    report = {
        "status": "implementation_check_not_identifiability_outcome",
        "source_raw_file": str(shard["source_path"]),
        "sweeps": len(shard["sweep_spec_no"]),
        "production_corrected_parity": parity,
        "canonical_vs_equatorial_interpretation": shift_summary(
            canonical_lat,
            canonical_lon,
            shard["equatorial_latitude"],
            shard["equatorial_longitude"],
        ),
        "canonical_vs_frame_mixed": shift_summary(
            canonical_lat,
            canonical_lon,
            shard["frame_mixed_latitude"],
            shard["frame_mixed_longitude"],
        ),
        "canonical_vs_legacy_attitude": shift_summary(
            canonical_lat,
            canonical_lon,
            shard["legacy_latitude"],
            shard["legacy_longitude"],
        ),
        "canonical_vs_submitted_legacy": shift_summary(
            canonical_lat,
            canonical_lon,
            shard["submitted_legacy_latitude"],
            shard["submitted_legacy_longitude"],
        ),
    }
    return report


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shard", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = validate(args.shard)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="ascii")
    print(json.dumps(report, indent=2))
    return 0 if report["production_corrected_parity"]["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
