#!/usr/bin/env python3
"""Synthesize and numerically validate an isolated Experiment 008 replay."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from replay_context import (
    ARTIFACT_DIR,
    CENSUS_PATH,
    EXPERIMENT_DIR,
    RESULTS_DIR,
    RUN_ID,
)

sys.path.insert(0, str(EXPERIMENT_DIR))
import observation_operator as operator
import run_analysis

if TYPE_CHECKING:
    from pathlib import Path


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object."""
    with path.open(encoding="ascii") as stream:
        return dict(json.load(stream))


def recoverable_window_classes(grid: pd.DataFrame) -> list[str]:
    """Return frozen independent-window classes that pass all gates."""
    passing = grid.loc[grid["class_pass"]].sort_values(["duration_hours", "lmax"])
    result = []
    for row in passing.itertuples(index=False):
        duration = f"{row.duration_hours:.6f}".rstrip("0").rstrip(".")
        result.append(f"{duration}h_lmax{int(row.lmax)}")
    return result


def final_summary() -> dict[str, Any]:
    """Apply the unchanged outcome tree to the corrected replay outputs."""
    census = pd.read_csv(CENSUS_PATH)
    primary = pd.read_csv(RESULTS_DIR / "primary-windows.csv")
    grid = pd.read_csv(RESULTS_DIR / "window-grid-summary.csv")
    static = pd.read_csv(RESULTS_DIR / "static-controls.csv")
    temporal = pd.read_csv(RESULTS_DIR / "temporal-controls.csv")
    blocked = pd.read_csv(RESULTS_DIR / "blocked-recovery.csv")

    primary_pass = bool(
        np.mean(~primary["unstable"].to_numpy(dtype=bool)) >= 0.9
        and np.mean(primary["recovery_pass"].to_numpy(dtype=bool)) >= 0.9
    )
    coarse_classes = recoverable_window_classes(grid)
    static_pass = bool(
        np.all(static["recovery_pass"]) and not np.any(static["unstable"])
    )
    temporal_pass = bool(
        np.all(temporal["recovery_pass"]) and not np.any(temporal["unstable"])
    )
    decision = (
        "A_submitted_model_recoverable"
        if primary_pass
        else (
            "C_conditional_coarse_recovery"
            if coarse_classes and static_pass and temporal_pass
            else "B_submitted_model_nonrecoverable"
        )
    )
    visibility = primary["visibility_min"].to_numpy(dtype=float)
    return {
        "blocked_shared_full_rank_records": int(
            np.sum(blocked["rank"] == blocked["parameters"])
        ),
        "canonical_connected_sweeps": int(census["corrected_connected_sweeps"].sum()),
        "canonical_complete_sweeps": int(census["corrected_complete_sweeps"].sum()),
        "decision": decision,
        "geometry_file_failures": int(np.sum(census["error"].fillna("") != "")),
        "primary": {
            "eligible_windows": len(primary),
            "formal_full_rank_windows": int(
                np.sum(primary["exact_rank"] == primary["parameters"])
            ),
            "lmax": 5,
            "recovery_pass_windows": int(np.sum(primary["recovery_pass"])),
            "stable_windows": int(np.sum(~primary["unstable"])),
            "visibility_max": float(np.max(visibility)),
            "visibility_median": float(np.median(visibility)),
            "visibility_min": float(np.min(visibility)),
            "window_hours": 24.0,
        },
        "recoverable_classes": {
            "independent_windows": coarse_classes,
            "shared_temporal": (
                "constant_and_constant_synodic_through_lmax10"
                if temporal_pass
                else "none"
            ),
            "static": "through_lmax20" if static_pass else "none",
        },
        "run_id": RUN_ID,
        "schema_version": 2,
        "spacecraft_potential_gauge_resolved": False,
        "structurally_valid_sweeps": int(census["structurally_valid_sweeps"].sum()),
        "total_exact_15_row_sweeps": int(census["exact_15_row_sweeps"].sum()),
        "total_raw_files": len(census),
    }


def _temporal_cache_checks() -> tuple[float, float]:
    """Check the cached tensor operator against its spatial and direct forms."""
    basis = np.load(
        ARTIFACT_DIR / "canonical-sweep-basis-l10.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    temporal = np.load(
        ARTIFACT_DIR / "canonical-sweep-temporal-synodic-l10.npy",
        mmap_mode="r",
        allow_pickle=False,
    )
    constant_difference = 0.0
    for start in range(0, len(basis), 20_000):
        stop = min(start + 20_000, len(basis))
        constant_difference = max(
            constant_difference,
            float(np.max(np.abs(temporal[start:stop, :121] - basis[start:stop]))),
        )

    geometry = operator.load_sweep_geometry(ARTIFACT_DIR / "geometry.npz")
    indices = np.linspace(0, geometry.size - 1, 17, dtype=np.int64)
    direct = operator.sweep_tensor_harmonics(
        run_analysis.slice_geometry(geometry, indices),
        10,
        "constant_synodic",
        period_days=29.530588,
        epoch_ns=0,
    )
    direct_difference = float(np.max(np.abs(direct - temporal[indices])))
    return constant_difference, direct_difference


def _grid_witness_rms() -> list[float]:
    """Validate representative 100 V witnesses on an area-weighted 2-degree grid."""
    with np.load(RESULTS_DIR / "primary-spectra-witnesses.npz") as data:
        witnesses = np.asarray(data["witness_coefficients_100v"])
    latitude = np.arange(-89.0, 90.0, 2.0)
    longitude = np.arange(0.0, 360.0, 2.0)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    basis = operator.real_spherical_harmonics(lat_grid, lon_grid, 5)
    weights = np.cos(np.deg2rad(lat_grid))
    result = []
    for index in (0, len(witnesses) // 2, len(witnesses) - 1):
        field = basis @ witnesses[index]
        result.append(float(np.sqrt(np.sum(weights * field**2) / np.sum(weights))))
    return result


def numerical_validation() -> dict[str, Any]:
    """Cross-check independent implementations and duplicate result paths."""
    primary = pd.read_csv(RESULTS_DIR / "primary-windows.csv")
    grid = pd.read_csv(RESULTS_DIR / "window-grid-summary.csv")
    gates = pd.read_csv(RESULTS_DIR / "injection-window-gates.csv")
    comparison = gates.merge(
        primary[["window_id", "recovery_median_relative", "recovery_p90_relative"]],
        on="window_id",
        validate="one_to_one",
    )
    grid_primary = grid.loc[
        (grid["duration_hours"] == 24.0) & (grid["lmax"] == 5)
    ].iloc[0]
    temporal_constant, temporal_direct = _temporal_cache_checks()
    return {
        "explicit_covariance_median_ratio_median": float(
            np.median(
                comparison["median_relative"] / comparison["recovery_median_relative"]
            )
        ),
        "explicit_covariance_p90_ratio_median": float(
            np.median(comparison["p90_relative"] / comparison["recovery_p90_relative"])
        ),
        "explicit_injection_passing_windows": int(np.sum(gates["passes"])),
        "physical_area_grid_rms_representative_v": _grid_witness_rms(),
        "primary_grid_recovery_fraction_difference": float(
            np.mean(primary["recovery_pass"]) - grid_primary["recovery_pass_fraction"]
        ),
        "primary_grid_stable_fraction_difference": float(
            np.mean(~primary["unstable"]) - grid_primary["stable_fraction"]
        ),
        "primary_grid_visibility_median_difference": float(
            np.median(primary["visibility_min"]) - grid_primary["visibility_median"]
        ),
        "schema_version": 2,
        "temporal_cache_constant_block_max_abs_difference": temporal_constant,
        "temporal_cache_sampled_direct_max_abs_difference": temporal_direct,
    }


def comparison_to_frozen(summary: dict[str, Any]) -> dict[str, Any]:
    """Compare corrected outcomes with the immutable original Experiment 008 run."""
    original_results = EXPERIMENT_DIR / "results"
    original = load_json(original_results / "final-summary.json")
    original_static = pd.read_csv(original_results / "static-controls.csv")
    corrected_static = pd.read_csv(RESULTS_DIR / "static-controls.csv")
    original_temporal = pd.read_csv(original_results / "temporal-controls.csv")
    corrected_temporal = pd.read_csv(RESULTS_DIR / "temporal-controls.csv")
    old_l20 = original_static.loc[original_static["lmax"] == 20].iloc[0]
    new_l20 = corrected_static.loc[corrected_static["lmax"] == 20].iloc[0]

    def temporal_filter(frame: pd.DataFrame) -> pd.Series:
        return frame.loc[
            (frame["model"] == "constant_synodic")
            & (frame["lmax"] == 10)
            & (frame["estimand"] == "oracle_with_monopole")
        ].iloc[0]

    old_synodic = temporal_filter(original_temporal)
    new_synodic = temporal_filter(corrected_temporal)
    old_visibility = float(original["primary"]["visibility_median"])
    new_visibility = float(summary["primary"]["visibility_median"])
    return {
        "canonical_connected_sweep_change": int(
            summary["canonical_connected_sweeps"]
            - original["canonical_connected_sweeps"]
        ),
        "canonical_connected_sweep_fractional_change": float(
            summary["canonical_connected_sweeps"]
            / original["canonical_connected_sweeps"]
            - 1.0
        ),
        "corrected_decision": summary["decision"],
        "decision_changed": summary["decision"] != original["decision"],
        "original_decision": original["decision"],
        "primary_visibility_median_fractional_change": float(
            new_visibility / old_visibility - 1.0
        ),
        "primary_visibility_median_original": old_visibility,
        "primary_visibility_median_corrected": new_visibility,
        "recoverable_independent_classes_changed": (
            summary["recoverable_classes"]["independent_windows"]
            != original["recoverable_classes"]["independent_windows"]
        ),
        "schema_version": 1,
        "static_l20_condition_corrected": float(new_l20["condition_number"]),
        "static_l20_condition_original": float(old_l20["condition_number"]),
        "synodic_l10_condition_corrected": float(new_synodic["condition_number"]),
        "synodic_l10_condition_original": float(old_synodic["condition_number"]),
    }


def main() -> int:
    """Write deterministic replay synthesis products."""
    if not RUN_ID:
        raise RuntimeError("Set EXPERIMENT_008_RUN_ID for replay synthesis")
    summary = final_summary()
    run_analysis.write_json(RESULTS_DIR / "final-summary.json", summary)
    run_analysis.write_json(
        RESULTS_DIR / "numerical-validation.json", numerical_validation()
    )
    run_analysis.write_json(
        RESULTS_DIR / "comparison-to-frozen.json", comparison_to_frozen(summary)
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
