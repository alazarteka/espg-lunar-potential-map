#!/usr/bin/env python3
"""Run Experiment 005 Attempt 03 split calibration and validation.

Run from the repository root:

    uv run python experiments/experiment-005/calibrate_v3.py
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import beam
import beam_v3
import calibrate
import numpy as np


def background_family(replicate: int) -> str:
    """Return the frozen 50/25/25 background mixture deterministically."""
    families = (
        "single_logistic",
        "single_logistic",
        "double_transition",
        "asymmetric_shoulder",
    )
    return families[replicate % len(families)]


def _logistic(
    pitches_deg: np.ndarray, boundary_deg: np.ndarray, width_deg: float
) -> np.ndarray:
    return 1.0 / (
        1.0 + np.exp(-np.clip((boundary_deg - pitches_deg) / width_deg, -60.0, 60.0))
    )


def simulate_observation(
    support: beam.BeamObservation,
    *,
    center_ev: float,
    amplitude: float,
    pitch_sigma_deg: float,
    log_noise_sigma: float,
    rng: np.random.Generator,
    field_ratio: float,
    cone_floor: float,
    boundary_width_deg: float,
    family: str,
) -> beam.BeamObservation:
    """Simulate a beam over one of the frozen continuum families."""
    boundaries = calibrate.relative_boundary_pitch(
        support.sample_energies_ev, -center_ev, field_ratio
    )
    contrast = 1.0 - cone_floor
    primary = _logistic(support.pitches_deg, boundaries, boundary_width_deg)
    if family == "single_logistic":
        continuum = cone_floor + contrast * primary
    elif family == "double_transition":
        offset = float(rng.uniform(8.0, 24.0))
        second_boundary = np.clip(boundaries - offset, 90.0, 180.0)
        second = _logistic(
            support.pitches_deg,
            second_boundary,
            float(rng.uniform(1.4, 2.2)) * boundary_width_deg,
        )
        fraction = float(rng.uniform(0.35, 0.70))
        continuum = cone_floor + contrast * (
            fraction * primary + (1.0 - fraction) * second
        )
    elif family == "asymmetric_shoulder":
        continuum = cone_floor + contrast * primary
        shoulder_center = float(rng.uniform(118.0, 142.0))
        shoulder_width = float(rng.uniform(10.0, 24.0))
        energy_phase = np.sin(
            1.7 * np.log(support.sample_energies_ev / np.min(support.energies_ev))
        )
        shoulder = np.exp(
            -0.5 * np.square((support.pitches_deg - shoulder_center) / shoulder_width)
        )
        continuum += float(rng.uniform(0.05, 0.18)) * shoulder * (1.2 + energy_phase)
    else:
        raise ValueError(f"Unknown background family: {family}")
    shape = beam.beam_shape(
        support.sample_energies_ev,
        support.pitches_deg,
        center_ev,
        pitch_sigma_deg,
        0.48,
    )
    expectation = continuum + amplitude * shape
    noise = rng.normal(
        -0.5 * log_noise_sigma**2,
        log_noise_sigma,
        size=len(expectation),
    )
    return replace(support, values=np.maximum(expectation * np.exp(noise), 1e-8))


def split_name(kind: str, replicate: int) -> str:
    if kind == "beam":
        return "calibration" if replicate <= 99 else "validation"
    return "calibration" if replicate <= 249 else "validation"


def _run_case(task: dict[str, Any]) -> dict[str, Any]:
    configuration = task["configuration"]
    cell = task["cell"]
    replicate = int(task["replicate"])
    kind = str(task["kind"])
    support = task["support"]
    rng = np.random.default_rng(int(task["seed"]))
    if kind == "beam":
        center = float(cell["center_ev"])
        amplitude = float(cell["amplitude"])
        pitch_sigma = float(cell["pitch_sigma_deg"])
        noise_sigma = float(cell["log_noise_sigma"])
    else:
        center = float(rng.choice([65.0, 150.0, 350.0, 800.0]))
        amplitude = 0.0
        pitch_sigma = float(rng.uniform(12.0, 42.0))
        noise_sigma = float(rng.choice([0.20, 0.25, 0.35, 0.45]))
    family = background_family(replicate)
    observation = simulate_observation(
        support,
        center_ev=center,
        amplitude=amplitude,
        pitch_sigma_deg=pitch_sigma,
        log_noise_sigma=noise_sigma,
        rng=rng,
        field_ratio=float(rng.uniform(0.88, 1.02)),
        cone_floor=float(rng.uniform(0.08, 0.40)),
        boundary_width_deg=float(rng.uniform(4.0, 14.0)),
        family=family,
    )
    assessment = beam_v3.assess_smooth_beam(
        observation,
        observation,
        configuration["beam"],
        configuration["beam_qc"],
        reuse_identical_sensitivity=True,
    )
    fitted = assessment.primary
    true_center = center if kind == "beam" else float("nan")
    error = fitted.center_ev - true_center if kind == "beam" else float("nan")
    return {
        "kind": kind,
        "split": split_name(kind, replicate),
        "background_family": family,
        "cell_id": int(task["cell_id"]),
        "replicate": replicate,
        "support_spec_no": int(task["support_spec_no"]),
        "seed": int(task["seed"]),
        "true_center_ev": true_center,
        "true_amplitude": amplitude,
        "true_pitch_sigma_deg": pitch_sigma,
        "log_noise_sigma": noise_sigma,
        "fitted_center_ev": fitted.center_ev,
        "fitted_amplitude": fitted.amplitude,
        "fitted_pitch_sigma_deg": fitted.pitch_sigma_deg,
        "center_standard_error_ev": fitted.center_standard_error_ev,
        "amplitude_standard_error": fitted.amplitude_standard_error,
        "center_error_ev": error,
        "rms_log_residual": fitted.rms_log_residual,
        "covariance_rank": fitted.covariance_rank,
        "covariance_dimension": fitted.covariance_dimension,
        "continuum_bound_contacts": fitted.continuum_bound_contacts,
        "positive_excess_rows": assessment.positive_excess_rows,
        "normalization_difference_ev": assessment.normalization_difference_ev,
        "leave_one_energy_out_max_shift_ev": (
            assessment.leave_one_energy_out_max_shift_ev
        ),
        "base_valid": assessment.base_valid,
        "reason_codes": ";".join(assessment.reason_codes),
    }


def fit_empirical_correction(
    rows: list[dict[str, Any]], configuration: dict[str, Any]
) -> dict[str, list[float]]:
    """Learn bias and interval scales from only the calibration split."""
    anchors = np.asarray(
        configuration["calibration"]["empirical_correction"]["energy_anchors_ev"],
        dtype=np.float64,
    )
    biases = []
    scale_68 = []
    scale_95 = []
    retained_counts = []
    calibration_rows = [
        row
        for row in rows
        if row["kind"] == "beam" and row["split"] == "calibration" and row["base_valid"]
    ]
    for anchor in anchors:
        selected = [row for row in calibration_rows if row["true_center_ev"] == anchor]
        retained_counts.append(len(selected))
        if not selected:
            biases.append(float("nan"))
            scale_68.append(float("nan"))
            scale_95.append(float("nan"))
            continue
        errors = np.array([row["center_error_ev"] for row in selected], dtype=float)
        standard_errors = np.array(
            [row["center_standard_error_ev"] for row in selected], dtype=float
        )
        bias = float(np.median(errors))
        standardized = np.abs(errors - bias) / standard_errors
        biases.append(bias)
        scale_68.append(float(np.quantile(standardized, 0.68)))
        scale_95.append(float(np.quantile(standardized, 0.95)))
    return {
        "energy_anchors_ev": anchors.tolist(),
        "median_signed_bias_ev": biases,
        "scale_68": scale_68,
        "scale_95": scale_95,
        "retained_calibration_counts": retained_counts,
    }


def interpolate_correction(
    center_ev: float, correction: dict[str, list[float]]
) -> tuple[float, float, float]:
    anchors = np.asarray(correction["energy_anchors_ev"], dtype=float)
    log_center = np.log(np.clip(center_ev, anchors[0], anchors[-1]))
    log_anchors = np.log(anchors)
    values = [
        float(np.interp(log_center, log_anchors, np.asarray(correction[key])))
        for key in ("median_signed_bias_ev", "scale_68", "scale_95")
    ]
    return values[0], values[1], values[2]


def apply_empirical_correction(
    rows: list[dict[str, Any]],
    correction: dict[str, list[float]],
    configuration: dict[str, Any],
) -> None:
    """Attach frozen correction and final interval QC to every fitted row."""
    max_width_fraction = float(
        configuration["beam_qc"]["calibrated_95_width_response_fwhm_max"]
    )
    for row in rows:
        center = float(row["fitted_center_ev"])
        standard_error = float(row["center_standard_error_ev"])
        bias, scale_68, scale_95 = interpolate_correction(center, correction)
        corrected = center - bias
        half_68 = scale_68 * standard_error
        half_95 = scale_95 * standard_error
        response_fwhm = 0.48 * center
        interval_resolved = bool(
            np.isfinite(half_95) and 2.0 * half_95 <= max_width_fraction * response_fwhm
        )
        row.update(
            {
                "empirical_bias_ev": bias,
                "empirical_scale_68": scale_68,
                "empirical_scale_95": scale_95,
                "corrected_center_ev": corrected,
                "interval_68_low_ev": corrected - half_68,
                "interval_68_high_ev": corrected + half_68,
                "interval_95_low_ev": corrected - half_95,
                "interval_95_high_ev": corrected + half_95,
                "calibrated_interval_resolved": interval_resolved,
                "final_valid": bool(row["base_valid"] and interval_resolved),
            }
        )
        if row["kind"] == "beam":
            truth = float(row["true_center_ev"])
            row["corrected_error_ev"] = corrected - truth
            row["corrected_absolute_error_response_fwhm"] = abs(corrected - truth) / (
                0.48 * truth
            )
            row["coverage_68"] = bool(
                corrected - half_68 <= truth <= corrected + half_68
            )
            row["coverage_95"] = bool(
                corrected - half_95 <= truth <= corrected + half_95
            )
        else:
            row["corrected_error_ev"] = float("nan")
            row["corrected_absolute_error_response_fwhm"] = float("nan")
            row["coverage_68"] = False
            row["coverage_95"] = False


def summarize_validation(
    rows: list[dict[str, Any]], configuration: dict[str, Any]
) -> dict[str, Any]:
    calibration = configuration["calibration"]
    validation_beams = [
        row for row in rows if row["kind"] == "beam" and row["split"] == "validation"
    ]
    cell_summaries = []
    eligible_passes = []
    for cell_id, cell in enumerate(calibration["cells"]):
        cell_rows = [row for row in validation_beams if row["cell_id"] == cell_id]
        retained = [row for row in cell_rows if row["final_valid"]]
        retained_count = len(retained)
        median_error = (
            float(
                np.median(
                    [row["corrected_absolute_error_response_fwhm"] for row in retained]
                )
            )
            if retained
            else float("nan")
        )
        coverage_68 = (
            float(np.mean([row["coverage_68"] for row in retained]))
            if retained
            else float("nan")
        )
        coverage_95 = (
            float(np.mean([row["coverage_95"] for row in retained]))
            if retained
            else float("nan")
        )
        eligible = retained_count >= int(
            calibration["minimum_qc_retained_validation_for_cell_gate"]
        )
        passed = bool(
            eligible
            and median_error
            <= float(calibration["median_corrected_absolute_error_response_fwhm_max"])
            and calibration["coverage_68_bounds"][0]
            <= coverage_68
            <= calibration["coverage_68_bounds"][1]
            and calibration["coverage_95_bounds"][0]
            <= coverage_95
            <= calibration["coverage_95_bounds"][1]
        )
        if eligible:
            eligible_passes.append(passed)
        cell_summaries.append(
            {
                "cell_id": cell_id,
                **cell,
                "attempted_validation": len(cell_rows),
                "final_retained_validation": retained_count,
                "retained_fraction": retained_count / max(len(cell_rows), 1),
                "median_corrected_absolute_error_response_fwhm": median_error,
                "coverage_68": coverage_68,
                "coverage_95": coverage_95,
                "coverage_gate_eligible": eligible,
                "passed": passed,
            }
        )
    retained_validation = [row for row in validation_beams if row["final_valid"]]
    validation_nulls = [
        row for row in rows if row["kind"] == "null" and row["split"] == "validation"
    ]
    false_detections = int(sum(row["final_valid"] for row in validation_nulls))
    false_detection_rate = false_detections / max(len(validation_nulls), 1)
    enough_retained = len(retained_validation) >= int(
        calibration["minimum_qc_retained_validation_total"]
    )
    false_detection_pass = false_detection_rate <= float(
        calibration["false_detection_rate_max"]
    )
    calibration_pass = bool(
        enough_retained
        and eligible_passes
        and all(eligible_passes)
        and false_detection_pass
    )
    return {
        "calibration_pass": calibration_pass,
        "attempted_validation_beams": len(validation_beams),
        "retained_validation_beams": len(retained_validation),
        "minimum_retained_validation_pass": enough_retained,
        "attempted_validation_nulls": len(validation_nulls),
        "false_detections": false_detections,
        "false_detection_rate": false_detection_rate,
        "false_detection_rate_pass": false_detection_pass,
        "eligible_cells": len(eligible_passes),
        "eligible_cells_passed": int(sum(eligible_passes)),
        "cells": cell_summaries,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=directory / "configuration-attempt-03.json",
    )
    parser.add_argument(
        "--output", type=Path, default=directory / "calibration_results_attempt_03.csv"
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=directory / "calibration_summary_attempt_03.json",
    )
    parser.add_argument("--replicates-per-cell", type=int)
    parser.add_argument("--null-replicates", type=int)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    configuration = beam_v3.load_configuration(args.configuration)
    calibration_config = configuration["calibration"]
    replicates = int(
        calibration_config["simulation_replicates_per_cell"]
        if args.replicates_per_cell is None
        else args.replicates_per_cell
    )
    null_replicates = int(
        calibration_config["beam_absent_replicates"]
        if args.null_replicates is None
        else args.null_replicates
    )
    support_specs = [int(value) for value in calibration_config["support_spec_nos"]]
    spectra = beam.load_spectra(
        Path(configuration["input"]["data_path"]), support_specs
    )
    supports = {
        spec_no: beam.build_observation(
            spectra[spec_no],
            configuration["beam"]["primary_normalization"],
            configuration["beam"],
        )
        for spec_no in support_specs
    }
    tasks = []
    seed_base = int(configuration["beam"]["random_seed"])
    for cell_id, cell in enumerate(calibration_config["cells"]):
        for replicate in range(replicates):
            support_spec = support_specs[(cell_id + replicate) % len(support_specs)]
            tasks.append(
                {
                    "kind": "beam",
                    "cell_id": cell_id,
                    "cell": cell,
                    "replicate": replicate,
                    "support_spec_no": support_spec,
                    "support": supports[support_spec],
                    "seed": seed_base + cell_id * 100_000 + replicate,
                    "configuration": configuration,
                }
            )
    for replicate in range(null_replicates):
        support_spec = support_specs[replicate % len(support_specs)]
        tasks.append(
            {
                "kind": "null",
                "cell_id": -1,
                "cell": {},
                "replicate": replicate,
                "support_spec_no": support_spec,
                "support": supports[support_spec],
                "seed": seed_base + 9_000_000 + replicate,
                "configuration": configuration,
            }
        )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_case, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 25 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["kind"], row["cell_id"], row["replicate"]))
    correction = fit_empirical_correction(rows, configuration)
    apply_empirical_correction(rows, correction, configuration)
    write_rows(args.output, rows)
    summary = summarize_validation(rows, configuration)
    summary["empirical_correction"] = correction
    summary["execution"] = {
        "replicates_per_cell": replicates,
        "null_replicates": null_replicates,
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_scale": bool(
            replicates == calibration_config["simulation_replicates_per_cell"]
            and null_replicates == calibration_config["beam_absent_replicates"]
        ),
    }
    safe_summary = calibrate.json_safe(summary)
    args.summary.write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    print(json.dumps(safe_summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
