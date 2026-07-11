#!/usr/bin/env python3
"""Calibrate the Experiment 005 beam estimator before the one-day pilot.

Run from the repository root:

    uv run python experiments/experiment-005/calibrate.py

Non-default replicate arguments are development checks and must not replace the
frozen calibration outputs used for the experiment decision.
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
import numpy as np


def relative_boundary_pitch(
    energies_ev: np.ndarray, u_relative_v: float, field_ratio: float
) -> np.ndarray:
    """Evaluate the frozen spacecraft-relative adiabatic boundary."""
    sin2 = field_ratio * (1.0 + u_relative_v / energies_ev)
    alpha = np.degrees(np.arcsin(np.sqrt(np.clip(sin2, 0.0, 1.0))))
    return 180.0 - alpha


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
) -> beam.BeamObservation:
    """Generate a beam on a broadened loss-cone background and real support."""
    boundaries = relative_boundary_pitch(
        support.sample_energies_ev, -center_ev, field_ratio
    )
    transition = 1.0 / (
        1.0
        + np.exp(
            -np.clip(
                (boundaries - support.pitches_deg) / boundary_width_deg,
                -60.0,
                60.0,
            )
        )
    )
    continuum = cone_floor + (1.0 - cone_floor) * transition
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
    values = np.maximum(expectation * np.exp(noise), 1e-8)
    return replace(support, values=values)


def _run_case(task: dict[str, Any]) -> dict[str, Any]:
    configuration = task["configuration"]
    cell = task["cell"]
    replicate = int(task["replicate"])
    kind = str(task["kind"])
    support = task["support"]
    seed = int(task["seed"])
    rng = np.random.default_rng(seed)
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
    field_ratio = float(rng.uniform(0.88, 1.02))
    cone_floor = float(rng.uniform(0.08, 0.40))
    boundary_width = float(rng.uniform(4.0, 14.0))
    observation = simulate_observation(
        support,
        center_ev=center,
        amplitude=amplitude,
        pitch_sigma_deg=pitch_sigma,
        log_noise_sigma=noise_sigma,
        rng=rng,
        field_ratio=field_ratio,
        cone_floor=cone_floor,
        boundary_width_deg=boundary_width,
    )
    sensitivity = replace(observation, normalization="conjugate_sector")
    assessment = beam.assess_beam(
        observation,
        sensitivity,
        configuration["beam"],
        configuration["beam_qc"],
        rng,
        bootstrap_replicates=int(task["bootstrap_replicates"]),
    )
    fitted = assessment.primary
    bootstrap = assessment.bootstrap
    true_center = center if kind == "beam" else float("nan")
    error = fitted.center_ev - true_center if kind == "beam" else float("nan")
    response_fwhm = 0.48 * true_center if kind == "beam" else float("nan")
    return {
        "kind": kind,
        "cell_id": int(task["cell_id"]),
        "replicate": replicate,
        "support_spec_no": int(task["support_spec_no"]),
        "seed": seed,
        "true_center_ev": true_center,
        "true_amplitude": amplitude,
        "true_pitch_sigma_deg": pitch_sigma,
        "log_noise_sigma": noise_sigma,
        "field_ratio": field_ratio,
        "cone_floor": cone_floor,
        "boundary_width_deg": boundary_width,
        "fitted_center_ev": fitted.center_ev,
        "fitted_amplitude": fitted.amplitude,
        "fitted_pitch_sigma_deg": fitted.pitch_sigma_deg,
        "center_error_ev": error,
        "absolute_error_response_fwhm": (
            abs(error) / response_fwhm if kind == "beam" else float("nan")
        ),
        "bootstrap_q16_ev": bootstrap.center_q16_ev,
        "bootstrap_q84_ev": bootstrap.center_q84_ev,
        "bootstrap_q025_ev": bootstrap.center_q025_ev,
        "bootstrap_q975_ev": bootstrap.center_q975_ev,
        "coverage_68": bool(
            kind == "beam"
            and bootstrap.center_q16_ev <= true_center <= bootstrap.center_q84_ev
        ),
        "coverage_95": bool(
            kind == "beam"
            and bootstrap.center_q025_ev <= true_center <= bootstrap.center_q975_ev
        ),
        "bootstrap_successes": bootstrap.successes,
        "positive_excess_rows": assessment.positive_excess_rows,
        "normalization_difference_ev": assessment.normalization_difference_ev,
        "leave_one_energy_out_max_shift_ev": (
            assessment.leave_one_energy_out_max_shift_ev
        ),
        "beam_valid": assessment.beam_valid,
        "reason_codes": ";".join(assessment.reason_codes),
    }


def summarize_calibration(
    rows: list[dict[str, Any]], configuration: dict[str, Any]
) -> dict[str, Any]:
    """Apply the frozen calibration gates to simulation rows."""
    calibration = configuration["calibration"]
    cell_summaries = []
    eligible_cell_passes = []
    beam_rows = [row for row in rows if row["kind"] == "beam"]
    for cell_id in range(len(calibration["cells"])):
        cell_rows = [row for row in beam_rows if row["cell_id"] == cell_id]
        accepted = [row for row in cell_rows if row["beam_valid"]]
        accepted_count = len(accepted)
        median_error = (
            float(np.median([row["absolute_error_response_fwhm"] for row in accepted]))
            if accepted
            else float("nan")
        )
        coverage_68 = (
            float(np.mean([row["coverage_68"] for row in accepted]))
            if accepted
            else float("nan")
        )
        coverage_95 = (
            float(np.mean([row["coverage_95"] for row in accepted]))
            if accepted
            else float("nan")
        )
        eligible = accepted_count >= int(
            calibration["minimum_qc_retained_for_cell_gate"]
        )
        passed = bool(
            eligible
            and median_error <= float(calibration["median_bias_response_fwhm_max"])
            and calibration["coverage_68_bounds"][0]
            <= coverage_68
            <= calibration["coverage_68_bounds"][1]
            and calibration["coverage_95_bounds"][0]
            <= coverage_95
            <= calibration["coverage_95_bounds"][1]
        )
        if eligible:
            eligible_cell_passes.append(passed)
        cell_summaries.append(
            {
                "cell_id": cell_id,
                **calibration["cells"][cell_id],
                "attempted": len(cell_rows),
                "qc_retained": accepted_count,
                "qc_retained_fraction": accepted_count / max(len(cell_rows), 1),
                "median_absolute_error_response_fwhm": median_error,
                "coverage_68": coverage_68,
                "coverage_95": coverage_95,
                "coverage_gate_eligible": eligible,
                "passed": passed,
            }
        )
    accepted_beams = [row for row in beam_rows if row["beam_valid"]]
    null_rows = [row for row in rows if row["kind"] == "null"]
    false_detection_rate = float(np.mean([row["beam_valid"] for row in null_rows]))
    enough_retained = len(accepted_beams) >= int(
        calibration["minimum_qc_retained_total"]
    )
    false_detection_pass = false_detection_rate <= float(
        calibration["false_detection_rate_max"]
    )
    calibration_pass = bool(
        enough_retained
        and eligible_cell_passes
        and all(eligible_cell_passes)
        and false_detection_pass
    )
    return {
        "calibration_pass": calibration_pass,
        "attempted_beam_simulations": len(beam_rows),
        "qc_retained_beam_simulations": len(accepted_beams),
        "minimum_qc_retained_total_pass": enough_retained,
        "attempted_null_simulations": len(null_rows),
        "false_detections": int(sum(row["beam_valid"] for row in null_rows)),
        "false_detection_rate": false_detection_rate,
        "false_detection_rate_pass": false_detection_pass,
        "eligible_cells": len(eligible_cell_passes),
        "eligible_cells_passed": int(sum(eligible_cell_passes)),
        "cells": cell_summaries,
    }


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def json_safe(value: Any) -> Any:
    """Replace non-finite floats with JSON null recursively."""
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configuration",
        type=Path,
        default=Path(__file__).with_name("configuration.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("calibration_results.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(__file__).with_name("calibration_summary.json"),
    )
    parser.add_argument("--replicates-per-cell", type=int)
    parser.add_argument("--null-replicates", type=int)
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    return parser.parse_args()


def main() -> None:
    started = time.perf_counter()
    args = parse_args()
    configuration = beam.load_configuration(args.configuration)
    calibration = configuration["calibration"]
    replicates = int(
        calibration["simulation_replicates_per_cell"]
        if args.replicates_per_cell is None
        else args.replicates_per_cell
    )
    null_replicates = int(
        calibration["beam_absent_replicates"]
        if args.null_replicates is None
        else args.null_replicates
    )
    bootstrap_replicates = int(
        configuration["beam"]["bootstrap_replicates"]
        if args.bootstrap_replicates is None
        else args.bootstrap_replicates
    )
    support_specs = [int(value) for value in calibration["support_spec_nos"]]
    spectra = beam.load_spectra(
        Path(configuration["input"]["data_path"]), support_specs
    )
    missing = sorted(set(support_specs) - set(spectra))
    if missing:
        raise RuntimeError(f"Missing calibration support spectra: {missing}")
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
    for cell_id, cell in enumerate(calibration["cells"]):
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
                    "bootstrap_replicates": bootstrap_replicates,
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
                "bootstrap_replicates": bootstrap_replicates,
                "configuration": configuration,
            }
        )
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(_run_case, task) for task in tasks]
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == len(futures):
                print(f"completed {completed}/{len(futures)}", flush=True)
    rows.sort(key=lambda row: (row["kind"], row["cell_id"], row["replicate"]))
    write_rows(args.output, rows)
    summary = summarize_calibration(rows, configuration)
    summary["execution"] = {
        "replicates_per_cell": replicates,
        "null_replicates": null_replicates,
        "bootstrap_replicates": bootstrap_replicates,
        "workers": args.workers,
        "elapsed_seconds": time.perf_counter() - started,
        "frozen_scale": bool(
            replicates == calibration["simulation_replicates_per_cell"]
            and null_replicates == calibration["beam_absent_replicates"]
            and bootstrap_replicates == configuration["beam"]["bootstrap_replicates"]
        ),
    }
    safe_summary = json_safe(summary)
    args.summary.write_text(
        json.dumps(safe_summary, indent=2, allow_nan=False) + "\n", encoding="ascii"
    )
    print(json.dumps(safe_summary, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
