"""Aggregate isolated Experiment 003 subprocess results.

Usage:
    .venv/bin/python experiments/experiment-003/aggregate.py
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import shutil
from dataclasses import asdict
from pathlib import Path

import numpy as np
from run import (
    METHODS,
    FitResult,
    _plot_comparison,
    _summary,
)


def _bool(value: str) -> bool:
    return value.lower() == "true"


def read_results(path: Path) -> list[FitResult]:
    with path.open(encoding="ascii", newline="") as source:
        return [
            FitResult(
                panel=row["panel"],
                spec_no=int(row["spec_no"]),
                utc=row["utc"],
                method=row["method"],
                seed=int(row["seed"]),
                u_surface_v=float(row["u_surface_v"]),
                u_relative_v=float(row["u_relative_v"]),
                bs_over_bm=float(row["bs_over_bm"]),
                beam_amp=float(row["beam_amp"]),
                objective_per_bin=float(row["objective_per_bin"]),
                median_abs_residual=float(row["median_abs_residual"]),
                reflected_residual_rms=float(row["reflected_residual_rms"]),
                incident_residual_rms=float(row["incident_residual_rms"]),
                n_valid=int(row["n_valid"]),
                boundary_hit=_bool(row["boundary_hit"]),
                optimizer_success=_bool(row["optimizer_success"]),
                elapsed_s=float(row["elapsed_s"]),
            )
            for row in csv.DictReader(source)
        ]


def write_results(path: Path, results: list[FitResult]) -> None:
    with path.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=tuple(asdict(results[0])))
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)


def extended_summary(
    results: list[FitResult], seeds: list[FitResult], coverage: list[dict[str, str]]
) -> dict[str, object]:
    summary = _summary(results, 0.13168487690063668, 0.42094407555567925)
    confirmation = [result for result in results if result.panel == "confirmation"]
    summary["confirmation_by_method"] = {}
    for method in METHODS:
        selected = [result for result in confirmation if result.method == method]
        voltage = np.array([result.u_relative_v for result in selected])
        summary["confirmation_by_method"][method] = {
            "n": len(selected),
            "median_u_relative_v": float(np.median(voltage)),
            "iqr_u_relative_v": float(
                np.quantile(voltage, 0.75) - np.quantile(voltage, 0.25)
            ),
            "boundary_hit_fraction": float(
                np.mean([result.boundary_hit for result in selected])
            ),
            "median_abs_residual": float(
                np.median([result.median_abs_residual for result in selected])
            ),
        }

    summary["primary_seed_stability"] = {}
    for method in METHODS:
        selected = [result for result in seeds if result.method == method]
        voltage = np.array([result.u_relative_v for result in selected])
        summary["primary_seed_stability"][method] = {
            "min_u_relative_v": float(np.min(voltage)),
            "max_u_relative_v": float(np.max(voltage)),
            "range_v": float(np.ptp(voltage)),
            "all_beam_at_upper_bound": bool(
                all(result.beam_amp >= 4.95 for result in selected)
            ),
        }
    summary["matched_profile_coverage"] = {
        method: {
            row["nominal_level"]: float(row["evaluation_coverage"])
            for row in coverage
            if row["method"] == method
            and row["threshold_source"] == "empirical_calibration"
        }
        for method in METHODS
    }
    return summary


def combine_coverage(output_dir: Path) -> list[dict[str, str]]:
    paths = sorted(output_dir.glob("coverage_*.csv"))
    rows = []
    for path in paths:
        with path.open(encoding="ascii", newline="") as source:
            rows.extend(csv.DictReader(source))
    if len(paths) != 3 or len(rows) != 10:
        raise ValueError(
            f"Expected 3 coverage files and 10 rows, got {len(paths)} and {len(rows)}"
        )
    with (output_dir / "coverage.csv").open(
        "w", encoding="ascii", newline=""
    ) as target:
        writer = csv.DictWriter(target, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--discovery-results",
        type=Path,
        default=Path("/tmp/experiment-003-discovery-final/results.csv"),
    )
    parser.add_argument(
        "--seed-results",
        type=Path,
        default=Path("/tmp/experiment-003-seeds/results.csv"),
    )
    parser.add_argument(
        "--confirmation-glob",
        default="/tmp/experiment-003-confirm-*/results.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent)
    args = parser.parse_args()

    base = read_results(args.discovery_results)
    seeds = read_results(args.seed_results)
    confirmation_paths = sorted(
        Path(path) for path in glob.glob(args.confirmation_glob)
    )
    confirmation = [
        result for path in confirmation_paths for result in read_results(path)
    ]
    if len(base) != 30:
        raise ValueError(f"Expected 30 benchmark/discovery fits, got {len(base)}")
    if len(confirmation) != 150:
        raise ValueError(f"Expected 150 confirmation fits, got {len(confirmation)}")
    if len(seeds) != 9:
        raise ValueError(f"Expected 9 seed-stability fits, got {len(seeds)}")
    confirmation_keys = {(result.spec_no, result.method) for result in confirmation}
    expected_keys = {(spec, method) for spec in range(669, 719) for method in METHODS}
    if confirmation_keys != expected_keys:
        raise ValueError("Confirmation spectra/methods are incomplete or duplicated")

    extra_seeds = [result for result in seeds if result.seed != 42]
    results = sorted(
        base + confirmation + extra_seeds,
        key=lambda result: (result.spec_no, result.method, result.seed),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_results(args.output_dir / "results.csv", results)
    write_results(args.output_dir / "seed_stability.csv", seeds)
    coverage = combine_coverage(args.output_dir)
    summary = extended_summary(results, seeds, coverage)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="ascii"
    )
    _plot_comparison(args.output_dir / "comparison.png", results)
    discovery_atlas = args.discovery_results.parent / "residual_atlas.pdf"
    shutil.copyfile(discovery_atlas, args.output_dir / "residual_atlas.pdf")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
