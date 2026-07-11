#!/usr/bin/env python3
"""Plot and tabulate the frozen Experiment 005 calibration result.

Run from the repository root:

    uv run python experiments/experiment-005/plot_calibration.py
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    directory = Path(__file__).parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results", type=Path, default=directory / "calibration_results.csv"
    )
    parser.add_argument(
        "--summary", type=Path, default=directory / "calibration_summary.json"
    )
    parser.add_argument(
        "--figure", type=Path, default=directory / "calibration_summary.png"
    )
    parser.add_argument(
        "--cells", type=Path, default=directory / "calibration_cell_summary.csv"
    )
    parser.add_argument(
        "--reasons",
        type=Path,
        default=directory / "calibration_failure_reasons.csv",
    )
    return parser.parse_args()


def write_cell_summary(path: Path, cells: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cells[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(cells)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="ascii"))
    with args.results.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    beam_rows = [row for row in rows if row["kind"] == "beam"]
    reasons = Counter(
        reason
        for row in beam_rows
        for reason in row["reason_codes"].split(";")
        if reason
    )
    reason_rows = [
        {
            "reason_code": reason,
            "count": count,
            "fraction_of_1600_attempted": count / len(beam_rows),
        }
        for reason, count in reasons.most_common()
    ]
    with args.reasons.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(reason_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(reason_rows)
    cells = summary["cells"]
    write_cell_summary(args.cells, cells)

    labels = [
        f"{cell['center_ev']:.0f} eV\nA={cell['amplitude']:.0f}, "
        f"sigma={cell['pitch_sigma_deg']:.0f} deg"
        for cell in cells
    ]
    x = np.arange(len(cells))
    retained = np.array([cell["qc_retained_fraction"] for cell in cells])
    median_error = np.array(
        [cell["median_absolute_error_response_fwhm"] for cell in cells]
    )
    coverage_68 = np.array([cell["coverage_68"] for cell in cells], dtype=float)
    coverage_95 = np.array([cell["coverage_95"] for cell in cells], dtype=float)
    coverage_eligible = np.array(
        [cell["coverage_gate_eligible"] for cell in cells], dtype=bool
    )

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    axis = axes[0, 0]
    axis.bar(x, retained, color="#2878b5")
    axis.set_ylabel("QC-retained fraction")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("A. Detection and QC retention")

    axis = axes[0, 1]
    axis.bar(x, median_error, color="#e07a1f")
    axis.axhline(0.10, color="black", linestyle="--", linewidth=1.2, label="gate")
    axis.set_ylabel("Median |error| / response FWHM")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("B. Point-estimate error after QC")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    width = 0.36
    axis.bar(
        x - width / 2,
        np.where(coverage_eligible, coverage_68, np.nan),
        width,
        label="68%",
        color="#3a923a",
    )
    axis.bar(
        x + width / 2,
        np.where(coverage_eligible, coverage_95, np.nan),
        width,
        label="95%",
        color="#8b5aa5",
    )
    axis.axhspan(0.60, 0.76, color="#3a923a", alpha=0.10)
    axis.axhspan(0.88, 0.99, color="#8b5aa5", alpha=0.10)
    axis.set_ylabel("Empirical coverage")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("C. Residual-bootstrap interval calibration")
    axis.legend(frameon=False)
    for cell_x in x[~coverage_eligible]:
        axis.text(cell_x, 0.04, "n < 50", ha="center", va="bottom", rotation=90)

    axis = axes[1, 1]
    top_reasons = reason_rows[:6][::-1]
    axis.barh(
        [row["reason_code"].replace("_", " ") for row in top_reasons],
        [row["count"] for row in top_reasons],
        color="#b64040",
    )
    axis.set_xlabel("Nonexclusive failures among 1,600 injections")
    axis.set_title("D. QC failure anatomy")
    axis.text(
        0.98,
        0.05,
        "0 / 500 nulls accepted",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
    )

    figure.suptitle(
        "Experiment 005 frozen calibration: detection works, uncertainty does not",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
