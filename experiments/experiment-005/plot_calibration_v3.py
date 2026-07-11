#!/usr/bin/env python3
"""Plot Experiment 005 Attempt 03 held-out validation results."""

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
        "--results",
        type=Path,
        default=directory / "calibration_results_attempt_03.csv",
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=directory / "calibration_summary_attempt_03.json",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=directory / "calibration_summary_attempt_03.png",
    )
    parser.add_argument(
        "--cells",
        type=Path,
        default=directory / "calibration_cells_attempt_03.csv",
    )
    parser.add_argument(
        "--reasons",
        type=Path,
        default=directory / "calibration_reasons_attempt_03.csv",
    )
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="ascii"))
    with args.results.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    validation_beams = [
        row for row in rows if row["kind"] == "beam" and row["split"] == "validation"
    ]
    reasons = Counter(
        reason
        for row in validation_beams
        for reason in row["reason_codes"].split(";")
        if reason
    )
    reason_rows = [
        {
            "reason_code": reason,
            "count": count,
            "fraction_of_800_validation_beams": count / len(validation_beams),
        }
        for reason, count in reasons.most_common()
    ]
    write_csv(args.cells, summary["cells"])
    write_csv(args.reasons, reason_rows)

    cells = summary["cells"]
    labels = [
        f"{cell['center_ev']:.0f} eV\nA={cell['amplitude']:.0f}, "
        f"sigma={cell['pitch_sigma_deg']:.0f} deg"
        for cell in cells
    ]
    x = np.arange(len(cells))
    retained = np.array([cell["retained_fraction"] for cell in cells])
    errors = np.array(
        [
            np.nan
            if cell["median_corrected_absolute_error_response_fwhm"] is None
            else cell["median_corrected_absolute_error_response_fwhm"]
            for cell in cells
        ]
    )
    coverage_68 = np.array(
        [
            np.nan if cell["coverage_68"] is None else cell["coverage_68"]
            for cell in cells
        ]
    )
    coverage_95 = np.array(
        [
            np.nan if cell["coverage_95"] is None else cell["coverage_95"]
            for cell in cells
        ]
    )
    eligible = np.array([cell["coverage_gate_eligible"] for cell in cells], dtype=bool)
    passed = np.array([cell["passed"] for cell in cells], dtype=bool)

    figure, axes = plt.subplots(2, 2, figsize=(13.5, 8.5), constrained_layout=True)
    colors = np.where(passed, "#2f8f46", "#b64242")
    axis = axes[0, 0]
    axis.bar(x, retained, color=colors)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Held-out retained fraction")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("A. Final validation retention")

    axis = axes[0, 1]
    axis.bar(x, errors, color=colors)
    axis.axhline(0.10, color="black", linestyle="--", linewidth=1.2, label="gate")
    axis.set_ylabel("Median corrected |error| / response FWHM")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("B. Corrected point-estimate error")
    axis.legend(frameon=False)

    axis = axes[1, 0]
    width = 0.36
    axis.bar(
        x - width / 2,
        np.where(eligible, coverage_68, np.nan),
        width,
        label="68%",
        color="#3a923a",
    )
    axis.bar(
        x + width / 2,
        np.where(eligible, coverage_95, np.nan),
        width,
        label="95%",
        color="#8b5aa5",
    )
    axis.axhspan(0.60, 0.76, color="#3a923a", alpha=0.10)
    axis.axhspan(0.88, 0.99, color="#8b5aa5", alpha=0.10)
    for cell_x in x[~eligible]:
        axis.text(cell_x, 0.04, "n < 25", ha="center", va="bottom", rotation=90)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Held-out empirical coverage")
    axis.set_xticks(x, labels, rotation=25, ha="right")
    axis.set_title("C. Empirically calibrated intervals")
    axis.legend(frameon=False)

    axis = axes[1, 1]
    top_reasons = reason_rows[:6][::-1]
    axis.barh(
        [row["reason_code"].replace("_", " ") for row in top_reasons],
        [row["count"] for row in top_reasons],
        color="#b64040",
    )
    axis.set_xlabel("Nonexclusive failures among 800 validation beams")
    axis.set_title("D. Base-QC failure anatomy")
    axis.text(
        0.98,
        0.05,
        "15 / 250 held-out nulls accepted (6.0%)\ngate: <= 5%",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontweight="bold",
    )
    figure.suptitle(
        "Experiment 005 Attempt 03: middle energies pass, full gate fails",
        fontsize=15,
        fontweight="bold",
    )
    figure.savefig(args.figure, dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
