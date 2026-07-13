#!/usr/bin/env python3
"""Generate compact publication-style figures for Experiment 008."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
FIGURE_DIR = RESULTS_DIR / "figures"

COLORS = {
    "red": "#b33a3a",
    "blue": "#3066a6",
    "green": "#2f7d5c",
    "gold": "#b07a1c",
    "gray": "#6b6f76",
}


def configure() -> None:
    """Apply restrained, legible plot defaults."""
    plt.rcParams.update(
        {
            "axes.spines.right": False,
            "axes.spines.top": False,
            "figure.dpi": 160,
            "font.size": 9,
            "savefig.bbox": "tight",
        }
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def save(figure: plt.Figure, name: str) -> None:
    """Save raster and vector copies."""
    figure.savefig(FIGURE_DIR / f"{name}.png")
    figure.savefig(FIGURE_DIR / f"{name}.pdf")
    plt.close(figure)


def plot_primary() -> None:
    """Plot daily visibility and recovery distributions."""
    data = pd.read_csv(RESULTS_DIR / "primary-windows.csv")
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    axes[0].hist(
        np.log10(data["visibility_min"]),
        bins=35,
        color=COLORS["blue"],
        edgecolor="white",
        linewidth=0.3,
    )
    axes[0].axvline(np.log10(0.1), color=COLORS["red"], linestyle="--")
    axes[0].set_xlabel(r"$\log_{10}$ weakest visibility")
    axes[0].set_ylabel("Daily windows")
    axes[0].set_title("24 h, degree 5")

    axes[1].hist(
        np.log10(data["recovery_median_relative"]),
        bins=35,
        color=COLORS["gold"],
        edgecolor="white",
        linewidth=0.3,
    )
    axes[1].axvline(np.log10(0.25), color=COLORS["red"], linestyle="--")
    axes[1].set_xlabel(r"$\log_{10}$ median relative RMS error")
    axes[1].set_ylabel("Daily windows")
    axes[1].set_title("25 V sweep noise")
    figure.tight_layout()
    save(figure, "primary-daily-failure")


def plot_grid() -> None:
    """Plot stable and recovery fractions over the frozen window grid."""
    data = pd.read_csv(RESULTS_DIR / "window-grid-summary.csv")
    durations = list(dict.fromkeys(data["duration_hours"]))
    degrees = sorted(data["lmax"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.5), constrained_layout=True)
    for axis, column, title in (
        (axes[0], "stable_fraction", "Stable-window fraction"),
        (axes[1], "recovery_pass_fraction", "Recovery-pass fraction"),
    ):
        matrix = data.pivot(index="lmax", columns="duration_hours", values=column).loc[
            degrees, durations
        ]
        image = axis.imshow(matrix, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
        axis.set_xticks(range(len(durations)))
        axis.set_xticklabels([f"{value:g}" for value in durations], rotation=45)
        axis.set_yticks(range(len(degrees)))
        axis.set_yticklabels(degrees)
        axis.set_xlabel("Window duration (h)")
        axis.set_ylabel(r"$l_{\max}$")
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                value = matrix.iloc[row, column_index]
                axis.text(
                    column_index,
                    row,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="white" if value < 0.65 else "black",
                    fontsize=7,
                )
    figure.colorbar(image, ax=axes, label="Fraction", shrink=0.85)
    save(figure, "window-degree-boundary")


def plot_controls() -> None:
    """Plot whole-mission static and synodic minimum visibility."""
    static = pd.read_csv(RESULTS_DIR / "static-controls.csv")
    temporal = pd.read_csv(RESULTS_DIR / "temporal-controls.csv")
    synodic = temporal[
        (temporal["model"] == "constant_synodic")
        & (temporal["estimand"] == "oracle_with_monopole")
    ]
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.plot(
        static["lmax"],
        static["visibility_min"],
        "o-",
        color=COLORS["green"],
        label="Static",
    )
    axis.plot(
        synodic["lmax"],
        synodic["visibility_min"],
        "s-",
        color=COLORS["blue"],
        label="Constant + synodic",
    )
    axis.axhline(0.1, color=COLORS["red"], linestyle="--", label="Frozen gate")
    axis.set_xlabel(r"$l_{\max}$")
    axis.set_ylabel("Weakest physical visibility")
    axis.set_ylim(0.0, 1.0)
    axis.legend(frameon=False)
    figure.tight_layout()
    save(figure, "whole-mission-controls")


def plot_injections() -> None:
    """Compare frozen estimators for 25 V broadband injections."""
    data = pd.read_csv(RESULTS_DIR / "injection-summary.csv")
    data = data[(data["truth_class"] == "broadband") & (data["noise_sigma_v"] == 25)]
    order = [
        "unregularized",
        "tsvd_1e-09",
        "tsvd_1e-06",
        "tsvd_0.001",
        "ridge_1e-06",
        "ridge_0.001",
        "ridge_1",
        "ridge_1000",
    ]
    data = data.set_index("method").loc[order].reset_index()
    labels = [
        value.replace("tsvd_", "TSVD ").replace("ridge_", "ridge ") for value in order
    ]
    figure, axis = plt.subplots(figsize=(7.4, 3.6))
    x_values = np.arange(len(data))
    axis.bar(x_values, data["error_median_relative"], color=COLORS["blue"])
    axis.scatter(
        x_values,
        data["error_p90_relative"],
        color=COLORS["red"],
        s=18,
        label="90th percentile",
    )
    axis.axhline(0.25, color=COLORS["green"], linestyle="--", label="Median gate")
    axis.set_yscale("log")
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.set_ylabel("Relative global RMS error")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    save(figure, "injection-estimators")


def plot_sensitivities() -> None:
    """Plot daily median visibility for frozen operator sensitivities."""
    data = pd.read_csv(RESULTS_DIR / "sensitivity-summary.csv")
    selected = data[
        data["name"].isin(
            [
                "canonical_native_mean",
                "canonical_midpoint_row",
                "canonical_complete_15_rows",
                "legacy_native",
                "frame_mixed_native",
                "submitted_legacy_native",
                "solar_corotating_native",
                "contrast_without_l0",
                "row_footprints_total_sweep_weight",
                "invalid_independent_rows",
            ]
        )
    ]
    labels = [value.replace("_", " ") for value in selected["name"]]
    figure, axis = plt.subplots(figsize=(7.8, 4.0))
    axis.barh(labels, selected["visibility_median"], color=COLORS["gray"])
    axis.axvline(0.1, color=COLORS["red"], linestyle="--")
    axis.set_xscale("log")
    axis.set_xlabel("Median weakest visibility")
    axis.invert_yaxis()
    figure.tight_layout()
    save(figure, "sensitivity-robustness")


def main() -> int:
    """Generate every tracked figure."""
    configure()
    plot_primary()
    plot_grid()
    plot_controls()
    plot_injections()
    plot_sensitivities()
    print(f"Wrote figures to {FIGURE_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
