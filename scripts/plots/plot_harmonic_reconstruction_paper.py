#!/usr/bin/env python3
"""
Create spherical harmonic reconstruction map for paper.

Reconstructs global surface potential from temporal coefficients at a specific time.

Example:
    uv run python scripts/plots/plot_harmonic_reconstruction_paper.py \\
        --input data/temporal_coefficients_1998_april.npz \\
        --time-index 15 \\
        --output plots/publish/harmonic_reconstruction.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.temporal import load_temporal_coefficients, reconstruct_global_map


def create_harmonic_reconstruction_plot(
    coeff_file: Path,
    time_index: int,
    output_path: Path,
    dpi: int,
    projection: str,
) -> None:
    """
    Create reconstructed global map from spherical harmonic coefficients.

    Args:
        coeff_file: Path to temporal coefficients NPZ file
        time_index: Index of time window to plot (0-based)
        output_path: Where to save the figure
        dpi: Resolution for output
        projection: Matplotlib projection ('rect' or 'mollweide')
    """
    print(f"Loading temporal coefficients from {coeff_file.name}...")
    dataset = load_temporal_coefficients(coeff_file)

    times = dataset.times
    coeffs = dataset.coeffs
    lmax = dataset.lmax

    # Validate time index
    if time_index < 0 or time_index >= len(times):
        raise ValueError(
            f"Time index {time_index} out of range [0, {len(times)-1}]"
        )

    print("\nDataset Info:")
    print(f"  Time range: {times[0]} → {times[-1]}")
    print(f"  Total windows: {len(times)}")
    print(f"  Max degree: lmax = {lmax}")
    print(f"  Selected time: {times[time_index]} (index {time_index})")

    # Reconstruct map at selected time
    print("\nReconstructing global map...")
    lats, lons, potential = reconstruct_global_map(coeffs[time_index], lmax)

    # Create figure
    if projection == "mollweide":
        fig, ax = plt.subplots(
            figsize=(12, 6),
            subplot_kw={"projection": "mollweide"},
            constrained_layout=True,
            dpi=dpi,
        )
        # Convert to radians for Mollweide
        lon_rad = np.deg2rad(lons)
        lat_rad = np.deg2rad(lats)
        mesh = ax.pcolormesh(
            lon_rad, lat_rad, potential, shading="auto", cmap="viridis"
        )
        ax.set_xlabel("Longitude", fontsize=12)
        ax.set_ylabel("Latitude", fontsize=12)
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    else:  # rectangular
        fig, ax = plt.subplots(
            figsize=(10, 6), constrained_layout=True, dpi=dpi
        )
        mesh = ax.pcolormesh(
            lons, lats, potential, shading="auto", cmap="viridis"
        )
        ax.set_xlabel("Longitude (°)", fontsize=12)
        ax.set_ylabel("Latitude (°)", fontsize=12)
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    # Add colorbar
    cbar = fig.colorbar(mesh, ax=ax, label="Φ_surface (V)")

    # Add title with timestamp
    time_str = str(times[time_index])[:19]  # Trim to datetime only
    ax.set_title(
        f"Reconstructed Surface Potential (l_max={lmax}) at {time_str}",
        fontsize=13,
    )

    # Add statistics box
    valid_pot = potential[np.isfinite(potential)]
    textstr = (
        f"Min: {np.min(valid_pot):.1f} V\n"
        f"Median: {np.median(valid_pot):.1f} V\n"
        f"Max: {np.max(valid_pot):.1f} V"
    )
    ax.text(
        0.02,
        0.98,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved to {output_path}")
    print(
        f"Potential range: {np.min(valid_pot):.1f}V to {np.max(valid_pot):.1f}V"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create spherical harmonic reconstruction map for paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NPZ file with temporal coefficients",
    )
    parser.add_argument(
        "--time-index",
        type=int,
        required=True,
        help="Time window index to plot (0-based)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=["rect", "mollweide"],
        default="rect",
        help="Map projection type",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    create_harmonic_reconstruction_plot(
        args.input,
        args.time_index,
        args.output,
        args.dpi,
        args.projection,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
