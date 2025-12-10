#!/usr/bin/env python3
"""
Create 2x3 grid of spherical harmonic reconstructions from monthly data.

Selects 6 days from a monthly temporal coefficient file and plots them
in a grid layout with a shared colorbar.

Example:
    uv run python scripts/plots/plot_harmonic_reconstruction_multiday.py \\
        --input data/temporal_coefficients_1998_april.npz \\
        --output plots/publish/harmonic_reconstruction_apr.png \\
        --time-indices 0 5 10 15 20 25

    # Or auto-select evenly spaced days:
    uv run python scripts/plots/plot_harmonic_reconstruction_multiday.py \\
        --input data/temporal_coefficients_1998_april.npz \\
        --output plots/publish/harmonic_reconstruction_apr.png \\
        --auto-select
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.temporal import load_temporal_coefficients, reconstruct_global_map
from src.visualization import style, utils


def create_multiday_reconstruction_plot(
    coeff_file: Path,
    time_indices: list[int],
    output_path: Path,
    dpi: int,
    projection: str,
    vmin: float | None = None,
    vmax: float | None = None,
    title: str | None = None,
) -> None:
    """
    Create 2x3 grid of reconstructed maps from spherical harmonic coefficients.

    Args:
        coeff_file: Path to temporal coefficients NPZ file
        time_indices: List of 6 time indices to plot (0-based)
        output_path: Where to save the figure
        dpi: Resolution for output
        projection: Matplotlib projection ('rect' or 'mollweide')
        vmin: Minimum value for colorbar (auto if None)
        vmax: Maximum value for colorbar (auto if None)
        title: Optional override for the figure title
    """
    if len(time_indices) != 6:
        raise ValueError(f"Expected 6 time indices, got {len(time_indices)}")

    print(f"Loading temporal coefficients from {coeff_file.name}...")
    dataset = load_temporal_coefficients(coeff_file)

    times = dataset.times
    coeffs = dataset.coeffs
    lmax = dataset.lmax

    # Validate time indices
    for idx in time_indices:
        if idx < 0 or idx >= len(times):
            raise ValueError(f"Time index {idx} out of range [0, {len(times) - 1}]")

    print("\nDataset Info:")
    print(f"  Time range: {times[0]} → {times[-1]}")
    print(f"  Total windows: {len(times)}")
    print(f"  Max degree: lmax = {lmax}")
    print("  Selected times:")
    for i, idx in enumerate(time_indices, 1):
        print(f"    {i}. {times[idx]} (index {idx})")

    # Reconstruct all maps first to determine shared color scale
    print(f"\nReconstructing {len(time_indices)} global maps...")
    maps_data = []
    for idx in time_indices:
        lats, lons, potential = reconstruct_global_map(coeffs[idx], lmax)
        maps_data.append((lats, lons, potential, times[idx]))

    # Determine global color scale
    if vmin is None or vmax is None:
        all_potentials = np.concatenate(
            [pot[np.isfinite(pot)] for _, _, pot, _ in maps_data]
        )
        if vmin is None:
            vmin = np.min(all_potentials)
        if vmax is None:
            vmax = np.max(all_potentials)
    print(f"Color scale: {vmin:.1f}V to {vmax:.1f}V")

    # Create 2x3 subplot grid
    if projection == "mollweide":
        fig, axes = plt.subplots(
            2,
            3,
            figsize=(18, 10),
            subplot_kw={"projection": "mollweide"},
            constrained_layout=True,
            dpi=dpi,
        )
    else:  # rectangular
        fig, axes = plt.subplots(
            2, 3, figsize=(18, 10), constrained_layout=True, dpi=dpi
        )

    axes = axes.flatten()

    # Plot each map
    for i, (lats, lons, potential, time) in enumerate(maps_data):
        ax = axes[i]

        if projection == "mollweide":
            # Convert to radians for Mollweide
            lon_rad = np.deg2rad(lons)
            lat_rad = np.deg2rad(lats)
            mesh = ax.pcolormesh(
                lon_rad,
                lat_rad,
                potential,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Longitude", fontsize=style.FONT_SIZE_LABEL - 2)
            ax.set_ylabel("Latitude", fontsize=style.FONT_SIZE_LABEL - 2)
        else:  # rectangular
            mesh = ax.pcolormesh(
                lons,
                lats,
                potential,
                shading="auto",
                cmap="viridis",
                vmin=vmin,
                vmax=vmax,
            )
            ax.set_xlabel("Longitude (°)", fontsize=style.FONT_SIZE_LABEL - 2)
            ax.set_ylabel("Latitude (°)", fontsize=style.FONT_SIZE_LABEL - 2)
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)
            ax.set_aspect("equal")

        # Apply shared style
        style.apply_paper_style(ax)

        # Add title with timestamp
        time_str = str(time)[:10]  # Just the date
        ax.set_title(f"{time_str}", fontsize=style.FONT_SIZE_TITLE - 2, pad=5)

        # Add statistics in corner
        valid_pot = potential[np.isfinite(potential)]
        textstr = (
            f"Min: {np.min(valid_pot):.1f}V\n"
            f"Med: {np.median(valid_pot):.1f}V\n"
            f"Max: {np.max(valid_pot):.1f}V"
        )
        utils.add_stats_box(ax, textstr, loc="upper left")

    # Add shared colorbar
    cbar = fig.colorbar(
        mesh,
        ax=axes,
        label="Φ_surface (V)",
        orientation="horizontal",
        fraction=0.05,
        pad=0.05,
    )

    # Add overall title
    if title:
        fig_title = title
    else:
        month_str = str(times[0])[:7]  # YYYY-MM
        fig_title = f"Reconstructed Surface Potential (l_max={lmax}) - {month_str}"
    fig.suptitle(fig_title, fontsize=style.FONT_SIZE_TITLE + 1, y=0.98)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved to {output_path}")
    print(f"Global potential range: {vmin:.1f}V to {vmax:.1f}V")


def auto_select_indices(total_windows: int, n_select: int = 6) -> list[int]:
    """
    Automatically select evenly-spaced time indices from available windows.

    Args:
        total_windows: Total number of time windows available
        n_select: Number of indices to select (default 6)

    Returns:
        List of evenly-spaced indices
    """
    if total_windows < n_select:
        raise ValueError(f"Not enough windows ({total_windows}) to select {n_select}")

    # Select evenly spaced indices
    indices = np.linspace(0, total_windows - 1, n_select, dtype=int).tolist()
    return indices


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create 2x3 grid of harmonic reconstruction maps",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NPZ file with temporal coefficients",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--time-indices",
        type=int,
        nargs=6,
        help="Six time window indices to plot (0-based)",
    )
    parser.add_argument(
        "--auto-select",
        action="store_true",
        help="Automatically select 6 evenly-spaced time windows",
    )
    parser.add_argument(
        "--projection",
        type=str,
        choices=["rect", "mollweide"],
        default="rect",
        help="Map projection type",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        help="Minimum value for colorbar (auto if not specified)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="Maximum value for colorbar (auto if not specified)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="Output resolution",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override figure title",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    # Determine time indices to use
    if args.auto_select and args.time_indices:
        print("Error: Cannot use both --auto-select and --time-indices")
        return 1

    if args.auto_select:
        # Load to get total windows
        dataset = load_temporal_coefficients(args.input)
        time_indices = auto_select_indices(len(dataset.times), n_select=6)
        print(f"Auto-selected indices: {time_indices}")
    elif args.time_indices:
        time_indices = args.time_indices
    else:
        print("Error: Must specify either --time-indices or --auto-select")
        return 1

    create_multiday_reconstruction_plot(
        args.input,
        time_indices,
        args.output,
        args.dpi,
        args.projection,
        args.vmin,
        args.vmax,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
