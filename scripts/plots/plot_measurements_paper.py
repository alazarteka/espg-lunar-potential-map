#!/usr/bin/env python3
"""
Create surface measurement projection plot for paper.

Shows raw footprint measurements as scatter points on lunar surface.

Example:
    # Single day
    uv run python scripts/plots/plot_measurements_paper.py \\
        --date 1998-04-15 \\
        --output plots/publish/surface_measurements_19980415.png

    # Date range (combined)
    uv run python scripts/plots/plot_measurements_paper.py \\
        --start 1998-04-01 \\
        --end 1998-04-30 \\
        --output plots/publish/surface_measurements_april1998.png
"""

from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a Python date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc


def _date_range(start_day: date, end_day: date) -> list[date]:
    """Inclusive list of days between start_day and end_day."""
    if end_day < start_day:
        raise ValueError("--end must be >= --start")
    span = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(span + 1)]


def load_measurements(
    cache_dir: Path, start_day: date, end_day: date
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load measurement footprints for date range.

    Returns:
        lats: Footprint latitudes
        lons: Footprint longitudes
        potentials: Surface potentials
        in_sun: Illumination flags
    """
    days = _date_range(start_day, end_day)
    pattern_list = [f"3D{day.strftime('%y%m%d')}.npz" for day in days]

    files = []
    for pattern in pattern_list:
        matches = list(cache_dir.rglob(pattern))
        if matches:
            files.append(matches[0])

    if not files:
        raise FileNotFoundError(
            f"No cache files found for {start_day} to {end_day} in {cache_dir}"
        )

    print(f"Found {len(files)} files for {start_day} to {end_day}")

    all_lats = []
    all_lons = []
    all_pots = []
    all_sun = []

    for npz_file in files:
        print(f"Loading {npz_file.name}...")
        with np.load(npz_file) as data:
            lats = data["rows_projection_latitude"]
            lons = data["rows_projection_longitude"]
            pots = data["rows_projected_potential"]
            in_sun = data["rows_projection_in_sun"]

            # Filter valid measurements
            valid = np.isfinite(lats) & np.isfinite(lons) & np.isfinite(pots)

            all_lats.append(lats[valid])
            all_lons.append(lons[valid])
            all_pots.append(pots[valid])
            all_sun.append(in_sun[valid])

    return (
        np.concatenate(all_lats),
        np.concatenate(all_lons),
        np.concatenate(all_pots),
        np.concatenate(all_sun),
    )


def create_measurements_plot(
    cache_dir: Path,
    start_day: date,
    end_day: date,
    output_path: Path,
    point_size: float,
    dpi: int,
) -> None:
    """
    Create surface measurement projection plot.

    Args:
        cache_dir: Directory containing cached NPZ files
        start_day: Start date
        end_day: End date
        output_path: Where to save the figure
        point_size: Scatter point size
        dpi: Resolution for output
    """
    # Load data
    lats, lons, potentials, in_sun = load_measurements(cache_dir, start_day, end_day)

    print(f"\nLoaded {len(lats)} valid measurements")
    print(f"  Sunlit: {np.sum(in_sun)}")
    print(f"  Shadowed: {np.sum(~in_sun)}")
    print(f"  Potential range: {np.min(potentials):.1f}V to {np.max(potentials):.1f}V")

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True, dpi=dpi)

    # Scatter plot
    scatter = ax.scatter(
        lons,
        lats,
        c=potentials,
        cmap="viridis",
        s=point_size,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.2,
        rasterized=True,
    )

    # Optional: highlight sunlit points
    # ax.scatter(lons[in_sun], lats[in_sun], s=point_size*2,
    #            facecolors='none', edgecolors='white', linewidths=0.5)

    # Formatting
    ax.set_xlabel("Longitude (°)", fontsize=12)
    ax.set_ylabel("Latitude (°)", fontsize=12)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)

    # Title
    if start_day == end_day:
        date_str = start_day.strftime("%Y-%m-%d")
    else:
        date_str = f"{start_day.strftime('%Y-%m-%d')} to {end_day.strftime('%Y-%m-%d')}"
    ax.set_title(f"Surface Potential Measurements ({date_str})", fontsize=13)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, label="Φ_surface (V)")

    # Statistics box
    textstr = (
        f"Total: {len(lats):,}\n"
        f"Min: {np.min(potentials):.1f} V\n"
        f"Median: {np.median(potentials):.1f} V\n"
        f"Max: {np.max(potentials):.1f} V"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create surface measurement projection plot for paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Option 1: Single date
    parser.add_argument(
        "--date",
        type=_parse_iso_date,
        help="Single date to plot (YYYY-MM-DD)",
    )

    # Option 2: Date range
    parser.add_argument(
        "--start",
        type=_parse_iso_date,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
        help="End date (YYYY-MM-DD)",
    )

    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("data/potential_cache"),
        help="Cache directory containing NPZ files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for PNG file",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=3.0,
        help="Scatter point size",
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

    # Validate date arguments
    if args.date:
        start_day = end_day = args.date
    elif args.start and args.end:
        start_day = args.start
        end_day = args.end
    else:
        print("Error: Must provide either --date or both --start and --end")
        return 1

    if not args.cache_dir.exists():
        print(f"Error: Cache directory {args.cache_dir} not found")
        return 1

    create_measurements_plot(
        args.cache_dir,
        start_day,
        end_day,
        args.output,
        args.point_size,
        args.dpi,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
