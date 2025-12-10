#!/usr/bin/env python3
"""
Create surface measurement projection plot for paper.

Plots cached potential measurements on a global Equirectangular map.

Example:
    uv run python scripts/plots/plot_measurements_paper.py \\
        --date 1998-04-15 \\
        --output plots/publish/measurements_map.png
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization import loaders, style, utils


def create_measurements_plot(
    cache_dir: Path,
    start_day: date,
    end_day: date,
    output_path: Path,
    point_size: float,
    dpi: int,
    title: str | None = None,
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
        title: Optional title override
    """
    # Load data
    lats, lons, potentials, in_sun = loaders.load_measurements(
        cache_dir, start_day, end_day
    )

    print(f"\nLoaded {len(lats)} valid measurements")
    print(f"  Sunlit: {np.sum(in_sun)}")
    print(f"  Shadowed: {np.sum(~in_sun)}")
    print(
        f"  Potential range: {np.min(potentials):.1f}V to {np.max(potentials):.1f}V"
    )

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

    # Apply shared style
    style.apply_paper_style(ax)

    # Formatting overrides
    ax.set_xlabel("Longitude (°)", fontsize=style.FONT_SIZE_LABEL)
    ax.set_ylabel("Latitude (°)", fontsize=style.FONT_SIZE_LABEL)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_aspect("equal")

    # Title
    if title:
        plot_title = title
    else:
        if start_day == end_day:
            date_str = start_day.strftime("%Y-%m-%d")
        else:
            date_str = (
                f"{start_day.strftime('%Y-%m-%d')} to {end_day.strftime('%Y-%m-%d')}"
            )
        plot_title = f"Surface Potential Measurements ({date_str})"

    ax.set_title(plot_title, fontsize=style.FONT_SIZE_TITLE)

    # Colorbar
    cbar = fig.colorbar(scatter, ax=ax, label="Φ_surface (V)")

    # Statistics box
    textstr = (
        f"Total: {len(lats):,}\n"
        f"Min: {np.min(potentials):.1f} V\n"
        f"Median: {np.median(potentials):.1f} V\n"
        f"Max: {np.max(potentials):.1f} V"
    )
    utils.add_stats_box(ax, textstr, loc="upper left")

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
        type=utils.parse_iso_date,
        help="Single date to plot (YYYY-MM-DD)",
    )

    # Option 2: Date range
    parser.add_argument(
        "--start",
        type=utils.parse_iso_date,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=utils.parse_iso_date,
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
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Override plot title",
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
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
