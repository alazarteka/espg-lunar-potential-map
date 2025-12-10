#!/usr/bin/env python3
"""
Create terminator potential profile plot for paper.

Analyzes potential vs Solar Zenith Angle (SZA) to show day/night/terminator transitions.

Example:
    uv run python scripts/plots/plot_terminator_profile_paper.py \\
        --start 1998-04-01 --end 1998-04-30 \\
        --output plots/publish/terminator_profile.png
"""

from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization import loaders, style, utils


def compute_binned_statistics(
    sza: np.ndarray, potentials: np.ndarray, bin_width: float = 5.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute binned median and MAD for potential vs SZA.

    Args:
        sza: Solar zenith angles
        potentials: Surface potentials
        bin_width: Bin width in degrees

    Returns:
        bin_centers: Center of each SZA bin
        medians: Median potential in each bin
        mads: Median absolute deviation in each bin
        counts: Number of samples in each bin
    """
    bins = np.arange(0, 180 + bin_width, bin_width)
    bin_centers = (bins[:-1] + bins[1:]) / 2

    medians = []
    mads = []
    counts = []

    for i in range(len(bins) - 1):
        mask = (sza >= bins[i]) & (sza < bins[i + 1])
        bin_data = potentials[mask]

        if len(bin_data) > 0:
            med = np.median(bin_data)
            mad = np.median(np.abs(bin_data - med))
            medians.append(med)
            mads.append(mad)
            counts.append(len(bin_data))
        else:
            medians.append(np.nan)
            mads.append(np.nan)
            counts.append(0)

    return bin_centers, np.array(medians), np.array(mads), np.array(counts)


def create_terminator_profile_plot(
    cache_dir: Path,
    start_day: date,
    end_day: date,
    output_path: Path,
    bin_width: float,
    dpi: int,
    title: str | None = None,
) -> None:
    """
    Create terminator potential profile plot.

    Args:
        cache_dir: Directory containing cached NPZ files
        start_day: Start date
        end_day: End date
        output_path: Where to save the figure
        bin_width: Bin width for statistics (degrees)
        dpi: Resolution for output
        title: Optional title override
    """
    # Load data
    sza, potentials, in_sun = loaders.load_date_range_data_with_sza(
        cache_dir, start_day, end_day
    )

    print(f"\nLoaded {len(sza)} measurements")
    print(f"Sunlit: {np.sum(in_sun)}, Shadowed: {np.sum(~in_sun)}")

    # Compute binned statistics
    bin_centers, medians, mads, counts = compute_binned_statistics(
        sza, potentials, bin_width
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True, dpi=dpi)

    # Plot individual measurements as scatter
    sunlit_mask = in_sun.astype(bool)
    ax.scatter(
        sza[sunlit_mask],
        potentials[sunlit_mask],
        c=style.COLOR_SUNLIT,
        s=1,
        alpha=0.3,
        label="Sunlit (SZA ≤ 88°)",
        rasterized=True,
    )
    ax.scatter(
        sza[~sunlit_mask],
        potentials[~sunlit_mask],
        c=style.COLOR_SHADOWED,
        s=1,
        alpha=0.3,
        label="Shadowed (SZA ≥ 92°)",
        rasterized=True,
    )

    # Plot binned median ± MAD
    valid = ~np.isnan(medians) & (counts > 10)
    ax.plot(
        bin_centers[valid],
        medians[valid],
        "k-",
        linewidth=2,
        label=f"Median ({bin_width:.0f}° bins)",
    )
    ax.fill_between(
        bin_centers[valid],
        medians[valid] - mads[valid],
        medians[valid] + mads[valid],
        color="gray",
        alpha=0.3,
        label="Median ± MAD",
    )

    # Highlight terminator region
    ax.axvspan(
        88, 92, color=style.COLOR_TERMINATOR, alpha=0.1, label="Terminator (88°-92°)"
    )
    ax.axvline(90, color=style.COLOR_TERMINATOR, linestyle="--", linewidth=1, alpha=0.5)

    # Add subsolar and anti-solar point annotations
    ax.axvline(
        0, color=style.COLOR_SUBSOLAR, linestyle=":", linewidth=1.5, alpha=0.7
    )
    ax.text(
        2,
        ax.get_ylim()[1] * 0.95,
        "Subsolar\nPoint",
        fontsize=style.FONT_SIZE_TEXT,
        color=style.COLOR_SUBSOLAR,
        weight="bold",
        va="top",
    )

    ax.axvline(
        180, color=style.COLOR_ANTISOLAR, linestyle=":", linewidth=1.5, alpha=0.7
    )
    ax.text(
        178,
        ax.get_ylim()[1] * 0.95,
        "Anti-Solar\nPoint",
        fontsize=style.FONT_SIZE_TEXT,
        color=style.COLOR_ANTISOLAR,
        weight="bold",
        va="top",
        ha="right",
    )

    # Apply shared style
    style.apply_paper_style(ax, grid=True)

    # Formatting
    ax.set_xlabel("Solar Zenith Angle (°)", fontsize=style.FONT_SIZE_LABEL)
    ax.set_ylabel("Surface Potential (V)", fontsize=style.FONT_SIZE_LABEL)

    if title:
        plot_title = title
    else:
        date_range_str = f"{start_day.strftime('%b %Y')}"
        if start_day.year != end_day.year or start_day.month != end_day.month:
            date_range_str = (
                f"{start_day.strftime('%b %Y')} - {end_day.strftime('%b %Y')}"
            )
        plot_title = f"Lunar Surface Potential vs Solar Zenith Angle ({date_range_str})"

    ax.set_title(plot_title, fontsize=style.FONT_SIZE_TITLE)

    ax.legend(loc="best", fontsize=style.FONT_SIZE_TEXT, markerscale=5)
    ax.set_xlim(0, 180)

    # Add text annotations via shared util
    textstr = (
        f"Total measurements: {len(sza):,}\n"
        f"Sunlit: {np.sum(sunlit_mask):,}\n"
        f"Shadowed: {np.sum(~sunlit_mask):,}"
    )
    utils.add_stats_box(ax, textstr, loc="upper left")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved to {output_path}")

    # Print statistics
    sunlit_pot = potentials[sunlit_mask]
    shadow_pot = potentials[~sunlit_mask]
    print(f"\nSunlit potential: {np.median(sunlit_pot):.1f} V (median)")
    print(f"Shadowed potential: {np.median(shadow_pot):.1f} V (median)")
    print(
        f"Contrast (sunlit - shadowed): "
        f"{np.median(sunlit_pot) - np.median(shadow_pot):.1f} V"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create terminator potential profile plot for paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=utils.parse_iso_date,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=utils.parse_iso_date,
        required=True,
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
        "--bin-width",
        type=float,
        default=5.0,
        help="SZA bin width in degrees for statistics",
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

    if not args.cache_dir.exists():
        print(f"Error: Cache directory {args.cache_dir} not found")
        return 1

    create_terminator_profile_plot(
        args.cache_dir,
        args.start,
        args.end,
        args.output,
        args.bin_width,
        args.dpi,
        title=args.title,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
