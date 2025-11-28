#!/usr/bin/env python3
"""
Create terminator potential profile plot for paper.

Shows surface potential vs solar zenith angle with both:
- Individual measurement scatter points
- Binned statistics (median ± MAD)

Example:
    uv run python scripts/plots/plot_terminator_profile_paper.py \\
        --start 1998-04-01 \\
        --end 1998-04-30 \\
        --output plots/publish/terminator_potential_profile.png
"""
from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from src import config
from src.potential_mapper import spice as spice_loader
from src.utils import spice_ops


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


def load_date_range_data(
    cache_dir: Path, start_day: date, end_day: date
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cached potential data for date range and compute SZA.

    Args:
        cache_dir: Directory containing NPZ cache files
        start_day: Start date (inclusive)
        end_day: End date (inclusive)

    Returns:
        sza: Solar zenith angles (degrees)
        potentials: Surface potentials (V)
        in_sun: Boolean array for illumination
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
            f"No cache files found for date range {start_day} to {end_day} in {cache_dir}"
        )

    print(f"Found {len(files)} files for {start_day} to {end_day}")

    # Load SPICE kernels
    spice_loader.load_spice_files()

    all_sza = []
    all_potentials = []
    all_in_sun = []

    for npz_file in files:
        print(f"Loading {npz_file.name}...")
        with np.load(npz_file) as data:
            lats = data["rows_projection_latitude"]
            lons = data["rows_projection_longitude"]
            # rows_projected_potential is UM (surface potential relative to plasma)
            pots = data["rows_projected_potential"]
            utcs = data["rows_utc"]
            in_sun = data["rows_projection_in_sun"]

            # Compute SZA for each measurement
            for lat, lon, pot, utc_str, sun_flag in zip(lats, lons, pots, utcs, in_sun):
                if not np.isfinite(pot):
                    continue

                try:
                    et = spice.utc2et(utc_str)
                    sun_vec = spice_ops.get_sun_vector_wrt_moon(et)

                    # Convert lat/lon to cartesian
                    lat_rad = np.radians(lat)
                    lon_rad = np.radians(lon)
                    point_vec = np.array(
                        [
                            np.cos(lat_rad) * np.cos(lon_rad),
                            np.cos(lat_rad) * np.sin(lon_rad),
                            np.sin(lat_rad),
                        ]
                    )

                    # Compute SZA (angle between surface normal and sun vector)
                    cos_sza = np.dot(point_vec, sun_vec) / (
                        np.linalg.norm(point_vec) * np.linalg.norm(sun_vec)
                    )
                    cos_sza = np.clip(cos_sza, -1.0, 1.0)
                    sza = np.degrees(np.arccos(cos_sza))

                    all_sza.append(sza)
                    all_potentials.append(pot)
                    all_in_sun.append(sun_flag)

                except Exception:
                    continue

    return np.array(all_sza), np.array(all_potentials), np.array(all_in_sun)


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
    """
    # Load data
    sza, potentials, in_sun = load_date_range_data(cache_dir, start_day, end_day)

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
        c="#FDB462",  # Orange for sunlit
        s=1,
        alpha=0.3,
        label="Sunlit (SZA ≤ 88°)",
        rasterized=True,
    )
    ax.scatter(
        sza[~sunlit_mask],
        potentials[~sunlit_mask],
        c="#8DD3C7",  # Teal for shadowed
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
    ax.axvspan(88, 92, color="red", alpha=0.1, label="Terminator (88°-92°)")
    ax.axvline(90, color="red", linestyle="--", linewidth=1, alpha=0.5)

    # Add subsolar and anti-solar point annotations
    ax.axvline(0, color="orange", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(2, ax.get_ylim()[1] * 0.95, "Subsolar\nPoint",
            fontsize=9, color="orange", weight="bold", va="top")

    ax.axvline(180, color="navy", linestyle=":", linewidth=1.5, alpha=0.7)
    ax.text(178, ax.get_ylim()[1] * 0.95, "Anti-Solar\nPoint",
            fontsize=9, color="navy", weight="bold", va="top", ha="right")

    # Formatting
    ax.set_xlabel("Solar Zenith Angle (°)", fontsize=12)
    ax.set_ylabel("Surface Potential (V)", fontsize=12)

    date_range_str = f"{start_day.strftime('%b %Y')}"
    if start_day.year != end_day.year or start_day.month != end_day.month:
        date_range_str = f"{start_day.strftime('%b %Y')} - {end_day.strftime('%b %Y')}"

    ax.set_title(
        f"Lunar Surface Potential vs Solar Zenith Angle ({date_range_str})",
        fontsize=13,
    )
    ax.grid(True, alpha=0.3, linestyle="--", linewidth=0.4)
    ax.legend(loc="best", fontsize=9, markerscale=5)
    ax.set_xlim(0, 180)

    # Add text annotations
    textstr = f"Total measurements: {len(sza):,}\nSunlit: {np.sum(sunlit_mask):,}\nShadowed: {np.sum(~sunlit_mask):,}"
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

    # Print statistics
    sunlit_pot = potentials[sunlit_mask]
    shadow_pot = potentials[~sunlit_mask]
    print(f"\nSunlit potential: {np.median(sunlit_pot):.1f} V (median)")
    print(f"Shadowed potential: {np.median(shadow_pot):.1f} V (median)")
    print(
        f"Contrast (sunlit - shadowed): {np.median(sunlit_pot) - np.median(shadow_pot):.1f} V"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create terminator potential profile plot for paper",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        type=_parse_iso_date,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
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
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
