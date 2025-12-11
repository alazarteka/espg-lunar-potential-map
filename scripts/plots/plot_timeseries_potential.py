#!/usr/bin/env python3
"""
Plot potential timeseries at a specific lunar surface location.

Given harmonic coefficients, latitude, and longitude, reconstructs
the surface potential across all time windows and plots the timeseries.

Example:
    uv run python scripts/plots/plot_timeseries_potential.py \\
        --input artifacts/paper/harmonics/april_1998_lmax10.npz \\
        --lat 0 --lon 0 \\
        --output plots/timeseries_equator.png

    # Multiple locations:
    uv run python scripts/plots/plot_timeseries_potential.py \\
        --input artifacts/paper/harmonics/april_1998_lmax10.npz \\
        --lat 0 45 -45 --lon 0 90 -90 \\
        --output plots/timeseries_multi.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import sph_harm_y

from src.temporal import load_temporal_coefficients
from src.visualization import style


def _sph_harm(m: int, ell: int, phi: float, theta: float) -> complex:
    """Evaluate spherical harmonic Y_l^m at (phi, theta)."""
    return sph_harm_y(ell, m, theta, phi)


def reconstruct_potential_at_point(
    coeffs: np.ndarray,
    lmax: int,
    lat: float,
    lon: float,
) -> float:
    """
    Reconstruct surface potential at a single lat/lon point.

    Args:
        coeffs: Spherical harmonic coefficients for one time window
        lmax: Maximum spherical harmonic degree
        lat: Latitude in degrees (-90 to 90)
        lon: Longitude in degrees (-180 to 180)

    Returns:
        Reconstructed potential in Volts
    """
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    colatitude = (np.pi / 2.0) - lat_rad

    potential = 0.0 + 0.0j
    col_idx = 0
    for ell in range(lmax + 1):
        for m in range(-ell, ell + 1):
            Y_lm = _sph_harm(m, ell, lon_rad, colatitude)
            potential += coeffs[col_idx] * Y_lm
            col_idx += 1

    return float(np.real(potential))


def compute_timeseries(
    coeffs: np.ndarray,
    lmax: int,
    lat: float,
    lon: float,
) -> np.ndarray:
    """
    Compute potential timeseries at a single location across all time windows.

    Args:
        coeffs: Array of shape (n_times, n_coeffs)
        lmax: Maximum spherical harmonic degree
        lat: Latitude in degrees
        lon: Longitude in degrees

    Returns:
        Array of potential values, shape (n_times,)
    """
    n_times = coeffs.shape[0]
    potentials = np.empty(n_times, dtype=np.float64)

    for i in range(n_times):
        potentials[i] = reconstruct_potential_at_point(coeffs[i], lmax, lat, lon)

    return potentials


def create_timeseries_plot(
    coeff_file: Path,
    latitudes: list[float],
    longitudes: list[float],
    output_path: Path,
    dpi: int = 150,
    title: str | None = None,
    figsize: tuple[float, float] = (12, 6),
) -> None:
    """
    Create timeseries plot of potential at specified location(s).

    Args:
        coeff_file: Path to temporal coefficients NPZ file
        latitudes: List of latitudes (degrees)
        longitudes: List of longitudes (degrees)
        output_path: Where to save the figure
        dpi: Figure resolution
        title: Optional figure title override
        figsize: Figure size in inches
    """
    if len(latitudes) != len(longitudes):
        raise ValueError(
            f"Number of latitudes ({len(latitudes)}) must match "
            f"longitudes ({len(longitudes)})"
        )

    print(f"Loading temporal coefficients from {coeff_file.name}...")
    dataset = load_temporal_coefficients(coeff_file)

    times = dataset.times
    coeffs = dataset.coeffs
    lmax = dataset.lmax

    print("\nDataset Info:")
    print(f"  Time range: {times[0]} → {times[-1]}")
    print(f"  Total windows: {len(times)}")
    print(f"  Max degree: lmax = {lmax}")

    # Convert times to matplotlib-compatible format
    times_dt = times.astype("datetime64[s]").astype("datetime64[us]")
    times_plot = times_dt.astype(object)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi, constrained_layout=True)

    # Plot each location
    colors = plt.cm.tab10.colors
    for i, (lat, lon) in enumerate(zip(latitudes, longitudes, strict=True)):
        print(f"\nComputing timeseries for ({lat:.1f}°, {lon:.1f}°)...")
        potentials = compute_timeseries(coeffs, lmax, lat, lon)

        label = f"({lat:.0f}°, {lon:.0f}°)"
        color = colors[i % len(colors)]
        ax.plot(times_plot, potentials, "-o", markersize=4, label=label, color=color)

        # Print stats
        print(f"  Min: {potentials.min():.1f} V")
        print(f"  Max: {potentials.max():.1f} V")
        print(f"  Mean: {potentials.mean():.1f} V")
        print(f"  Std: {potentials.std():.1f} V")

    # Style
    style.apply_paper_style(ax)

    ax.set_xlabel("Date", fontsize=style.FONT_SIZE_LABEL)
    ax.set_ylabel("Surface Potential (V)", fontsize=style.FONT_SIZE_LABEL)

    # Format x-axis for dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    fig.autofmt_xdate(rotation=45)

    # Add horizontal line at 0V
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Legend
    if len(latitudes) > 1:
        ax.legend(
            loc="best",
            fontsize=style.FONT_SIZE_LABEL - 2,
            framealpha=0.9,
        )

    # Title
    if title:
        fig_title = title
    else:
        month_str = str(times[0])[:7]
        if len(latitudes) == 1:
            fig_title = (
                f"Surface Potential at ({latitudes[0]:.0f}°, {longitudes[0]:.0f}°) "
                f"- {month_str}"
            )
        else:
            fig_title = f"Surface Potential Timeseries - {month_str}"

    ax.set_title(fig_title, fontsize=style.FONT_SIZE_TITLE, pad=10)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"\nSaved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot potential timeseries at specific lunar location(s)",
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
        "--lat",
        type=float,
        nargs="+",
        required=True,
        help="Latitude(s) in degrees (-90 to 90)",
    )
    parser.add_argument(
        "--lon",
        type=float,
        nargs="+",
        required=True,
        help="Longitude(s) in degrees (-180 to 180)",
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
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=[12, 6],
        help="Figure size (width height) in inches",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.input.exists():
        print(f"Error: Input file {args.input} not found")
        return 1

    # Validate lat/lon counts
    if len(args.lat) != len(args.lon):
        print(
            f"Error: Number of latitudes ({len(args.lat)}) must match "
            f"number of longitudes ({len(args.lon)})"
        )
        return 1

    # Validate ranges
    for lat in args.lat:
        if not -90 <= lat <= 90:
            print(f"Error: Latitude {lat} out of range [-90, 90]")
            return 1
    for lon in args.lon:
        if not -180 <= lon <= 180:
            print(f"Error: Longitude {lon} out of range [-180, 180]")
            return 1

    create_timeseries_plot(
        args.input,
        args.lat,
        args.lon,
        args.output,
        dpi=args.dpi,
        title=args.title,
        figsize=tuple(args.figsize),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
