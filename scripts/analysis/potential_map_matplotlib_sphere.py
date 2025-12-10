"""Plot cached potential mapper results on a 3D lunar sphere for a date range."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization import loaders, style, utils

# Default cache root mirrors the batch runner's output
DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")


def _sample_rows(
    lat: np.ndarray,
    lon: np.ndarray,
    pot: np.ndarray,
    sample: int | None,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if sample is None or sample <= 0 or pot.size <= sample:
        return lat, lon, pot
    rng = np.random.default_rng(seed)
    idx = rng.choice(pot.size, size=sample, replace=False)
    return lat[idx], lon[idx], pot[idx]


def _plot_sphere(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    pot: np.ndarray,
    title: str,
    colormap: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[plt.Figure, plt.Axes]:
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)

    sc = ax.scatter(x, y, z, c=pot, cmap=colormap, s=6, alpha=0.9, vmin=vmin, vmax=vmax)

    phi = np.linspace(0, np.pi, 30)
    theta = np.linspace(0, 2 * np.pi, 60)
    sphere_x = np.outer(np.sin(phi), np.cos(theta))
    sphere_y = np.outer(np.sin(phi), np.sin(theta))
    sphere_z = np.outer(np.cos(phi), np.ones_like(theta))

    # Use shared grid style where applicable (wireframe isn't exactly a grid but close)
    ax.plot_wireframe(
        sphere_x, sphere_y, sphere_z,
        color="lightgray",
        linewidth=style.GRID_STYLE["linewidth"],
        alpha=style.GRID_STYLE["alpha"]
    )

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()

    ax.set_title(title, fontsize=style.FONT_SIZE_TITLE)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Surface potential Φ_surface (V)", fontsize=style.FONT_SIZE_LABEL)
    cbar.ax.tick_params(labelsize=style.FONT_SIZE_TEXT)

    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cached lunar surface potentials on a sphere for a date range",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=utils.parse_iso_date,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=utils.parse_iso_date,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Root directory containing potential_cache NPZ files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Randomly down-sample to at most this many rows before plotting",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for the random sampler when --sample is provided",
    )
    parser.add_argument(
        "--colormap",
        default="viridis",
        help="Matplotlib colormap for surface potentials",
    )
    parser.add_argument(
        "--min-pot",
        type=float,
        default=None,
        help="Lower clip for surface potential color scale (V)",
    )
    parser.add_argument(
        "--max-pot",
        type=float,
        default=None,
        help="Upper clip for surface potential color scale (V)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to save the figure",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Show the plot interactively",
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

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    # Use shared loader
    try:
        lat, lon, pot, _ = loaders.load_measurements(
            args.cache_dir, args.start, args.end
        )
    except FileNotFoundError:
        print("No cached rows found in the requested date range.")
        return 1

    if lat.size == 0:
        print("No cached rows found in the requested date range.")
        return 1

    lat, lon, pot = _sample_rows(lat, lon, pot, args.sample, args.seed)

    if args.title:
        title = args.title
    else:
        title = f"Φ_surface {str(args.start)} → {str(args.end)} | rows={pot.size:n}"

    fig, _ = _plot_sphere(
        lat=lat,
        lon=lon,
        pot=pot,
        title=title,
        colormap=args.colormap,
        vmin=args.min_pot,
        vmax=args.max_pot,
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=200)
        print(f"Saved plot to {args.output}")

    if args.display:
        plt.show()

    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
