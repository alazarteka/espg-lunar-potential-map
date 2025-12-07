"""Plot cached potential mapper results on a 3D lunar sphere for a date range."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Default cache root mirrors the batch runner's output
DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")


def _parse_iso_date(value: str) -> np.datetime64:
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc
    return np.datetime64(dt.date())


def _discover_npz(cache_dir: Path) -> list[Path]:
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")
    return sorted(p for p in cache_dir.rglob("*.npz") if p.is_file())


def _load_rows(
    files: Iterable[Path],
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lats: list[np.ndarray] = []
    lons: list[np.ndarray] = []
    potentials: list[np.ndarray] = []

    start_str = str(start_ts.astype("datetime64[s]"))
    end_str = str(end_ts_exclusive.astype("datetime64[s]"))

    for path in files:
        with np.load(path) as data:
            utc = data["rows_utc"]  # dtype <U64
            proj_lat = data["rows_projection_latitude"]
            proj_lon = data["rows_projection_longitude"]
            proj_pot = data["rows_projected_potential"]

            if utc.size == 0:
                continue

            valid_time = utc != ""
            if not np.any(valid_time):
                continue

            mask = valid_time & (utc >= start_str) & (utc < end_str)
            if not np.any(mask):
                continue

            finite_mask = (
                np.isfinite(proj_lat)
                & np.isfinite(proj_lon)
                & np.isfinite(proj_pot)
            )
            mask &= finite_mask
            if not np.any(mask):
                continue

            lats.append(proj_lat[mask])
            lons.append(proj_lon[mask])
            potentials.append(proj_pot[mask])

    if not lats:
        return (np.array([]), np.array([]), np.array([]))

    return (
        np.concatenate(lats),
        np.concatenate(lons),
        np.concatenate(potentials),
    )


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
    ax.plot_wireframe(sphere_x, sphere_y, sphere_z, color="lightgray", linewidth=0.3, alpha=0.3)

    ax.set_box_aspect([1, 1, 1])
    ax.set_axis_off()
    ax.set_title(title)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Surface potential Φ_surface (V)")

    return fig, ax


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot cached lunar surface potentials on a sphere for a date range",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=_parse_iso_date,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=_parse_iso_date,
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.end < args.start:
        raise SystemExit("--end must be >= --start")

    start_ts = args.start.astype("datetime64[s]")
    end_ts_exclusive = (args.end + np.timedelta64(1, "D")).astype("datetime64[s]")

    files = _discover_npz(args.cache_dir)
    lat, lon, pot = _load_rows(files, start_ts, end_ts_exclusive)

    if lat.size == 0:
        print("No cached rows found in the requested date range.")
        return 1

    lat, lon, pot = _sample_rows(lat, lon, pot, args.sample, args.seed)

    title = (
        f"Φ_surface {str(args.start)} → {str(args.end)}"
        f" | rows={pot.size:n}"
    )
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
