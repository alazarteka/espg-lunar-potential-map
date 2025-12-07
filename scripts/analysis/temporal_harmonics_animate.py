"""
Generate animated visualizations of time-dependent spherical harmonic reconstructions.

The script consumes the same NPZ temporal coefficient bundles produced by
`scripts/dev/temporal_harmonic_coefficients.py` and generates two animations:

1. Hemispheric view (polar projection of northern and southern hemispheres)
2. Global equirectangular map

Both animations show the evolution of the reconstructed surface potential across
time windows.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from src.temporal import (
    compute_cell_edges,
    compute_color_limits,
    compute_potential_series,
    format_timestamp,
    load_temporal_coefficients,
)


def _prepare_writer(name: str, fps: int) -> animation.AnimationWriter:
    """Instantiate matplotlib writer by name."""
    if name == "pillow":
        return animation.PillowWriter(fps=fps)
    if name == "ffmpeg":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "ffmpeg writer requested but not available. Install ffmpeg or "
                "choose --writer pillow."
            )
        return animation.FFMpegWriter(fps=fps)
    raise ValueError(f"Unsupported writer '{name}'. Choose from pillow, ffmpeg.")


def create_hemisphere_animation(
    output_path: Path,
    times: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    maps: np.ndarray,
    vmin: float,
    vmax: float,
    fps: int,
    dpi: int,
    writer: animation.AnimationWriter,
) -> None:
    """Create animation showing northern and southern hemisphere panels."""
    north_indices = np.where(latitudes >= 0.0)[0]
    south_indices = np.where(latitudes <= 0.0)[0]
    north_lats = latitudes[north_indices]
    south_lats = latitudes[south_indices]

    north_maps = maps[:, north_indices, :]
    south_maps = maps[:, south_indices, :]

    lon_edges = compute_cell_edges(longitudes, -180.0, 180.0)
    theta_edges = np.deg2rad(lon_edges)

    north_edges = compute_cell_edges(north_lats, 0.0, 90.0)
    south_edges = compute_cell_edges(south_lats, -90.0, 0.0)
    north_r_edges = 90.0 - north_edges  # colatitude (0 = pole, 90 = equator)
    south_r_edges = south_edges + 90.0  # distance from south pole (0 = pole)

    theta_north, radius_north = np.meshgrid(theta_edges, north_r_edges)
    theta_south, radius_south = np.meshgrid(theta_edges, south_r_edges)

    fig, (ax_north, ax_south) = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )

    cmap = plt.get_cmap("viridis")
    north_mesh = ax_north.pcolormesh(
        theta_north,
        radius_north,
        north_maps[0],
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )
    south_mesh = ax_south.pcolormesh(
        theta_south,
        radius_south,
        south_maps[0],
        cmap=cmap,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
    )

    for ax in (ax_north, ax_south):
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, 90.0)
        ax.set_xticks(np.deg2rad(np.arange(0, 360, 60)))
        ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"])
        ax.set_yticks([0.0, 30.0, 60.0, 90.0])

    ax_north.set_yticklabels(["90°N", "60°N", "30°N", "Equator"])
    ax_south.set_yticklabels(["90°S", "60°S", "30°S", "Equator"])
    ax_north.set_title("Northern Hemisphere Potential")
    ax_south.set_title("Southern Hemisphere Potential")

    cbar = fig.colorbar(north_mesh, ax=[ax_north, ax_south], label="Φ_surface (V)")
    cbar.ax.tick_params(labelsize=9)

    title = fig.suptitle(f"t = {format_timestamp(times[0])}")

    def update(frame: int):
        north_mesh.set_array(north_maps[frame].ravel())
        south_mesh.set_array(south_maps[frame].ravel())
        title.set_text(f"t = {format_timestamp(times[frame])}")
        return north_mesh, south_mesh, title

    ani = animation.FuncAnimation(
        fig, update, frames=maps.shape[0], interval=1000 / fps, blit=False
    )
    try:
        ani.save(str(output_path), writer=writer, dpi=dpi)
    finally:
        plt.close(fig)


def create_global_animation(
    output_path: Path,
    times: np.ndarray,
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    maps: np.ndarray,
    vmin: float,
    vmax: float,
    fps: int,
    dpi: int,
    writer: animation.AnimationWriter,
) -> None:
    """Create equirectangular global animation."""
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    cmap = plt.get_cmap("viridis")
    lon_edges = compute_cell_edges(longitudes, -180.0, 180.0)
    lat_edges = compute_cell_edges(latitudes, -90.0, 90.0)
    im = ax.imshow(
        maps[0],
        extent=(lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]),
        origin="lower",
        aspect="auto",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title("Global Surface Potential")
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    cbar = fig.colorbar(im, ax=ax, label="Φ_surface (V)")

    title = fig.suptitle(f"t = {format_timestamp(times[0])}")

    def update(frame: int):
        im.set_data(maps[frame])
        title.set_text(f"t = {format_timestamp(times[frame])}")
        return im, title

    ani = animation.FuncAnimation(
        fig, update, frames=maps.shape[0], interval=1000 / fps, blit=False
    )
    try:
        ani.save(str(output_path), writer=writer, dpi=dpi)
    finally:
        plt.close(fig)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Animate temporal spherical harmonic reconstructions",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input NPZ file with temporal harmonic coefficients",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/plots/temporal_harmonics_animation"),
        help="Destination directory for animation files",
    )
    parser.add_argument(
        "--lat-steps",
        type=int,
        default=181,
        help="Number of latitude samples for reconstruction grid",
    )
    parser.add_argument(
        "--lon-steps",
        type=int,
        default=361,
        help="Number of longitude samples for reconstruction grid",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=10,
        help="Frames per second for animations",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=120,
        help="Rendered DPI for animation frames",
    )
    parser.add_argument(
        "--writer",
        choices=("pillow", "ffmpeg"),
        default="pillow",
        help="Animation writer backend (gif via pillow or mp4 via ffmpeg)",
    )
    parser.add_argument(
        "--symmetric-percentile",
        type=float,
        default=99.0,
        help=(
            "Clip color scale using symmetric percentile of |Φ| (set to 100 to disable)"
        ),
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Lower color scale bound (defaults to dataset minimum)",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Upper color scale bound (defaults to dataset maximum)",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames (useful for quick previews)",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if not args.input.exists():
        raise FileNotFoundError(f"Input file {args.input} not found")

    dataset = load_temporal_coefficients(args.input)
    times: np.ndarray = np.asarray(dataset.times)
    coeffs: np.ndarray = np.asarray(dataset.coeffs)
    lmax: int = int(dataset.lmax)

    if args.limit_frames is not None:
        max_frames = min(len(times), args.limit_frames)
        times = times[:max_frames]
        coeffs = coeffs[:max_frames]

    if args.fps <= 0:
        raise ValueError("Frames per second (--fps) must be positive")

    print(f"Loaded {len(times)} time windows with lmax={lmax}")
    print("Reconstructing potential maps for all time windows...")

    latitudes, longitudes, maps = compute_potential_series(
        coeffs, lmax, lat_steps=args.lat_steps, lon_steps=args.lon_steps
    )

    percentile = args.symmetric_percentile
    if percentile is not None:
        percentile = max(0.0, min(100.0, percentile))
    vmin, vmax = compute_color_limits(maps, args.vmin, args.vmax, percentile)
    print(f"Color scale: vmin={vmin:.2f} V, vmax={vmax:.2f} V")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    hemisphere_path = args.output_dir / "hemisphere_potential.gif"
    global_path = args.output_dir / "global_potential.gif"
    if args.writer == "ffmpeg":
        hemisphere_path = hemisphere_path.with_suffix(".mp4")
        global_path = global_path.with_suffix(".mp4")

    writer = _prepare_writer(args.writer, fps=args.fps)

    print(f"Saving hemisphere animation to {hemisphere_path}")
    create_hemisphere_animation(
        hemisphere_path,
        times,
        longitudes,
        latitudes,
        maps,
        vmin,
        vmax,
        fps=args.fps,
        dpi=args.dpi,
        writer=writer,
    )

    writer = _prepare_writer(args.writer, fps=args.fps)
    print(f"Saving global animation to {global_path}")
    create_global_animation(
        global_path,
        times,
        longitudes,
        latitudes,
        maps,
        vmin,
        vmax,
        fps=args.fps,
        dpi=args.dpi,
        writer=writer,
    )

    print(f"Animations written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
