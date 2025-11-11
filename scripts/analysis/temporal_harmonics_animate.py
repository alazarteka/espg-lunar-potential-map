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
from pathlib import Path
from typing import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from scipy.special import sph_harm_y


def _sph_harm(m: int, l: int, phi, theta):
    """Evaluate spherical harmonics using SciPy's sph_harm_y (θ=colat, φ=azimuth)."""
    return sph_harm_y(l, m, theta, phi)


def _parse_datetime(ts: np.datetime64) -> str:
    """Format numpy datetime64 for titles."""
    return np.datetime_as_string(ts, unit="m")


def load_temporal_coefficients(path: Path) -> dict[str, np.ndarray | int]:
    """Load temporal coefficient dataset saved as NPZ."""
    with np.load(path) as data:
        return {
            "times": data["times"],
            "lmax": int(data["lmax"]),
            "coeffs": data["coeffs"],
            "n_samples": data.get("n_samples"),
            "spatial_coverage": data.get("spatial_coverage"),
            "rms_residuals": data.get("rms_residuals"),
        }


def reconstruct_global_map(
    coeffs: np.ndarray,
    lmax: int,
    lat_steps: int = 181,
    lon_steps: int = 361,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reconstruct global potential map from spherical harmonic coefficients."""
    latitudes = np.linspace(-90.0, 90.0, lat_steps)
    longitudes = np.linspace(-180.0, 180.0, lon_steps)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    lat_rad = np.deg2rad(lat_grid.ravel())
    lon_rad = np.deg2rad(lon_grid.ravel())
    colatitudes = (np.pi / 2.0) - lat_rad

    n_points = lat_rad.size
    n_coeffs = coeffs.size
    design = np.empty((n_points, n_coeffs), dtype=np.complex128)

    col_idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            design[:, col_idx] = _sph_harm(m, l, lon_rad, colatitudes)
            col_idx += 1

    potential_flat = np.real(design @ coeffs)
    potential_map = potential_flat.reshape(lat_grid.shape)
    return latitudes, longitudes, potential_map


def compute_potential_series(
    coeffs: np.ndarray,
    lmax: int,
    lat_steps: int,
    lon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute reconstructed potential map for each time index."""
    latitudes: np.ndarray | None = None
    longitudes: np.ndarray | None = None
    maps = np.empty((coeffs.shape[0], lat_steps, lon_steps), dtype=np.float32)

    for idx in range(coeffs.shape[0]):
        lats, lons, potential = reconstruct_global_map(
            coeffs[idx], lmax, lat_steps=lat_steps, lon_steps=lon_steps
        )
        if latitudes is None:
            latitudes = lats
        if longitudes is None:
            longitudes = lons
        maps[idx] = potential
    assert latitudes is not None and longitudes is not None  # for mypy
    return latitudes, longitudes, maps


def _compute_edges(values: np.ndarray, clamp_min: float, clamp_max: float) -> np.ndarray:
    """Derive cell-edge coordinates from monotonically increasing centers."""
    diffs = np.diff(values) / 2.0
    edges = np.empty(values.size + 1, dtype=values.dtype)
    edges[1:-1] = values[:-1] + diffs
    edges[0] = values[0] - diffs[0]
    edges[-1] = values[-1] + diffs[-1]
    return np.clip(edges, clamp_min, clamp_max)


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

    lon_edges = _compute_edges(longitudes, -180.0, 180.0)
    theta_edges = np.deg2rad(lon_edges)

    north_edges = _compute_edges(north_lats, 0.0, 90.0)
    south_edges = _compute_edges(south_lats, -90.0, 0.0)
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

    title = fig.suptitle(f"t = {_parse_datetime(times[0])}")

    def update(frame: int):
        north_mesh.set_array(north_maps[frame].ravel())
        south_mesh.set_array(south_maps[frame].ravel())
        title.set_text(f"t = {_parse_datetime(times[frame])}")
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
    lon_edges = _compute_edges(longitudes, -180.0, 180.0)
    lat_edges = _compute_edges(latitudes, -90.0, 90.0)
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

    title = fig.suptitle(f"t = {_parse_datetime(times[0])}")

    def update(frame: int):
        im.set_data(maps[frame])
        title.set_text(f"t = {_parse_datetime(times[frame])}")
        return im, title

    ani = animation.FuncAnimation(
        fig, update, frames=maps.shape[0], interval=1000 / fps, blit=False
    )
    try:
        ani.save(str(output_path), writer=writer, dpi=dpi)
    finally:
        plt.close(fig)


def compute_color_limits(
    maps: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    symmetric_percentile: float | None,
) -> tuple[float, float]:
    """Determine global color scale."""
    if (
        symmetric_percentile is not None
        and 0.0 < symmetric_percentile <= 100.0
        and (vmin is None or vmax is None)
    ):
        percentile_value = float(
            np.nanpercentile(np.abs(maps), symmetric_percentile)
        )
        if percentile_value > 0:
            if vmin is None:
                vmin = -percentile_value
            if vmax is None:
                vmax = percentile_value

    if vmin is None:
        vmin = float(np.nanmin(maps))
    if vmax is None:
        vmax = float(np.nanmax(maps))
    if np.isclose(vmin, vmax):
        delta = max(1.0, abs(vmin) * 0.1 + 1.0)
        vmin -= delta
        vmax += delta
    return vmin, vmax


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
        default=Path("plots/temporal_harmonics_animation"),
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

    data = load_temporal_coefficients(args.input)
    times: np.ndarray = np.asarray(data["times"])
    coeffs: np.ndarray = np.asarray(data["coeffs"])
    lmax: int = int(data["lmax"])

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
