"""
Render surface potential measurements in hemispheric and global projections.

The script can produce either a static figure for a single mission day or an
animated GIF/MP4 covering a date range. The visual style mirrors
``animate_temporal_harmonics.py`` but uses the actual footprint measurements
from the cached potential rows.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection

DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")
DEFAULT_OUTPUT_DIR = Path("artifacts/plots/daily_measurements")

POINT_SIZE_POLAR = 20.0
POINT_SIZE_GLOBAL = 12.0
SUN_SIZE_POLAR = 48.0
SUN_SIZE_GLOBAL = 36.0


def _parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a Python date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - argparse emits message
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc


def _date_range(start_day: date, end_day: date) -> list[date]:
    """Inclusive list of days between start_day and end_day."""
    if end_day < start_day:
        raise SystemExit("--end must be >= --start")
    span = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(span + 1)]


def _format_date_label(day: date) -> str:
    """Render day for titles."""
    return day.strftime("%Y-%m-%d")


def _prepare_writer(name: str, fps: int) -> animation.AnimationWriter:
    """Instantiate matplotlib animation writer by name."""
    if name == "pillow":
        return animation.PillowWriter(fps=fps)
    if name == "ffmpeg":
        if not animation.writers.is_available("ffmpeg"):
            raise RuntimeError(
                "ffmpeg writer requested but not available. Install ffmpeg or choose --writer pillow."
            )
        return animation.FFMpegWriter(fps=fps)
    raise ValueError(f"Unsupported writer '{name}'. Choose from pillow, ffmpeg.")


def _find_daily_file(cache_dir: Path, day: date) -> Path:
    """Locate NPZ cache file matching the requested date."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")

    pattern = f"3D{day.strftime('%y%m%d')}.npz"
    matches = list(cache_dir.rglob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No cached potential file named {pattern} found under {cache_dir}"
        )
    if len(matches) > 1:
        raise RuntimeError(
            f"Multiple files matched {pattern}; please disambiguate by passing --input explicitly: {matches}"
        )
    return matches[0]


@dataclass(slots=True)
class MeasurementPoints:
    """Measurement bundle for a day."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    potentials: np.ndarray
    in_sun: np.ndarray


@dataclass(slots=True)
class FrameData:
    """Prepared geometry for a single animation frame."""

    day: date
    label: str
    count: int
    north_offsets: np.ndarray
    north_potentials: np.ndarray
    north_sun_offsets: np.ndarray
    south_offsets: np.ndarray
    south_potentials: np.ndarray
    south_sun_offsets: np.ndarray
    global_offsets: np.ndarray
    global_potentials: np.ndarray
    global_sun_offsets: np.ndarray


def _load_measurements(path: Path) -> MeasurementPoints:
    """Load projected surface measurements from cache."""
    with np.load(path) as data:
        potentials = data["rows_projected_potential"].astype(np.float64)
        latitudes = data["rows_projection_latitude"].astype(np.float64)
        longitudes = data["rows_projection_longitude"].astype(np.float64)
        in_sun = data.get("rows_projection_in_sun")
        if in_sun is None:
            in_sun = np.zeros_like(potentials, dtype=bool)
        else:
            in_sun = in_sun.astype(bool)

    mask = np.isfinite(potentials) & np.isfinite(latitudes) & np.isfinite(longitudes)
    if not np.any(mask):
        raise RuntimeError(f"{path} contains no projected measurement footprints")

    return MeasurementPoints(
        latitudes=latitudes[mask],
        longitudes=longitudes[mask],
        potentials=potentials[mask],
        in_sun=in_sun[mask],
    )


def _sample_measurements(
    points: MeasurementPoints,
    sample: int | None,
    rng: np.random.Generator | None,
) -> MeasurementPoints:
    """Down-sample measurement points if requested."""
    total = points.latitudes.size
    if sample is None or sample <= 0 or sample >= total:
        return points
    rng = rng or np.random.default_rng()
    idx = rng.choice(total, size=sample, replace=False)
    return MeasurementPoints(
        latitudes=points.latitudes[idx],
        longitudes=points.longitudes[idx],
        potentials=points.potentials[idx],
        in_sun=points.in_sun[idx],
    )


def _compute_color_limits(
    values: np.ndarray,
    vmin: float | None,
    vmax: float | None,
    symmetric_percentile: float | None,
) -> tuple[float, float]:
    """Derive color scale bounds from measurement potentials."""
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError("Measurement potentials contain no finite values")

    data = values[finite]
    if (
        symmetric_percentile is not None
        and 0.0 < symmetric_percentile <= 100.0
        and (vmin is None or vmax is None)
    ):
        clip = float(np.nanpercentile(np.abs(data), symmetric_percentile))
        if clip > 0:
            if vmin is None:
                vmin = -clip
            if vmax is None:
                vmax = clip

    if vmin is None:
        vmin = float(np.nanmin(data))
    if vmax is None:
        vmax = float(np.nanmax(data))
    if np.isclose(vmin, vmax):
        expand = max(1.0, abs(vmin) * 0.1 + 1.0)
        vmin -= expand
        vmax += expand
    return vmin, vmax


def _column_stack_or_empty(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Stack two equally shaped arrays or return empty offsets when size is zero."""
    if first.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    return np.column_stack((first, second))


def _frame_from_points(day: date, points: MeasurementPoints) -> FrameData:
    """Convert measurements into polar/global offsets for plotting."""
    label = _format_date_label(day)
    count = points.latitudes.size

    north_mask = points.latitudes >= 0.0
    north_theta = np.deg2rad(points.longitudes[north_mask])
    north_radius = 90.0 - points.latitudes[north_mask]
    north_pot = points.potentials[north_mask]
    north_sun = points.in_sun[north_mask]

    south_mask = points.latitudes <= 0.0
    south_theta = np.deg2rad(points.longitudes[south_mask])
    south_radius = points.latitudes[south_mask] + 90.0
    south_pot = points.potentials[south_mask]
    south_sun = points.in_sun[south_mask]

    global_offsets = _column_stack_or_empty(points.longitudes, points.latitudes)
    global_sun_offsets = _column_stack_or_empty(
        points.longitudes[points.in_sun], points.latitudes[points.in_sun]
    )

    return FrameData(
        day=day,
        label=label,
        count=count,
        north_offsets=_column_stack_or_empty(north_theta, north_radius),
        north_potentials=north_pot,
        north_sun_offsets=_column_stack_or_empty(
            north_theta[north_sun], north_radius[north_sun]
        ),
        south_offsets=_column_stack_or_empty(south_theta, south_radius),
        south_potentials=south_pot,
        south_sun_offsets=_column_stack_or_empty(
            south_theta[south_sun], south_radius[south_sun]
        ),
        global_offsets=global_offsets,
        global_potentials=points.potentials,
        global_sun_offsets=global_sun_offsets,
    )


def _configure_polar_axes(ax: plt.Axes, hemisphere: str) -> None:
    """Apply consistent styling to polar axes."""
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0.0, 90.0)
    ax.set_xticks(np.deg2rad(np.arange(0, 360, 60)))
    ax.set_xticklabels(["0°", "60°E", "120°E", "180°", "120°W", "60°W"])
    ax.set_yticks([0.0, 30.0, 60.0, 90.0])
    if hemisphere == "north":
        ax.set_yticklabels(["90°N", "60°N", "30°N", "Equator"])
        ax.set_title("Northern Hemisphere Measurements")
    else:
        ax.set_yticklabels(["90°S", "60°S", "30°S", "Equator"])
        ax.set_title("Southern Hemisphere Measurements")


def _create_base_figure(
    frame: FrameData,
    vmin: float,
    vmax: float,
    dpi: int,
    highlight_sun: bool,
) -> tuple[plt.Figure, dict[str, PathCollection | plt.Text | None]]:
    """Create the figure and initial artists for animation/static rendering."""
    fig = plt.figure(figsize=(12, 5.5), constrained_layout=True, dpi=dpi)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 1.3))
    ax_north = fig.add_subplot(grid[0, 0], projection="polar")
    ax_south = fig.add_subplot(grid[0, 1], projection="polar")
    ax_global = fig.add_subplot(grid[0, 2])

    _configure_polar_axes(ax_north, "north")
    _configure_polar_axes(ax_south, "south")

    scatter_north = ax_north.scatter(
        frame.north_offsets[:, 0],
        frame.north_offsets[:, 1],
        c=frame.north_potentials,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=POINT_SIZE_POLAR,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.2,
    )
    scatter_south = ax_south.scatter(
        frame.south_offsets[:, 0],
        frame.south_offsets[:, 1],
        c=frame.south_potentials,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=POINT_SIZE_POLAR,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.2,
    )
    scatter_global = ax_global.scatter(
        frame.global_offsets[:, 0],
        frame.global_offsets[:, 1],
        c=frame.global_potentials,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        s=POINT_SIZE_GLOBAL,
        alpha=0.85,
        edgecolors="black",
        linewidths=0.2,
    )

    sun_north: PathCollection | None = None
    sun_south: PathCollection | None = None
    sun_global: PathCollection | None = None
    if highlight_sun:
        sun_north = ax_north.scatter(
            frame.north_sun_offsets[:, 0],
            frame.north_sun_offsets[:, 1],
            s=SUN_SIZE_POLAR,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )
        sun_south = ax_south.scatter(
            frame.south_sun_offsets[:, 0],
            frame.south_sun_offsets[:, 1],
            s=SUN_SIZE_POLAR,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )
        sun_global = ax_global.scatter(
            frame.global_sun_offsets[:, 0],
            frame.global_sun_offsets[:, 1],
            s=SUN_SIZE_GLOBAL,
            facecolors="none",
            edgecolors="white",
            linewidths=0.6,
        )

    ax_global.set_xlim(-180, 180)
    ax_global.set_ylim(-90, 90)
    ax_global.set_xlabel("Longitude (°)")
    ax_global.set_ylabel("Latitude (°)")
    ax_global.set_title("Global Surface Measurements")
    ax_global.grid(True, linestyle="--", linewidth=0.4, alpha=0.4)

    fig.colorbar(
        scatter_global,
        ax=(ax_north, ax_south, ax_global),
        label="Φ_surface (V)",
        fraction=0.046,
        pad=0.02,
    )

    title = fig.suptitle(
        f"Lunar Surface Potential Measurements — {frame.label} (n={frame.count})"
    )

    return fig, {
        "north": scatter_north,
        "south": scatter_south,
        "global": scatter_global,
        "sun_north": sun_north,
        "sun_south": sun_south,
        "sun_global": sun_global,
        "title": title,
    }


def _update_scatter(
    collection: PathCollection, offsets: np.ndarray, size: float
) -> None:
    """Update offsets and marker sizes for a PathCollection."""
    collection.set_offsets(offsets)
    collection.set_sizes(np.full(offsets.shape[0], size))


def _update_colored_scatter(
    collection: PathCollection,
    offsets: np.ndarray,
    values: np.ndarray,
    size: float,
) -> None:
    """Update offsets, color array, and marker sizes."""
    _update_scatter(collection, offsets, size)
    collection.set_array(values)


def _update_artists(
    frame: FrameData,
    artists: dict[str, PathCollection | plt.Text | None],
    highlight_sun: bool,
) -> Sequence[PathCollection | plt.Text]:
    """Apply new frame data to the matplotlib artists."""
    scatter_north = artists["north"]
    scatter_south = artists["south"]
    scatter_global = artists["global"]
    if not isinstance(scatter_north, PathCollection) or not isinstance(
        scatter_south, PathCollection
    ):
        raise RuntimeError("Scatter artists not initialized correctly.")
    if not isinstance(scatter_global, PathCollection):
        raise RuntimeError("Global scatter artist not initialized correctly.")

    _update_colored_scatter(
        scatter_north, frame.north_offsets, frame.north_potentials, POINT_SIZE_POLAR
    )
    _update_colored_scatter(
        scatter_south, frame.south_offsets, frame.south_potentials, POINT_SIZE_POLAR
    )
    _update_colored_scatter(
        scatter_global,
        frame.global_offsets,
        frame.global_potentials,
        POINT_SIZE_GLOBAL,
    )

    artist_list: list[PathCollection | plt.Text] = [
        scatter_north,
        scatter_south,
        scatter_global,
    ]

    if highlight_sun:
        sun_north = artists["sun_north"]
        sun_south = artists["sun_south"]
        sun_global = artists["sun_global"]
        if isinstance(sun_north, PathCollection):
            _update_scatter(sun_north, frame.north_sun_offsets, SUN_SIZE_POLAR)
            artist_list.append(sun_north)
        if isinstance(sun_south, PathCollection):
            _update_scatter(sun_south, frame.south_sun_offsets, SUN_SIZE_POLAR)
            artist_list.append(sun_south)
        if isinstance(sun_global, PathCollection):
            _update_scatter(sun_global, frame.global_sun_offsets, SUN_SIZE_GLOBAL)
            artist_list.append(sun_global)

    title = artists["title"]
    if isinstance(title, plt.Text):
        title.set_text(
            f"Lunar Surface Potential Measurements — {frame.label} (n={frame.count})"
        )
        artist_list.append(title)

    return artist_list


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description=(
            "Plot measured lunar surface potentials in hemispheric and global projections. "
            "Produces an animation when multiple days are requested."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--start",
        "--date",
        dest="start",
        required=True,
        type=_parse_iso_date,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
        help="End date (YYYY-MM-DD, inclusive). Defaults to --start when omitted.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Base directory containing cached measurement NPZ files",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Optional explicit NPZ file for single-day rendering (bypasses discovery)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Destination directory for generated plots",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Explicit output file path (overrides default naming)",
    )
    parser.add_argument(
        "--output-format",
        choices=("png", "pdf", "svg"),
        default="png",
        help="Static figure format when only one day is requested",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=4,
        help="Frames per second when rendering animations",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI when rendering to raster formats",
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
        help="Symmetric percentile clip for |Φ| when deriving color scale",
    )
    parser.add_argument(
        "--vmin",
        type=float,
        default=None,
        help="Optional fixed lower bound for color scale",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=None,
        help="Optional fixed upper bound for color scale",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Uniformly down-sample measurement points for readability",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used when down-sampling points",
    )
    parser.add_argument(
        "--no-sun-highlight",
        action="store_true",
        help="Disable special outlines for points in sunlight",
    )
    parser.add_argument(
        "--limit-frames",
        type=int,
        default=None,
        help="Optional cap on number of frames (useful for quick previews)",
    )
    parser.add_argument(
        "--force-animation",
        action="store_true",
        help="Render an animation even when only one day is requested",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the interactive window after saving the output",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    if args.fps <= 0:
        raise ValueError("Frames per second (--fps) must be positive")

    start_day: date = args.start
    end_day: date = args.end or start_day

    if args.input is not None:
        if not args.input.exists():
            raise FileNotFoundError(f"Input file {args.input} not found")
        # Treat explicit input as a single-day render.
        points = _load_measurements(args.input)
        rng = np.random.default_rng(args.seed) if args.sample else None
        points = _sample_measurements(points, args.sample, rng)
        frames = [_frame_from_points(start_day, points)]
    else:
        days = _date_range(start_day, end_day)
        if args.limit_frames is not None and args.limit_frames > 0:
            days = days[: args.limit_frames]
        rng = np.random.default_rng(args.seed) if args.sample else None
        frames: list[FrameData] = []
        for day in days:
            try:
                daily_path = _find_daily_file(args.cache_dir, day)
            except FileNotFoundError as exc:
                print(f"Skipping {day:%Y-%m-%d}: {exc}")
                continue
            try:
                points = _load_measurements(daily_path)
            except RuntimeError as exc:
                print(f"Skipping {day:%Y-%m-%d}: {exc}")
                continue
            points = _sample_measurements(points, args.sample, rng)
            frames.append(_frame_from_points(day, points))

    if not frames:
        raise RuntimeError("No frames available to plot")

    all_potentials = np.concatenate([frame.global_potentials for frame in frames])
    percentile = args.symmetric_percentile
    if percentile is not None:
        percentile = max(0.0, min(100.0, percentile))
    vmin, vmax = _compute_color_limits(all_potentials, args.vmin, args.vmax, percentile)

    highlight_sun = not args.no_sun_highlight
    is_animation = args.force_animation or len(frames) > 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output is not None:
        output_path = args.output
    else:
        if is_animation:
            suffix = ".gif" if args.writer == "pillow" else ".mp4"
            output_name = (
                f"surface_measurements_{frames[0].day.strftime('%Y%m%d')}_"
                f"{frames[-1].day.strftime('%Y%m%d')}{suffix}"
            )
        else:
            output_name = (
                f"surface_measurements_{frames[0].day.strftime('%Y%m%d')}"
                f".{args.output_format}"
            )
        output_path = args.output_dir / output_name

    if is_animation:
        writer = _prepare_writer(args.writer, fps=args.fps)
        fig, artists = _create_base_figure(
            frames[0], vmin=vmin, vmax=vmax, dpi=args.dpi, highlight_sun=highlight_sun
        )

        def update(frame_index: int):
            return _update_artists(
                frames[frame_index], artists, highlight_sun=highlight_sun
            )

        ani = animation.FuncAnimation(
            fig,
            update,
            frames=len(frames),
            interval=1000 / args.fps,
            blit=False,
        )
        print(f"Saving animation to {output_path}")
        try:
            ani.save(str(output_path), writer=writer, dpi=args.dpi)
        finally:
            if args.show:
                plt.show()
            else:
                plt.close(fig)
    else:
        fig, artists = _create_base_figure(
            frames[0], vmin=vmin, vmax=vmax, dpi=args.dpi, highlight_sun=highlight_sun
        )
        # Ensure artists reflect any color scaling adjustments.
        _update_artists(frames[0], artists, highlight_sun=highlight_sun)
        fig.savefig(output_path, dpi=args.dpi)
        print(f"Saved plot to {output_path}")
        if args.show:
            plt.show()
        else:
            plt.close(fig)

    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
