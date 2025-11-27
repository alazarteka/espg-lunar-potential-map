"""Interactive Plotly visualization of cached lunar surface potentials."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

import src.config as config

# Default cache directory mirrors the batch runner output
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
            utc = data["rows_utc"]
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


def _load_moon_texture(max_lat: int = 480, max_lon: int = 960) -> np.ndarray | None:
    texture_path = Path(config.DATA_DIR) / config.MOON_MAP_FILE
    if not texture_path.exists():
        logging.debug("Moon texture %s not found", texture_path)
        return None

    try:
        import matplotlib.image as mpimg

        img = mpimg.imread(texture_path)
    except Exception as exc:
        logging.debug("Failed to load moon texture: %s", exc)
        return None

    if img.ndim == 2:
        img = np.repeat(img[:, :, None], 3, axis=2)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    if img.max() <= 1.0:
        img = (img * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)

    lat_idx = np.linspace(0, img.shape[0] - 1, min(max_lat, img.shape[0]), dtype=int)
    lon_idx = np.linspace(0, img.shape[1] - 1, min(max_lon, img.shape[1]), dtype=int)
    texture = img[np.ix_(lat_idx, lon_idx)]
    texture = np.flipud(texture)
    return texture


def _sphere_coordinates(n_lat: int = 160, n_lon: int = 320) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lat_vals = np.linspace(np.pi / 2, -np.pi / 2, n_lat)
    lon_vals = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    return x, y, z


def _build_texture_surface(texture: np.ndarray, *, scale: float = 1.0) -> dict:
    n_lat = max(2, int(round(texture.shape[0] * scale)))
    n_lon = max(2, int(round(texture.shape[1] * scale)))
    x, y, z = _sphere_coordinates(n_lat, n_lon)

    palette, idx_map = _quantize_texture(texture)
    if scale != 1.0:
        idx_map = _resize_index_map(idx_map, n_lat, n_lon)

    if palette.shape[0] <= 1:
        rgb = palette[0] if palette.shape[0] else np.array([90, 90, 90], dtype=np.uint8)
        colorscale = [
            [0.0, f"rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})"],
            [1.0, f"rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})"],
        ]
        cmin, cmax = 0.0, 1.0
        surfacecolor = np.zeros_like(idx_map, dtype=float)
    else:
        colorscale = [
            [i / (palette.shape[0] - 1), f"rgb({int(rgb[0])}, {int(rgb[1])}, {int(rgb[2])})"]
            for i, rgb in enumerate(palette)
        ]
        cmin, cmax = 0.0, float(palette.shape[0] - 1)
        surfacecolor = idx_map.astype(float)

    return dict(
        x=x,
        y=y,
        z=z,
        surfacecolor=surfacecolor,
        colorscale=colorscale,
        cmin=cmin,
        cmax=cmax,
    )


def _quantize_texture(texture: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    try:
        from PIL import Image
    except ImportError:
        palette, inverse = np.unique(texture.reshape(-1, 3), axis=0, return_inverse=True)
        palette = palette[:256]
        idx_map = inverse.reshape(texture.shape[0], texture.shape[1])
        idx_map = np.clip(idx_map, 0, palette.shape[0] - 1)
        return palette, idx_map

    img = Image.fromarray(texture)
    pal_img = img.convert("P", palette=Image.ADAPTIVE, colors=256)
    palette = np.array(pal_img.getpalette(), dtype=np.uint8).reshape(-1, 3)[:256]
    idx_map = np.array(pal_img, dtype=np.uint8)
    return palette, idx_map


def _resize_index_map(idx_map: np.ndarray, n_lat: int, n_lon: int) -> np.ndarray:
    from PIL import Image

    img = Image.fromarray(idx_map)
    resized = img.resize((n_lon, n_lat), resample=Image.NEAREST)
    return np.array(resized, dtype=np.uint8)


def _camera_positions(
    n_frames: int,
    radius: float = 2.5,
    elevation_deg: float = 25.0,
) -> list[dict]:
    if n_frames <= 0:
        return []
    elevation_rad = np.deg2rad(elevation_deg)
    cos_el = np.cos(elevation_rad)
    sin_el = np.sin(elevation_rad)
    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)
    cameras: list[dict] = []
    for theta in angles:
        x = radius * cos_el * np.cos(theta)
        y = radius * cos_el * np.sin(theta)
        z = radius * sin_el
        cameras.append(
            dict(
                eye=dict(x=float(x), y=float(y), z=float(z)),
                center=dict(x=0.0, y=0.0, z=0.0),
                up=dict(x=0.0, y=0.0, z=1.0),
            )
        )
    return cameras


def _add_animation(fig: go.Figure, cameras: list[dict], duration_s: float) -> None:
    if not cameras:
        return

    duration_ms = max(1, int(1000 * duration_s / len(cameras)))
    frames = [
        go.Frame(name=f"cam_{i}", layout=dict(scene=dict(camera=cam)))
        for i, cam in enumerate(cameras)
    ]
    fig.frames = frames
    fig.update_layout(scene=dict(camera=cameras[0]))

    play_args = [
        None,
        {
            "frame": {"duration": duration_ms, "redraw": False},
            "transition": {"duration": 0},
            "fromcurrent": True,
            "mode": "immediate",
        },
    ]

    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="▶ Play", method="animate", args=play_args),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}],
                    ),
                ],
                pad=dict(r=10, t=10),
                x=0.0,
                y=0.0,
            )
        ]
    )


def _render_mp4(
    fig: go.Figure,
    cameras: list[dict],
    output_path: Path,
    *,
    fps: int,
    width: int,
    height: int,
) -> None:
    if not cameras:
        print("No animation frames available; skipping MP4 export.")
        return
    try:
        import imageio.v3 as iio  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"imageio unavailable ({exc}); skipping MP4 export.")
        return

    try:
        fig_mp4 = go.Figure(fig)
        frames: list[np.ndarray] = []
        for cam in cameras:
            fig_mp4.update_layout(scene=dict(camera=cam))
            img_bytes = pio.to_image(
                fig_mp4,
                format="png",
                width=width,
                height=height,
                engine="kaleido",
            )
            frame = iio.imread(img_bytes, extension=".png")
            frames.append(frame)
        iio.imwrite(str(output_path), frames, fps=fps)
        print(f"Saved MP4 animation to {output_path}")
    except Exception as exc:  # pragma: no cover - rendering guard
        print(f"Failed to create MP4 ({exc}).")


def _build_figure(
    *,
    lat: np.ndarray,
    lon: np.ndarray,
    pot: np.ndarray,
    colorscale: str,
    cmin: float | None,
    cmax: float | None,
    title: str,
    include_texture: bool,
    texture_scale: float,
) -> tuple[go.Figure, bool]:
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    xs = cos_lat * np.cos(lon_rad)
    ys = cos_lat * np.sin(lon_rad)
    zs = np.sin(lat_rad)

    radial_offset = 1.01
    xs_off = xs * radial_offset
    ys_off = ys * radial_offset
    zs_off = zs * radial_offset

    fig = go.Figure()

    texture_payload: dict | None = None
    texture_payload: dict | None = None
    if include_texture:
        texture = _load_moon_texture()
        if texture is not None:
            texture_payload = _build_texture_surface(texture, scale=texture_scale)
        else:
            logging.debug("Texture unavailable; proceeding with a neutral sphere.")

    has_texture = texture_payload is not None and include_texture

    if not has_texture:
        x, y, z = _sphere_coordinates()
        texture_payload = dict(
            x=x,
            y=y,
            z=z,
            surfacecolor=np.zeros_like(x),
            colorscale=[[0.0, "rgb(70, 70, 70)"], [1.0, "rgb(110, 110, 110)"]],
            cmin=0.0,
            cmax=1.0,
        )

    fig.add_surface(
        x=texture_payload["x"],
        y=texture_payload["y"],
        z=texture_payload["z"],
        surfacecolor=texture_payload["surfacecolor"],
        colorscale=texture_payload["colorscale"],
        cmin=texture_payload.get("cmin"),
        cmax=texture_payload.get("cmax"),
        showscale=False,
        hoverinfo="skip",
        opacity=1.0 if has_texture else 0.8,
    )

    marker_opacity = 0.6 if include_texture and has_texture else 0.9

    scatter = go.Scatter3d(
        x=xs_off,
        y=ys_off,
        z=zs_off,
        mode="markers",
        marker=dict(
            size=3,
            color=pot,
            colorscale=colorscale,
            cmin=cmin,
            cmax=cmax,
            opacity=marker_opacity,
            colorbar=dict(title="Φ_surface (V)"),
        ),
        text=[
            f"lat={lat_i:.2f}°<br>lon={lon_i:.2f}°<br>Φ={pot_i:.2f} V"
            for lat_i, lon_i, pot_i in zip(lat, lon, pot, strict=True)
        ],
        hovertemplate="%{text}<extra></extra>",
    )
    fig.add_trace(scatter)

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            aspectmode="data",
        ),
    )

    return fig, has_texture


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive Plotly sphere of cached surface potentials.",
    )
    parser.add_argument("--start", required=True, type=_parse_iso_date, help="Start date (YYYY-MM-DD, inclusive)")
    parser.add_argument("--end", required=True, type=_parse_iso_date, help="End date (YYYY-MM-DD, inclusive)")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Root directory with potential_cache NPZ files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Down-sample to at most this many rows before plotting",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed when --sample is used")
    parser.add_argument(
        "--colorscale",
        default="Viridis",
        help="Plotly colorscale for potentials (e.g., Viridis, Plasma)",
    )
    parser.add_argument("--min-pot", type=float, default=None, help="Lower clip for potentials (V)")
    parser.add_argument("--max-pot", type=float, default=None, help="Upper clip for potentials (V)")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("potential_map.html"),
        help="Destination HTML file",
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "inline"],
        default="cdn",
        help="How to embed Plotly JS in the HTML output",
    )
    parser.add_argument(
        "--no-texture",
        action="store_true",
        help="Skip loading the moon texture background",
    )
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Open the resulting HTML in a browser",
    )
    parser.add_argument(
        "--animate",
        action="store_true",
        help="Add orbit animation controls and (optionally) export an MP4",
    )
    parser.add_argument(
        "--animation-frames",
        type=int,
        default=120,
        help="Number of frames for a full orbit when --animate is set",
    )
    parser.add_argument(
        "--animation-duration",
        type=float,
        default=10.0,
        help="Seconds per full orbit in the HTML animation",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=24,
        help="Frames per second for MP4 export (requires --mp4-output)",
    )
    parser.add_argument(
        "--mp4-output",
        type=Path,
        default=None,
        help="Optional path to save an MP4 orbit (requires kaleido + imageio)",
    )
    parser.add_argument(
        "--animation-width",
        type=int,
        default=1280,
        help="Frame width in pixels for MP4 export",
    )
    parser.add_argument(
        "--animation-height",
        type=int,
        default=960,
        help="Frame height in pixels for MP4 export",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

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

    cmin = args.min_pot if args.min_pot is not None else None
    cmax = args.max_pot if args.max_pot is not None else None

    title = (
        f"Φ_surface {str(args.start)} → {str(args.end)}"
        f" | rows={pot.size:n}"
    )
    texture_scale = 1.0
    if args.animate or args.mp4_output is not None:
        if args.animation_frames > 60:
            texture_scale = 0.75
    fig, _has_texture = _build_figure(
        lat=lat,
        lon=lon,
        pot=pot,
        colorscale=args.colorscale,
        cmin=cmin,
        cmax=cmax,
        title=title,
        include_texture=not args.no_texture,
        texture_scale=texture_scale,
    )

    cameras: list[dict] = []
    if args.animate or args.mp4_output is not None:
        cameras = _camera_positions(
            max(1, args.animation_frames),
            radius=2.5,
            elevation_deg=25.0,
        )
        if args.animate:
            _add_animation(fig, cameras, max(args.animation_duration, 0.1))

    if args.mp4_output is not None:
        output_path = args.mp4_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        _render_mp4(
            fig,
            cameras,
            output_path,
            fps=max(1, args.animation_fps),
            width=max(100, args.animation_width),
            height=max(100, args.animation_height),
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        str(args.output),
        include_plotlyjs=args.include_plotlyjs,
        auto_open=args.auto_open,
        auto_play=args.animate,
        full_html=True,
    )
    print(f"Saved interactive map to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
