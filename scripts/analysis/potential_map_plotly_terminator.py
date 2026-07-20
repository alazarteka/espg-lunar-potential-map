"""Plot cached surface potentials with day/night shading from SPICE geometry."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from src.potential_mapper.cache_io import (
    datetime64_midpoint,
    discover_npz,
    load_rows_in_window,
    sample_indices,
    sun_geometry_at,
)

# Default cache directory mirrors the batch runner output
DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")



@dataclass(slots=True)
class RowBundle:
    utc: np.ndarray  # dtype datetime64[ns]
    lat: np.ndarray  # projection latitude degrees
    lon: np.ndarray  # projection longitude degrees
    potential: np.ndarray  # projected potential volts
    projection_in_sun: np.ndarray  # bool per row


def _parse_iso_date(value: str) -> np.datetime64:
    """Parse YYYY-MM-DD into a numpy datetime64 day."""
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc
    return np.datetime64(dt.date())


def _load_rows(
    files: list[Path],
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
) -> RowBundle:
    data = load_rows_in_window(
        files,
        start_ts,
        end_ts_exclusive,
        fields=("utc", "lat", "lon", "potential", "projection_in_sun"),
    )
    return RowBundle(
        utc=data["utc"],
        lat=data["lat"],
        lon=data["lon"],
        potential=data["potential"],
        projection_in_sun=data["projection_in_sun"],
    )


def _sample_rows(bundle: RowBundle, sample: int | None, seed: int | None) -> RowBundle:
    idx = sample_indices(bundle.utc.size, sample, seed)
    if idx is None:
        return bundle
    return RowBundle(
        utc=bundle.utc[idx],
        lat=bundle.lat[idx],
        lon=bundle.lon[idx],
        potential=bundle.potential[idx],
        projection_in_sun=bundle.projection_in_sun[idx],
    )


def _sphere_coordinates(
    n_lat: int = 160, n_lon: int = 320
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return x/y/z coordinates for a unit sphere mesh."""
    lat_vals = np.linspace(np.pi / 2, -np.pi / 2, n_lat)
    lon_vals = np.linspace(-np.pi, np.pi, n_lon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    return x, y, z


def _day_night_surface(sun_unit: np.ndarray, scale: float = 1.0) -> dict:
    """Build a two-color surface payload based on Sun direction."""
    base_lat, base_lon = 160, 320
    n_lat = max(2, int(round(base_lat * scale)))
    n_lon = max(2, int(round(base_lon * scale)))
    x, y, z = _sphere_coordinates(n_lat, n_lon)
    dots = sun_unit[0] * x + sun_unit[1] * y + sun_unit[2] * z
    surfacecolor = np.where(dots >= 0, 0.0, 1.0)
    colorscale = [
        [0.0, "rgb(245, 245, 245)"],
        [0.4999, "rgb(245, 245, 245)"],
        [0.5, "rgb(160, 28, 36)"],
        [1.0, "rgb(160, 28, 36)"],
    ]
    return dict(
        x=x,
        y=y,
        z=z,
        surfacecolor=surfacecolor,
        colorscale=colorscale,
        cmin=0.0,
        cmax=1.0,
    )


def _terminator_curve(
    sun_unit: np.ndarray, n_points: int = 360, radius: float = 1.002
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return xyz points along the great circle separating day/night."""
    sun_hat = sun_unit / np.linalg.norm(sun_unit)
    ref = np.array([0.0, 0.0, 1.0])
    cross = np.cross(sun_hat, ref)
    if np.linalg.norm(cross) == 0.0:
        ref = np.array([1.0, 0.0, 0.0])
        cross = np.cross(sun_hat, ref)
    u = cross / np.linalg.norm(cross)
    v = np.cross(sun_hat, u)

    angles = np.linspace(0.0, 2 * np.pi, n_points, endpoint=True)
    circle = np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    circle *= radius
    return circle[:, 0], circle[:, 1], circle[:, 2]


def _scatter_coordinates(
    lat_deg: np.ndarray, lon_deg: np.ndarray, radius: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert lat/lon in degrees to xyz at the requested radius."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat_rad)
    x = radius * cos_lat * np.cos(lon_rad)
    y = radius * cos_lat * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)
    return x, y, z


def _predict_daylit(
    lat_deg: np.ndarray, lon_deg: np.ndarray, sun_unit: np.ndarray
) -> np.ndarray:
    """Return boolean mask indicating whether surface normal faces the Sun."""
    x, y, z = _scatter_coordinates(lat_deg, lon_deg, radius=1.0)
    dots = sun_unit[0] * x + sun_unit[1] * y + sun_unit[2] * z
    return dots >= 0.0


def _build_figure(
    bundle: RowBundle,
    geometry: dict[str, float | np.ndarray | str],
    *,
    colorscale: str,
    cmin: float | None,
    cmax: float | None,
    title: str,
) -> go.Figure:
    """Compose the Plotly figure with shaded surface and potential markers."""
    sun_unit = geometry["sun_unit"]
    surface_payload = _day_night_surface(sun_unit, scale=1.0)
    fig = go.Figure()

    fig.add_surface(
        x=surface_payload["x"],
        y=surface_payload["y"],
        z=surface_payload["z"],
        surfacecolor=surface_payload["surfacecolor"],
        colorscale=surface_payload["colorscale"],
        cmin=surface_payload["cmin"],
        cmax=surface_payload["cmax"],
        showscale=False,
        hoverinfo="skip",
        opacity=0.98,
        name="Moon",
    )

    finite_mask = (
        np.isfinite(bundle.lat)
        & np.isfinite(bundle.lon)
        & np.isfinite(bundle.potential)
    )
    if np.any(finite_mask):
        xs, ys, zs = _scatter_coordinates(
            bundle.lat[finite_mask], bundle.lon[finite_mask], radius=1.01
        )
        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=zs,
                mode="markers",
                name="Φ_surface",
                marker=dict(
                    size=3,
                    color=bundle.potential[finite_mask],
                    colorscale=colorscale,
                    cmin=cmin,
                    cmax=cmax,
                    opacity=0.9,
                    line=dict(width=0),
                    colorbar=dict(title="Φ_surface (V)"),
                ),
                text=[
                    f"UTC={np.datetime_as_string(t.astype('datetime64[s]'), unit='s')}<br>"
                    f"lat={lat:.2f}° lon={lon:.2f}°<br>"
                    f"Φ={pot:.2f} V"
                    for t, lat, lon, pot in zip(
                        bundle.utc[finite_mask],
                        bundle.lat[finite_mask],
                        bundle.lon[finite_mask],
                        bundle.potential[finite_mask],
                        strict=True,
                    )
                ],
                hovertemplate="%{text}<extra></extra>",
            )
        )
    else:
        logging.warning(
            "No finite projected potentials found; only the shaded Moon is displayed."
        )

    # Add subsolar point marker
    sub_lat = float(geometry["subsolar_lat"])
    sub_lon = float(geometry["subsolar_lon"])
    sub_x, sub_y, sub_z = _scatter_coordinates(
        np.array([sub_lat]), np.array([sub_lon]), radius=1.02
    )
    fig.add_trace(
        go.Scatter3d(
            x=sub_x,
            y=sub_y,
            z=sub_z,
            mode="markers",
            name="Subsolar point",
            marker=dict(
                size=8, color="gold", symbol="circle", line=dict(color="black", width=1)
            ),
            hovertemplate=(
                f"Subsolar<br>lat={sub_lat:.2f}°<br>lon={sub_lon:.2f}°<br>"
                + f"midpoint UTC={geometry['utc']}"
                + "<extra></extra>"
            ),
        )
    )

    # Great-circle outline of the terminator
    tx, ty, tz = _terminator_curve(sun_unit)
    fig.add_trace(
        go.Scatter3d(
            x=tx,
            y=ty,
            z=tz,
            mode="lines",
            name="Terminator",
            line=dict(color="black", width=2, dash="dash"),
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=50, b=0),
        scene=dict(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            zaxis=dict(showgrid=False, zeroline=False, visible=False),
            aspectmode="data",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an interactive Plotly sphere of cached surface potentials with day/night shading.",
    )
    parser.add_argument(
        "--start",
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
        help="Root directory with potential_cache NPZ files",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Down-sample to at most this many rows before plotting",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed when --sample is used"
    )
    parser.add_argument(
        "--colorscale",
        default="Viridis",
        help="Plotly colorscale for potentials (e.g., Viridis, Plasma)",
    )
    parser.add_argument(
        "--min-pot", type=float, default=None, help="Lower clip for potentials (V)"
    )
    parser.add_argument(
        "--max-pot", type=float, default=None, help="Upper clip for potentials (V)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("potential_terminator.html"),
        help="Destination HTML file",
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "inline"],
        default="cdn",
        help="How to embed Plotly JS in the HTML output",
    )
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Open the resulting HTML in a browser",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    end_date = args.end if args.end is not None else args.start
    if end_date < args.start:
        raise SystemExit("--end must be >= --start")

    start_ts = args.start.astype("datetime64[s]")
    end_ts_exclusive = (end_date + np.timedelta64(1, "D")).astype("datetime64[s]")

    files = discover_npz(args.cache_dir)
    bundle = _load_rows(files, start_ts, end_ts_exclusive)
    if bundle.utc.size == 0:
        print("No cached rows found in the requested date range.")
        return 1

    bundle = _sample_rows(bundle, args.sample, args.seed)
    order = np.argsort(bundle.utc)
    bundle = RowBundle(
        utc=bundle.utc[order],
        lat=bundle.lat[order],
        lon=bundle.lon[order],
        potential=bundle.potential[order],
        projection_in_sun=bundle.projection_in_sun[order],
    )

    midpoint = datetime64_midpoint(bundle.utc)
    geometry = sun_geometry_at(midpoint)

    sun_unit = geometry["sun_unit"]
    valid_rows = (
        np.isfinite(bundle.lat)
        & np.isfinite(bundle.lon)
        & (bundle.projection_in_sun.size == bundle.lat.size)
    )
    if np.any(valid_rows):
        predicted = _predict_daylit(
            bundle.lat[valid_rows], bundle.lon[valid_rows], sun_unit
        )
        recorded = bundle.projection_in_sun[valid_rows]
        mismatches = int(np.count_nonzero(predicted ^ recorded))
        total = int(valid_rows.sum())
        print(
            f"Illumination agreement: {mismatches}/{total} rows differ from cached projection_in_sun"
        )

    midpoint_str = np.datetime_as_string(midpoint.astype("datetime64[s]"), unit="s")
    print(
        f"Midpoint UTC: {midpoint_str} | "
        f"Subsolar lat={geometry['subsolar_lat']:.2f}° "
        f"lon={geometry['subsolar_lon']:.2f}°"
    )

    cmin = args.min_pot if args.min_pot is not None else None
    cmax = args.max_pot if args.max_pot is not None else None
    title = (
        f"Φ_surface {str(args.start)} → {str(end_date)}"
        f" | midpoint {midpoint_str}"
        f" | rows={bundle.potential.size:n}"
    )

    fig = _build_figure(
        bundle,
        geometry,
        colorscale=args.colorscale,
        cmin=cmin,
        cmax=cmax,
        title=title,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pio.write_html(
        fig,
        str(args.output),
        include_plotlyjs=args.include_plotlyjs,
        auto_open=args.auto_open,
        full_html=True,
    )
    print(f"Saved interactive terminator map to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
