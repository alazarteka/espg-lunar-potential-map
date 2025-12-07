"""Plot time-series of cached potentials with angular distance to subsolar point."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import spiceypy as spice
from plotly.subplots import make_subplots

from src.potential_mapper.spice import load_spice_files
from src.utils.spice_ops import get_sun_vector_wrt_moon

# Default cache directory mirrors the batch runner output
DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")

# Global guard so we only load kernels once
_SPICE_LOADED = False


@dataclass(slots=True)
class TimeSeriesRows:
    utc: np.ndarray  # datetime64[ns]
    spacecraft_potential: np.ndarray  # Φ_sc per row (V)
    surface_potential: np.ndarray  # Φ_surface per row (V)
    projection_lat: np.ndarray  # degrees
    projection_lon: np.ndarray  # degrees
    projection_in_sun: np.ndarray  # bool


def _parse_iso_date(value: str) -> np.datetime64:
    """Parse YYYY-MM-DD into a numpy datetime64 day."""
    try:
        dt = datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc
    return np.datetime64(dt.date())


def _discover_npz(cache_dir: Path) -> list[Path]:
    """Return all NPZ cache files under cache_dir."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")
    return sorted(p for p in cache_dir.rglob("*.npz") if p.is_file())


def _load_rows(
    files: Iterable[Path],
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
) -> TimeSeriesRows:
    """Load rows inside the requested UTC window."""
    utc_parts: list[np.ndarray] = []
    sc_pot_parts: list[np.ndarray] = []
    surf_pot_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    proj_sun_parts: list[np.ndarray] = []

    start_str = str(start_ts.astype("datetime64[s]"))
    end_str = str(end_ts_exclusive.astype("datetime64[s]"))

    for path in files:
        with np.load(path) as data:
            utc = data["rows_utc"]
            sc_pot = data["rows_spacecraft_potential"].astype(np.float64)
            surf_pot = data["rows_projected_potential"].astype(np.float64)
            lat = data["rows_projection_latitude"].astype(np.float64)
            lon = data["rows_projection_longitude"].astype(np.float64)
            proj_in_sun = data.get("rows_projection_in_sun")

        if utc.size == 0:
            continue

        valid_time = utc != ""
        if not np.any(valid_time):
            continue

        mask = valid_time & (utc >= start_str) & (utc < end_str)
        if not np.any(mask):
            continue

        try:
            utc_vals = np.array(utc[mask], dtype="datetime64[ns]")
        except ValueError:
            logging.debug("Failed to parse UTC strings in %s; skipping", path)
            continue

        utc_parts.append(utc_vals)
        sc_pot_parts.append(sc_pot[mask])
        surf_pot_parts.append(surf_pot[mask])
        lat_parts.append(lat[mask])
        lon_parts.append(lon[mask])
        if proj_in_sun is not None:
            proj_sun_parts.append(proj_in_sun[mask].astype(bool))
        else:
            proj_sun_parts.append(np.zeros(mask.sum(), dtype=bool))

    if not utc_parts:
        empty_time = np.array([], dtype="datetime64[ns]")
        empty_float = np.array([])
        return TimeSeriesRows(
            utc=empty_time,
            spacecraft_potential=empty_float,
            surface_potential=empty_float,
            projection_lat=empty_float,
            projection_lon=empty_float,
            projection_in_sun=np.array([], dtype=bool),
        )

    return TimeSeriesRows(
        utc=np.concatenate(utc_parts),
        spacecraft_potential=np.concatenate(sc_pot_parts),
        surface_potential=np.concatenate(surf_pot_parts),
        projection_lat=np.concatenate(lat_parts),
        projection_lon=np.concatenate(lon_parts),
        projection_in_sun=np.concatenate(proj_sun_parts),
    )


def _sample_rows(
    rows: TimeSeriesRows, sample: int | None, seed: int | None
) -> TimeSeriesRows:
    """Uniformly down-sample the row bundle if needed."""
    size = rows.utc.size
    if sample is None or sample <= 0 or size <= sample:
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(size, size=sample, replace=False)
    return TimeSeriesRows(
        utc=rows.utc[idx],
        spacecraft_potential=rows.spacecraft_potential[idx],
        surface_potential=rows.surface_potential[idx],
        projection_lat=rows.projection_lat[idx],
        projection_lon=rows.projection_lon[idx],
        projection_in_sun=rows.projection_in_sun[idx],
    )


def _ensure_spice_loaded() -> None:
    """Load SPICE kernels once per process."""
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        load_spice_files()
        _SPICE_LOADED = True


def _datetime64_midpoint(times: np.ndarray) -> np.datetime64:
    """Return midpoint between earliest and latest timestamps."""
    t_min = times.min()
    t_max = times.max()
    delta = t_max - t_min
    return t_min + delta // 2


def _utc_string(dt64: np.datetime64) -> str:
    """Render datetime64 with millisecond precision for SPICE."""
    dt_ms = dt64.astype("datetime64[ms]")
    return np.datetime_as_string(dt_ms, unit="ms")


def _compute_sun_geometry(
    mid_time: np.datetime64,
) -> dict[str, float | np.ndarray | str]:
    """Compute midpoint Sun direction in IAU_MOON."""
    _ensure_spice_loaded()
    utc_str = _utc_string(mid_time)
    try:
        et = spice.utc2et(utc_str)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert UTC {utc_str} to ET") from exc

    sun_vec = get_sun_vector_wrt_moon(et)
    if sun_vec is None:
        raise RuntimeError("SPICE returned no Moon→Sun vector.")

    sun_vec = np.asarray(sun_vec, dtype=np.float64)
    norm = np.linalg.norm(sun_vec)
    if norm == 0.0:
        raise RuntimeError("Sun vector magnitude is zero; cannot derive geometry.")
    sun_unit = sun_vec / norm

    subsolar_lat = np.degrees(np.arcsin(sun_unit[2]))
    subsolar_lon = np.degrees(np.arctan2(sun_unit[1], sun_unit[0]))

    return {
        "utc": utc_str,
        "et": et,
        "sun_unit": sun_unit,
        "subsolar_lat": subsolar_lat,
        "subsolar_lon": subsolar_lon,
    }


def _angular_distance_deg(
    lat_deg: np.ndarray, lon_deg: np.ndarray, sun_unit: np.ndarray
) -> np.ndarray:
    """Great-circle distance between projection point and subsolar point."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    cos_lat = np.cos(lat_rad)
    x = cos_lat * np.cos(lon_rad)
    y = cos_lat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    dots = sun_unit[0] * x + sun_unit[1] * y + sun_unit[2] * z
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _detect_transitions(times: np.ndarray, sunlit: np.ndarray) -> list[dict]:
    """Locate indices where the sunlit flag flips state."""
    if times.size < 2:
        return []
    changes = np.flatnonzero(np.diff(sunlit.astype(int)) != 0)
    events: list[dict] = []
    for idx in changes:
        t0 = times[idx].astype("datetime64[ns]")
        t1 = times[idx + 1].astype("datetime64[ns]")
        delta = t1 - t0
        if delta <= np.timedelta64(0, "ns"):
            crossing = t1
        else:
            crossing = t0 + delta // 2
        direction = "sunlit→shadow" if sunlit[idx] else "shadow→sunlit"
        events.append(
            {
                "index": idx,
                "next_index": idx + 1,
                "time": crossing,
                "direction": direction,
            }
        )
    return events


def _build_figure(
    rows: TimeSeriesRows,
    angles_deg: np.ndarray,
    transitions: list[dict],
    colors: dict[str, str],
) -> go.Figure:
    """Create time-series figure with secondary y-axis for angular distance."""
    time_strings = np.datetime_as_string(rows.utc.astype("datetime64[s]"), unit="s")
    epoch_ms = rows.utc.astype("datetime64[ms]").astype(np.int64)
    time_values = [datetime.utcfromtimestamp(ms / 1000.0) for ms in epoch_ms]

    fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

    # Spacecraft potential
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=rows.spacecraft_potential,
            mode="lines+markers",
            name="Φ_sc (V)",
            line=dict(color=colors["spacecraft"], width=2),
            marker=dict(size=5, color=colors["spacecraft"]),
            hovertemplate="UTC=%{x}<br>Φ_sc=%{y:.2f} V<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Surface potential
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=rows.surface_potential,
            mode="lines+markers",
            name="Φ_surface (V)",
            line=dict(color=colors["surface"], width=2, dash="dot"),
            marker=dict(size=5, color=colors["surface"]),
            hovertemplate="UTC=%{x}<br>Φ_surface=%{y:.2f} V<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    # Angular distance
    fig.add_trace(
        go.Scatter(
            x=time_values,
            y=angles_deg,
            mode="lines",
            name="Angular distance (deg)",
            line=dict(color=colors["angle"], width=2),
            hovertemplate="UTC=%{x}<br>Angle=%{y:.2f}°<extra></extra>",
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    for event in transitions:
        ms = event["time"].astype("datetime64[ms]").astype(np.int64)
        x_val = datetime.utcfromtimestamp(ms / 1000.0)
        fig.add_shape(
            type="line",
            x0=x_val,
            x1=x_val,
            y0=0.0,
            y1=1.0,
            xref="x",
            yref="paper",
            line=dict(color="#666", dash="dash"),
        )
        fig.add_annotation(
            x=x_val,
            y=1.01,
            xref="x",
            yref="paper",
            text=event["direction"],
            showarrow=False,
            yanchor="bottom",
            xanchor="left",
            bgcolor="rgba(255,255,255,0.6)",
        )

    fig.update_layout(
        margin=dict(l=60, r=60, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
    )
    fig.update_yaxes(title_text="Potential (V)", secondary_y=False)
    fig.update_yaxes(title_text="Angle to subsolar (deg)", secondary_y=True)
    t_min = min(time_values)
    t_max = max(time_values)
    pad = timedelta(minutes=5)
    fig.update_xaxes(title_text="UTC", range=[t_min - pad, t_max + pad])

    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spacecraft/surface potentials and angular distance from the subsolar point.",
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
        "--output",
        type=Path,
        default=Path("potential_timeseries.html"),
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

    files = _discover_npz(args.cache_dir)
    rows = _load_rows(files, start_ts, end_ts_exclusive)
    if rows.utc.size == 0:
        print("No cached rows found in the requested date range.")
        return 1

    rows = _sample_rows(rows, args.sample, args.seed)
    order = np.argsort(rows.utc)
    rows = TimeSeriesRows(
        utc=rows.utc[order],
        spacecraft_potential=rows.spacecraft_potential[order],
        surface_potential=rows.surface_potential[order],
        projection_lat=rows.projection_lat[order],
        projection_lon=rows.projection_lon[order],
        projection_in_sun=rows.projection_in_sun[order],
    )

    midpoint = _datetime64_midpoint(rows.utc)
    geometry = _compute_sun_geometry(midpoint)
    sun_unit = geometry["sun_unit"]  # type: ignore[assignment]

    angles_deg = _angular_distance_deg(
        rows.projection_lat, rows.projection_lon, sun_unit
    )
    transitions = _detect_transitions(rows.utc, rows.projection_in_sun)

    midpoint_str = np.datetime_as_string(midpoint.astype("datetime64[s]"), unit="s")
    print(
        f"Midpoint UTC: {midpoint_str} | Subsolar lat={geometry['subsolar_lat']:.2f}° lon={geometry['subsolar_lon']:.2f}°"
    )
    if transitions:
        first = transitions[0]
        first_str = np.datetime_as_string(
            first["time"].astype("datetime64[s]"), unit="s"
        )
        print(f"First terminator crossing near {first_str} ({first['direction']}).")
    else:
        print("No terminator crossings detected in the selected window.")

    if np.any(np.isfinite(rows.surface_potential)):
        sun_mask = rows.projection_in_sun & np.isfinite(rows.surface_potential)
        shadow_mask = (~rows.projection_in_sun) & np.isfinite(rows.surface_potential)
        if sun_mask.any():
            mean_sun = float(np.nanmean(rows.surface_potential[sun_mask]))
            mean_shadow = (
                float(np.nanmean(rows.surface_potential[shadow_mask]))
                if shadow_mask.any()
                else float("nan")
            )
            print(
                f"Surface Φ mean (sunlit)={mean_sun:.2f} V | (shadow)={mean_shadow:.2f} V"
            )

    colors = {
        "spacecraft": "#2E7D32",
        "surface": "#1565C0",
        "angle": "#D84315",
    }
    fig = _build_figure(rows, angles_deg, transitions, colors)

    fig.update_layout(
        title=(
            f"Potentials vs. subsolar angle {str(args.start)} → {str(end_date)}"
            f" | rows={rows.utc.size:n}"
            f" | midpoint {midpoint_str}"
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(
        str(args.output),
        include_plotlyjs=args.include_plotlyjs,
        auto_open=args.auto_open,
        full_html=True,
    )
    print(f"Saved time-series plot to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
