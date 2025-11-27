"""Dev helper to inspect cached potential NPZ bundles.

This script intentionally mirrors the CLI of `scripts/analysis/potential_timeseries_plotly.py`
so it can be swapped into existing workflows while experimenting with new analyses.
It currently loads the same cached NPZ rows and emits a textual summary (optionally
wrapped in a minimal HTML shell when `--output` ends in `.html`). Extend the
`summarize_rows` function with any ad-hoc diagnostics you need.
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Iterable
import webbrowser

import numpy as np
from numpy.linalg import LinAlgError, lstsq, solve

try:
    # SciPy ≥1.15
    from scipy.special import sph_harm_y as _sph_harm
except ImportError:  # pragma: no cover - fallback for older SciPy
    from scipy.special import sph_harm as _sph_harm

# Default cache directory mirrors the batch runner output
DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")
POTENTIAL_COLORMAP = "viridis"


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


def _sort_rows(rows: TimeSeriesRows) -> TimeSeriesRows:
    if rows.utc.size == 0:
        return rows
    order = np.argsort(rows.utc)
    return TimeSeriesRows(
        utc=rows.utc[order],
        spacecraft_potential=rows.spacecraft_potential[order],
        surface_potential=rows.surface_potential[order],
        projection_lat=rows.projection_lat[order],
        projection_lon=rows.projection_lon[order],
        projection_in_sun=rows.projection_in_sun[order],
    )


def _nan_stats(values: np.ndarray) -> tuple[float | None, float | None, float | None]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None, None
    return (
        float(np.min(finite)),
        float(np.mean(finite)),
        float(np.max(finite)),
    )


def _format_value(value: float | None, precision: int = 2) -> str:
    if value is None or np.isnan(value):
        return "n/a"
    return f"{value:.{precision}f}"


def summarize_rows(
    rows: TimeSeriesRows, cache_dir: Path, files_scanned: int
) -> str:
    """Return a multi-line textual summary of the loaded rows."""
    if rows.utc.size == 0:
        return "No cached rows found in the requested date range."

    start = np.datetime_as_string(rows.utc.min().astype("datetime64[s]"), unit="s")
    end = np.datetime_as_string(rows.utc.max().astype("datetime64[s]"), unit="s")
    sc_stats = _nan_stats(rows.spacecraft_potential)
    surf_stats = _nan_stats(rows.surface_potential)
    lat_min = np.nanmin(rows.projection_lat) if rows.projection_lat.size else float("nan")
    lat_max = np.nanmax(rows.projection_lat) if rows.projection_lat.size else float("nan")
    lon_min = np.nanmin(rows.projection_lon) if rows.projection_lon.size else float("nan")
    lon_max = np.nanmax(rows.projection_lon) if rows.projection_lon.size else float("nan")
    sunlit_fraction = (
        float(rows.projection_in_sun.mean()) if rows.projection_in_sun.size else float("nan")
    )

    lines = [
        f"Cache directory : {cache_dir}",
        f"Files scanned   : {files_scanned}",
        f"Rows loaded     : {rows.utc.size}",
        f"UTC span        : {start} → {end}",
        (
            "Φ_sc (V)       : "
            f"min={_format_value(sc_stats[0])} "
            f"mean={_format_value(sc_stats[1])} "
            f"max={_format_value(sc_stats[2])}"
        ),
        (
            "Φ_surface (V)  : "
            f"min={_format_value(surf_stats[0])} "
            f"mean={_format_value(surf_stats[1])} "
            f"max={_format_value(surf_stats[2])}"
        ),
        (
            "Projection lat : "
            f"min={_format_value(float(lat_min))} "
            f"max={_format_value(float(lat_max))}"
        ),
        (
            "Projection lon : "
            f"min={_format_value(float(lon_min))} "
            f"max={_format_value(float(lon_max))}"
        ),
        f"Sunlit fraction : {_format_value(sunlit_fraction, precision=3)}",
    ]
    return "\n".join(lines)


def _write_output(summary: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() == ".html":
        html_text = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <title>NPZ summary</title>
  </head>
  <body>
    <pre>{escape(summary)}</pre>
  </body>
</html>
"""
        destination.write_text(html_text, encoding="utf-8")
    else:
        destination.write_text(summary, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspect cached spacecraft/surface potential NPZ files and emit a textual summary. "
            "Arguments intentionally match scripts/analysis/potential_timeseries_plotly.py for CLI parity."
        ),
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
        help="Down-sample to at most this many rows before summarizing",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="Random seed when --sample is used"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("potential_timeseries.html"),
        help=(
            "Destination output file. When the suffix is .html a small HTML wrapper is produced; "
            "otherwise raw text is written."
        ),
    )
    parser.add_argument(
        "--include-plotlyjs",
        choices=["cdn", "inline"],
        default="cdn",
        help="Kept for CLI compatibility; currently unused.",
    )
    parser.add_argument(
        "--auto-open",
        action="store_true",
        help="Open the resulting output file in a browser when it ends in .html",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    parser.add_argument(
        "--lmax",
        type=int,
        default=10,
        help="Maximum spherical harmonic degree to fit. Use a negative value to skip the fit.",
    )
    parser.add_argument(
        "--regularize-l2",
        type=float,
        default=0.0,
        help="Non-negative ridge penalty applied to spherical harmonic coefficients.",
    )
    parser.add_argument(
        "--plot-output",
        type=Path,
        default=None,
        help="Path to save a latitude/longitude heatmap of the fitted spherical harmonic (requires matplotlib).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the latitude/longitude heatmap interactively after it is generated.",
    )
    parser.add_argument(
        "--plot-lat-steps",
        type=int,
        default=181,
        help="Number of latitude samples for the plotting grid.",
    )
    parser.add_argument(
        "--plot-lon-steps",
        type=int,
        default=361,
        help="Number of longitude samples for the plotting grid.",
    )
    parser.add_argument(
        "--plot-measurements",
        action="store_true",
        help="Overlay actual NPZ measurement points on the plot.",
    )
    return parser.parse_args()


def _harmonic_coefficient_count(lmax: int) -> int:
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    return (lmax + 1) ** 2


def _valid_harmonic_mask(rows: TimeSeriesRows) -> np.ndarray:
    """Return mask selecting rows safe for harmonic fitting."""
    return (
        np.isfinite(rows.surface_potential)
        & np.isfinite(rows.projection_lat)
        & np.isfinite(rows.projection_lon)
    )


def _build_harmonic_design(lat_deg: np.ndarray, lon_deg: np.ndarray, lmax: int) -> np.ndarray:
    """Return design matrix of spherical harmonics evaluated at each location."""
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    colatitudes = (np.pi / 2.0) - lat_rad

    n_rows = lat_rad.size
    n_cols = _harmonic_coefficient_count(lmax)
    design = np.empty((n_rows, n_cols), dtype=np.complex128)
    col_idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            design[:, col_idx] = _sph_harm(m, l, lon_rad, colatitudes)
            col_idx += 1
    return design


def spherical_harmonics(
    measurements: TimeSeriesRows, lmax: int, l2_penalty: float = 0.0
) -> np.ndarray:
    """Fit spherical harmonic coefficients to the surface potential samples."""
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    if l2_penalty < 0.0:
        raise ValueError("l2_penalty must be non-negative")

    mask = _valid_harmonic_mask(measurements)
    if not np.any(mask):
        raise ValueError("No finite samples available for harmonic fit.")

    latitudes = measurements.projection_lat[mask]
    longitudes = measurements.projection_lon[mask]
    potentials = measurements.surface_potential[mask].astype(np.float64)

    design = _build_harmonic_design(latitudes, longitudes, lmax)
    n_coeffs = design.shape[1]
    if potentials.size < n_coeffs:
        raise ValueError(
            f"Need at least {n_coeffs} finite rows for lmax={lmax}, got {potentials.size}"
        )
    potentials_complex = potentials.astype(np.complex128, copy=False)
    if l2_penalty > 0.0:
        gram = design.conj().T @ design
        rhs = design.conj().T @ potentials_complex
        diag = np.diag_indices_from(gram)
        gram[diag] += l2_penalty
        coeffs = solve(gram, rhs)
    else:
        coeffs, *_ = lstsq(design, potentials_complex, rcond=None)
    return coeffs


def evaluate_spherical_harmonics(
    coeffs: np.ndarray, latitudes: np.ndarray, longitudes: np.ndarray, lmax: int
) -> np.ndarray:
    """Evaluate a fitted coefficient vector at the provided positions."""
    expected = _harmonic_coefficient_count(lmax)
    if coeffs.size != expected:
        raise ValueError(f"Expected {expected} coefficients for lmax={lmax}, got {coeffs.size}")
    design = _build_harmonic_design(latitudes, longitudes, lmax)
    return np.real(design @ coeffs)


def plot_harmonics_latlon(
    coeffs: np.ndarray,
    lmax: int,
    destination: Path | None,
    show: bool,
    lat_steps: int,
    lon_steps: int,
    measurement_lat: np.ndarray,
    measurement_lon: np.ndarray,
    measurement_potential: np.ndarray,
    show_measurements: bool,
) -> None:
    """Render spherical harmonic map on a latitude/longitude grid and optionally overlay measurements."""
    if lat_steps < 2 or lon_steps < 2:
        raise ValueError("Latitude/longitude steps must both be >= 2 for plotting.")

    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("matplotlib is required for plotting output.") from exc

    latitudes = np.linspace(-90.0, 90.0, lat_steps)
    longitudes = np.linspace(-180.0, 180.0, lon_steps)
    lon_grid, lat_grid = np.meshgrid(longitudes, latitudes)

    values = evaluate_spherical_harmonics(
        coeffs, lat_grid.ravel(), lon_grid.ravel(), lmax
    ).reshape(lat_grid.shape)

    meas_lat = meas_lon = meas_pot = None
    if show_measurements:
        finite_measurements = (
            np.isfinite(measurement_lat)
            & np.isfinite(measurement_lon)
            & np.isfinite(measurement_potential)
        )
        meas_lat = measurement_lat[finite_measurements]
        meas_lon = measurement_lon[finite_measurements]
        meas_pot = measurement_potential[finite_measurements]

    if meas_pot is not None and meas_pot.size > 0:
        vmin = float(np.min(meas_pot))
        vmax = float(np.max(meas_pot))
    else:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    norm = Normalize(vmin=vmin, vmax=vmax)

    fig, ax = plt.subplots(figsize=(11, 5))
    mesh = ax.pcolormesh(
        lon_grid,
        lat_grid,
        values,
        shading="auto",
        cmap=POTENTIAL_COLORMAP,
        norm=norm,
    )
    cbar = fig.colorbar(mesh, ax=ax)
    cbar.set_label("Φ_surface (V)")
    ax.set_xlabel("Longitude (deg)")
    ax.set_ylabel("Latitude (deg)")
    ax.set_title(f"Surface potential (lmax={lmax})")
    ax.set_xlim(longitudes.min(), longitudes.max())
    ax.set_ylim(latitudes.min(), latitudes.max())

    if show_measurements and meas_lat is not None and meas_lat.size > 0:
        ax.scatter(
            meas_lon,
            meas_lat,
            c=meas_pot,
            cmap=POTENTIAL_COLORMAP,
            norm=norm,
            s=10,
            alpha=0.8,
            edgecolors="white",
            linewidths=0.2,
            label="Measurements",
        )
        ax.legend(loc="upper right", frameon=False)

    if destination is not None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(destination, dpi=200, bbox_inches="tight")
        logging.info("Saved lat/lon heatmap to %s", destination)

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level))

    end_date = args.end if args.end is not None else args.start
    if end_date < args.start:
        raise SystemExit("--end must be >= --start")

    start_ts = args.start.astype("datetime64[s]")
    end_ts_exclusive = (end_date + np.timedelta64(1, "D")).astype("datetime64[s]")

    try:
        files = _discover_npz(args.cache_dir)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    rows = _load_rows(files, start_ts, end_ts_exclusive)
    if rows.utc.size == 0:
        print("No cached rows found in the requested date range.")
        return 1
    
    if args.lmax >= 0:
        try:
            coeffs = spherical_harmonics(
                rows, lmax=args.lmax, l2_penalty=args.regularize_l2
            )
        except (ValueError, LinAlgError) as exc:
            logging.error("Unable to compute spherical harmonics (lmax=%s): %s", args.lmax, exc)
            return 1
        if args.regularize_l2 > 0.0:
            print(
                f"\nSpherical harmonic coefficients (lmax={args.lmax}, l2={args.regularize_l2:g}):"
            )
        else:
            print(f"\nSpherical harmonic coefficients (lmax={args.lmax}):")
        for idx, coeff in enumerate(coeffs):
            real_part = f"{coeff.real:+.6f}"
            imag_part = f"{coeff.imag:+.6f}j"
            if coeff.real != 0 or coeff.imag != 0:
                print(f"C_{idx:03d}: {real_part} {imag_part}")

        if args.plot_output is not None or args.show_plot:
            try:
                plot_harmonics_latlon(
                    coeffs=coeffs,
                    lmax=args.lmax,
                    destination=args.plot_output,
                    show=args.show_plot,
                    lat_steps=args.plot_lat_steps,
                    lon_steps=args.plot_lon_steps,
                    measurement_lat=rows.projection_lat,
                    measurement_lon=rows.projection_lon,
                    measurement_potential=rows.surface_potential,
                    show_measurements=args.plot_measurements,
                )
            except (ValueError, ImportError) as exc:
                logging.error("Unable to render latitude/longitude heatmap: %s", exc)
                return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
