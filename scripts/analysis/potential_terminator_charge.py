"""Estimate surface charge density across the lunar terminator from cached NPZ rows."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable

import numpy as np
import spiceypy as spice

import src.config as config
from src.potential_mapper.spice import load_spice_files
from src.utils.spice_ops import get_sun_vector_wrt_moon
from scripts.analysis.potential_charge_report_md import _render_markdown

# Default cache directory mirrors the batch runner output
DEFAULT_CACHE_DIR = Path("data/potential_cache")

# Global guard so we only load kernels once
_SPICE_LOADED = False


@dataclass(slots=True)
class RowData:
    utc_ns: np.ndarray  # datetime64[ns]
    utc_str: np.ndarray  # dtype '<U'
    lat: np.ndarray  # projection latitude (deg)
    lon: np.ndarray  # projection longitude (deg)
    surface_potential: np.ndarray  # rows_projected_potential (V)
    spacecraft_potential: np.ndarray  # rows_spacecraft_potential (V)
    projection_in_sun: np.ndarray  # bool


def _parse_iso_date(value: str) -> np.datetime64:
    """Parse YYYY-MM-DD into a numpy datetime64 date."""
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
    path: Path,
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
) -> RowData | None:
    """Load rows from a NPZ file that fall inside the requested UTC window."""
    start_str = str(start_ts.astype("datetime64[s]"))
    end_str = str(end_ts_exclusive.astype("datetime64[s]"))

    with np.load(path) as data:
        utc = data["rows_utc"]
        sc_pot = data["rows_spacecraft_potential"].astype(np.float64)
        surface_pot = data["rows_projected_potential"].astype(np.float64)
        lat = data["rows_projection_latitude"].astype(np.float64)
        lon = data["rows_projection_longitude"].astype(np.float64)
        proj_in_sun = data.get("rows_projection_in_sun")

    if utc.size == 0:
        return None

    valid_time = utc != ""
    if not np.any(valid_time):
        return None

    mask = valid_time & (utc >= start_str) & (utc < end_str)
    if not np.any(mask):
        return None

    utc_masked = utc[mask]
    try:
        utc_ns = np.array(utc_masked, dtype="datetime64[ns]")
    except ValueError:
        logging.debug("Failed to parse UTC strings in %s; skipping file.", path)
        return None

    if proj_in_sun is None:
        proj_in_sun = np.zeros_like(mask, dtype=bool)

    return RowData(
        utc_ns=utc_ns,
        utc_str=utc_masked,
        lat=lat[mask],
        lon=lon[mask],
        surface_potential=surface_pot[mask],
        spacecraft_potential=sc_pot[mask],
        projection_in_sun=proj_in_sun[mask].astype(bool),
    )


def _sample_rows(rows: RowData, sample: int | None, seed: int | None) -> RowData:
    """Down-sample row bundle uniformly if requested."""
    size = rows.utc_ns.size
    if sample is None or sample <= 0 or size <= sample:
        return rows
    rng = np.random.default_rng(seed)
    idx = rng.choice(size, size=sample, replace=False)
    return RowData(
        utc_ns=rows.utc_ns[idx],
        utc_str=rows.utc_str[idx],
        lat=rows.lat[idx],
        lon=rows.lon[idx],
        surface_potential=rows.surface_potential[idx],
        spacecraft_potential=rows.spacecraft_potential[idx],
        projection_in_sun=rows.projection_in_sun[idx],
    )


def _ensure_spice_loaded() -> None:
    """Load SPICE kernels once per process."""
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        load_spice_files()
        _SPICE_LOADED = True


def _compute_sza(rows: RowData) -> np.ndarray:
    """Compute solar zenith angle (deg) for each projection footprint."""
    _ensure_spice_loaded()

    utc_strings = rows.utc_str.astype(str)
    et = np.array([spice.utc2et(u) for u in utc_strings], dtype=float)
    sun_vecs = np.vstack([get_sun_vector_wrt_moon(e) for e in et])
    norms = np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    sun_unit = sun_vecs / np.where(norms == 0, np.nan, norms)

    lat_rad = np.deg2rad(rows.lat)
    lon_rad = np.deg2rad(rows.lon)
    surface_normals = np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )
    dots = np.sum(surface_normals * sun_unit, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def _find_crossing(sza: np.ndarray) -> int | None:
    """Locate first shadow->sunlit crossing index based on SZA."""
    if sza.size < 2:
        return None
    shadow = sza > 90.0
    transitions = np.where(shadow[:-1] & ~shadow[1:])[0]
    if transitions.size == 0:
        return None
    return int(transitions[0])


def _robust_stats(values: np.ndarray) -> tuple[float, float, int]:
    """Median and MAD (≈1σ) for finite values."""
    finite = values[np.isfinite(values)]
    n = finite.size
    if n == 0:
        return (np.nan, np.nan, 0)
    median = float(np.median(finite))
    mad = float(1.4826 * np.median(np.abs(finite - median)))
    return median, mad, n


def _sun_shadow_masks(
    sza: np.ndarray, tol: float = 2.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean masks for sunlit and shadowed samples using angular bands."""
    sun_mask = sza <= (90.0 - tol)
    shadow_mask = sza >= (90.0 + tol)
    return sun_mask, shadow_mask


def _spacecraft_delta(
    rows: RowData, sun_mask: np.ndarray, shadow_mask: np.ndarray
) -> tuple[float, float]:
    """Robust change in spacecraft potential between sun and shadow."""
    sc = rows.spacecraft_potential
    sun_med, sun_mad, _ = _robust_stats(sc[sun_mask])
    shadow_med, shadow_mad, _ = _robust_stats(sc[shadow_mask])
    if not np.isfinite(sun_med) or not np.isfinite(shadow_med):
        return (np.nan, np.nan)
    delta = sun_med - shadow_med
    sigma = np.hypot(sun_mad, shadow_mad)
    return float(delta), float(sigma)


def _local_poly_at_90(sza: np.ndarray, phi: np.ndarray) -> tuple[float, float, float]:
    """Fit a robust quadratic to Φ(SZA) near 90°; return Φ(90), dΦ/dθ, curvature."""
    mask = np.isfinite(phi) & (sza >= 80.0) & (sza <= 100.0)
    if mask.sum() < 20:
        return (np.nan, np.nan, np.nan)

    z = sza[mask] - 90.0
    y = phi[mask]
    beta = np.zeros(3)
    for _ in range(4):
        A = np.column_stack([np.ones_like(z), z, 0.5 * z**2])
        beta, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ beta
        scale = 1.4826 * np.median(np.abs(resid)) + 1e-9
        weights = 1.0 / np.maximum(1.0, np.abs(resid) / (2.5 * scale))
        Aw = A * weights[:, None]
        yw = y * weights
        beta, *_ = np.linalg.lstsq(Aw, yw, rcond=None)
    phi90 = float(beta[0])
    dphi_dtheta = float(beta[1])
    curvature = float(beta[2])
    return phi90, dphi_dtheta, curvature


def _great_circle_km(phi1: float, lam1: float, phi2: float, lam2: float) -> float:
    """Great-circle distance on the Moon between two lat/lon pairs."""
    R = 1737.4  # km
    dphi = np.radians(phi2 - phi1)
    dlam = np.radians(lam2 - lam1)
    a = (
        np.sin(dphi / 2.0) ** 2
        + np.cos(np.radians(phi1)) * np.cos(np.radians(phi2)) * np.sin(dlam / 2.0) ** 2
    )
    return float(2.0 * R * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0))))


def _lateral_width_km(sza: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> float:
    """Estimate lateral width (SZA 88° -> 92°) in km."""
    if sza.size < 2:
        return np.nan
    try:
        idx88 = np.where(sza <= 88.0)[0]
        idx92 = np.where(sza >= 92.0)[0]
        if idx88.size == 0 or idx92.size == 0:
            return np.nan
        i0 = idx88[-1]
        i1 = idx92[0]
        if i1 <= i0:
            return np.nan
        return _great_circle_km(lat[i0], lon[i0], lat[i1], lon[i1])
    except Exception:
        return np.nan


def _monte_carlo_sigma(
    phi_sun: float,
    mad_sun: float,
    phi_shadow: float,
    mad_shadow: float,
    lam_day_range: tuple[float, float],
    lam_night_range: tuple[float, float],
    nsamp: int = 20000,
    seed: int | None = 0,
) -> dict[str, list[float]] | None:
    """Monte Carlo surface-charge quantiles; returns dict with percentile lists."""
    if not (np.isfinite(phi_sun) and np.isfinite(phi_shadow)):
        return None
    rng = np.random.default_rng(seed)
    sig_day = rng.normal(phi_sun, max(mad_sun, 1e-6), nsamp)
    sig_night = rng.normal(phi_shadow, max(mad_shadow, 1e-6), nsamp)

    lam_day = np.exp(
        rng.uniform(np.log(lam_day_range[0]), np.log(lam_day_range[1]), nsamp)
    )
    lam_night = np.exp(
        rng.uniform(np.log(lam_night_range[0]), np.log(lam_night_range[1]), nsamp)
    )

    eps0 = 8.8541878128e-12  # F/m
    sigma_day = eps0 * sig_day / lam_day
    sigma_night = eps0 * sig_night / lam_night
    delta_sigma = sigma_day - sigma_night

    def _quantiles(arr: np.ndarray) -> list[float]:
        p05, p50, p95 = np.nanpercentile(arr, [5, 50, 95])
        return [float(p05), float(p50), float(p95)]

    return {
        "sigma_day_q": _quantiles(sigma_day),
        "sigma_night_q": _quantiles(sigma_night),
        "delta_sigma_q": _quantiles(delta_sigma),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Correlation between two arrays, guarding against small samples."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    try:
        corr = np.corrcoef(x[mask], y[mask])[0, 1]
        if np.isfinite(corr):
            return float(corr)
    except Exception:
        return None
    return None


def _compute_geometry_at_midpoint(rows: RowData) -> dict[str, float | str]:
    """Compute midpoint Sun geometry for reporting."""
    midpoint = rows.utc_ns.min() + (rows.utc_ns.max() - rows.utc_ns.min()) // 2
    midpoint_str = np.datetime_as_string(midpoint.astype("datetime64[s]"), unit="s")
    _ensure_spice_loaded()
    et = spice.utc2et(midpoint_str)
    sun_vec = get_sun_vector_wrt_moon(et)
    sun_vec = np.asarray(sun_vec, dtype=np.float64)
    norm = np.linalg.norm(sun_vec)
    sun_unit = sun_vec / norm
    sub_lat = float(np.degrees(np.arcsin(sun_unit[2])))
    sub_lon = float(np.degrees(np.arctan2(sun_unit[1], sun_unit[0])))
    return {
        "midpoint_utc": midpoint_str,
        "subsolar_lat": sub_lat,
        "subsolar_lon": sub_lon,
        "sun_unit": sun_unit,
    }


def _analyze_rows(
    path: Path,
    rows: RowData,
    lam_day_range: tuple[float, float],
    lam_night_range: tuple[float, float],
    nsamp: int,
    seed: int | None,
) -> dict:
    """Compute all metrics for a row bundle."""
    report: dict[str, object] = {"file": str(path)}

    if rows.utc_ns.size < 4:
        report["error"] = "insufficient_rows"
        return report

    sza = _compute_sza(rows)
    geometry = _compute_geometry_at_midpoint(rows)
    report["subsolar"] = {
        "midpoint_utc": geometry["midpoint_utc"],
        "latitude_deg": geometry["subsolar_lat"],
        "longitude_deg": geometry["subsolar_lon"],
    }

    sun_mask, shadow_mask = _sun_shadow_masks(sza)
    phi_sun, mad_sun, count_sun = _robust_stats(rows.surface_potential[sun_mask])
    phi_shadow, mad_shadow, count_shadow = _robust_stats(
        rows.surface_potential[shadow_mask]
    )
    delta_phi = (
        float(phi_sun - phi_shadow)
        if np.isfinite(phi_sun) and np.isfinite(phi_shadow)
        else np.nan
    )
    delta_phi_sigma = (
        float(np.hypot(mad_sun, mad_shadow))
        if np.isfinite(mad_sun) and np.isfinite(mad_shadow)
        else np.nan
    )

    report["phi_sunlit"] = {
        "median_V": phi_sun if np.isfinite(phi_sun) else None,
        "mad_V": mad_sun if np.isfinite(mad_sun) else None,
        "count": count_sun,
    }
    report["phi_shadow"] = {
        "median_V": phi_shadow if np.isfinite(phi_shadow) else None,
        "mad_V": mad_shadow if np.isfinite(mad_shadow) else None,
        "count": count_shadow,
    }
    report["delta_phi"] = {
        "median_V": delta_phi if np.isfinite(delta_phi) else None,
        "mad_V": delta_phi_sigma if np.isfinite(delta_phi_sigma) else None,
    }

    delta_phi_sc, delta_phi_sc_sigma = _spacecraft_delta(rows, sun_mask, shadow_mask)
    report["delta_phi_spacecraft"] = {
        "median_V": delta_phi_sc if np.isfinite(delta_phi_sc) else None,
        "mad_V": delta_phi_sc_sigma if np.isfinite(delta_phi_sc_sigma) else None,
    }

    phi90, dphi_dtheta, curvature = _local_poly_at_90(sza, rows.surface_potential)
    report["phi_at_terminator"] = {
        "phi90_V": phi90 if np.isfinite(phi90) else None,
        "dphi_dtheta_V_per_deg": dphi_dtheta if np.isfinite(dphi_dtheta) else None,
        "curvature_V_per_deg2": curvature if np.isfinite(curvature) else None,
    }

    width_km = _lateral_width_km(sza, rows.lat, rows.lon)
    report["lateral_width_km"] = width_km if np.isfinite(width_km) else None

    mc = _monte_carlo_sigma(
        phi_sun,
        mad_sun,
        phi_shadow,
        mad_shadow,
        lam_day_range,
        lam_night_range,
        nsamp=nsamp,
        seed=seed,
    )
    if mc is not None:
        report["sigma_day_C_m2"] = mc["sigma_day_q"]
        report["sigma_night_C_m2"] = mc["sigma_night_q"]
        report["delta_sigma_C_m2"] = mc["delta_sigma_q"]
    else:
        report["sigma_day_C_m2"] = None
        report["sigma_night_C_m2"] = None
        report["delta_sigma_C_m2"] = None

    crossing_idx = _find_crossing(sza)
    if crossing_idx is not None:
        crossing_time = rows.utc_ns[crossing_idx]
        report["terminator_crossing_utc"] = np.datetime_as_string(
            crossing_time.astype("datetime64[s]"), unit="s"
        )
    else:
        report["terminator_crossing_utc"] = None

    agreement = np.mean(rows.projection_in_sun == (sza < 90.0))
    corr_surface_sc = _safe_corr(
        rows.surface_potential[sun_mask | shadow_mask],
        rows.spacecraft_potential[sun_mask | shadow_mask],
    )

    quality_notes: list[str] = []
    if count_sun < 10:
        quality_notes.append("few_sunlit_samples")
    if count_shadow < 10:
        quality_notes.append("few_shadow_samples")
    if crossing_idx is None:
        quality_notes.append("no_shadow_to_sunlit_crossing")
    if agreement < 0.9:
        quality_notes.append("illumination_flag_mismatch")
    if mc is None:
        quality_notes.append("sigma_estimate_unavailable")

    report["quality"] = {
        "flag_agreement": float(agreement),
        "n_sunlit": int(count_sun),
        "n_shadow": int(count_shadow),
        "sunlit_sza_range_deg": [
            float(np.nanmin(sza[sun_mask])) if np.any(sun_mask) else None,
            float(np.nanmax(sza[sun_mask])) if np.any(sun_mask) else None,
        ],
        "shadow_sza_range_deg": [
            float(np.nanmin(sza[shadow_mask])) if np.any(shadow_mask) else None,
            float(np.nanmax(sza[shadow_mask])) if np.any(shadow_mask) else None,
        ],
        "correlation_surface_vs_spacecraft": corr_surface_sc,
        "notes": quality_notes,
    }

    report["rows_analyzed"] = int(rows.utc_ns.size)
    report["analysis_window_seconds"] = int(
        (rows.utc_ns.max() - rows.utc_ns.min()) / np.timedelta64(1, "s")
    )

    return report


def _parse_range(arg: str) -> tuple[float, float]:
    """Parse comma-separated numeric range."""
    try:
        lo_str, hi_str = arg.split(",", 1)
        lo = float(lo_str.strip())
        hi = float(hi_str.strip())
    except Exception as exc:
        raise argparse.ArgumentTypeError("Expected range in form lo,hi") from exc
    if not (lo > 0 and hi > lo):
        raise argparse.ArgumentTypeError("Range must satisfy 0 < lo < hi")
    return (lo, hi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate surface charge density across the lunar terminator using cached NPZ rows.",
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
        "--files",
        nargs="*",
        default=None,
        help="Optional explicit NPZ file paths. When provided, discovery is skipped.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Down-sample to at most this many rows per file before analysis",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed when --sample is used or for Monte Carlo",
    )
    parser.add_argument(
        "--lambda-day",
        default="0.5,2.0",
        help="Effective sheath thickness range (m) for sunlit samples (comma separated)",
    )
    parser.add_argument(
        "--lambda-night",
        default="10.0,50.0",
        help="Effective sheath thickness range (m) for shadowed samples (comma separated)",
    )
    parser.add_argument(
        "--mc-samples",
        type=int,
        default=20000,
        help="Monte Carlo samples for sigma estimation",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("terminator_charge_report.json"),
        help="Destination JSON file",
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=None,
        help="Optional Markdown output path. Defaults to JSON path with .md",
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

    lam_day_range = _parse_range(args.lambda_day)
    lam_night_range = _parse_range(args.lambda_night)

    start_ts = args.start.astype("datetime64[s]")
    end_ts_exclusive = (end_date + np.timedelta64(1, "D")).astype("datetime64[s]")

    if args.files:
        files = [Path(f) for f in args.files]
    else:
        files = _discover_npz(args.cache_dir)

    reports: list[dict] = []
    for path in files:
        try:
            rows = _load_rows(path, start_ts, end_ts_exclusive)
            if rows is None:
                continue
            if args.sample is not None:
                rows = _sample_rows(rows, args.sample, args.seed)
            report = _analyze_rows(
                path,
                rows,
                lam_day_range=lam_day_range,
                lam_night_range=lam_night_range,
                nsamp=max(int(args.mc_samples), 1000),
                seed=args.seed,
            )
            reports.append(report)
        except Exception as exc:
            logging.exception("Failed to analyse %s: %s", path, exc)
            reports.append({"file": str(path), "error": str(exc)})

    if not reports:
        logging.warning("No NPZ rows matched the requested filters; nothing to report.")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        json.dump(reports, fh, indent=2)
    logging.info("Wrote %d record(s) to %s", len(reports), args.output)

    md_path = args.markdown
    if md_path is None:
        md_path = args.output.with_suffix(".md")
    try:
        md_content = _render_markdown(reports)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(md_content, encoding="utf-8")
        logging.info("Wrote Markdown summary to %s", md_path)
    except Exception as exc:
        logging.warning("Failed to render Markdown summary: %s", exc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
