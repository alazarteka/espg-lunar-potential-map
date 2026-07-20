"""Shared NPZ cache discovery/loading and SPICE sun-geometry helpers."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, TypedDict

import numpy as np
import spiceypy as spice

from src.potential_mapper.spice import load_spice_files
from src.utils.spice_ops import get_sun_vector_wrt_moon

if TYPE_CHECKING:
    from pathlib import Path

DEFAULT_U_IDENTIFIABLE_KEY = "rows_u_is_identifiable_lhs_dchi2red_0p001"

_SPICE_LOADED = False

# Canonical NPZ field names → short keys used by loaders
_FIELD_MAP: dict[str, str] = {
    "utc": "rows_utc",
    "lat": "rows_projection_latitude",
    "lon": "rows_projection_longitude",
    "potential": "rows_projected_potential",
    "spacecraft_potential": "rows_spacecraft_potential",
    "projection_in_sun": "rows_projection_in_sun",
}


class SunGeometry(TypedDict):
    utc: str
    et: float
    sun_unit: np.ndarray
    subsolar_lat: float
    subsolar_lon: float


def discover_npz(cache_dir: Path) -> list[Path]:
    """Return all NPZ cache files under cache_dir, sorted."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")
    return sorted(p for p in cache_dir.rglob("*.npz") if p.is_file())


def ensure_spice_loaded() -> None:
    """Load SPICE kernels once per process."""
    global _SPICE_LOADED
    if not _SPICE_LOADED:
        load_spice_files()
        _SPICE_LOADED = True


def datetime64_midpoint(times: np.ndarray) -> np.datetime64:
    """Return midpoint between earliest and latest timestamps."""
    t_min = times.min()
    t_max = times.max()
    return t_min + (t_max - t_min) // 2


def utc_string_ms(dt64: np.datetime64) -> str:
    """Render datetime64 with millisecond precision for SPICE."""
    return np.datetime_as_string(dt64.astype("datetime64[ms]"), unit="ms")


def sun_geometry_at(mid_time: np.datetime64) -> SunGeometry:
    """Compute Moon→Sun unit vector and subsolar lat/lon at mid_time."""
    ensure_spice_loaded()
    utc_str = utc_string_ms(mid_time)
    try:
        et = spice.utc2et(utc_str)
    except Exception as exc:
        raise RuntimeError(f"Failed to convert UTC {utc_str} to ET") from exc

    sun_vec_raw = get_sun_vector_wrt_moon(et)
    if sun_vec_raw is None:
        raise RuntimeError("SPICE returned no Moon→Sun vector.")

    sun_vec = np.asarray(sun_vec_raw, dtype=np.float64)
    norm = np.linalg.norm(sun_vec)
    if norm == 0.0:
        raise RuntimeError("Sun vector magnitude is zero; cannot derive geometry.")
    sun_unit = sun_vec / norm

    return {
        "utc": utc_str,
        "et": float(et),
        "sun_unit": sun_unit,
        "subsolar_lat": float(np.degrees(np.arcsin(sun_unit[2]))),
        "subsolar_lon": float(np.degrees(np.arctan2(sun_unit[1], sun_unit[0]))),
    }


def sza_from_latlon_utc(
    lat_deg: np.ndarray, lon_deg: np.ndarray, utc: np.ndarray
) -> np.ndarray:
    """Vectorized solar zenith angle (deg) for projection footprints."""
    ensure_spice_loaded()
    utc_strings = np.asarray(utc).astype(str)
    et = np.array([spice.utc2et(u) for u in utc_strings], dtype=float)
    sun_vecs = np.vstack([get_sun_vector_wrt_moon(e) for e in et])
    norms = np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    sun_unit = sun_vecs / np.where(norms == 0, np.nan, norms)

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    surface_normals = np.column_stack(
        [
            np.cos(lat_rad) * np.cos(lon_rad),
            np.cos(lat_rad) * np.sin(lon_rad),
            np.sin(lat_rad),
        ]
    )
    dots = np.clip(np.sum(surface_normals * sun_unit, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def sample_indices(n: int, sample: int | None, seed: int | None) -> np.ndarray | None:
    """Return subsample indices, or None if no downsampling is needed."""
    if sample is None or sample <= 0 or n <= sample:
        return None
    rng = np.random.default_rng(seed)
    return rng.choice(n, size=sample, replace=False)


def load_rows_in_window(
    files: list[Path],
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
    fields: tuple[str, ...] = (
        "utc",
        "lat",
        "lon",
        "potential",
        "spacecraft_potential",
        "projection_in_sun",
    ),
) -> dict[str, np.ndarray]:
    """
    Load selected NPZ row fields inside ``[start_ts, end_ts_exclusive)``.

    ``fields`` entries are short names from ``_FIELD_MAP``. Always includes
    parsed ``utc`` as datetime64[ns]. Missing optional ``projection_in_sun``
    is filled with False.
    """
    unknown = set(fields) - set(_FIELD_MAP)
    if unknown:
        raise ValueError(f"Unknown fields: {sorted(unknown)}")

    start_str = str(start_ts.astype("datetime64[s]"))
    end_str = str(end_ts_exclusive.astype("datetime64[s]"))
    parts: dict[str, list[np.ndarray]] = {name: [] for name in fields}

    for path in files:
        with np.load(path) as data:
            utc = data["rows_utc"]
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

            for name in fields:
                npz_key = _FIELD_MAP[name]
                if name == "utc":
                    parts[name].append(utc_vals)
                    continue
                if name == "projection_in_sun":
                    raw = data.get(npz_key)
                    if raw is None:
                        parts[name].append(np.zeros(mask.sum(), dtype=bool))
                    else:
                        parts[name].append(raw[mask].astype(bool))
                    continue
                arr = data[npz_key].astype(np.float64)
                parts[name].append(arr[mask])

    result: dict[str, np.ndarray] = {}
    for name in fields:
        if not parts[name]:
            if name == "utc":
                result[name] = np.array([], dtype="datetime64[ns]")
            elif name == "projection_in_sun":
                result[name] = np.array([], dtype=bool)
            else:
                result[name] = np.array([])
        else:
            result[name] = np.concatenate(parts[name])
    return result


def load_projection_rows(
    files: list[Path],
    start_ts: np.datetime64,
    end_ts_exclusive: np.datetime64,
    *,
    require_u_identifiable: bool = False,
    u_identifiable_key: str = DEFAULT_U_IDENTIFIABLE_KEY,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load utc/lat/lon/potential for temporal reconstruction.

    When ``require_u_identifiable`` is True, rows lacking the QC flag key are
    skipped and non-identifiable potentials are filtered out.
    """
    start_str = str(start_ts.astype("datetime64[s]"))
    end_str = str(end_ts_exclusive.astype("datetime64[s]"))

    utc_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    lon_parts: list[np.ndarray] = []
    pot_parts: list[np.ndarray] = []

    for path in files:
        with np.load(path) as data:
            utc = data["rows_utc"]
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
                logging.debug("Failed to parse UTC in %s", path)
                continue

            lat = data["rows_projection_latitude"].astype(np.float64)
            lon = data["rows_projection_longitude"].astype(np.float64)
            pot = data["rows_projected_potential"].astype(np.float64)

            if require_u_identifiable:
                try:
                    u_identifiable = data[u_identifiable_key].astype(bool)
                except KeyError:
                    logging.warning(
                        "Missing %s in %s; skipping (require_u_identifiable=True)",
                        u_identifiable_key,
                        path,
                    )
                    continue
            else:
                u_identifiable = np.ones_like(pot, dtype=bool)

            finite_mask = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(pot)
            mask_refined = mask & finite_mask & u_identifiable
            if not np.any(mask_refined):
                continue

            utc_parts.append(utc_vals[(finite_mask & u_identifiable)[mask]])
            lat_parts.append(lat[mask_refined])
            lon_parts.append(lon[mask_refined])
            pot_parts.append(pot[mask_refined])

    if not utc_parts:
        return (
            np.array([], dtype="datetime64[ns]"),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    return (
        np.concatenate(utc_parts),
        np.concatenate(lat_parts),
        np.concatenate(lon_parts),
        np.concatenate(pot_parts),
    )
