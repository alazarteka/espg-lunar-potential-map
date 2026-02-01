#!/usr/bin/env python3
"""
Generate per-day time-series PNGs from monthly potential cache NPZ files.

This script reads the *monthly* batch outputs produced by:

  uv run python -m src.potential_mapper.batch --fast --year YYYY --month MM ...

and writes one PNG per UTC day under:

  artifacts/plots/timeseries/daily/YYYY/MM/DD.png

The plot uses spectrum-level fits (spec_u_surface) to keep files small and fast
to render. When the LHS identifiability QC flag is present
(`spec_u_is_identifiable_lhs_dchi2red_0p001`), the plot includes **only**
identifiable spectra.

Examples
--------
Generate daily plots for all cached months:

  uv run python scripts/analysis/potential_timeseries_daily_png.py

Generate daily plots only for 1998:

  uv run python scripts/analysis/potential_timeseries_daily_png.py --years 1998

Overwrite existing PNGs:

  uv run python scripts/analysis/potential_timeseries_daily_png.py --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm


def _discover_monthly_npzs(cache_root: Path) -> list[Path]:
    if not cache_root.exists():
        raise FileNotFoundError(f"Cache root does not exist: {cache_root}")

    paths: list[Path] = []
    for month_dir in sorted(cache_root.glob("*_u0_lillis")):
        parts = month_dir.name.split("_")
        # Expect: YYYY_MM_u0_lillis (4 parts)
        if len(parts) != 4:
            continue
        year = parts[0]
        month = parts[1]
        if not (year.isdigit() and month.isdigit()):
            continue
        npz = month_dir / f"potential_batch_{int(year):04d}_{int(month):02d}.npz"
        if npz.exists():
            paths.append(npz)
    return paths


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render daily potential time-series PNGs from monthly cache NPZs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("artifacts/potential_cache"),
        help="Root directory containing monthly *_u0_lillis caches.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts/plots/timeseries/daily"),
        help="Root directory for per-day PNG outputs.",
    )
    parser.add_argument(
        "--years",
        type=str,
        default="1998,1999",
        help="Comma-separated list of years to include (e.g., '1998,1999').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs.",
    )
    parser.add_argument(
        "--min-fit",
        type=int,
        default=1,
        help="Skip days with fewer than this many finite fitted spectra.",
    )
    parser.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="If >0, limit to at most this many days (for quick testing).",
    )
    return parser.parse_args()

def _write_day_plot(
    *,
    out_png: Path,
    date_str: str,
    hour: np.ndarray,
    u_surface: np.ndarray,
    n_fit_total: int,
    n_ident_fit: int,
    n_specs_total: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(9.5, 3.5), constrained_layout=True)

    if hour.size and u_surface.size:
        ax.scatter(hour, u_surface, s=6, alpha=0.7, linewidths=0)

    if n_fit_total > 0:
        frac = float(n_ident_fit) / float(n_fit_total)
        ax.set_title(
            f"{date_str}  (ident_fit={n_ident_fit}/{n_fit_total}, specs={n_specs_total}, frac={frac:.2f})"
        )
    else:
        ax.set_title(f"{date_str}  (no fits)")

    ax.set_xlim(0, 24)
    ax.set_xlabel("UTC hour")
    ax.set_ylabel("U_surface [V]")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.2)
    ax.grid(True, alpha=0.2)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> int:
    args = _parse_args()

    years = {int(y.strip()) for y in str(args.years).split(",") if y.strip()}

    monthly_npzs = _discover_monthly_npzs(Path(args.cache_root))
    if not monthly_npzs:
        raise FileNotFoundError(
            f"No monthly NPZ files found under {args.cache_root} (expected *_u0_lillis/)"
        )

    days_written = 0
    max_days = int(args.max_days) if int(args.max_days) > 0 else None

    for npz_path in tqdm(monthly_npzs, desc="Months", unit="npz"):
        # Filter by year from filename (potential_batch_YYYY_MM.npz).
        parts = npz_path.stem.split("_")
        if len(parts) >= 4 and parts[2].isdigit():
            year = int(parts[2])
            if year not in years:
                continue

        data = np.load(npz_path, allow_pickle=False)
        ts_str = np.char.strip(data["spec_time_start"].astype("U64", copy=False))
        valid_time = ts_str != ""
        if not np.any(valid_time):
            continue

        sec_all = ts_str[valid_time].astype(np.float64).astype(np.int64)
        dt_all = sec_all.astype("datetime64[s]")
        day_all = dt_all.astype("datetime64[D]")

        u_all = data["spec_u_surface"].astype(np.float64, copy=False)[valid_time]
        has_fit_all = data["spec_has_fit"].astype(bool, copy=False)[valid_time]

        ident_all = None
        if "spec_u_is_identifiable_lhs_dchi2red_0p001" in data.files:
            ident_all = data["spec_u_is_identifiable_lhs_dchi2red_0p001"].astype(
                bool, copy=False
            )[valid_time]

        for day in tqdm(np.unique(day_all), desc=str(npz_path.name), leave=False):
            date_str = np.datetime_as_string(day, unit="D")
            y, m, d = date_str.split("-")
            out_png = Path(args.output_root) / y / m / f"{int(d):02d}.png"
            if out_png.exists() and not bool(args.overwrite):
                continue

            in_day = day_all == day
            if not np.any(in_day):
                continue

            u = u_all[in_day]
            has_fit = has_fit_all[in_day]
            finite_fit = has_fit & np.isfinite(u)
            n_fit = int(np.sum(finite_fit))
            n_specs = int(in_day.sum())
            if n_fit < int(args.min_fit):
                continue

            ident_fit = finite_fit
            if ident_all is not None:
                ident_fit = finite_fit & ident_all[in_day]

            n_ident = int(np.sum(ident_fit))
            sec_day = sec_all[in_day]
            day_start_s = int(day.astype("datetime64[s]").astype(np.int64))

            sec_plot = sec_day[ident_fit]
            hour = (sec_plot - day_start_s) / 3600.0

            _write_day_plot(
                out_png=out_png,
                date_str=date_str,
                hour=hour,
                u_surface=u[ident_fit],
                n_fit_total=n_fit,
                n_ident_fit=n_ident,
                n_specs_total=n_specs,
            )

            days_written += 1
            if max_days is not None and days_written >= max_days:
                return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
