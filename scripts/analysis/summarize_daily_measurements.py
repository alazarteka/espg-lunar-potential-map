"""
Summarize cached surface potential measurements over a date range.

The script mirrors ``plot_daily_measurements.py`` in how it locates cached
measurement bundles but instead of rendering figures it reports aggregate
statistics for sunlit and shaded regions separately.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np

DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")

TEXT_METRIC_HEADERS = (
    ("count", "Count"),
    ("minimum", "Min"),
    ("percentile_5", '5th pct ("95% min")'),
    ("median", "Median"),
    ("mean", "Mean"),
    ("percentile_95", "95th pct"),
    ("maximum", "Max"),
)


def _parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a Python date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:  # pragma: no cover - argparse emits message
        raise argparse.ArgumentTypeError(
            "Dates must be provided as YYYY-MM-DD"
        ) from exc


def _date_range(start_day: date, end_day: date) -> list[date]:
    """Inclusive list of days between start_day and end_day."""
    if end_day < start_day:
        raise SystemExit("--end must be >= --start")
    span = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(span + 1)]


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

    potentials: np.ndarray
    in_sun: np.ndarray


def _load_measurements(path: Path) -> MeasurementPoints:
    """Load projected surface measurements from cache."""
    with np.load(path) as data:
        potentials = data["rows_projected_potential"].astype(np.float64)
        in_sun = data.get("rows_projection_in_sun")
        if in_sun is None:
            in_sun = np.zeros_like(potentials, dtype=bool)
        else:
            in_sun = in_sun.astype(bool)

    mask = np.isfinite(potentials)
    if not np.any(mask):
        raise RuntimeError(f"{path} contains no projected measurement footprints")

    return MeasurementPoints(potentials=potentials[mask], in_sun=in_sun[mask])


def _concatenate(values: Iterable[np.ndarray]) -> np.ndarray:
    """Join value arrays, skipping empties."""
    arrays = [array for array in values if array.size]
    if not arrays:
        return np.empty(0, dtype=np.float64)
    return np.concatenate(arrays).astype(np.float64, copy=False)


def _compute_region_stats(values: np.ndarray) -> dict[str, float]:
    """Compute summary statistics for a set of measurements."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No valid measurements found for requested range")
    pct5, median, pct95 = np.percentile(finite, [5.0, 50.0, 95.0])
    return {
        "count": int(finite.size),
        "minimum": float(np.min(finite)),
        "percentile_5": float(pct5),
        "median": float(median),
        "mean": float(np.mean(finite)),
        "percentile_95": float(pct95),
        "maximum": float(np.max(finite)),
    }


def _load_range(
    days: Sequence[date], cache_dir: Path
) -> tuple[np.ndarray, np.ndarray, list[date]]:
    """Aggregate sunlit and shaded potentials across a date sequence."""
    sunlit = []
    shaded = []
    missing: list[date] = []
    for day in days:
        try:
            path = _find_daily_file(cache_dir, day)
        except FileNotFoundError:
            missing.append(day)
            continue
        points = _load_measurements(path)
        sunlit.append(points.potentials[points.in_sun])
        shaded.append(points.potentials[~points.in_sun])
    return _concatenate(sunlit), _concatenate(shaded), missing


def _render_text(results: dict[str, dict[str, float]]) -> str:
    """Render results table for console output."""
    lines = []
    header = ["Region"] + [label for _, label in TEXT_METRIC_HEADERS]
    widths = [max(len(column), 6) for column in header]
    rows = []
    for region, stats in results.items():
        row = [region]
        for key, _ in TEXT_METRIC_HEADERS:
            value = stats[key]
            if key == "count":
                row.append(f"{value:d}")
            else:
                row.append(f"{value: .3f}")
        rows.append(row)
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))

    def _format(row: list[str]) -> str:
        padded = [
            cell.rjust(widths[idx]) if idx else cell.ljust(widths[idx])
            for idx, cell in enumerate(row)
        ]
        return "  ".join(padded)

    lines.append(_format(header))
    lines.append("  ".join("-" * width for width in widths))
    lines.extend(_format(row) for row in rows)
    return "\n".join(lines)


def _write_csv(path: Path, results: dict[str, dict[str, float]]) -> None:
    """Serialize results as a flat CSV table."""
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "metric", "value"])
        for region, stats in results.items():
            for key, _ in TEXT_METRIC_HEADERS:
                writer.writerow([region, key, stats[key]])


def _write_json(path: Path, results: dict[str, dict[str, float]]) -> None:
    """Serialize results as JSON."""
    with path.open("w") as handle:
        json.dump(results, handle, indent=2, sort_keys=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compute summary statistics for cached potential measurements over a date range."
        )
    )
    parser.add_argument(
        "--start", type=_parse_iso_date, required=True, help="Start date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--end", type=_parse_iso_date, required=True, help="End date (YYYY-MM-DD)."
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing daily NPZ caches (default: artifacts/potential_cache).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional file to write results. Use --format to choose encoding.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "csv", "json"),
        default="text",
        help="Output format when writing to --output (default: text).",
    )
    parser.add_argument(
        "--require-all",
        action="store_true",
        help="Raise an error if any day in the range is missing from the cache.",
    )
    args = parser.parse_args()

    days = _date_range(args.start, args.end)
    sunlit, shaded, missing = _load_range(days, args.cache_dir)

    if missing:
        missing_str = ", ".join(day.strftime("%Y-%m-%d") for day in missing)
        message = f"{len(missing)} day(s) skipped because no cache file was found: {missing_str}"
        if args.require_all:
            raise SystemExit(message)
        print(f"Warning: {message}", file=sys.stderr)

    try:
        results = {
            "Sunlit": _compute_region_stats(sunlit),
            "Shaded": _compute_region_stats(shaded),
        }
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    rendered = _render_text(results)
    print(rendered)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if args.format == "text":
            args.output.write_text(rendered + "\n", encoding="utf-8")
        elif args.format == "csv":
            _write_csv(args.output, results)
        else:
            _write_json(args.output, results)


if __name__ == "__main__":
    main()
