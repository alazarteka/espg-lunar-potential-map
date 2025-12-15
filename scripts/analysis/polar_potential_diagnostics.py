"""
Polar potential diagnostics script.

Validates whether kV-scale polar potentials are real physics or
artifacts from the harmonic reconstruction. This script examines
raw fitted potentials (before harmonic reconstruction) for polar
regions (|lat| > threshold).

Diagnostics produced:
1. Histogram with median, 5th, 95th percentiles
2. Max magnitude per day
3. Fraction of measurements below voltage thresholds (-500V, -1000V, etc.)

Usage:
    uv run python -m scripts.analysis.polar_potential_diagnostics \
        --cache-dir artifacts/potential_cache \
        --lat-threshold 75 \
        --output-dir artifacts/diagnostics/polar
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.visualization.style import GRID_STYLE, apply_paper_style

DEFAULT_CACHE_DIR = Path("artifacts/potential_cache")
DEFAULT_OUTPUT_DIR = Path("artifacts/diagnostics/polar")
DEFAULT_LAT_THRESHOLD = 75.0
VOLTAGE_THRESHOLDS = [-100.0, -250.0, -500.0, -1000.0, -2000.0]

# Fitter bounds (from src/flux.py LossConeFitter._fit_surface_potential)
FITTER_BOUND_LOW = -2000.0
FITTER_BOUND_HIGH = 2000.0
BOUND_TOLERANCE = 1.0  # Consider within ±1V as "at bound"


@dataclass(slots=True)
class PolarMeasurements:
    """Polar region measurement bundle."""

    utc: np.ndarray  # datetime64[ns]
    surface_potential: np.ndarray  # V
    projection_lat: np.ndarray  # degrees
    projection_in_sun: np.ndarray  # bool


def _parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a Python date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc


def _date_range(start_day: date, end_day: date) -> list[date]:
    """Generate inclusive list of dates."""
    days: list[date] = []
    current = start_day
    while current <= end_day:
        days.append(current)
        current += timedelta(days=1)
    return days


def _discover_npz(cache_dir: Path) -> list[Path]:
    """Return all NPZ cache files under cache_dir."""
    if not cache_dir.exists():
        raise FileNotFoundError(f"Cache directory {cache_dir} does not exist")
    return sorted(p for p in cache_dir.rglob("*.npz") if p.is_file())


def _load_polar_measurements(
    files: Iterable[Path],
    lat_threshold: float,
    start_day: date | None = None,
    end_day: date | None = None,
) -> PolarMeasurements:
    """Load measurements with |lat| > threshold from NPZ cache files."""
    utc_parts: list[np.ndarray] = []
    pot_parts: list[np.ndarray] = []
    lat_parts: list[np.ndarray] = []
    sun_parts: list[np.ndarray] = []

    # Build date filter strings if specified
    start_str = end_str = None
    if start_day is not None:
        start_str = start_day.strftime("%Y-%m-%dT00:00:00")
    if end_day is not None:
        end_str = (end_day + timedelta(days=1)).strftime("%Y-%m-%dT00:00:00")

    for path in files:
        try:
            with np.load(path) as data:
                utc = data["rows_utc"]
                pot = data["rows_projected_potential"].astype(np.float64)
                lat = data["rows_projection_latitude"].astype(np.float64)
                proj_in_sun = data.get("rows_projection_in_sun")
        except Exception as e:
            logging.warning("Failed to load %s: %s", path, e)
            continue

        if utc.size == 0:
            continue

        # Apply date range filter
        valid_time = utc != ""
        if not np.any(valid_time):
            continue

        mask = valid_time
        if start_str is not None:
            mask = mask & (utc >= start_str)
        if end_str is not None:
            mask = mask & (utc < end_str)

        # Filter by polar region and finite potential
        polar_mask = mask & (np.abs(lat) > lat_threshold) & np.isfinite(pot)

        if not np.any(polar_mask):
            continue

        try:
            utc_vals = np.array(utc[polar_mask], dtype="datetime64[ns]")
        except ValueError:
            logging.debug("Failed to parse UTC strings in %s; skipping", path)
            continue

        utc_parts.append(utc_vals)
        pot_parts.append(pot[polar_mask])
        lat_parts.append(lat[polar_mask])
        if proj_in_sun is not None:
            sun_parts.append(proj_in_sun[polar_mask].astype(bool))
        else:
            sun_parts.append(np.zeros(polar_mask.sum(), dtype=bool))

    if not utc_parts:
        empty_time = np.array([], dtype="datetime64[ns]")
        empty_float = np.array([])
        return PolarMeasurements(
            utc=empty_time,
            surface_potential=empty_float,
            projection_lat=empty_float,
            projection_in_sun=np.array([], dtype=bool),
        )

    return PolarMeasurements(
        utc=np.concatenate(utc_parts),
        surface_potential=np.concatenate(pot_parts),
        projection_lat=np.concatenate(lat_parts),
        projection_in_sun=np.concatenate(sun_parts),
    )


def _compute_daily_stats(
    measurements: PolarMeasurements,
) -> tuple[list[date], list[float], list[float], list[float], list[int]]:
    """Compute daily max magnitude and percentile stats."""
    if measurements.utc.size == 0:
        return [], [], [], [], []

    # Convert to dates
    days_np = measurements.utc.astype("datetime64[D]")
    unique_days = np.unique(days_np)

    dates: list[date] = []
    max_magnitudes: list[float] = []
    p5_list: list[float] = []
    p95_list: list[float] = []
    counts: list[int] = []

    for day in unique_days:
        mask = days_np == day
        pots = measurements.surface_potential[mask]
        if pots.size == 0:
            continue

        # Convert numpy datetime64 to Python date
        # numpy datetime64[D] -> int64 -> timedelta from epoch -> date
        epoch = np.datetime64(0, "D")
        day_offset = int((day - epoch) / np.timedelta64(1, "D"))
        dates.append(date.fromordinal(day_offset + date(1970, 1, 1).toordinal()))
        max_magnitudes.append(float(np.max(np.abs(pots))))
        p5_list.append(float(np.percentile(pots, 5)))
        p95_list.append(float(np.percentile(pots, 95)))
        counts.append(int(pots.size))

    return dates, max_magnitudes, p5_list, p95_list, counts


def _compute_threshold_fractions(
    measurements: PolarMeasurements,
    thresholds: list[float],
) -> dict[float, float]:
    """Compute fraction of measurements below each threshold."""
    if measurements.surface_potential.size == 0:
        return {t: 0.0 for t in thresholds}

    total = measurements.surface_potential.size
    fractions = {}
    for thresh in thresholds:
        count_below = np.sum(measurements.surface_potential < thresh)
        fractions[thresh] = float(count_below) / total

    return fractions


@dataclass(slots=True)
class BoundHitStats:
    """Statistics about fitter bound saturation."""

    total: int
    exact_neg_bound: int  # Exactly at -2000V
    exact_pos_bound: int  # Exactly at +2000V
    near_neg_bound: int  # Within tolerance of -2000V
    near_pos_bound: int  # Within tolerance of +2000V
    clean_count: int  # After removing bound-hitters
    clean_min: float
    clean_max: float
    clean_p5: float
    clean_p25: float
    clean_median: float
    clean_p75: float
    clean_p95: float
    clean_mean: float
    clean_std: float

    @property
    def frac_neg_bound(self) -> float:
        return self.near_neg_bound / self.total if self.total > 0 else 0.0

    @property
    def frac_pos_bound(self) -> float:
        return self.near_pos_bound / self.total if self.total > 0 else 0.0

    @property
    def frac_any_bound(self) -> float:
        return (self.near_neg_bound + self.near_pos_bound) / self.total if self.total > 0 else 0.0

    @property
    def frac_clean(self) -> float:
        return self.clean_count / self.total if self.total > 0 else 0.0


def _compute_bound_hit_stats(potentials: np.ndarray) -> BoundHitStats:
    """Compute comprehensive bound-hit statistics."""
    total = len(potentials)
    if total == 0:
        return BoundHitStats(
            total=0,
            exact_neg_bound=0,
            exact_pos_bound=0,
            near_neg_bound=0,
            near_pos_bound=0,
            clean_count=0,
            clean_min=np.nan,
            clean_max=np.nan,
            clean_p5=np.nan,
            clean_p25=np.nan,
            clean_median=np.nan,
            clean_p75=np.nan,
            clean_p95=np.nan,
            clean_mean=np.nan,
            clean_std=np.nan,
        )

    exact_neg = np.sum(potentials == FITTER_BOUND_LOW)
    exact_pos = np.sum(potentials == FITTER_BOUND_HIGH)
    near_neg = np.sum(potentials <= FITTER_BOUND_LOW + BOUND_TOLERANCE)
    near_pos = np.sum(potentials >= FITTER_BOUND_HIGH - BOUND_TOLERANCE)

    # Clean data: exclude bound-hitters
    clean_mask = (potentials > FITTER_BOUND_LOW + BOUND_TOLERANCE) & (
        potentials < FITTER_BOUND_HIGH - BOUND_TOLERANCE
    )
    clean = potentials[clean_mask]

    if len(clean) == 0:
        return BoundHitStats(
            total=total,
            exact_neg_bound=int(exact_neg),
            exact_pos_bound=int(exact_pos),
            near_neg_bound=int(near_neg),
            near_pos_bound=int(near_pos),
            clean_count=0,
            clean_min=np.nan,
            clean_max=np.nan,
            clean_p5=np.nan,
            clean_p25=np.nan,
            clean_median=np.nan,
            clean_p75=np.nan,
            clean_p95=np.nan,
            clean_mean=np.nan,
            clean_std=np.nan,
        )

    return BoundHitStats(
        total=total,
        exact_neg_bound=int(exact_neg),
        exact_pos_bound=int(exact_pos),
        near_neg_bound=int(near_neg),
        near_pos_bound=int(near_pos),
        clean_count=int(len(clean)),
        clean_min=float(np.min(clean)),
        clean_max=float(np.max(clean)),
        clean_p5=float(np.percentile(clean, 5)),
        clean_p25=float(np.percentile(clean, 25)),
        clean_median=float(np.percentile(clean, 50)),
        clean_p75=float(np.percentile(clean, 75)),
        clean_p95=float(np.percentile(clean, 95)),
        clean_mean=float(np.mean(clean)),
        clean_std=float(np.std(clean)),
    )


def _plot_histogram(
    measurements: PolarMeasurements,
    lat_threshold: float,
    output_dir: Path,
    show: bool = False,
) -> Path:
    """Plot histogram of polar potentials with percentile markers."""
    potentials = measurements.surface_potential
    p5 = np.percentile(potentials, 5)
    p50 = np.percentile(potentials, 50)
    p95 = np.percentile(potentials, 95)

    fig, ax = plt.subplots(figsize=(10, 6))

    # Histogram with logarithmic y-axis for better visibility
    bins = np.linspace(potentials.min(), potentials.max(), 100)
    ax.hist(potentials, bins=bins, color="steelblue", alpha=0.7, edgecolor="white")
    ax.set_yscale("log")

    # Add percentile lines
    ax.axvline(p5, color="red", linestyle="--", linewidth=2, label=f"P5: {p5:.0f} V")
    ax.axvline(p50, color="green", linestyle="-", linewidth=2, label=f"P50: {p50:.0f} V")
    ax.axvline(p95, color="orange", linestyle="--", linewidth=2, label=f"P95: {p95:.0f} V")

    # Add threshold markers at bottom
    for thresh in VOLTAGE_THRESHOLDS:
        if thresh >= potentials.min():
            ax.axvline(thresh, color="gray", linestyle=":", alpha=0.5)

    ax.set_xlabel("Surface Potential (V)")
    ax.set_ylabel("Count")
    ax.set_title(f"Raw Fitted Potentials in Polar Regions (|lat| > {lat_threshold}°)")
    ax.legend(loc="upper right")
    apply_paper_style(ax)

    # Add text box with stats
    stats_text = (
        f"N = {len(potentials):,}\n"
        f"Min = {potentials.min():.0f} V\n"
        f"Max = {potentials.max():.0f} V\n"
        f"Mean = {np.mean(potentials):.0f} V"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    output_path = output_dir / "polar_potential_histogram.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info("Saved histogram to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _plot_daily_max(
    dates: list[date],
    max_magnitudes: list[float],
    lat_threshold: float,
    output_dir: Path,
    show: bool = False,
) -> Path:
    """Plot max potential magnitude per day."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(dates, max_magnitudes, color="steelblue", alpha=0.8, width=0.8)

    # Add kV threshold lines
    ax.axhline(500, color="orange", linestyle="--", linewidth=1.5, label="500 V")
    ax.axhline(1000, color="red", linestyle="--", linewidth=1.5, label="1 kV")
    ax.axhline(2000, color="darkred", linestyle="-", linewidth=2, label="2 kV")

    ax.set_xlabel("Date")
    ax.set_ylabel("Max |Potential| (V)")
    ax.set_title(f"Daily Maximum Potential Magnitude (|lat| > {lat_threshold}°)")
    ax.legend(loc="upper right")
    apply_paper_style(ax)

    # Rotate x-axis labels
    fig.autofmt_xdate()

    output_path = output_dir / "polar_daily_max_magnitude.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info("Saved daily max plot to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _plot_threshold_fractions(
    fractions: dict[float, float],
    lat_threshold: float,
    output_dir: Path,
    show: bool = False,
) -> Path:
    """Bar chart of fraction below each threshold."""
    thresholds = sorted(fractions.keys())
    frac_values = [fractions[t] * 100 for t in thresholds]  # Convert to percent
    labels = [f"< {int(t)} V" for t in thresholds]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars = ax.bar(labels, frac_values, color="steelblue", alpha=0.8)

    # Add percentage labels on bars
    for bar, val in zip(bars, frac_values):
        if val > 0.01:  # Only label if visible
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{val:.1f}%",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Fraction of Measurements (%)")
    ax.set_title(f"Polar Measurements Below Voltage Thresholds (|lat| > {lat_threshold}°)")
    ax.set_ylim(0, max(frac_values) * 1.2 if max(frac_values) > 0 else 1)
    apply_paper_style(ax)

    output_path = output_dir / "polar_threshold_fractions.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info("Saved threshold fractions to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _plot_daily_percentiles(
    dates: list[date],
    p5_list: list[float],
    p95_list: list[float],
    lat_threshold: float,
    output_dir: Path,
    show: bool = False,
) -> Path:
    """Plot daily P5 and P95 over time."""
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.fill_between(dates, p5_list, p95_list, alpha=0.3, color="steelblue", label="P5-P95 range")
    ax.plot(dates, p5_list, color="blue", linewidth=1.5, label="P5")
    ax.plot(dates, p95_list, color="orange", linewidth=1.5, label="P95")

    # Add reference lines
    ax.axhline(0, color="gray", linestyle="-", alpha=0.5)
    ax.axhline(-500, color="orange", linestyle="--", alpha=0.7, label="-500 V")
    ax.axhline(-1000, color="red", linestyle="--", alpha=0.7, label="-1 kV")

    ax.set_xlabel("Date")
    ax.set_ylabel("Potential (V)")
    ax.set_title(f"Daily P5 and P95 Potentials (|lat| > {lat_threshold}°)")
    ax.legend(loc="lower left")
    apply_paper_style(ax)

    fig.autofmt_xdate()

    output_path = output_dir / "polar_daily_percentiles.png"
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    logging.info("Saved daily percentiles to %s", output_path)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return output_path


def _generate_summary(
    measurements: PolarMeasurements,
    lat_threshold: float,
    fractions: dict[float, float],
    bound_stats: BoundHitStats,
    output_dir: Path,
) -> Path:
    """Generate a text summary of the polar potential diagnostics."""
    potentials = measurements.surface_potential

    if potentials.size == 0:
        summary = "No polar measurements found in the specified date range."
    else:
        p5 = np.percentile(potentials, 5)
        p25 = np.percentile(potentials, 25)
        p50 = np.percentile(potentials, 50)
        p75 = np.percentile(potentials, 75)
        p95 = np.percentile(potentials, 95)

        lines = [
            "=" * 70,
            "POLAR POTENTIAL DIAGNOSTICS SUMMARY",
            "=" * 70,
            "",
            f"Latitude threshold: |lat| > {lat_threshold}°",
            f"Total measurements: {len(potentials):,}",
            f"Fitter bounds: [{FITTER_BOUND_LOW:.0f}, {FITTER_BOUND_HIGH:.0f}] V",
            "",
            "=" * 70,
            "CRITICAL: FITTER BOUND SATURATION ANALYSIS",
            "=" * 70,
            "",
            f"  Exactly at -2000 V: {bound_stats.exact_neg_bound:>8,} ({bound_stats.exact_neg_bound/bound_stats.total*100:>5.1f}%)",
            f"  Exactly at +2000 V: {bound_stats.exact_pos_bound:>8,} ({bound_stats.exact_pos_bound/bound_stats.total*100:>5.1f}%)",
            f"  Near -2000 V (±1V): {bound_stats.near_neg_bound:>8,} ({bound_stats.frac_neg_bound*100:>5.1f}%)",
            f"  Near +2000 V (±1V): {bound_stats.near_pos_bound:>8,} ({bound_stats.frac_pos_bound*100:>5.1f}%)",
            f"  TOTAL BOUND-HITTERS: {bound_stats.near_neg_bound + bound_stats.near_pos_bound:>6,} ({bound_stats.frac_any_bound*100:>5.1f}%)",
            "",
            ">>> THESE ARE NOT REAL MEASUREMENTS - THEY ARE FITTER FAILURES <<<",
            "",
            "=" * 70,
            "RAW STATISTICS (INCLUDING BOUND-HITTERS)",
            "=" * 70,
            "",
            f"  Min:     {potentials.min():>10.1f} V",
            f"  P5:      {p5:>10.1f} V",
            f"  P25:     {p25:>10.1f} V",
            f"  Median:  {p50:>10.1f} V",
            f"  P75:     {p75:>10.1f} V",
            f"  P95:     {p95:>10.1f} V",
            f"  Max:     {potentials.max():>10.1f} V",
            "",
            "=" * 70,
            "CLEAN STATISTICS (AFTER REMOVING BOUND-HITTERS)",
            "=" * 70,
            "",
            f"  Valid measurements: {bound_stats.clean_count:,} ({bound_stats.frac_clean*100:.1f}% of total)",
            f"  Min:     {bound_stats.clean_min:>10.1f} V",
            f"  P5:      {bound_stats.clean_p5:>10.1f} V",
            f"  P25:     {bound_stats.clean_p25:>10.1f} V",
            f"  Median:  {bound_stats.clean_median:>10.1f} V",
            f"  P75:     {bound_stats.clean_p75:>10.1f} V",
            f"  P95:     {bound_stats.clean_p95:>10.1f} V",
            f"  Max:     {bound_stats.clean_max:>10.1f} V",
            f"  Mean:    {bound_stats.clean_mean:>10.1f} V",
            f"  Std:     {bound_stats.clean_std:>10.1f} V",
            "",
            "=" * 70,
            "DIAGNOSIS",
            "=" * 70,
            "",
        ]

        # Diagnosis based on bound-hit fraction
        if bound_stats.frac_neg_bound > 0.15:
            lines.extend([
                f"  ⚠️  SEVERE BOUND SATURATION: {bound_stats.frac_neg_bound*100:.1f}% of fits hit -2000V floor",
                "",
                "  The 'persistent polar kV' feature is DRIVEN BY FITTER FAILURES, not physics.",
                "",
                "  Evidence:",
                f"    • {bound_stats.exact_neg_bound:,} fits exactly at -2000V (hard cap)",
                f"    • P25 raw = {p25:.0f}V (stacked near bound)",
                f"    • Clean median = {bound_stats.clean_median:.0f}V (much less extreme)",
                "",
                "  Halekas (2008) sanity check:",
                "    • Tail lobes: ~-100V",
                "    • Plasma sheet: -200V to -1kV",
                "    • Wake edges: ~-200V",
                "    • SEP events: up to ~-4kV (rare)",
                "",
                f"  Your clean median ({bound_stats.clean_median:.0f}V) is more consistent with",
                "  plasma sheet / wake conditions than the raw median.",
                "",
                "  RECOMMENDED ACTIONS:",
                "    1. Treat bound-hitters as INVALID/CENSORED (U ≤ -2000V, not U = -2000V)",
                "    2. Recompute harmonic fits excluding bound-hitters",
                "    3. Inspect spectra at bound-hits to understand failure mode",
                "    4. Consider widening fitter bounds or flagging unconstrained fits",
            ])
        elif bound_stats.frac_neg_bound > 0.05:
            lines.extend([
                f"  ⚠️  MODERATE BOUND SATURATION: {bound_stats.frac_neg_bound*100:.1f}% at -2000V",
                "",
                "  Some fits are hitting bounds, inflating kV statistics.",
                f"  Clean median = {bound_stats.clean_median:.0f}V vs raw median = {p50:.0f}V",
                "",
                "  Recommend excluding bound-hitters before analysis.",
            ])
        else:
            lines.extend([
                f"  ✓ Low bound saturation: {bound_stats.frac_neg_bound*100:.1f}% at bounds",
                "",
                "  Most fits are well-constrained. Statistics are reliable.",
            ])

        lines.extend(["", "=" * 70])
        summary = "\n".join(lines)

    summary_path = output_dir / "polar_diagnostics_summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    logging.info("Saved summary to %s", summary_path)

    return summary_path


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose whether kV-scale polar potentials are physics or artifacts. "
            "Examines raw fitted potentials before harmonic reconstruction."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help="Directory containing potential cache NPZ files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save diagnostic outputs",
    )
    parser.add_argument(
        "--lat-threshold",
        type=float,
        default=DEFAULT_LAT_THRESHOLD,
        help="Latitude threshold for polar region (|lat| > threshold)",
    )
    parser.add_argument(
        "--start",
        type=_parse_iso_date,
        default=None,
        help="Start date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--end",
        type=_parse_iso_date,
        default=None,
        help="End date (YYYY-MM-DD, inclusive)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    """Main entry point."""
    args = parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # Validate and create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Discover cache files
    try:
        files = _discover_npz(args.cache_dir)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    logging.info("Found %d NPZ files in %s", len(files), args.cache_dir)

    # Load polar measurements
    logging.info("Loading polar measurements (|lat| > %.1f°)...", args.lat_threshold)
    measurements = _load_polar_measurements(
        files,
        lat_threshold=args.lat_threshold,
        start_day=args.start,
        end_day=args.end,
    )

    if measurements.utc.size == 0:
        logging.warning("No polar measurements found matching criteria.")
        return 1

    logging.info("Loaded %d polar measurements", measurements.utc.size)

    # Compute statistics
    logging.info("Computing daily statistics...")
    dates, max_mags, p5_list, p95_list, counts = _compute_daily_stats(measurements)

    logging.info("Computing threshold fractions...")
    fractions = _compute_threshold_fractions(measurements, VOLTAGE_THRESHOLDS)

    logging.info("Computing bound-hit statistics...")
    bound_stats = _compute_bound_hit_stats(measurements.surface_potential)

    # Generate plots
    logging.info("Generating diagnostic plots...")
    _plot_histogram(measurements, args.lat_threshold, args.output_dir, args.show)
    if dates:
        _plot_daily_max(dates, max_mags, args.lat_threshold, args.output_dir, args.show)
        _plot_daily_percentiles(
            dates, p5_list, p95_list, args.lat_threshold, args.output_dir, args.show
        )
    _plot_threshold_fractions(fractions, args.lat_threshold, args.output_dir, args.show)

    # Generate summary
    summary_path = _generate_summary(
        measurements, args.lat_threshold, fractions, bound_stats, args.output_dir
    )

    # Print summary to console
    print(summary_path.read_text())

    logging.info("Diagnostics complete. Outputs saved to %s", args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
