"""Convert terminator charge JSON reports into readable Markdown summaries."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable

import numpy as np


def _fmt_float(value: float | None, fmt: str = "{:.2f}") -> str:
    if value is None:
        return "N/A"
    return fmt.format(value)


def _fmt_scientific(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2e}"


def _fmt_range(values: list[float | None], fmt: str = "{:.2f}") -> str:
    if not values or all(v is None for v in values):
        return "N/A"
    a = fmt.format(values[0]) if values[0] is not None else "N/A"
    b = fmt.format(values[1]) if values[1] is not None else "N/A"
    return f"{a} → {b}"


def _fmt_sigma_quantiles(values: list[float] | None) -> str:
    if not values or len(values) != 3:
        return "N/A"
    q05, q50, q95 = values
    return f"{q50:.2e} (5% {q05:.2e}, 95% {q95:.2e})"


def _render_record(record: dict) -> str:
    if record.get("error"):
        return f"- **{record['file']}**\n  - ERROR: {record['error']}\n"

    lines: list[str] = []
    header = f"- **{record['file']}**"
    lines.append(header)

    subsolar = record.get("subsolar", {})
    midpoint = subsolar.get("midpoint_utc")
    lat = subsolar.get("latitude_deg")
    lon = subsolar.get("longitude_deg")
    if midpoint and lat is not None and lon is not None:
        lines.append(
            f"  - Midpoint UTC `{midpoint}` | Subsolar lat {lat:.2f}°, lon {lon:.2f}°"
        )

    sun = record.get("phi_sunlit", {})
    shadow = record.get("phi_shadow", {})
    delta = record.get("delta_phi", {})

    def fmt_pot(block: dict) -> str:
        med = block.get("median_V")
        mad = block.get("mad_V")
        cnt = block.get("count")
        med_str = "N/A" if med is None else f"{med:.2f}"
        if mad is None or np.isnan(mad):
            mad_str = "N/A"
        else:
            mad_str = f"± {mad:.2f}"
        if cnt is not None:
            return f"{med_str} {mad_str} V (n={cnt})"
        return f"{med_str} {mad_str} V"

    sun_desc = fmt_pot(sun)
    shadow_desc = fmt_pot(shadow)
    lines.append(f"  - Surface potentials: sunlit {sun_desc} | shadow {shadow_desc}")

    def fmt_delta(block: dict, label: str) -> str:
        med = block.get("median_V")
        mad = block.get("mad_V")
        med_str = "N/A" if med is None or np.isnan(med) else f"{med:.2f}"
        if mad is None or np.isnan(mad):
            mad_str = "± N/A"
        else:
            mad_str = f"± {mad:.2f}"
        return f"    {label} = {med_str} {mad_str} V"

    lines.append(fmt_delta(delta, "ΔΦ"))
    lines.append(fmt_delta(record.get("delta_phi_spacecraft", {}), "ΔΦ_spacecraft"))

    phi_term = record.get("phi_at_terminator", {})
    phi90_val = phi_term.get("phi90_V")
    dphi_val = phi_term.get("dphi_dtheta_V_per_deg")
    lines.append(
        "  - Φ(90°) = "
        + (
            f"{phi90_val:.2f} V"
            if phi90_val is not None and not np.isnan(phi90_val)
            else "N/A"
        )
        + " | dΦ/dθ = "
        + (
            f"{dphi_val:.2f} V/deg"
            if dphi_val is not None and not np.isnan(dphi_val)
            else "N/A"
        )
    )

    width = record.get("lateral_width_km")
    if width is None or np.isnan(width):
        lines.append("  - Lateral width (SZA 88→92): N/A")
    else:
        lines.append(f"  - Lateral width (SZA 88→92): {width:.2f} km")

    sigma_day = _fmt_sigma_quantiles(record.get("sigma_day_C_m2"))
    sigma_night = _fmt_sigma_quantiles(record.get("sigma_night_C_m2"))
    delta_sigma = _fmt_sigma_quantiles(record.get("delta_sigma_C_m2"))
    lines.append(f"  - σ_day = {sigma_day}")
    lines.append(f"  - σ_night = {sigma_night}")
    lines.append(f"  - Δσ = {delta_sigma}")

    crossing = record.get("terminator_crossing_utc")
    if crossing:
        lines.append(f"  - First shadow→sunlit crossing: `{crossing}`")

    quality = record.get("quality", {})
    flag_agree = quality.get("flag_agreement")
    lines.append(
        f"  - Illumination flag agreement = {_fmt_float(flag_agree, '{:.3f}')}"
    )
    sun_sza = _fmt_range(quality.get("sunlit_sza_range_deg", [None, None]))
    shadow_sza = _fmt_range(quality.get("shadow_sza_range_deg", [None, None]))
    lines.append(f"    Sunlit SZA range: {sun_sza}°")
    lines.append(f"    Shadow SZA range: {shadow_sza}°")

    corr = quality.get("correlation_surface_vs_spacecraft")
    lines.append(f"    Corr(surface, spacecraft) = {_fmt_float(corr, '{:.3f}')}")

    notes = quality.get("notes") or []
    if notes:
        lines.append(f"    Notes: {', '.join(notes)}")

    rows_total = record.get("rows_analyzed")
    window_sec = record.get("analysis_window_seconds")
    if rows_total is not None and window_sec is not None:
        window_hr = window_sec / 3600.0
        lines.append(f"  - Rows analysed {rows_total:,} over {window_hr:.1f} h window")

    return "\n".join(lines) + "\n"


def _render_markdown(records: Iterable[dict]) -> str:
    lines = [
        "# Terminator Charge Report",
        "",
        f"_Generated: {datetime.now(UTC).isoformat(timespec='seconds')}_",
        "",
    ]
    for record in records:
        lines.append(_render_record(record))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert terminator charge JSON report to Markdown."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Input JSON report generated by potential_terminator_charge.py",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination Markdown file.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = json.loads(args.input.read_text(encoding="utf-8"))
    md = _render_markdown(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
