# Terminator Charge Monthly Summary (1998-03 → 1998-10)

This document consolidates the outputs of `scripts/analysis/potential_terminator_charge.py` for six months in 1998 (March, April, May, July, August, October). Each month’s detailed JSON and Markdown artifacts live under `reports/monthly/YYYY-MM/terminator_charge.{json,md}`.

## Inputs and Calculation Pipeline

- **Source data** – Cached NPZ tiles under `data/potential_cache/**/3DYYMMDD.npz`. Each NPZ stores projected footprint locations, spacecraft/surface potentials, UTC strings, and illumination flags.
- **Geometry & classification** – For every row, the analysis script recomputes solar zenith angle (SZA) via SPICE (`spice.utc2et`, `src.utils.spice_ops.get_sun_vector_wrt_moon`). Samples with `SZA ≤ 88°` become “sunlit”; `SZA ≥ 92°` become “shadowed”, ensuring a 4° guard band instead of trusting the cached `projection_in_sun` flag (the latter is only used for QA agreement metrics).
- **Robust statistics** – Median/MAD estimates feed the sun/shadow surface potentials, spacecraft potential deltas, and the quadratic fit at SZA≈90°. Monte Carlo sampling (default 20 k draws) perturbs Φ_sun/Φ_shadow across the requested sheath ranges to estimate charge density quantiles (`σ_day`, `σ_night`, `Δσ`).
- **Outputs** – For each NPZ day inside the requested window the script emits a record containing the metrics above, QA notes, run length, and whether a shadow→sun terminator crossing was observed. Markdown summaries reuse `_render_markdown` from `potential_charge_report_md.py`.

## Execution details

All reports were generated with UV’s managed environment:

```bash
uv run python scripts/analysis/potential_terminator_charge.py --start YYYY-MM-01 --end YYYY-MM-<last> \
  --output reports/monthly/YYYY-MM/terminator_charge.json \
  --markdown reports/monthly/YYYY-MM/terminator_charge.md \
  --log-level INFO
```

Months executed: `1998-03`, `1998-04`, `1998-05`, `1998-07`, `1998-08`, `1998-10`.

## Monthly aggregates

| Month | Files (ok/total) | Φ_sun_med (V) | Φ_shadow_med (V) | ΔΦ_med (V) | ΔΦ range (V) | Δσ_med (C/m²) | Rows_med | Crossing % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1998-03 | 32/32 | -396.6 | -279.1 | -89.6 | -751.2…476.3 | -3.15e-09 | 15352 | 96.9% |
| 1998-04 | 30/30 | -486.2 | -240.3 | -151.6 | -813.0…329.1 | -4.01e-09 | 15465 | 96.7% |
| 1998-05 | 32/32 | -588.3 | -439.5 | -66.8 | -681.5…346.2 | -4.70e-09 | 15452 | 96.9% |
| 1998-07 | 27/27 | -373.3 | -357.4 | -10.4 | -178.5…604.2 | -2.93e-09 | 15330 | 100.0% |
| 1998-08 | 30/30 | -337.3 | -326.5 | -46.1 | -440.5…338.4 | -2.73e-09 | 15330 | 96.7% |
| 1998-10 | 30/30 | -352.8 | -243.9 | -115.3 | -408.6…111.4 | -2.89e-09 | 15308 | 96.7% |

_Notes_: Rows_med is the median number of NPZ samples per daily run. “Crossing %” is the fraction of days where an SZA-based shadow→sun transition was detected.

## Interpretation highlights

- **Spring (Mar–May)** – Potentials trend more negative overall, with April showing the strongest sun/shadow contrast (ΔΦ ≈ −152 V) and the largest inferred sheet charge magnitude (~4×10⁻⁹ C/m²). Nearly every run captures a terminator crossing despite the strict SZA masks.
- **Mid-year (Jul–Aug)** – Sunlit and shadowed medians converge, driving ΔΦ toward zero even though σ estimates remain on the order of −3×10⁻⁹ C/m². July also exhibits the widest ΔΦ spread (up to +604 V) tied to local geometry, plus virtually no QA warnings.
- **Autumn (Oct)** – Potentials separate again (ΔΦ ≈ −115 V) while σ magnitudes stabilize near earlier levels. Spacecraft/surface correlation skews more negative, hinting at geometry or plasma-state changes worth a deeper look.
- **Quality observations** – Median illumination-flag agreement stays at 1.0 for every month; QA warnings are rare and limited to isolated “few_*_samples” or missing crossings. Lateral width estimates remain `None` because no run accumulated sufficient samples in the SZA 88–92° band—if that metric is needed, we should widen `_lateral_width_km`’s angular gate or extend the sampling window.

## Next questions for review

1. Do the seasonal swings in ΔΦ and |σ| align with expectations from upstream solar-wind or terminator geometry models?
2. Is the persistent absence of lateral-width estimates acceptable, or should we revisit the algorithm/thresholds to recover that diagnostic?
3. Would adding percentile spans (e.g., 5–95%) for ΔΦ help compare with independent instrument products?

Feel free to point collaborators to the per-month Markdown files for the full per-day breakdowns referenced here.
