# Engineering Products

Generate per-site engineering statistics, and diagnostic visualizations of the
temporal-harmonic reconstruction, from `src.temporal` coefficient files.

**Framing note:** the `map_*.png` outputs below are grid evaluations of the
spatiotemporal reconstruction described in
[temporal.md](temporal.md) — that reconstruction is the identifiability
analysis behind the paper's negative result (a global lunar surface-potential
map is not recoverable from LP ER data), so these images are illustrative
diagnostics, not a validated global potential-map product. The valid,
supported deliverable here is `site_analysis.csv`: per-site aggregate
statistics over the measurements at each site's coordinates.

## Usage

```bash
uv run python -m src.engineering COEFFS_FILE [OPTIONS]
```

## Arguments

| Argument | Type | Description |
|----------|------|-------------|
| `COEFFS_FILE` | path | Temporal coefficient NPZ file from `src.temporal` |

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | path | `artifacts/engineering` | Directory for maps and tables |
| `--current-density` | float | 1e-6 | Representative current density (A/m^2) |
| `--lat-res` | float | 1.0 | Latitude resolution in degrees |
| `--lon-res` | float | 1.0 | Longitude resolution in degrees |
| `-v`, `--verbose` | flag | False | Enable DEBUG-level logging |

## Outputs

Default outputs in `artifacts/engineering/`:
- `map_mean_power.png` – mean power density, reconstruction diagnostic (not a validated global map)
- `map_p95_potential.png` – 95th percentile |U|, reconstruction diagnostic (not a validated global map)
- `map_risk_1kV.png` – fraction of time |U| > 1 kV, reconstruction diagnostic (not a validated global map)
- `site_analysis.csv` – per-site summary table (the supported deliverable)
- `summary_report.md` – plain-language summary
- `global_stats.npz` – raw reconstruction-diagnostic arrays for downstream use

## Site List

Sites are defined in `src/engineering/sites.py`. Update that list to add or
remove targets of interest.

## Example

```bash
uv run python -m src.engineering artifacts/jan98_harmonics.npz \
  --output-dir artifacts/engineering/jan98 \
  --current-density 1e-6
```
