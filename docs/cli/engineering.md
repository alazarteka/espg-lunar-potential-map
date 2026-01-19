# Engineering Products

Generate engineering maps and site summaries from temporal harmonic coefficients.

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
- `map_mean_power.png` – global mean power density map
- `map_p95_potential.png` – 95th percentile |U| map
- `map_risk_1kV.png` – fraction of time |U| > 1 kV
- `site_analysis.csv` – per-site summary table
- `summary_report.md` – plain-language summary
- `global_stats.npz` – raw arrays for downstream use

## Site List

Sites are defined in `src/engineering/sites.py`. Update that list to add or
remove targets of interest.

## Example

```bash
uv run python -m src.engineering artifacts/jan98_harmonics.npz \
  --output-dir artifacts/engineering/jan98 \
  --current-density 1e-6
```
