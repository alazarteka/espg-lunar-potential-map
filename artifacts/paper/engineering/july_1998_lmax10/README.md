# July 1998 Engineering Analysis (lmax=10)

## Generation Command

```bash
uv run python -m src.engineering \
  artifacts/paper/harmonics/july_1998_lmax10.npz \
  --output-dir artifacts/paper/engineering/july_1998_lmax10 \
  -v
```

## Input

- **Harmonics file**: `artifacts/paper/harmonics/july_1998_lmax10.npz`
- **lmax**: 10
- **Temporal basis**: constant, synodic1,2,3
- **L2 penalty**: 100.0
- **Current density**: 1.0 µA/m² (default)

## Outputs

| File | Description |
|------|-------------|
| `map_mean_power.png` | Expected ESPG power density (mW/m²) |
| `map_p95_potential.png` | 95th percentile surface potential (V) |
| `map_risk_1kV.png` | Fraction of time |U| > 1 kV |
| `site_analysis.csv` | Per-site statistics (Apollo 11/16, Tycho, etc.) |
| `summary_report.md` | Interpretive summary with key findings |
| `global_stats.npz` | Raw gridded data for further analysis |

Generated: 2025-12-10
