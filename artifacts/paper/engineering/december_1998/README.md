# December 1998 Engineering Analysis

## Generation Command

```bash
uv run python -m src.engineering \
  artifacts/paper/harmonics/december_1998_lmax10.npz \
  --output-dir artifacts/paper/engineering/december_1998 \
  -v
```

## Output Files

- `map_mean_power.png` — Mean charging power density map
- `map_p95_potential.png` — 95th percentile potential magnitude
- `map_risk_1kV.png` — Risk map for >1kV potentials
- `summary_report.md` — Detailed statistics and hazard assessment

## Parameters

- **Source**: december_1998_lmax10.npz harmonics
- **Grid resolution**: 181 × 361 (1° spacing)

Generated: 2025-12-12
