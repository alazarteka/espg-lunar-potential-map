# December 1998 Residual Analysis

## Generation Command

```bash
uv run python scripts/analysis/residual_analysis.py \
  --start 1998-12-01 --end 1998-12-31 \
  --lmax 10 \
  --temporal-basis "constant,synodic,synodic2,synodic3" \
  --output-dir artifacts/paper/residuals/december_1998
```

## Output Files

- `global_residual_map.png` — Spatial distribution of residuals
- `residual_vs_sza.png` — Residuals as function of solar zenith angle
- `residual_by_environment.png` — Boxplot by environment classification

## Statistics

- **Date range**: 1998-12-01 → 1998-12-31
- **Total samples**: 161,887
- **RMS residual**: 635.33 V

Generated: 2025-12-12
