# July 1998 Residual Analysis (lmax=10)

## Generation Command

```bash
uv run python scripts/analysis/residual_analysis.py \
  --start 1998-07-01 --end 1998-07-31 \
  --lmax 10 \
  --temporal-basis "constant,synodic,synodic2,synodic3" \
  --cache-dir artifacts/potential_cache \
  --l2-penalty 100.0 \
  --output-dir artifacts/paper/residuals/july_1998_lmax10 \
  -v
```

## Parameters

- **Date range**: 1998-07-01 → 1998-07-31
- **lmax**: 10
- **Temporal basis**: constant, synodic, synodic2, synodic3
- **L2 penalty**: 100.0 (default)
- **Total points**: 137,892
- **Global RMS residual**: 664.42 V

## Results

| Environment | Mean | Std | N |
|-------------|------|-----|---|
| Solar Wind | -2.99 | 743.45 | 56,163 |
| Wake | -14.51 | 629.36 | 48,927 |
| Plasma Sheet | -9.90 | 521.98 | 22,032 |
| Lobes | 18.36 | 641.49 | 10,770 |

Generated: 2025-12-10
