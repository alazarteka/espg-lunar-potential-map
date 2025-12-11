# April 1998 Residual Analysis (lmax=10)

## Generation Command

```bash
uv run python scripts/analysis/residual_analysis.py \
  --start 1998-04-01 --end 1998-04-30 \
  --lmax 10 \
  --temporal-basis "constant,synodic,synodic2,synodic3" \
  --cache-dir artifacts/potential_cache \
  --l2-penalty 100.0 \
  --output-dir artifacts/paper/residuals/april_1998_lmax10 \
  -v
```

## Parameters

- **Date range**: 1998-04-01 → 1998-04-30
- **lmax**: 10
- **Temporal basis**: constant, synodic, synodic2, synodic3
- **L2 penalty**: 100.0 (default)
- **Total points**: 136,224
- **Global RMS residual**: 567.62 V

## Results

| Environment | Mean | Std | N |
|-------------|------|-----|---|
| Solar Wind | -7.51 | 589.16 | 48,226 |
| Wake | -7.58 | 575.58 | 46,358 |
| Plasma Sheet | -0.71 | 506.52 | 29,495 |
| Lobes | -15.77 | 589.25 | 12,145 |

Generated: 2025-12-10
