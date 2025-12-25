# December 1998 Harmonics (lmax=10)

## Generation Command

```bash
uv run python -m src.temporal \
  --start 1998-12-01 --end 1998-12-31 \
  --lmax 10 \
  --fit-mode basis \
  --temporal-basis "constant,synodic,synodic2,synodic3" \
  --cache-dir artifacts/potential_cache \
  --l2-penalty 100.0 \
  --output artifacts/paper/harmonics/december_1998_lmax10.npz \
  -v
```

## Parameters

- **Date range**: 1998-12-01 → 1998-12-31
- **lmax**: 10 (121 spherical harmonic coefficients)
- **Temporal basis**: constant, synodic_cos/sin, synodic2_cos/sin, synodic3_cos/sin (7 bases)
- **L2 penalty**: 100.0
- **Total parameters**: 847 (7 × 121)
- **Total samples**: 161,887
- **RMS residual**: 635.33 V

Generated: 2025-12-12
