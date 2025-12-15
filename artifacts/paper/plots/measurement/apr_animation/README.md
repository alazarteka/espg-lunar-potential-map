# April 1998 Measurement Animation

Animated surface potential measurements for April 1998, rendered in hemispheric (polar) and global projections.

## Files

- `surface_measurements_19980401_19980430.gif` — Full month animation

## Generation

```bash
uv run python scripts/plots/plot_measurements_hemispheric.py \
  --start 1998-04-01 \
  --end 1998-04-30 \
  --output-dir artifacts/paper/plots/measurement/apr_animation \
  --input artifacts/potential_cache/potential_batch_1998_04.npz \
  --no-sun-highlight
```
