# April 15 1998 Hemispheric Measurements

Surface potential measurements for April 15, 1998 rendered in hemispheric (polar) and global projections.

## Files

- `surface_measurements_19980415.png` — April 15, 1998 measurements

## Generation

```bash
uv run python scripts/plots/plot_measurements_hemispheric.py \
  --start 1998-04-15 \
  --output-dir artifacts/paper/plots/measurement/apr15_hemispheric \
  --input artifacts/potential_cache/potential_batch_1998_04.npz \
  --no-sun-highlight
```

### Key options

| Option | Description |
|--------|-------------|
| `--start` | Date to plot (YYYY-MM-DD) |
| `--input` | Batch NPZ file containing measurements |
| `--output-dir` | Output directory |
| `--no-sun-highlight` | Disable white ring outlines on sunlit points |
| `--vmin`, `--vmax` | Override color scale bounds |
| `--sample N` | Down-sample to N points for readability |
| `--output-format` | `png`, `pdf`, or `svg` |

Run `--help` for full options:
```bash
uv run python scripts/plots/plot_measurements_hemispheric.py --help
```
