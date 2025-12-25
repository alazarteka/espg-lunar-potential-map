# Loss Cone Fit Comparison

Observed electron flux vs model fit for April 29, 1999 spectrum 653.

## Files

- `losscone_fit_comparison.png` — Side-by-side heatmaps
- `losscone_fit_comparison.txt` — Fit parameters (generated with `--no-stats-box`)

## Generation

```bash
uv run python scripts/plots/plot_losscone_fit_paper.py \
  --input data/1999/091_120APR/3D990429.TAB \
  --spec-no 653 \
  --output artifacts/paper/plots/measurement/losscone_fit/losscone_fit_comparison.png \
  --normalization ratio \
  --usc 11.0 \
  --dpi 200 \
  --no-stats-box
```

### Key options

| Option | Description |
|--------|-------------|
| `--input` | ER .TAB data file |
| `--spec-no` | Spectrum number to plot |
| `--usc` | Spacecraft potential [V] |
| `--normalization` | `global`, `ratio`, `ratio2`, `ratio_rescaled` |
| `--no-stats-box` | Save parameters to .txt instead of on-plot box |
| `--paper-mode` | Use Halekas Figure 5 settings |
