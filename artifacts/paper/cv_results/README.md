# Temporal Basis Cross-Validation Results

Cross-validation results for the temporal basis fitting, testing model generalization.

## Files

- `april_1998_cv.txt` — April 1998 single-config CV
- `july_1998_cv.txt` — July 1998 single-config CV
- `april_1998_grid.csv` — April 1998 grid search (10 configs)
- `july_1998_grid.csv` — July 1998 grid search (10 configs)
- `december_1998_grid.csv` — December 1998 grid search (10 configs)

## Generation

```bash
# Single configuration
uv run python scripts/analysis/temporal_cv.py \
  --start 1998-04-01 --end 1998-04-30 \
  --lmax 10 --temporal-basis "constant,synodic,synodic2,synodic3"

# Grid search
uv run python scripts/analysis/temporal_cv_grid.py \
  --start 1998-04-01 --end 1998-04-30 \
  --lmax 10 --output artifacts/paper/cv_results/april_1998_grid.csv
```

## Grid Search Summary (Best R² per month)

| Month | Best Basis | R² | Skill | Overfit |
|-------|-----------|-----|-------|---------|
| April | synodic+sidereal+2nd harmonics | 0.115 | +5.9% | 1.067 |
| July | synodic 1-4 harmonics | 0.143 | +7.4% | 1.098 |
| December | synodic+sidereal+2nd harmonics | 0.170 | +8.9% | 1.042 |

### Key Findings

- All configurations show overfit ratio < 1.1 (good generalization)
- Higher-order synodic harmonics generally improve fit
- Combined synodic+sidereal performs comparably to synodic-only
- December shows best overall R² (0.170)
