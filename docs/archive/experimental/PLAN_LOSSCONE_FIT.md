# Loss Cone Fitting Improvement Plan

## Current State

**Reference case**: Spec 653 from April 29, 1999 (paper Figure 5)
- Paper values: U_surface = -160V, Bs/Bm = 0.975, U_SC = +11V
- Our best result (ratio norm): U_surface = -186V, Bs/Bm = 1.000, beam_amp = 1.634

**Gap**: 16% error on U_surface, Bs/Bm hitting upper bound

## Architecture Overview

```
Raw Flux Data (15 energies × 16 pitch angles)
    ↓
Normalization (ratio/global/ratio2/ratio_rescaled)
    ↓
norm2d array [0, ~2] or [0, 1]
    ↓
Log-space χ² = Σ(log(norm2d) - log(model))²
    ↓
differential_evolution optimizer
    ↓
Best fit: (U_surface, Bs/Bm, beam_amp)
```

## Hypotheses

### H1: Background Value Too High (0.05)
The model sets `model[outside_loss_cone] = 0.05` (5% of peak). In log-space:
- log(0.05) ≈ -3.0
- log(0.001) ≈ -6.9

A 0.05 background means mismatching the loss cone region costs less in χ². The optimizer may not be penalized enough for wrong boundary placement.

**Test**: Try background = 0.01, 0.001, and compare fitted values.

### H2: Beam Amplitude Absorbing Model Error
beam_amp is a free parameter (fitted to 1.634). The paper doesn't mention fitting beam amplitude — they may derive it from physics or use a fixed value. An unconstrained beam_amp can compensate for:
- Normalization differences
- Wrong loss cone shape
- Background level issues

**Test**: Fix beam_amp to values like 0.5, 1.0, 2.0 and see how U_surface changes.

### H3: Max vs Mean Normalization
We changed from mean to max for incident flux normalization:
```python
# Current (max):
incident_flux = np.max(electron_flux[incident_mask])

# Previous (mean):
incident_flux = np.mean(electron_flux[incident_mask])
```

The paper says "dividing reflected by incident" but doesn't specify max/mean. Max is more robust to outliers but may over-normalize.

**Test**: Revert to mean and compare.

### H4: Bs/Bm Bound Issue
Bs/Bm = 1.0 means no magnetic focusing (B_spacecraft = B_moon). The paper reports 0.975, suggesting slight focusing. Our optimizer hitting 1.0 could mean:
- The loss cone in data is slightly larger than model predicts
- Compensating for other model errors

**Test**: Extend bound to 1.05 or 1.1 to see if optimizer wants more.

### H5: Per-Pitch-Angle Normalization (Paper's Actual Method)
The paper says "dividing the reflected flux by the incident flux" which might mean:
- For each pitch angle pair (θ, 180°-θ), divide reflected by incident
- This is closest to our "ratio2" mode, which currently fails

**Test**: Debug ratio2 mode — why does it produce χ² = 1e30?

## Experiments

### Experiment 1: Background Sweep
```bash
# Modify src/model.py line 112, test each value:
for bg in 0.05 0.01 0.005 0.001; do
    # Edit background value
    uv run python scripts/dev/plot_losscone_3panel.py \
        --input data/1999/091_120APR/3D990429.TAB \
        --spec-no 653 --usc 11.0 --normalization ratio \
        --output artifacts/bg_test_${bg}.png
done
```

**Expected outcome**: Lower background → U_surface closer to -160V, Bs/Bm < 1.0

### Experiment 2: Fixed Beam Amplitude
```bash
for beam in 0.0 0.5 1.0 2.0; do
    uv run python scripts/dev/plot_losscone_3panel.py \
        --input data/1999/091_120APR/3D990429.TAB \
        --spec-no 653 --usc 11.0 --normalization ratio \
        --fixed-beam-amp $beam \
        --output artifacts/beam_test_${beam}.png
done
```

**Expected outcome**: Understand beam_amp's effect on U_surface recovery

### Experiment 3: Mean vs Max Normalization
Modify `src/flux.py` line 402:
```python
# Try mean instead of max:
incident_flux = float(np.mean(electron_flux[incident_mask]))
```

Run same test case and compare.

### Experiment 4: Debug ratio2 Mode
The ratio2 normalization produces χ² = 1e30 (failure). Investigate:
1. Check if norm2d contains NaN/Inf
2. Check if pitch angle pairing is correct
3. Visualize the normalized data before fitting

### Experiment 5: Extend Bs/Bm Bound
Modify `src/flux.py` line 631:
```python
bounds = [
    (-2000.0, 2000.0),  # U_surface
    (0.1, 1.1),          # bs_over_bm - extend upper bound
    ...
]
```

## Implementation Order

1. **Experiment 1** (background sweep) — most likely to improve results
2. **Experiment 2** (fixed beam) — understand parameter coupling
3. **Experiment 3** (mean vs max) — low effort, may help
4. **Experiment 4** (debug ratio2) — could be the "correct" approach per paper
5. **Experiment 5** (extend bounds) — diagnostic, not a fix

## Success Criteria

- U_surface within 10% of paper value (-160V ± 16V)
- Bs/Bm not hitting bounds (should be ~0.975)
- χ² should be reasonable (not 1e30, not too high)
- Visual match: loss cone boundary should track data in 3-panel plot

## Files to Modify

| File | Line | Change |
|------|------|--------|
| `src/model.py` | 112 | Background value |
| `src/flux.py` | 402 | max → mean normalization |
| `src/flux.py` | 631 | Bs/Bm upper bound |
| `src/flux.py` | 434-473 | Debug ratio2 logic |

## Notes

- Always test on spec 653 first (known ground truth)
- Keep artifacts for comparison
- Commit working improvements incrementally
- The paper's "blurred" loss cone edge suggests real data won't perfectly match the sharp model — some χ² residual is expected
