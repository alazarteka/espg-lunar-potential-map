# Beam Amplitude Sensitivity Analysis

**Date:** 2025-11-17
**Analysis Script:** `scripts/dev/test_beam_amp_sensitivity.py`
**Status:** RESOLVED - Updated `LOSS_CONE_BEAM_AMP_MAX` from 50 to 100

---

## Background

The loss cone fitter in `src/flux.py` fits three free parameters to match observed electron distributions:
1. **U_surface** - Surface potential (primary science output)
2. **Bs/Bm** - Magnetic field ratio
3. **beam_amp** - Amplitude of secondary electron beam

During initial testing, we observed that ~25% of fits were hitting the upper bound on `beam_amp` (originally set to 50). This raised concerns about:
- Physical meaning of high beam amplitudes
- Whether the bound was artificially constraining fits
- Sensitivity of U_surface estimates to beam_amp parameter choice

## Investigation Method

We conducted a systematic sensitivity test comparing four fitting strategies:

1. **Free beam_amp (0-50)** - Original implementation
2. **Fixed beam_amp=25** - Fixed to median value
3. **No beam (beam_amp=0)** - Binary loss cone only
4. **High bound (0-100)** - Test if fits want higher values

### Test Parameters

- **Dates tested:** 15 randomly selected files (seed=42 for reproducibility)
- **Chunks per date:** 50 (first 50 successful fits from each file)
- **Total samples:** 750 fits
- **Date range:** March 1998 - July 1999

### Selected Files

```
data/1998/060_090MAR/3D980320.TAB
data/1998/060_090MAR/3D980325.TAB
data/1998/060_090MAR/3D980327.TAB
data/1998/060_090MAR/3D980330.TAB
data/1998/091_120APR/3D980405.TAB
data/1998/091_120APR/3D980410.TAB
data/1998/121_151MAY/3D980521.TAB
data/1998/121_151MAY/3D980528.TAB
data/1998/182_212JUL/3D980723.TAB
data/1998/213_243AUG/3D980812.TAB
data/1998/335_365DEC/3D981204.TAB
data/1999/001_031JAN/3D990105.TAB
data/1999/060_090MAR/3D990330.TAB
data/1999/182_212JUL/3D990706.TAB
data/1999/182_212JUL/3D990714.TAB
```

## Results

### Baseline (Free beam_amp, bounds 0-50)

- Samples: 750
- **beam_amp hitting upper bound (≥49.9): 185 (24.7%)**
- beam_amp mean: 24.7 V
- beam_amp median: 20.8 V

### Strategy Comparison: U_surface Differences

| Strategy | Mean Δ(U_surface) | Std Δ(U_surface) | Max Δ(U_surface) | **RMS Δ(U_surface)** |
|----------|------------|-----------|-----------|---------------|
| Fixed beam_amp=25 | -5.4 V | 124.4 V | 1,211 V | **124.5 V** |
| No beam (=0) | -3.1 V | 238.0 V | 1,866 V | **238.0 V** |
| **High bound (0-100)** | **1.1 V** | **15.9 V** | **240 V** | **15.9 V** |

### High Bound Strategy Details

- beam_amp hitting new upper bound (≥99.9): 74 (9.9%)
- **beam_amp > 50: 179 (23.9%)**
- Median χ² ratio: 1.00 (no degradation in fit quality)

### Bs/Bm Sensitivity

| Strategy | Mean Δ(Bs/Bm) | Std Δ(Bs/Bm) | Max Δ(Bs/Bm) |
|----------|---------------|--------------|--------------|
| Fixed beam_amp=25 | -0.0022 | 0.537 | 5.83 |
| No beam (=0) | -0.0572 | 0.739 | 7.87 |
| High bound (0-100) | 0.0111 | 0.310 | 8.07 |

## Interpretation

### Why beam_amp wants to be large

The beam amplitude parameter acts as a **residual absorber** for features the binary loss cone model cannot capture:

1. **Partial loss cone filling** - Real loss cones may not be sharp step functions
2. **Energy-dependent effects** - Physics not captured by simple model
3. **Non-Gaussian beam shapes** - Secondary electron distribution may be more complex
4. **Other physical processes** - Additional electron populations

The fact that ~10% of fits still hit even a bound of 100 suggests the model has **systematic limitations**, not just that we chose the bound too low.

### Impact on U_surface (primary science output)

The **key finding** is that raising the bound from 50 to 100:
- Changes U_surface estimates by only **15.9 V RMS**
- This is **small compared to typical U_surface values** (100-1000 V)
- Max difference (240 V) occurs in <1% of cases
- **No degradation in fit quality** (χ² ratio = 1.00)

This demonstrates that **U_surface is relatively robust** to beam_amp parameter choice, as long as we don't over-constrain it.

### Why not fix beam_amp or remove it?

- **Fixed beam_amp=25**: RMS difference of 124 V is too large
- **No beam (beam_amp=0)**: RMS difference of 238 V, and χ² degrades
- Beam component **does improve fits** and **does matter** for U_surface accuracy

## Decision

**Update `LOSS_CONE_BEAM_AMP_MAX` from 50 to 100** because:

1. ✅ **Minimal impact on science**: Only 16 V RMS change in U_surface
2. ✅ **Removes artificial constraint**: ~24% of fits wanted beam_amp > 50
3. ✅ **Still bounded**: Prevents completely unphysical values
4. ✅ **Maintains fit quality**: No increase in χ²
5. ✅ **More honest**: The ~10% that still hit 100 are a warning flag (documented below)

## Limitations and Caveats

### Model Limitations

The fact that ~10% of fits still hit the upper bound of 100 indicates that the **binary loss cone + Gaussian beam model** may be too simple for some conditions. This could reflect:

- **Overfitting**: beam_amp compensating for model inadequacies
- **Physical complexity**: Real electron distributions are more complex than our model
- **Unknown physics**: Processes we haven't accounted for

### Not documented in Halekas et al. 2008

A review of Halekas et al. (2008) found **no mention of fitting beam amplitude as a free parameter**. Their methodology (Section 5.1) describes:

1. Creating synthetic distribution with loss cone based on U_surface and Bs/Bm
2. Adding an upward-going beam centered at energy U_surface - USC
3. Fitting for U_surface and Bs/Bm using least squares

They may have:
- Used beam **position** (energy) to determine U_surface
- But not fitted beam **amplitude** at all
- Fixed beam amplitude to some reasonable value

This suggests our addition of beam_amp as a third fitted parameter goes **beyond the published methodology** and should be carefully validated.

### Uncertainty Estimate

Based on this analysis, we estimate **beam_amp parameter choice contributes ~16 V RMS uncertainty** to U_surface estimates. This should be:

- Propagated through uncertainty analysis (if implemented)
- OR documented as a systematic uncertainty in publications
- Compared to other error sources (spacecraft potential uncertainty, model assumptions, etc.)

## Recommendations for Future Work

1. **Validate against Halekas methodology** - Try fitting with fixed beam_amp to match their approach more closely

2. **Test alternative beam models**:
   - Power-law instead of Gaussian
   - Dual-component beams
   - Energy-dependent beam parameters

3. **Investigate the ~10% that hit bound=100**:
   - Are they specific plasma conditions?
   - Specific energy ranges?
   - Particular times/locations?

4. **Consider removing beam from model** if:
   - Main science questions don't require it
   - Simpler model is more defensible
   - Fixed beam_amp proves adequate after further testing

5. **Cross-validate** - Compare U_surface estimates with/without beam on same data

## Running the Sensitivity Test

To reproduce this analysis:

```bash
# Single file (quick test)
uv run python scripts/dev/test_beam_amp_sensitivity.py \
    --er-file data/1998/060_090MAR/3D980323.TAB \
    --chunks-per-date 100 \
    --output scratch/beam_amp_sensitivity_single.png

# Multi-date (full analysis)
uv run python scripts/dev/test_beam_amp_sensitivity.py \
    --n-dates 15 \
    --chunks-per-date 50 \
    --output scratch/beam_amp_sensitivity_multidate.png
```

The script compares four strategies and outputs:
- Statistical summary
- Comparison plots (saved to `--output`)
- Recommendation based on RMS differences

## References

- Halekas, J. S., et al. (2008), Lunar Prospector observations of the electrostatic potential of the lunar surface and its response to incident currents, *J. Geophys. Res.*, 113, A09102, doi:10.1029/2008JA013194
- Analysis performed: 2025-11-17
- Configuration updated: `src/config.py:72`
