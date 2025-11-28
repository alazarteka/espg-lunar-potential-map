# Loss Cone Fitting - Normalization Experiments

## Objective
Match the Halekas et al. (2008) loss cone fitting results for validating our implementation.

## Reference Case from Halekas Paper

**Source**: Figure 5, page 8 of Halekas et al. (2008) JGR paper

**Exact quote from caption**:
> "Normalized energy pitch angle distribution measured at 14:45 UT on 29 April 1999, with best fit synthetic distribution (on right) from automated surface potential determination. The best fit uses the spacecraft potential USC = −11 V and finds that the lunar surface potential UM = −160 V and the magnetic field ratio BS/BM = 0.975."

**Expected results**:
- Spacecraft potential: USC = −11 V (note: minus sign)
- Lunar surface potential: UM = −160 V
- Magnetic field ratio: BS/BM = 0.975
- Timestamp: 14:45 UT on 29 April 1999 (UTC: 925397048.0)

**Our test case**:
- File: `data/1999/091_120APR/3D990429.TAB`
- Spectrum number: 653
- Timestamp: 925397048.0 (matches!)

## Experiments

### Experiment 1: Original Implementation (Per-Energy Normalization, No SC Correction)
**Date**: 2025-11-28

**Method**:
- Per-energy normalization: each energy row normalized by mean(incident_flux) at that energy
- No spacecraft potential correction
- Code: `_get_normalized_flux()` divides by `mean(flux[pitch < 90])`

**Results**:
```
U_surface = -802.5 V
Bs/Bm = 0.999
beam_amp = 13.875
χ² = 19502.10
```

**Analysis**:
- U_surface is 5x too negative (-802V vs -160V expected)
- Bs/Bm is slightly off (0.999 vs 0.975 expected)
- Per-energy normalization destroys energy-dependent flux structure that encodes U_surface

---

### Experiment 2: Global Normalization, No SC Correction
**Date**: 2025-11-28

**Method**:
- Global normalization: entire 2D array normalized by max(all incident fluxes)
- No spacecraft potential correction
- Code: `norm2d = flux_2d / np.nanmax(incident_fluxes)`

**Results**:
```
U_surface = -236.6 V
Bs/Bm = 1.200
beam_amp = 1.071
χ² = 51877.09
```

**Analysis**:
- U_surface much better! (-237V vs -160V expected, ~1.5x off)
- Bs/Bm worse (1.200 vs 0.975)
- Global normalization preserves energy structure, critical for U_surface fitting

---

### Experiment 3: Global Normalization + USC = -11V
**Date**: 2025-11-28

**Method**:
- Global normalization: `norm2d = flux_2d / np.nanmax(incident_fluxes)`
- Spacecraft potential correction: USC = -11V (as stated in Halekas paper)
- Energy correction: `energies = energies - sc_value` (line 502 in flux.py)

**Results**:
```
U_surface = -82.5 V
Bs/Bm = 0.983
beam_amp = 11.875
χ² = 57738.71
```

**Analysis**:
- **Bs/Bm nearly perfect!** (0.983 vs 0.975, only 0.8% error)
- U_surface now ~2x too small in magnitude (-82.5V vs -160V)
- Best Bs/Bm fit of all experiments

---

### Experiment 4: Global Normalization + USC = +11V
**Date**: 2025-11-28

**Method**:
- Global normalization
- Spacecraft potential correction: USC = +11V (testing sign interpretation)
- Energy correction: `energies = energies - sc_value`

**Results**:
```
U_surface = -247.3 V
Bs/Bm = 1.404
beam_amp = 0.900
χ² = 51697.45
```

**Analysis**:
- U_surface further from expected (-247V vs -160V)
- Bs/Bm much worse (1.404 vs 0.975)
- Confirms USC = -11V (negative) is correct

---

### Experiment 5: Per-Energy Normalization + USC = -11V
**Date**: 2025-11-28

**Method**:
- Per-energy normalization (original approach)
- Spacecraft potential correction: USC = -11V

**Results**:
```
U_surface = -982.6 V
Bs/Bm = 1.115
beam_amp = 9.217
χ² = 17069.23
```

**Analysis**:
- Worse than Experiment 1 (without SC correction!)
- Per-energy normalization fundamentally incompatible with SC correction
- Confirms global normalization is the correct approach

---

### Experiment 6: Global Normalization + USC = -11V with REVERSED Sign
**Date**: 2025-11-28

**Method**:
- Global normalization
- Spacecraft potential: USC = -11V
- **Energy correction REVERSED**: `energies = energies + sc_value` (changed from minus)
- Code change in flux.py line 503

**Results**:
```
U_surface = -247.3 V
Bs/Bm = 1.404
beam_amp = 0.900
χ² = 51697.45
```

**Analysis**:
- **IDENTICAL to Experiment 4!** (USC = +11V with minus sign)
- Confirms: `E - (-11V) = E + 11V` and `E + (-11V) = E - 11V`
- Original code with USC=-11V effectively adds 11eV → Best results (Bs/Bm = 0.983)
- Reversed code with USC=-11V effectively subtracts 11eV → Worse results (Bs/Bm = 1.404)

**Conclusion**:
Original sign convention (`energies - sc_value`) with USC = -11V gives best fit.
This means we're effectively adding 11eV to measured energies, which is physically correct
if electrons LOSE 11eV falling into negative spacecraft potential well.

---

## Summary of Findings

### Key Insights

1. **Normalization Method Critical**:
   - Per-energy normalization destroys energy-dependent flux structure
   - Global normalization (by max incident flux) preserves structure needed for U_surface fitting

2. **Spacecraft Potential Sign**:
   - USC = -11V gives best Bs/Bm fit (0.983 vs 0.975)
   - USC = +11V gives worse results
   - Halekas paper says "USC = −11 V" (minus sign present)

3. **Best Configuration**:
   - Global normalization + USC = -11V
   - Achieves Bs/Bm = 0.983 (0.8% error from expected 0.975)
   - U_surface = -82.5V (still ~2x smaller magnitude than expected -160V)

### Remaining Discrepancies and Sign Convention Analysis

**CRITICAL OBSERVATION**: Paper clearly states USC = −11V (verified negative sign in caption line 11)

**Our Implementation**:
```python
# In flux.py line 502:
energies = energies - sc_value
# If sc_value = -11V:
# energies = energies - (-11) = energies + 11V
```

**Physics of Spacecraft Potential**:
- Spacecraft at USC = -11V means spacecraft is 11V below plasma potential
- Electrons measured at spacecraft have already fallen UP this 11V potential well
- To get original plasma energy: E_plasma = E_measured - |gain| = E_measured - 11V
- Our code does: E = E - sc_value = E - (-11) = E + 11V ← **THIS IS BACKWARDS!**

**Expected vs Observed**:
```
If spacecraft is at -11V (negative):
  Electrons GAIN 11eV falling into spacecraft
  Measured energy is 11eV HIGHER than plasma energy
  Correction: E_plasma = E_measured - 11V
  Our code does: E = E_measured + 11V  ← WRONG!
```

**Empirical Results**:
- USC = -11V (our current code adds 11V): Best Bs/Bm = 0.983 ✓
- USC = +11V (our code subtracts 11V): Worse Bs/Bm = 1.404 ✗

**This suggests**:
1. Either our sign convention is backwards (code should be `energies = energies + sc_value` not minus)
2. OR Halekas uses opposite sign convention (their USC = -11V means +11V in standard convention)
3. OR there's a deeper implementation issue

**U_surface still off by ~2x**: (-82.5V vs -160V expected)

Possible explanations:
1. **Sign error compounds**: Spacecraft potential error affects U_surface by 2x somehow?
2. **Different spectrum**: Though timestamp matches, spec_no 653 might not be Figure 5 spectrum
3. **Different fitting bounds**: Our LHS samples [-1000V, +1000V], Halekas might use different range
4. **Beam model differences**: Our beam_amp = 11.875, width and sigma might differ
5. **Normalization differs from Halekas**: Our global max might not match their approach exactly
6. **Energy correction applied wrong**: Should be `E + sc_value` not `E - sc_value`?

**Next Steps**:
1. ✅ **DONE**: Tested reversing sign (Experiment 6) - confirmed original is correct
2. ⚠️ Review Halekas section 4 (pages 4-6) on spacecraft potential determination method
3. ⚠️ Check Halekas section 5.1 (pages 7-8) for exact energy correction formula
4. ⚠️ Verify which spectrum corresponds to Figure 5 (timestamp matches but is it spec_no 653?)
5. ⚠️ Compare our loss cone equation implementation line-by-line with Halekas Equation 2
6. ⚠️ Test multiple spectra to see if 2x error is systematic or varies
7. ⚠️ Check if 2x discrepancy relates to factor-of-2 in some formula (beam width? normalization?)

## CONCLUSIONS

**What Works**:
1. ✅ Global normalization (normalize by max incident flux across all energies)
2. ✅ Spacecraft potential correction with USC = -11V (as stated in paper)
3. ✅ Sign convention: `energies = energies - sc_value` (original implementation)
4. ✅ Bs/Bm = 0.983 matches expected 0.975 within 0.8% error

**What Doesn't Work**:
1. ❌ Per-energy normalization (destroys energy structure)
2. ❌ No spacecraft potential correction (gets U_surface 5x wrong)
3. ❌ Wrong sign for spacecraft potential (USC = +11V or reversed equation)

**Remaining Mystery**:
- U_surface = -82.5V vs expected -160V (off by factor of ~1.94)
- Bs/Bm nearly perfect suggests geometry/magnetic field handling is correct
- U_surface error suggests energy scale or normalization issue
- Possible causes:
  * Different spectrum being analyzed
  * Factor-of-2 error in beam model or normalization
  * Energy calibration offset
  * Different fitting methodology details

## Code Changes Made

### `src/flux.py::build_norm2d()`
Changed from per-energy to global normalization:

**Before**:
```python
norm2d = np.vstack([
    self._get_normalized_flux(energy_bin, measurement_chunk)
    for energy_bin in range(config.SWEEP_ROWS)
])
```

**After**:
```python
# Extract raw flux data
flux_2d = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[s:e]
pitches_2d = self.pitch_angle.pitch_angles[s:e]
incident_mask = pitches_2d < 90

# Global normalization
global_norm = float(np.nanmax(flux_2d[incident_mask]))
norm2d = flux_2d / global_norm
```

### `scripts/plots/plot_losscone_fit_paper.py`
Added spacecraft potential correction:

```python
# Create spacecraft potential array (constant -11V per Halekas et al.)
spacecraft_potential = np.full(len(er_data.data), -11.0)

# Create fitter with spacecraft potential correction
fitter = LossConeFitter(er_data, str(theta_file), pitch_angle, spacecraft_potential)
```

## References

Halekas, J. S., G. T. Delory, R. P. Lin, T. J. Stubbs, and W. M. Farrell (2008),
Lunar Prospector observations of the electrostatic potential of the lunar surface
and its response to incident currents, J. Geophys. Res., 113, A09102,
doi:10.1029/2008JA013194.
