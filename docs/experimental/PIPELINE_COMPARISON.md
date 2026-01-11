# Pipeline Comparison: Paper vs Implementation

## Overview

This document compares the paper's methodology with your current implementation.

---

## 1. Spacecraft Potential (U_SC) Determination

### Paper's Approach

#### Sunlight (Dayside):
1. **Two-component fit**: Photoelectrons (Maxwellian) + Plasma electrons (Kappa)
   ```
   f_photo(v) = n·(m/2πkT)^{3/2}·exp[-mv²/2kT]   (Maxwellian)
   f_plasma(v) = Γ(κ+1)/... · [1 + v²/κΘ²]^{-κ-1}  (Kappa)
   ```
2. **Convolution**: Convolve with instrument energy resolution (ΔE/E ≈ 0.5)
3. **Iterative fit**: Find break point in spectrum → U_SC
4. **J-U curve**: `J = A·exp(-U/B) + C·exp(-U/D)`
   - Coefficients: A=1.07e-6, B=5.0, C=1.6e-9, D=60.0 A/m²

#### Shadow (Nightside):
1. **Current balance**: `J_e + J_i - J_see = 0`
2. **Sternglass yield**: `δ(E) = 7.4·δ_m·(E/E_m)·exp(-2√(E/E_m))`
3. **Grid search**: Optimize δ_m and E_m to minimize flux discontinuities at terminator
4. **Best fit**: δ_m = 1.5, E_m = 500 eV

### Your Implementation (`src/spacecraft_potential.py`, `src/kappa.py`)

#### Sunlight:
1. **Single-component fit**: Kappa only ❌ (missing Maxwellian photoelectrons)
2. **Convolution**: ✅ Uses log-energy response matrix
3. **J-U inversion**: ✅ Matches paper's curve
4. **Energy pre-shift + refit**: ✅ Iterates to refine

#### Shadow:
1. **Current balance**: ✅ `Je + Ji - Jsee = 0`
2. **Sternglass yield**: ✅ Identical formula
3. **Parameters**: ✅ E_m=500, δ_m=1.5 (hardcoded, matches paper's best fit)
4. **Brent root-finding**: ✅ Solves for U

### Gap Analysis: U_SC

| Aspect | Paper | Your Code | Match? |
|--------|-------|-----------|--------|
| Daytime: Kappa fit | ✓ | ✓ | ✅ |
| Daytime: Photoelectron model | Maxwellian | Missing | ❌ |
| Daytime: Convolution | ΔE/E=0.5 | Response matrix | ✅ |
| Daytime: J-U curve | Double exp | Double exp | ✅ |
| Nightside: Current balance | ✓ | ✓ | ✅ |
| Nightside: Sternglass | ✓ | ✓ | ✅ |
| Nightside: SEY params | δ_m=1.5, E_m=500 | δ_m=1.5, E_m=500 | ✅ |

**Critical Missing Piece**: The photoelectron Maxwellian component in sunlight. The paper explicitly fits a two-component distribution to separate low-energy photoelectrons from higher-energy plasma electrons. Your code fits only Kappa, which may conflate the two populations.

---

## 2. Lunar Surface Potential (U_M) Determination

### Paper's Approach

1. **Normalization**: "Dividing reflected flux by incident flux"
2. **Synthetic distribution**:
   - Loss cone: `sin²α_c = (B_S/B_M)·(1 + eU_M/(E - eU_SC))`
   - Secondary beam: centered at `E_beam = e(U_M - U_SC)`
3. **Fitting**: Least squares on log(synthetic) vs log(measured)
4. **Parameters**: Grid search over B_S/B_M and U_M

### Your Implementation (`src/flux.py`, `src/model.py`)

1. **Normalization**: Multiple modes (global/ratio/ratio2/ratio_rescaled)
   - `ratio` = per-energy normalization by max incident ← closest working
   - `ratio2` = pairwise incident/reflected ← closest to paper, but unstable
2. **Synthetic distribution**:
   - Loss cone: ✅ Same formula
   - Secondary beam: ✅ Centered at U_SC - U_surface (suppressed if U_SC <= U_surface)
   - Extra: beam_amp as free parameter (paper doesn't mention)
3. **Fitting**: ✅ Log-space least squares
4. **Optimizer**: Differential evolution (global)

### Gap Analysis: U_M

| Aspect | Paper | Your Code | Match? |
|--------|-------|-----------|--------|
| Loss cone formula | Pitch-conserving | Pitch-conserving | ✅ |
| Beam center | U_M - U_SC | U_SC - U_surface (suppressed if U_SC <= U_surface) | ✅ |
| Beam amplitude | Not mentioned | Free parameter | ⚠️ |
| Normalization | reflected/incident | Multiple modes | ⚠️ |
| Log least squares | ✓ | ✓ | ✅ |
| Background value | Not specified | 0.05 | ❓ |
| Fit parameters | 2 (B_S/B_M, U_M) | 3 (+beam_amp) | ⚠️ |

---

## 3. Identified Issues

### High Priority

1. **Missing photoelectron model** (U_SC daytime)
   - Impact: U_SC in sunlight may be inaccurate
   - Location: `src/kappa.py`, `src/spacecraft_potential.py`
   - Fix: Add Maxwellian component for low energies, fit both

2. **Normalization mismatch** (U_M fitting)
   - Impact: "ratio2" mode closest to paper but unstable
   - Location: `src/flux.py:434-473`
   - Fix: Debug ratio2 or understand what paper actually does

### Medium Priority

3. **Extra beam_amp parameter**
   - Impact: May compensate for other model errors
   - Test: Fix beam_amp and see if U_M improves

4. **B_S/B_M not constrained**
   - Paper may have used magnetometer data
   - Test: Try constraining B_S/B_M near 1.0 for plasma sheet

### Low Priority

5. **Background value (0.05)**
   - Empirical choice, affects χ² surface
   - Paper doesn't specify their value

---

## 4. Recommendations

### Immediate (for current fitting)
- Use `ratio` normalization (working, gets U_M ≈ -200V)
- Consider fixing beam_amp to reduce degrees of freedom
- Accept ~20-30% discrepancy from paper as "different implementation"

### Future improvements
1. Add Maxwellian photoelectron model for daytime U_SC
2. Investigate ratio2 normalization failure
3. Consider external B_S/B_M constraint from magnetometer data
4. Compare U_SC values with paper's reported values if available

---

## 5. Test Results Summary

**Spec 653** (Paper Figure 5: U_M = -160V, B_S/B_M = 0.975, U_SC = +11V)

| Configuration | U_M | B_S/B_M | Notes |
|--------------|-----|---------|-------|
| Paper | -160 V | 0.975 | Reference |
| ratio + fixed beam=0 | -205 V | 0.873 | 28% off |
| ratio + free beam | -186 V | ~1.0 | 16% off, Bs/Bm at bound |
| ratio2 + bg=0.01 | -173 V | 0.378 | 8% off U_M, wrong B_S/B_M |

Visual fit quality with `ratio` normalization is good — the loss cone boundary tracks the data. Parameter discrepancy may be acceptable given implementation differences.
