# Technical Concerns & TODOs

**Created:** 2025-11-16
**Status:** Open for review and prioritization

This document tracks technical concerns identified during code review that may need investigation or documentation before publication.

---

## 🔴 HIGH PRIORITY

### 1. No Uncertainty Propagation
**Suspicion Level:** 8/10
**Impact:** Cannot assess confidence in final potential maps

**Issue:**
- Uncertainties are calculated at each step but never propagated through the pipeline
- `FitResults.params_uncertainty` exists but is not used downstream
- Error chain: density errors → κ fit errors → spacecraft potential errors → surface potential errors

**Affected Files:**
- `src/kappa.py` - calculates uncertainties but they're not propagated
- `src/spacecraft_potential.py` - doesn't track error propagation
- `src/potential_mapper/pipeline.py` - no uncertainty in final results

**Action Items:**
- [ ] Implement Monte Carlo uncertainty propagation OR
- [ ] Implement analytical error propagation (if tractable) OR
- [ ] Document why uncertainties are not propagated and provide qualitative assessment

**Questions:**
- What's the typical fractional uncertainty in κ fits?
- How does spacecraft potential uncertainty affect surface potential?
- Can we estimate total uncertainty from sensitivity analysis?

---

### 2. FIT_ERROR_THRESHOLD Decision
**Suspicion Level:** 7/10
**Impact:** Determines which data is included in analysis

**Issue:**
- Current code: `2.15×10¹⁰` (very permissive, ~99th+ percentile)
- Statistical analysis: `2.15×10⁵` (95th percentile)
- **100,000× difference** between code and documentation

**Affected Files:**
- `src/config.py:68`
- `src/kappa.py:528`
- `src/potential_mapper/pipeline.py:276` (actively filters data)
- `docs/analysis/fitter_error_analysis.md` (Section 5)

**Action Items:**
- [ ] Run test analysis with strict threshold (215,000) to assess data loss
- [ ] Check spatial coverage impact
- [ ] Examine quality of marginal fits (chi² between 215k and 21.5B)
- [ ] Make explicit decision: conservative (215k), middle (657k), or permissive (21.5B)
- [ ] Update code and docs to match final decision

**Options:**
1. **215,000** (95th percentile) - Strict quality control, may lose 5% of data
2. **657,000** (99th percentile) - Middle ground, loses ~1% of data
3. **21,500,000,000** (current) - Maximum coverage, accepts questionable fits

---

## 🟡 MEDIUM PRIORITY


---

## 🟢 LOWER PRIORITY

### 5. Energy Response Matrix (asinh transform)
**Suspicion Level:** 6/10
**Impact:** May mis-model instrument energy resolution, biasing κ fits

**Issue:**
```python
s = asinh(0.5 * width) / sqrt(2 * ln(2))
```
- Non-standard transformation (typically Gaussian in linear or log-E)
- No validation against instrument response function measurements
- Fixed across all energies (energy-independent resolution)

**Affected Files:**
- `src/kappa.py:288-314`

**Action Items:**
- [ ] Document why asinh transformation was chosen
- [ ] Compare to standard Gaussian blur in log-energy
- [ ] Validate against instrument calibration data if available
- [ ] Test sensitivity: do fits change significantly with standard Gaussian?

**Questions:**
- Is this from prior work?
- How does this compare to Halekas et al. implementation?
- Do we have instrument response function data to validate?

---

### 6. Fixed κ Bounds
**Suspicion Level:** 5/10
**Impact:** May exclude physically valid spectra at extremes

**Issue:**
- Current bounds: `κ ∈ [2.5, 6.0]`
- Literature reports κ < 2.5 (very hard spectra) and κ > 6 (near-Maxwellian)
- May bias fits toward mid-range κ

**Affected Files:**
- `src/kappa.py:64`

**Action Items:**
- [ ] Check how often fits hit the bounds (would indicate range is too restrictive)
- [ ] Review literature for expected κ range in lunar environment
- [ ] Consider widening bounds if justified

**Questions:**
- Do fits ever hit κ = 2.5 or κ = 6.0?
- What fraction of spectra might have κ outside this range?

---

### 7. Temporal Regularization Tuning
**Suspicion Level:** 4/10
**Impact:** Results sensitive to analyst choices, not fully reproducible

**Issue:**
- Manual L-curve analysis required to choose λ_spatial, λ_temporal
- No automated cross-validation or GCV (Generalized Cross Validation)
- Different analysts might choose different regularization parameters

**Affected Files:**
- `src/temporal/coefficients.py:233-329`

**Action Items:**
- [ ] Document how λ values were chosen for published results
- [ ] Consider implementing automated parameter selection (GCV, L-curve automation)
- [ ] Provide sensitivity analysis: how do maps change with different λ?

**Questions:**
- How sensitive are results to regularization parameter choice?
- Can we automate this for reproducibility?

---

## ✅ RESOLVED/JUSTIFIED

### Isotropy Assumptions (×2 and ×0.25 factors)
**Status:** JUSTIFIED (2025-11-16)
**Justification:**
- ×2 factor: Uses field-aligned hemisphere to sample clean upstream plasma, assumes ambient isotropy
- ×0.25 factor: Standard textbook conversion, matches Halekas et al. 2008 approach
- Bulk flow corrections small (~5%) in typical lunar environment
- See discussion in project review 2025-11-16

### Binary Loss Cone Model
**Status:** JUSTIFIED (2025-11-16)
**Justification:**
- Halekas et al. 2008 also used binary (step function) loss cone
- Created visualization tool (`scripts/dev/plot_loss_cone_fit.py`) to compare observed vs. model
- Residual analysis shows no systematic problems at loss cone boundary
- Data itself exhibits fairly sharp transitions, supporting step function approximation
- See `temp/loss_cone_comparison.png` and `todos/session_2025-11-16.md` for details
- Can be documented as known limitation if needed for publication

### Beam Model Parameters
**Status:** JUSTIFIED (2025-11-17)
**Justification:**
- **beam_amp bounds [0, 100]**: Extensively tested via sensitivity analysis on 750 fits across 15 dates. Raising from 50→100 changes ΔU by only 16V RMS. See `docs/analysis/beam_amplitude_sensitivity.md`
- **beam_width factor (0.5)**: Measurement energy bins have width 0.5U, so factor of 0.5 provides reasonable energy spread (~half the bin width). Pragmatic choice based on instrument resolution.
- **beam_pitch_sigma (7.5°)**: Reasonable spread for upward-going beam. Chosen to represent realistic angular distribution of secondary electrons without over-fitting.

These are **empirical choices** (not derived from first principles), which is appropriate for a phenomenological model. The key parameter (beam_amp) was validated to not significantly affect ΔU estimates.

### SEE Parameters
**Status:** JUSTIFIED
**Justification:**
- Using same parameters as Halekas et al. 2008 (work being replicated)
- `sey_E_m = 500 eV`, `sey_delta_m = 1.5`
- Appropriate for comparative study

### Ti = Te Assumption
**Status:** JUSTIFIED
**Justification:**
- Matches prior work for measurement reasons
- Standard assumption in absence of ion measurements

---

## Publication Checklist

Before publication, ensure:

- [ ] **HIGH PRIORITY items** are either resolved or explicitly documented as limitations
- [ ] FIT_ERROR_THRESHOLD decision is made and justified
- [ ] Uncertainty propagation is implemented OR lack thereof is discussed in limitations
- [ ] All empirical parameters are documented (origin, justification, sensitivity)
- [ ] Methods section clearly states assumptions (isotropy, binary loss cone, etc.)
- [ ] Limitations section addresses unresolved concerns

---

## Notes

Add notes here as you investigate each item:

