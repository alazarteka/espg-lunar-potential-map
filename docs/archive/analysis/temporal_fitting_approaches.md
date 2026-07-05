# Temporal Spherical Harmonic Fitting of Lunar Surface Potential

## Problem Context

We are reconstructing time-varying global maps of the lunar surface electrostatic potential using data from Lunar Prospector (LP). The goal is to produce moon-fixed potential maps that vary with time, capturing both the solar-driven day/night pattern and any genuine temporal variations.

---

## Data Characteristics

### Source
- **Lunar Prospector (LP)** electron reflectometer measurements  
- **Time range**: July 1998 (31 days analyzed)
- **Total measurements**: ~138,000 potential readings with (lat, lon, time)

### Orbital Geometry
LP flies in a **polar, inertially-fixed orbit** with:
- Orbital period: ~118 minutes
- Coverage: All latitudes sampled per orbit  
- **Key constraint**: Each day samples only a **~13° longitude band** in moon-fixed coordinates

### Sparse Sampling Problem
The moon rotates at ~13.2°/day (sidereal), while LP's orbit precesses slowly in solar local time (~1°/day). This creates a fundamental observational challenge:

```
Day 1:  Samples longitude 170°-183°   (13° band)
Day 2:  Samples longitude 157°-170°   (13° band)
Day 10: Samples longitude 40°-53°     (13° band)
...
Full 360° coverage takes ~27 days (one lunar sidereal month)
```

**Result**: Each 24-hour window severely under-samples the globe. A spherical harmonic expansion to degree L=10 has 121 coefficients, but we only observe ~13% of the surface per window.

### Physical Model
The surface potential has multiple contributors:
1. **Solar-driven pattern**: Dayside/nightside charging (dominant, ~synodic period = 29.53 days)
2. **Lunar geology**: Magnetic anomalies, surface composition (moon-fixed)
3. **Temporal transients**: Solar wind variations, SEP events

---

## Fitting Approaches Explored

### Approach 1: Per-Window Least Squares (Baseline)

**Method**: Fit each 24-hour window independently using:
```
Φ(lat, lon) = Σ_lm a_lm × Y_lm(lat, lon)
```

**Results**:
- Very high RMS residuals (~500-1100 V per window)
- **Massive extrapolation artifacts**: The fit is well-constrained only in the observed 13° band
- Unobserved regions show wild oscillations (spherical harmonic ringing)

**Verdict**: ❌ Not usable for global maps

---

### Approach 2: Temporal Regularization (First Enhancement)

**Method**: Fit all windows jointly with a penalty on coefficient changes between adjacent windows:
```
minimize ||Y @ a - Φ||² + λ_t × Σ_i ||(a_{i+1} - R_Δφ × a_i)||²
```
where `R_Δφ` optionally rotates coefficients to account for solar frame co-rotation.

**Multi-scale extension**: Connect windows at lags 1, 2, ..., N with exponentially decaying weights:
```
Weight at lag k = decay^(k-1)
```

**Results**:
- RMS residuals: 616-1122 V (slight improvement)
- Still shows **rotating stripe artifacts** in maps
- Regularization smooths coefficient evolution but doesn't propagate spatial information between windows

**Interpretation**: The problem is that each window's design matrix `Y` only covers ~13° of longitude. Regularizing coefficients doesn't magically teach window 1 about what window 10 observed.

**Verdict**: ⚠️ Marginal improvement, fundamental limitation remains

---

### Approach 3: Temporal Basis Functions ✅ (Current Best)

**Method**: Instead of fitting N_windows × N_coeffs parameters, parameterize the coefficient evolution with a small set of temporal basis functions:
```
a_lm(t) = Σ_k b_lmk × T_k(t)
```

**Available bases**:
- `constant`: T(t) = 1 (static pattern)
- `linear`: T(t) = t/T_total (secular drift)
- `synodic`: T_cos(t) = cos(2πt/T_syn), T_sin(t) = sin(2πt/T_syn) (solar-driven oscillation)

**Key insight**: ALL 138,000 measurements constrain the same K × (L_max+1)² parameters simultaneously. Over the month, the data covers ≈360° of longitude in the solar frame, making the problem well-posed.

**Results with lmax=20, basis=constant+synodic** (1323 parameters):
- RMS residual: **647 V** (best so far)
- Maps show smooth, physically plausible structure
- No extrapolation artifacts!
- Color scale: -2799 to +521 V (reasonable physical range)

**Results with lmax=20, basis=constant only** (441 parameters):
- RMS residual: 708 V
- Produces a single time-averaged map
- Color scale: -1825 to -475 V

**Implementation details**:
- Dense design matrix: 137,892 × 1,323 = 182 million complex elements
- Solved using `scipy.sparse.linalg.lsqr` with `damp` parameter for ridge regularization
- Solve time: ~70 seconds for lmax=20

**Verdict**: ✅ Best approach so far. Clean maps, physically interpretable.

---

### Approach 4: Kalman Filter/Smoother

**Method**: Model coefficients as a hidden state evolving according to stochastic dynamics:
```
State:     a(t+Δt) = F(Δt) × a(t) + w,   w ~ N(0, Q×Δt)
Obs:       Φ(t)    = H(t) × a(t) + v,   v ~ N(0, R)
```
where:
- `F(Δt)` = state transition (rotation for co-rotating solar frame)
- `H(t)` = spherical harmonic design matrix (different per window!)
- `Q` = process noise (allows temporal variation)
- `R` = measurement noise

**Implementation**:
- Forward Kalman filter updates state with each window's observations
- RTS backward smoother (Rauch-Tung-Striebel) for optimal bidirectional estimates

**Results**:
- Forward-only Kalman: Innovation RMS 534-1572 V, stable coefficients (max ~9000 V)
- RTS smoother: **Numerical instability** - coefficients explode to 10²⁷ V
  - Root cause: Backwards pass propagates instabilities from poorly-constrained early windows

**Tuned parameters**: `--process-noise 0.01 --measurement-noise 1000.0 --no-kalman-smooth`

**Verdict**: ⚠️ Forward-only works but less accurate than temporal basis. RTS smoother needs debugging.

---

## Summary Comparison

| Approach | Parameters | RMS (V) | Map Quality | Status |
|----------|------------|---------|-------------|--------|
| Per-Window LS | 27×121 = 3,267 | 500-1100 | ❌ Artifacts | Baseline |
| Temporal Regularization | 27×121 = 3,267 | 616-1122 | ⚠️ Rotating stripe | Insufficient |
| **Temporal Basis (synodic)** | 3×441 = 1,323 | **647** | ✅ Clean | **Best** |
| Temporal Basis (constant) | 1×441 = 441 | 708 | ✅ Clean (static) | Good |
| Kalman (forward only) | 121 state | 534-1572 | ⚠️ Noisier | Needs work |

---

## New Experiments (Following GPT 5 Pro Suggestions)

### 5.1. Linear Drift Basis

Added `linear` basis to capture secular trends:

**lmax=15 comparison:**
| Bases | Parameters | RMS |
|-------|------------|-----|
| constant + synodic | 768 | 665.69 V |
| constant + linear + synodic | 1024 | 659.87 V |

**Result**: ~6V improvement (1%). The pattern appears fairly stable over the month; linear drift doesn't help much.

---

### 5.2. Cross-Validation Experiments

Following GPT 5 Pro's suggestion, we tested whether the model is genuinely predictive.

#### Test 1: Temporal Split (First 75% → Last 25%)

| Metric | Value |
|--------|-------|
| Train RMS | 653 V |
| Test RMS | **1054 V** |
| Overfit ratio | 1.61 |
| vs Naive mean | **-26%** (worse!) |

**Failed badly.** But this is expected: the last 25% of time (~7 days) covers longitude bands that were **never observed** during training, due to LP's 13°/day drift in moon-fixed frame.

#### Test 2: Random Holdout (Random 75%/25% Split)

| Metric | Value |
|--------|-------|
| Train RMS | 684 V |
| Test RMS | **686 V** |
| Overfit ratio | **1.003** |
| vs Naive mean | **+8.7%** (better) |

**Passes.** Near-perfect generalization (overfit ratio ~1.00).

#### Interpretation

- **R² ≈ 17%**: The model explains ~17% of the variance in the data.
- The remaining 83% is measurement noise, unmodeled physics (plasma regimes, SEP events), and higher spatial frequencies.
- The overfit ratio of 1.003 confirms we're extracting **real structure**, not fitting noise.
- However, 17% explained variance is modest. The question is whether that 17% corresponds to the **correct** physical pattern (dayside/nightside, lunar geology).

#### Why Temporal Split Failed

The temporal split is actually a **harder** test than cross-validation should be. When you remove the last 7 days:
- Those days sample longitudes ~90-180° (in moon-fixed frame)
- The training data has **zero coverage** of those longitudes
- The model must extrapolate purely from the temporal basis structure

The failure here doesn't mean the model is wrong - it means you **cannot predict unseen longitudes from temporal structure alone**. You need spatial coverage to constrain the spatial pattern.

---

## Open Questions for GPT 5 Pro

### On the Cross-Validation Results

1. **Is 17% explained variance meaningful?**
   - The model explains R² ≈ 17% of variance, with RMS improving from 751V (naive) to 686V.
   - The overfit ratio is near-perfect (1.003), so this isn't noise-fitting.
   - But is 17% significant for this type of geophysical data, or concerning?

2. **What does the 83% unexplained variance represent?**
   - Measurement noise from the electron reflectometer?
   - Unmodeled physics (plasma regime variations, SEP events)?
   - Higher spatial frequencies beyond lmax=10?
   - Temporal variations faster than the synodic period?

3. **Is the temporal split failure expected or concerning?**
   - When we hold out the last 7 days (which cover different longitudes), the model performs *worse* than naive mean prediction.
   - We interpret this as: "you can't extrapolate to unseen longitudes from temporal structure alone."
   - Is this the correct interpretation, or does it reveal a fundamental flaw in the approach?

### On the Physical Model

4. **Is the synodic basis capturing real physics?**
   - We assume the dominant pattern rotates at the synodic period (29.53 days) relative to the moon-fixed frame.
   - But the actual physics (plasma sheaths, secondary emission) is nonlinear.
   - Does cos/sin adequately represent this, or should we consider other functional forms?

5. **Moon-fixed vs Solar-fixed frame separation**
   - Currently, `constant` captures moon-fixed structure and `synodic` captures solar-varying structure.
   - Would explicitly fitting two sets of coefficients (one in moon coordinates, one in solar coordinates) improve the decomposition?
   - Or is the current implicit separation adequate?

6. **What lmax is appropriate?**
   - We've tested lmax=10-20 (121-441 spatial coefficients).
   - LP's 100 km altitude limits spatial resolution.
   - Should we use power spectrum analysis to find where signal becomes noise?

### On Next Steps

7. **How do we validate against independent physics?**
   - The maps should show: dayside less negative than nightside, wake structure, possible crustal anomalies.
   - What quantitative checks would confirm the extracted pattern is physically correct?

8. **Should we add more temporal structure?**
   - Linear drift only improved RMS by 1% (6V).
   - Would higher synodic harmonics (2ω, 3ω) help, or risk overfitting?
   - Are there other physically motivated temporal bases we should consider?

---

## Current Recommendation

Use **Temporal Basis Fitting** with `constant + synodic` bases and lmax=10-20:

```bash
python -m src.temporal \
  --start 1998-07-01 --end 1998-07-31 \
  --lmax 20 --l2-penalty 100 \
  --fit-mode basis --temporal-basis constant,synodic \
  --output harmonics.npz
```

This produces:
- Global moon-fixed maps that vary at the synodic period
- Dayside/nightside pattern captured by sin/cos modulation
- Static component captures time-averaged moon-fixed features
- Well-posed problem: 138k measurements → 1323 parameters
