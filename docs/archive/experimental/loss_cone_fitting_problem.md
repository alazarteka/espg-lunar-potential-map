# Loss Cone Fitting Problem - Request for Analysis

## Problem Summary

We are implementing electron loss cone fitting to determine lunar surface potential, following Halekas et al. (2008) JGR paper. Our implementation produces physically reasonable results but shows systematic discrepancies from the published values. We need help identifying the root cause.

## Background

**Physical System:**
- Lunar Prospector spacecraft measures electron pitch angle distributions
- Electrons reflected from the lunar surface show a "loss cone" - absence of flux below a critical pitch angle
- The loss cone angle depends on lunar surface potential (U_surface) and magnetic field ratio (BS/BM)
- Loss cone equation: sin²(αc) = (BS/BM) × (1 + eU_surface/(E - U_spacecraft))

**Reference Paper:**
Halekas et al. (2008), Figure 5 (page 8):
- Timestamp: 14:45 UT on 29 April 1999
- Spacecraft potential: USC = +11 V
- Lunar surface potential: UM = -160 V
- Magnetic field ratio: BS/BM = 0.975

**Our Data:**
- File: `data/1999/091_120APR/3D990429.TAB`
- Spectrum number: 653
- Timestamp: 925397048.0 = 14:44:08 UT on 29 April 1999 ✓ (matches!)

## Implementation Details

### Model Function (`src/model.py::synth_losscone`)

```python
def synth_losscone(
    energy_grid: np.ndarray,          # Measured energies (nE,)
    pitch_grid: np.ndarray,           # Pitch angles (nE, nPitch)
    U_surface: float,                 # Lunar surface potential [V]
    U_spacecraft: float = 0.0,        # Spacecraft potential [V]
    bs_over_bm: float = 1.0,         # Magnetic field ratio
    beam_width_eV: float = 0.0,      # Secondary beam width
    beam_amp: float = 0.0,           # Secondary beam amplitude
    beam_pitch_sigma_deg: float = 0.0,  # Beam pitch spread
) -> np.ndarray:
```

**Key physics:**
1. Energy correction: `E_plasma = E_measured - U_spacecraft`
2. Loss cone calculation: `x = bs_over_bm * (1 + U_surface / E_plasma)`
3. Critical angle: `αc = arcsin(sqrt(x))` (clipped to [0, 1])
4. Model flux = 1.0 where `pitch <= 180° - αc`, else 0.0
5. Adds Gaussian secondary electron beam at energy `|U_surface|`

### Normalization Modes

**"ratio"** (per-energy):
```python
# Divide each energy row by its own mean incident flux (pitch < 90°)
norm[E, :] = flux[E, :] / mean(flux[E, pitch < 90])
```

**"global"** (implemented based on paper description):
```python
# Divide entire 2D array by maximum incident flux
global_max = max(flux[pitch < 90])
norm = flux / global_max
```

**"ratio2"** (pairwise incident/reflected):
```python
# For each energy, pair up incident/reflected angles symmetrically around 90°
# Set incident angles = 1.0, reflected angles = reflected_flux / incident_flux
mid = argmin(|pitch - 90°|)
for k in range(mid):
    i_inc = k
    i_ref = (2*mid - 1) - k  # mirror around 90°
    norm[E, i_inc] = 1.0
    norm[E, i_ref] = flux[E, i_ref] / flux[E, i_inc]
```
**Note**: ratio2 currently incompatible with synth_losscone model (different scaling)

### Fitting Procedure

1. **Latin Hypercube Sampling** (400 points):
   - U_surface: [-1000, 1000] V
   - bs_over_bm: [0.1, 1.0]
   - beam_amp: [config.min, config.max] or fixed to 1.0

2. **Vectorized χ² evaluation:**
   ```python
   models = synth_losscone(energies, pitches, U_surface_array, ...)  # Shape: (400, nE, nPitch)
   log_data = log(normalized_data + eps)
   log_models = log(models + eps)
   chi2 = sum((log_data - log_models)²) for each of 400 samples
   ```

3. **Local refinement** (from best LHS point):
   - Method: Nelder-Mead (unbounded, derivative-free)
   - Post-optimization clipping: bs_over_bm → [0.1, 1.0]
   - Iterations: 200 max
   - Tolerances: xatol=1e-3, fatol=1e-3

## Current Results

### With "global" normalization + USC=11V + beam_amp=1.0 (fixed):

| Spectrum | U_surface [V] | BS/BM | χ² |
|---|---|---|---|
| 100 | -82.5 | 0.983 | 83937 |
| 200 | -77.8 | 1.000 | 76226 |
| 653 (paper) | -247.5 | 1.000 | 51699 |

**Paper target for spec 653:** U_surface = -160V, BS/BM = 0.975

### With "ratio" normalization + USC=11V + beam_amp=1.0 (fixed):

| Spectrum | U_surface [V] | BS/BM | χ² |
|---|---|---|---|
| 653 (paper) | -821.1 | 0.874 | 21453 |

## Key Observations

1. **BS/BM behavior:**
   - With "global" norm: Often hits upper bound (1.000) and gets clipped
   - With "ratio" norm: Usually below 1.0 (e.g., 0.874)
   - Paper value: 0.975 (between our two modes)

2. **U_surface magnitude:**
   - "global" gives -247V for spec 653 (1.5× too large in magnitude)
   - "ratio" gives -821V (5× too large!)
   - Paper: -160V

3. **Normalization impact:**
   - "global" preserves energy-dependent structure → better U_surface
   - "ratio" destroys cross-energy relationships → poor U_surface
   - But neither matches paper exactly

4. **χ² values:**
   - "ratio" has lower χ² (21453) despite worse physical values
   - "global" has higher χ² (51699) but more reasonable physics
   - Suggests different normalization than either mode?

5. **Optimizer issues we discovered:**
   - L-BFGS-B: Froze BS/BM at initial LHS value (gradient ≈ 0)
   - Powell: Worked but gave worse fits than Nelder-Mead
   - Nelder-Mead: Best performance, requires post-clipping

## Questions for Analysis

### 1. **Normalization Method**
   The paper states: "normalizing the distribution by dividing both reflected and incident halves of the distribution by the incident half"

   **Our interpretations:**
   - "global": One global normalization factor (max of incident flux)
   - "ratio": Per-energy normalization (mean incident flux at each E)

   **Could the paper mean something else?** E.g.:
   - Normalize by total integrated incident flux?
   - Normalize only the reflected side, keep incident at measured values?
   - Some other energy-dependent normalization we haven't considered?

### 2. **Beam Model**
   Our implementation centers the beam at energy `U_spacecraft - U_surface` (clamped by beam width) and suppresses it when `U_spacecraft <= U_surface`.

   **Paper says:** "add an upward-going beam, centered at an electron energy of UM - USC"

   **Questions:**
   - Should we include an emission-energy offset (few eV) in the beam center?
   - How should beam amplitude scale? Currently we fix it to 1.0 for paper mode
   - Paper doesn't mention fitting beam amplitude - should it be determined by physics?

### 3. **Energy Correction**
   We apply: `E_plasma = E_measured - U_spacecraft`

   **Physical reasoning:** If spacecraft is at +11V above plasma, electrons gain 11eV falling to s/c, so we subtract to get plasma energy.

   **Is this correct?** The sign convention could be:
   - Our way: E_plasma = E_measured - U_spacecraft
   - Opposite: E_plasma = E_measured + U_spacecraft
   - The paper uses USC = +11V, which should mean spacecraft is positive

### 4. **Spectrum Identification**
   - Our spec #653 timestamp matches paper timestamp (14:44:08 ≈ 14:45 UT)
   - But could they be using a different spectrum number scheme?
   - Or averaging multiple spectra?

### 5. **Model vs Data Mismatch**
   - Our χ² values are quite high (50k-80k for "global", 20k for "ratio")
   - Does this suggest fundamental model limitation?
   - Paper's Figure 5 shows "modest slope" in loss cone vs our sharp step function
   - Should we add loss cone broadening (finite gyroradius effects)?

### 6. **Optimizer Behavior**
   - Why does BS/BM often hit the 1.0 bound with "global" normalization?
   - Is this suggesting our model or normalization doesn't match the data well?
   - Should we use wider bounds or different fitting strategy?

## Code Locations

- **Model**: `src/model.py::synth_losscone` (lines 13-133)
- **Normalization**: `src/flux.py::build_norm2d` (lines 404-448)
- **Fitting**: `src/flux.py::_fit_surface_potential` (lines 450-579)
- **Plotting**: `scripts/plots/plot_losscone_fit_paper.py`
- **Paper**: `scratch/2008_halekas_lunar prospector observations...pdf`

## Request

Please analyze this problem and suggest:
1. What could explain the factor of ~1.5-5× discrepancy in U_surface?
2. Is our normalization interpretation correct based on the paper's description?
3. Are there physical or numerical issues we're missing?
4. What additional experiments should we run to diagnose the problem?

The code is functional and produces self-consistent results - we just can't match the published values. Any insights would be greatly appreciated!
