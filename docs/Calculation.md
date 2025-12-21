# Calculation Pipeline

This document describes the end-to-end calculation flow for mapping lunar surface electrostatic potential, distinguishing between pure physics/math (forward calculations) and fitting problems (inverse calculations).

## Pipeline Overview

```
ER Flux Files (.TAB)
      │
      ▼
   Load & Validate (ERData)
      │
      ▼
   Coordinate Transforms (SCD → IAU_MOON via SPICE)
      │
      ▼
   Surface Intersection (ray-sphere along B-field)
      │
      ▼
   Spacecraft Potential (current balance + kappa fits)
      │
      ▼
   Surface Potential (loss-cone fitting)
      │
      ▼
   NPZ Cache → Temporal Reconstruction → Harmonic Maps
```

---

## Pure Physics/Math (Forward Calculations)

These are direct computations with closed-form solutions or deterministic algorithms.

| Component | Location | What it computes |
|-----------|----------|------------------|
| Coordinate transforms | `coordinates.py` | Rotation matrices SCD→J2000→IAU_MOON |
| Ray-sphere intersection | `utils/geometry.py` | Where B-field line hits lunar surface |
| Illumination check | `pipeline.py` | Dot product of surface normal with sun vector |
| Kappa flux model | `src/physics/kappa.py` | F(E) given (n, κ, θ) - analytic formula |
| Electron/ion currents | `src/physics/charging.py` | Je(U), Ji(U) from distribution integrals |
| Secondary yield | `spacecraft_potential.py` | Sternglass model - analytic |
| J-U curve | `src/physics/jucurve.py` | Photoemission current vs potential |
| Loss-cone forward model | `src/model.py::synth_losscone` | Predicted pitch-angle distribution given (U_surf, Bs/Bm, beam params) |
| Spherical harmonics | `src/temporal/` | Y_lm(θ,φ) evaluation - analytic |

---

## Fitting Problems (Inverse Calculations)

These are optimization or root-finding problems where we infer parameters from observations.

| Problem | Location | Method | What's fitted |
|---------|----------|--------|---------------|
| **Kappa fit** | `src/kappa.py::Kappa` | Least-squares / curve_fit | (n, κ, θ) from flux vs energy |
| **Spacecraft potential** | `spacecraft_potential.py` | Root-finding (current balance) | U_sc where Je + Ji - Jsee = 0 |
| **Surface potential** | `flux.py::LossConeFitter` | Global optimization (differential evolution) | (U_surf, Bs/Bm, beam_amp) from pitch-angle distribution |
| **Harmonic coefficients** | `temporal/coefficients.py` | Regularized least-squares | a_lm(t) from scattered measurements |

---

## Phase Details

### Phase 1: Data Loading & Validation

**Module:** `src/flux.py::ERData`

- Parse Lunar Prospector Electron Reflectometer (ER) flux files (`.TAB` format)
- Validate measurements: reject rows with invalid B-field magnitude (< 1e-9 or > 1000 nT) or bad timestamps
- Reconstruct electron counts from flux using geometric factor and integration time
- Merge multiple files into single dataset with renumbered spectrum indices

**Output:** Merged `ERData` with 88 energy channels, B-field components, pitch angles

---

### Phase 2: Coordinate Transforms

**Module:** `src/potential_mapper/coordinates.py::CoordinateCalculator`

**Type:** Pure math

Transforms all vectors from spacecraft frame to lunar body-fixed frame:

1. Load attitude data (spin axis RA/Dec from `data/attitude.tab`)
2. Build rotation matrices:
   - SCD (spacecraft) → J2000 using spin axis orientation
   - J2000 → IAU_MOON using SPICE planetary constants
3. Compute geometry arrays:
   - LP spacecraft position in IAU_MOON (km)
   - LP→Sun and Moon→Sun vectors
   - Per-row rotation matrices for vector transforms

---

### Phase 3: Surface Intersection & Illumination

**Module:** `src/utils/geometry.py`, `src/potential_mapper/pipeline.py`

**Type:** Pure math (geometry)

1. **B-field projection:** Transform measured B-field from SCD → IAU_MOON
2. **Ray-sphere intersection:** Find where field line from spacecraft intersects lunar sphere (R = 1737.4 km)
3. **Geodetic conversion:** Compute lat/lon for spacecraft and intersection points
4. **Illumination:** Check sun visibility for spacecraft and surface point

---

### Phase 4: Spacecraft Potential

**Module:** `src/spacecraft_potential.py`

**Type:** Fitting (root-finding)

Compute floating potential U_sc of Lunar Prospector spacecraft:

1. **Kappa fit** (fitting): Fit kappa distribution to electron flux spectrum → (n, κ, θ)
2. **Current balance** (root-finding): Solve for U_sc where net current = 0

**Dayside (sun-illuminated):**
- Dominant current: photoemission
- Use J-U curve inversion: `src/physics/jucurve.py::U_from_J()`

**Nightside (shadow):**
- Balance: Je(U_sc) + Ji(U_sc) - Jsee(U_sc) = 0
- Je: electron current from kappa distribution
- Ji: ion current (OML-like model)
- Jsee: secondary electron emission (Sternglass model)

---

### Phase 5: Surface Potential (Loss-Cone Fitting)

**Module:** `src/flux.py::LossConeFitter`

**Type:** Fitting (global optimization)

This is the **hardest inverse problem** in the pipeline.

**Forward model:** `src/model.py::synth_losscone()`
- Predicts pitch-angle-dependent flux given:
  - U_surface: lunar surface potential [V]
  - U_spacecraft: spacecraft potential [V] (from Phase 4)
  - Bs/Bm: surface-to-measurement magnetic field ratio
  - beam_amp: secondary electron beam amplitude
  - beam_width, beam_pitch_sigma: beam shape parameters

**Fitting procedure:**
1. Extract 15-row sweep (one energy scan)
2. Normalize flux (ratio of reflected/incident)
3. Initial sampling: Latin Hypercube with 400 points
4. Refinement: `scipy.optimize.differential_evolution()`
5. Minimize chi-squared between log(observed) and log(model)

**Quality control:**
- Reject fits with chi-squared > 657,000 (99th percentile threshold)
- Propagate NaN for failed fits

---

### Phase 6: Temporal Reconstruction

**Module:** `src/temporal/`

**Type:** Fitting (regularized least-squares)

Reconstruct time-varying global potential maps from scattered measurements:

1. **Load cached potentials** from `artifacts/potential_cache/*.npz`
2. **Time windowing:** Group measurements by time interval
3. **Spherical harmonic fit** (per window):
   - Design matrix A: Y_lm(lon, lat) at measurement points
   - Solve: a_lm = (A^T A + λI)^-1 A^T U_surface
   - λ: L2 regularization strength
4. **Reconstruction** (forward):
   - Evaluate U(lat, lon, t) = Σ_{lm} a_lm(t) Y_lm(lon, lat) on regular grid

---

## Pipeline Flow: Fitting vs Forward

```
Measured flux ──► FIT kappa params ──► FORWARD currents ──► FIT U_spacecraft
                                                                   │
                                                                   ▼
                                          FORWARD loss-cone model ◄── FIT U_surface
                                                                           │
                                                                           ▼
                                               Scattered potentials ──► FIT harmonics
                                                                           │
                                                                           ▼
                                                               FORWARD reconstruction
```

---

## Key Physics Models

### Kappa Distribution
**Location:** `src/physics/kappa.py`

Non-Maxwellian electron velocity distribution with high-energy tail:

```
F(E) ∝ (1 + E/(κθ²))^-(κ+1)
```

Parameters:
- n: density [cm⁻³]
- κ: shape parameter (κ→∞ recovers Maxwellian)
- θ: thermal speed [km/s]

### Loss-Cone Model
**Location:** `src/model.py::synth_losscone`

Pitch-angle-dependent electron flux accounting for:
- Magnetic mirror geometry (loss cone depletion)
- Surface potential energy barrier
- Secondary electron emission as Gaussian beam

### Current Balance (Spacecraft Charging)
**Location:** `src/physics/charging.py`

Spacecraft floats at potential where:
- Electron current (collection) = Ion current + Secondary emission + Photoemission

### Spherical Harmonics
**Location:** `src/temporal/`

Spatial basis functions Y_lm(θ,φ) for representing potential on sphere:
- Degree l: spatial scale (l=0 is global average)
- Order m: azimuthal variation (-l ≤ m ≤ l)

---

## Configuration & Constants

All physics and processing parameters in `src/config.py`:
- Physical constants: electron mass, charge, lunar radius, Boltzmann constant
- Instrument: geometric factor, 88 energy channels, 15 rows per sweep
- Fitting thresholds: chi-squared cutoff, beam amplitude bounds
- Beam model defaults: width factor, pitch sigma, background level

---

## Testing Architecture

### Pint Reference vs Fast Implementation

The codebase uses pint (unit-aware quantities) as a **reference oracle** for testing, not in production:

```
Slow + Units (pint)          Fast + Unitless (numpy)
       │                              │
       │   ← tests verify match →     │
       │                              │
Reference implementation         Production code
(verifiably correct)             (actually used)
```

| Module | Has pint reference? | Has fast version? | Cross-tested? |
|--------|---------------------|-------------------|---------------|
| `physics/kappa.py` | ✅ | ✅ | ✅ |
| `physics/charging.py` | ✅ | ✅ | ✅ |
| `physics/jucurve.py` | ✅ | ✅ | ✅ |
| `model.py` (loss-cone) | ❌ | ✅ | - |

### Ground Truth Tests

Physics constants should have tests that validate against their source:

```python
def test_coefficients_match_halekas_2008():
    """Verify coefficients match paper (doi:...) paragraph X."""
    assert coeff.A == 1.07e-6  # paper says 1.07 μA/m²
```

This catches transcription errors because expected values are written independently with the source cited.

---

## Physics Sources & Citations

### J-U Curve (Photoemission)

**Source:** [Halekas 2008](https://doi.org/10.1029/2008JA013194), paragraph 23

Double exponential: `J(U) = A·exp(-U/B) + C·exp(-U/D)`

| Coefficient | Value | Paper |
|-------------|-------|-------|
| A | 1.07e-6 A/m² | 1.07 μA/m² |
| B | 5.0 V | 5 V |
| C | 1.6e-8 A/m² | 0.016 μA/m² |
| D | 60.0 V | 60 V |

Note: Paper has typo listing A, B, A, B instead of A, B, C, D.

### Kappa Distribution

Standard form for non-Maxwellian plasma:
```
f(v) ∝ (1 + v²/(κθ²))^-(κ+1)
```

Validated by:
- Density integral test: ∫4πv²f(v)dv = n
- Ratio formula test: f(v₁)/f(v₂) matches analytic expression

### Loss-Cone Model

**TODO:** Document physics source for loss-cone angle formula:
```python
x = bs_over_bm * (1.0 + U_surface / E)
ac = arcsin(sqrt(x))
```

---

## Test Coverage Status

| Component | Physics Validated? | Notes |
|-----------|-------------------|-------|
| Kappa distribution | ✅ | Density integral, ratio formula |
| J-U curve | ✅ | Ground truth from Halekas 2008 |
| Charging currents | ⚠️ | Cross-tested, but 0.25 factor not validated |
| Ray-sphere intersection | ⚠️ | Basic cases only |
| Loss-cone model | ❌ | Only self-consistency (vectorized vs loop) |
| Spherical harmonics | ❌ | No direct tests |
| Coordinate transforms | ❌ | Test file empty |
