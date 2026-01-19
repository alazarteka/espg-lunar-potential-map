# ESPG Lunar Potential Map: Implementation Walkthrough

This document provides a comprehensive walkthrough of the codebase implementation. It is designed for someone who understands the underlying physics and Python, but needs to understand *how* the code implements the physics and *why* certain design decisions were made.

**Reading Strategy**: This document references specific files and line numbers. Keep the source code open alongside this document. When you see `file.py:123-456`, open that file and read lines 123-456.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Core Infrastructure](#2-core-infrastructure)
3. [Data Flow: From Raw to Results](#3-data-flow-from-raw-to-results)
4. [Physics Layer](#4-physics-layer)
5. [Fitting & Optimization: CPU vs GPU](#5-fitting--optimization-cpu-vs-gpu)
6. [Temporal Reconstruction](#6-temporal-reconstruction)
7. [Potential Mapper Orchestration](#7-potential-mapper-orchestration)
8. [Utilities Reference](#8-utilities-reference)
9. [Performance Architecture](#9-performance-architecture)

---

## 1. Architecture Overview

### 1.1 Conceptual Layers

The codebase is organized into four conceptual layers:

```
┌─────────────────────────────────────────────────────────────┐
│                    ORCHESTRATION LAYER                       │
│  potential_mapper/pipeline.py, batch.py, parallel_batch.py  │
│  temporal/cli.py, coefficients.py                           │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      FITTING LAYER                           │
│  flux.py (LossConeFitter), kappa.py (Kappa)                 │
│  model_torch.py, kappa_torch.py (GPU variants)              │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                      PHYSICS LAYER                           │
│  physics/kappa.py, physics/charging.py, physics/jucurve.py  │
│  model.py (loss-cone forward model)                          │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────┐
│                   INFRASTRUCTURE LAYER                       │
│  config.py, utils/units.py, utils/spice_ops.py              │
│  data_acquisition.py, utils/geometry.py                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Module Dependency Map

Key import relationships (read bottom-up):

```
potential_mapper/pipeline.py
├── flux.py (ERData, LossConeFitter)
├── kappa.py (Kappa fitter)
├── model_torch.py (LossConeFitterTorch) [optional]
├── kappa_torch.py (KappaFitterTorch) [optional]
├── potential_mapper/coordinates.py
│   ├── utils/spice_ops.py
│   ├── utils/attitude.py
│   └── utils/geometry.py
└── config.py
    └── utils/units.py

temporal/coefficients.py
├── temporal/_harmonics.py (scipy.special.sph_harm_y)
├── scipy.sparse (for regularization)
└── config.py
```

### 1.3 Two Parallel Implementations Pattern

A key architectural pattern you'll see repeatedly: **the same physics implemented twice**—once with full Pint units (slow but verifiable) and once with raw magnitudes (fast). This enables:

1. **Verification**: Run the Pint version to validate physics
2. **Performance**: Run the magnitude version in production

Examples:
- `physics/kappa.py`: `omnidirectional_flux()` vs `omnidirectional_flux_magnitude()`
- `physics/charging.py`: `electron_current_density()` vs `electron_current_density_magnitude()`
- `physics/jucurve.py`: `J_of_U_ref()` vs `J_of_U()`

---

## 2. Core Infrastructure

### 2.1 Configuration (`src/config.py`)

**Read**: `config.py:1-108`

This module is the single source of truth for all physical constants and configuration. Key sections:

| Lines | Content |
|-------|---------|
| 11-28 | Instrument specifications (CHANNELS=88, SWEEP_ROWS=15, ACCUMULATION_TIME) |
| 31-45 | Physical constants (LUNAR_RADIUS, ELECTRON_MASS, BOLTZMANN_CONSTANT) |
| 59-82 | Fitting parameters (FIT_ERROR_THRESHOLD, LOSS_CONE_* settings) |
| 99-108 | File paths (DATA_DIR, SPICE_KERNELS_DIR) |

**Critical values to understand**:
- `CHANNELS = 88`: Number of energy measurement channels per sweep
- `SWEEP_ROWS = 15`: Rows per spacecraft energy sweep
- `FIT_ERROR_THRESHOLD = 657000`: Chi-squared cutoff (99th percentile) for accepting fits

### 2.2 Unit System (`src/utils/units.py`)

**Read**: `units.py:1-62`

The project uses Pint for dimensional analysis. This module defines:

1. **Global unit registry** (line 6): `ureg = pint.UnitRegistry()`
2. **Type aliases** (lines 16-40): `LengthType`, `EnergyType`, `FluxType`, etc.
3. **Validation function** (lines 9-14): `validate_quantity()` for runtime type checking

The type aliases use Python's `Annotated` to document expected units:
```python
FluxType = Annotated[Quantity, ureg.particle / (ureg.cm**2 * ureg.s * ureg.sr * ureg.eV)]
```

### 2.3 Data Acquisition (`src/data_acquisition.py`)

**Read**: `data_acquisition.py:56-232` (the `DataManager` class)

The `DataManager` class handles downloading SPICE kernels and ER flux data from public repositories. Key methods:

| Method | Lines | Purpose |
|--------|-------|---------|
| `list_remote_dirs()` | 76-94 | Scrapes HTML directory listings |
| `download_file()` | 114-145 | Streams downloads with resume support |
| `download_files_in_parallel()` | 193-232 | ThreadPoolExecutor for concurrent downloads |

The main block (lines 235-355) downloads:
- SPICE ephemeris kernels from NASA JPL
- ER flux data from UCLA PDS
- Attitude data for spacecraft orientation

---

## 3. Data Flow: From Raw to Results

### 3.1 High-Level Pipeline

```
Raw .TAB files
    │
    ▼
ERData (src/flux.py:15-179)
    │ Load, validate, add count columns
    ▼
PitchAngle (src/flux.py:182-289)
    │ Calculate pitch angles from B-field
    ▼
LossConeFitter (src/flux.py:291-891)
    │ Normalize flux, fit loss-cone model
    ▼
PotentialResults (potential_mapper/results.py)
```

### 3.2 ERData: Loading and Validation

**Read**: `flux.py:15-179`

The `ERData` class is the primary data container. Key methods:

| Method | Lines | What it does |
|--------|-------|--------------|
| `load_data()` | 45-74 | Read .TAB file, parse timestamps, validate columns |
| `_clean_sweep_data()` | 77-114 | Remove incomplete sweeps, validate B-field |
| `_add_count_columns()` | 118-179 | Reconstruct electron counts from flux |

**Count reconstruction** (lines 137-165): The raw data contains flux values. To reconstruct counts:
```
Count = Flux × GeometricFactor × Energy × AccumulationTime
```
This is inverted from the instrument calibration equation.

### 3.3 Pitch Angle Calculation

**Read**: `flux.py:182-289`

The `PitchAngle` class converts detector coordinates to pitch angles relative to the magnetic field.

**Coordinate conversion** (lines 218-233): Detector angles (phi, theta) are converted to Cartesian unit vectors:
```
X = cos(φ)cos(θ)
Y = sin(φ)cos(θ)
Z = sin(θ)
```

**Pitch angle computation** (lines 269-289): Uses `np.einsum` for vectorized dot products:
```python
# Line 282-284: Batch dot product
dot_product = np.einsum('ijk,ijk->ij', B_normalized, detector_directions)
pitch_angles = np.degrees(np.arccos(np.clip(dot_product, -1, 1)))
```

### 3.4 Coordinate Transformations

**Read**: `potential_mapper/coordinates.py:42-175`

The `CoordinateCalculator` class handles all frame transformations. Key operations:

1. **UTC to ET** (line 73): Convert timestamp strings to SPICE ephemeris time
2. **SPICE batch calls** (lines 75-95): Get LP position, Sun vectors, frame matrices
3. **SCD→IAU_MOON transform** (lines 140-160): Chain through J2000 intermediate frame

The frame chain is:
```
SCD (Spacecraft Detector) → J2000 → IAU_MOON (Lunar body-fixed)
```

**Read**: `utils/coordinates.py:68-125` for `build_scd_to_j2000()` which constructs the rotation matrices.

---

## 4. Physics Layer

### 4.1 Kappa Distribution

**Read**: `physics/kappa.py:106-222`

The kappa distribution models non-Maxwellian plasmas with power-law tails. The core equation is:

```
f(v) = Γ(κ+1)/(π^1.5 · κ^1.5 · Γ(κ-0.5)) × (n/θ³) × [1 + (v/θ)²/κ]^(-κ-1)
```

**Implementation comparison** (same physics, different performance):

| Function | Lines | Input | Speed | Use Case |
|----------|-------|-------|-------|----------|
| `kappa_distribution()` | 106-145 | KappaParams + Quantity | Slow | Verification |
| `omnidirectional_flux()` | 189-222 | KappaParams + Quantity | Slow | Verification |
| `omnidirectional_flux_magnitude()` | 348-379 | Scalars (float) | **~50x faster** | Production |

The Numba-JIT version (lines 348-379) uses:
- `@jit(nopython=True, fastmath=True, cache=True)` for compilation
- Precomputed constants (lines 344-345) to avoid repeated calculations
- Direct `math.gamma()` instead of scipy for nopython mode

### 4.2 Loss-Cone Model

**Read**: `model.py:42-212`

The loss-cone model (Halekas 2008) predicts electron flux depletion at the lunar surface. The loss-cone angle is:

```
sin²(αc) = (B_s/B_m) × (1 + U_m / (E - U_spacecraft))
```

**Key functions**:

| Function | Lines | Purpose |
|----------|-------|---------|
| `_compute_loss_cone_angle()` | 42-69 | Vectorized loss-cone angle calculation |
| `_compute_beam()` | 72-111 | Secondary electron beam (Gaussian in E and pitch) |
| `synth_losscone_batch()` | 114-212 | **Vectorized**: Evaluate 400 models simultaneously |
| `synth_losscone()` | 215-274 | Wrapper for single parameter set |

**Broadcasting pattern** (lines 169-174):
```python
# Reshape for NumPy broadcasting
U_surface = U_surface.reshape(-1, 1, 1)      # (n_params, 1, 1)
energy = energy[None, :, None]                # (1, nE, 1)
pitch = pitch[None, :, :]                     # (1, nE, nPitch)
# Result: (n_params, nE, nPitch) via automatic broadcasting
```

### 4.3 Spacecraft Charging

**Read**: `physics/charging.py:9-138`

Two charging physics components:

1. **Secondary electron yield** (lines 9-43): Sternglass formula
   ```
   δ(E) = 7.4 × δ_m × (E/E_m) × exp(-2√(E/E_m))
   ```

2. **Electron current density** (lines 53-138): Integrated flux
   ```
   J_e = (1/4) × e × ∫ F_omni(E) dE
   ```

**Parallel implementations**:
- `electron_current_density()` (lines 53-97): Full Pint units
- `electron_current_density_magnitude()` (lines 100-138): Fast scalar version

### 4.4 J-U Curve

**Read**: `physics/jucurve.py:30-130`

Spacecraft photoemission modeled as a double exponential:
```
J(U) = A·exp(-U/B) + C·exp(-U/D)
```

The coefficients (lines 10-24) come from Mandell et al. (2008):
- A = 1.07×10⁻⁶ A/m², B = 5 V (fast decay component)
- C = 1.6×10⁻⁸ A/m², D = 60 V (slow decay component)

`U_from_J()` (lines 51-74) inverts this using Brent's root-finding method.

---

## 5. Fitting & Optimization: CPU vs GPU

This section compares the different implementations of the same fitting algorithms.

### 5.1 Kappa Fitting

**CPU Implementation**: `kappa.py:448-542`

The `Kappa.fit()` method uses multi-start L-BFGS-B optimization:

1. **Latin Hypercube sampling** (lines 470-476): Generate 10 initial guesses
2. **Objective function** (lines 361-400): Log-space chi-squared
3. **Optimization loop** (lines 483-500): scipy.optimize.minimize with early stopping

**GPU Implementation**: `kappa_torch.py:250-339`

The `KappaFitterTorch.fit_batch()` method uses batched Differential Evolution:

1. **All spectra in parallel** (line 290): Single tensor operation
2. **Population on GPU** (lines 270-280): (N_spectra, popsize, n_params)
3. **Batched evaluation** (line 316): `omnidirectional_flux_batch_torch()`

**Side-by-side comparison of forward model**:

| Aspect | CPU (`physics/kappa.py:189-222`) | GPU (`kappa_torch.py:59-121`) |
|--------|----------------------------------|-------------------------------|
| Input shape | Scalar energy | (E,) tensor |
| Batch dimension | None | (N, P, E) for N spectra, P candidates |
| Gamma function | `scipy.special.gamma()` | `torch.lgamma()` (log-space) |
| Memory | Python objects | Contiguous GPU tensors |

### 5.2 Loss-Cone Fitting

**CPU Implementation**: `flux.py:707-891`

The `LossConeFitter._fit_surface_potential()` method:

1. **LHS grid search** (lines 732-765): Evaluate 400 parameter combinations
2. **Scipy DE refinement** (lines 768-820): Polish the best candidate
3. **Per-chunk loop** (lines 866-891): Sequential processing

**GPU Implementation**: `model_torch.py:458-992`

The `LossConeFitterTorch` class provides three modes:

| Method | Lines | Strategy | Speedup |
|--------|-------|----------|---------|
| `_fit_surface_potential_torch()` | 458-606 | Single-chunk GPU | ~5x |
| `_fit_batch_lhs()` | 745-830 | Multi-chunk LHS | ~10x |
| `fit_surface_potential_batched()` | 925-992 | Mega-batching | ~20-28x |

**Multi-chunk batching** (lines 195-308): The key innovation is processing N chunks simultaneously:

```python
# Line 277-280: 4D tensor broadcasting
U_surface = U_surface.view(N_chunks, n_pop, 1, 1)  # (N, P, 1, 1)
energy = energy.view(N_chunks, 1, nE, 1)           # (N, 1, E, 1)
# Result: (N, P, E, A) for N chunks, P candidates, E energies, A angles
```

### 5.3 Differential Evolution Optimizer

**Read**: `utils/optimization.py:44-366`

The `BatchedDifferentialEvolution` class supports two modes:

| Mode | Population Shape | Fitness Shape | Use Case |
|------|------------------|---------------|----------|
| Single-spectrum | (popsize, n_params) | (popsize,) | One optimization |
| Multi-spectrum | (N, popsize, n_params) | (N, popsize) | N parallel optimizations |

**Key methods**:

| Method | Lines | Purpose |
|--------|-------|---------|
| `_init_population_multi()` | 153-183 | Sobol sequence initialization |
| `_mutate_multi()` | 206-219 | DE/best/1 mutation for all spectra |
| `_crossover_multi()` | 236-247 | Binomial crossover |
| `_optimize_multi()` | 317-366 | Main optimization loop |

The multi-spectrum mode keeps all N optimizations on GPU simultaneously, avoiding CPU-GPU transfers between iterations.

### 5.4 Vectorized Normalization

**Read**: `flux.py:554-705` (`build_norm2d_batch`)

This method pre-computes normalized flux arrays for multiple chunks at once, avoiding repeated data loading:

```python
# Line 595: Load all flux data once
flux_all = self.er_data.data[FLUX_COLS].to_numpy()  # (n_rows, 88)

# Lines 620-634: Vectorized per-chunk normalization
incident_mask = pitch_chunk < 90.0
norm_factors = np.nanmean(flux_for_norm, axis=1)
result[i, :] = flux_chunk / norm_factors[:, np.newaxis]
```

---

## 6. Temporal Reconstruction

### 6.1 Module Overview

The `src/temporal/` package fits spherical harmonic coefficients over time:

| File | Lines | Purpose |
|------|-------|---------|
| `coefficients.py` | ~995 | Core harmonic fitting |
| `basis.py` | ~325 | Temporal basis expansion |
| `reconstruction.py` | ~120 | Global map generation |
| `cli.py` | ~336 | Command-line interface |

### 6.2 Window-Based Fitting

**Read**: `temporal/coefficients.py:130-199` (window partitioning)

The `_partition_into_windows()` function creates time windows:

1. **Sort by UTC** (line 145)
2. **Create window boundaries** (lines 155-165): Start times with optional stride
3. **Collect measurements** (lines 175-190): Group by window interval
4. **Yield TimeWindow objects** (line 195)

**Read**: `temporal/coefficients.py:397-487` (per-window fitting)

The `_fit_window_harmonics()` function fits one window:

1. **Build design matrix** (line 420): Y_lm(lat, lon) for all measurements
2. **Ridge regularization** (lines 440-460): Optional degree weighting
3. **Solve least squares** (line 470): `np.linalg.lstsq()`
4. **Enforce reality** (line 480): Conjugate symmetry for real potentials

### 6.3 Coupled Multi-Window Fitting

**Read**: `temporal/coefficients.py:658-781`

The `_fit_coupled_windows()` function jointly fits all windows with temporal regularization:

```
Minimize: ||X_block @ a - Φ||² + λ_s||W @ a||² + λ_t||D_t @ a||²
```

Where:
- `X_block`: Block-diagonal design matrix (all windows stacked)
- `λ_s`: Spatial regularization (ridge on coefficients)
- `λ_t`: Temporal regularization (smoothness between windows)
- `D_t`: Finite-difference matrix for temporal coupling

**Key implementation details**:

| Lines | Feature |
|-------|---------|
| 690-710 | Temporal derivative matrix with multi-lag support |
| 720-740 | Optional co-rotating frame rotation |
| 750-760 | Sparse LSQR solver for memory efficiency |

### 6.4 Temporal Basis Expansion

**Read**: `temporal/basis.py:183-281`

Alternative approach: expand coefficients in temporal basis functions:

```
a_lm(t) = Σ_k b_lmk × T_k(t)
```

Available bases (lines 109-120):
- `constant`: DC component
- `synodic`, `synodic2-4`: Lunar synodic period (29.53 days) harmonics
- `sidereal`, `sidereal2-4`: Sidereal period (27.32 days) harmonics

The joint design matrix is constructed via Kronecker-like structure (lines 220-240):
```python
# Shape: (N_measurements, K × n_coeffs)
# Layout: [Y*T_0, Y*T_1, ..., Y*T_{K-1}]
```

### 6.5 Reconstruction

**Read**: `temporal/reconstruction.py:15-66`

`reconstruct_global_map()` evaluates the spherical harmonics on a global grid:

1. **Create lat/lon grid** (lines 20-25): 1° resolution (181×361)
2. **Build design matrix** (lines 30-35): Y_lm at each grid point
3. **Matrix multiply** (line 40): `potential = design @ coeffs`
4. **Reshape to map** (line 42): (lat_steps, lon_steps)

---

## 7. Potential Mapper Orchestration

### 7.1 Pipeline Architecture

**Read**: `potential_mapper/pipeline.py:608-820`

The `process_merged_data()` function orchestrates the full workflow:

| Step | Lines | Operation |
|------|-------|-----------|
| 1 | 620-640 | Load attitude, interpolate to measurement times |
| 2 | 645-680 | Calculate coordinates (SPICE transformations) |
| 3 | 685-700 | Project magnetic field to IAU_MOON |
| 4 | 705-720 | Find ray-surface intersections |
| 5 | 725-750 | Calculate illumination (sun line-of-sight) |
| 6 | 755-780 | **Compute spacecraft potential** (kappa fitting) |
| 7 | 785-810 | **Fit surface potential** (loss-cone fitting) |
| 8 | 815-820 | Assemble PotentialResults |

### 7.2 Processing Mode Selection

**Read**: `potential_mapper/pipeline.py:124-400`

Three modes for spacecraft potential:

| Function | Lines | Mode |
|----------|-------|------|
| `_spacecraft_potential_per_row()` | 124-180 | Sequential (scipy) |
| `_spacecraft_potential_per_row_parallel()` | 182-260 | Multiprocessing |
| `_spacecraft_potential_per_row_torch()` | 325-400 | **GPU batched** |

Selection logic (lines 755-780):
```python
if use_torch:
    spacecraft_potential = _spacecraft_potential_per_row_torch(...)
elif use_parallel:
    spacecraft_potential = _spacecraft_potential_per_row_parallel(...)
else:
    spacecraft_potential = _spacecraft_potential_per_row(...)
```

### 7.3 Batch Processing Modes

Three CLI entry points with different parallelization strategies:

| Script | File | Strategy |
|--------|------|----------|
| `python -m src.potential_mapper` | `cli.py` | Single merged dataset |
| `python -m src.potential_mapper.batch` | `batch.py` | Merged with NPZ output |
| `python -m src.potential_mapper.parallel_batch` | `parallel_batch.py` | Day-level parallelism |

**Day-level parallelism** (`parallel_batch.py:143-200`):
- Uses `ProcessPoolExecutor` with cpu_count-1 workers
- Each day processed independently in separate process
- Optional `--fast` flag enables GPU within each day

---

## 8. Utilities Reference

### 8.1 Geometry (`utils/geometry.py`)

**Read**: `geometry.py:47-114`

`get_intersections_or_none_batch()` finds ray-sphere intersections for N rays simultaneously using vectorized quadratic solving:

```python
# Lines 75-85: Vectorized quadratic coefficients
a = np.einsum('ij,ij->i', directions, directions)  # ||d||²
b = 2 * np.einsum('ij,ij->i', positions, directions)  # 2(p·d)
c = np.einsum('ij,ij->i', positions, positions) - radius**2
```

### 8.2 SPICE Operations (`utils/spice_ops.py`)

**Read**: `spice_ops.py:80-162`

Batch wrappers around single-time SPICE calls. The pattern is:

```python
def get_*_batch(times: np.ndarray) -> np.ndarray:
    result = np.empty((len(times), 3))
    for i, t in enumerate(times):
        try:
            result[i] = get_*_single(t)
        except:
            result[i] = np.nan
    return result
```

### 8.3 Attitude Interpolation (`utils/attitude.py`)

**Read**: `attitude.py:63-104`

`get_current_ra_dec_batch()` uses `np.searchsorted()` for O(log N) lookup per query:

```python
# Line 75: Binary search for all times at once
indices = np.searchsorted(et_spin, times, side='right') - 1
```

### 8.4 Synthetic Data (`utils/synthetic.py`)

**Read**: `synthetic.py:83-118`

`prepare_synthetic_er()` creates deterministic test fixtures:

1. Generate viewing angles (lines 90-95)
2. Compute theoretical kappa flux (lines 98-105)
3. Package as ERData (lines 110-118)

Used extensively in tests to avoid network dependencies.

---

## 9. Performance Architecture

### 9.1 Three-Tier Optimization

The codebase implements a consistent three-tier performance strategy:

| Tier | Technology | Purpose | Example |
|------|------------|---------|---------|
| 1 | Pint + Python | Verification | `omnidirectional_flux()` |
| 2 | NumPy + Numba | CPU production | `omnidirectional_flux_magnitude()` |
| 3 | PyTorch + GPU | Maximum throughput | `omnidirectional_flux_batch_torch()` |

### 9.2 Broadcasting Patterns

All optimized code uses broadcasting to eliminate loops. The pattern is:

```python
# Scalar approach (slow)
for i in range(n_params):
    for j in range(n_energies):
        result[i, j] = compute(params[i], energies[j])

# Broadcast approach (fast)
params = params.reshape(-1, 1)      # (n_params, 1)
energies = energies.reshape(1, -1)  # (1, n_energies)
result = compute(params, energies)  # (n_params, n_energies)
```

### 9.3 VRAM Management

**Read**: `model_torch.py:925-992`

The mega-batching strategy prevents GPU out-of-memory:

```python
# Formula from commit be74751
batch_size = int(VRAM_MB * 0.66 / 20.2)

# Process in chunks
for batch_start in range(0, n_chunks, batch_size):
    batch_indices = chunk_indices[batch_start:batch_start+batch_size]
    # ... process batch on GPU
```

### 9.4 Performance Summary

Measured speedups (approximate):

| Operation | Sequential | NumPy Vectorized | GPU Batched |
|-----------|------------|------------------|-------------|
| Kappa flux | 1x | ~50x (Numba) | ~200x |
| Loss-cone model | 1x | ~2-4x | ~5x per chunk |
| Multi-chunk loss-cone | 1x | N/A | ~20-28x |
| Spacecraft potential | 1x | N/A | ~1000x (batched) |

---

## Appendix: Key File Reference

| Purpose | Primary File | Support Files |
|---------|--------------|---------------|
| Configuration | `config.py` | `utils/units.py` |
| Data loading | `flux.py:15-179` | `data_acquisition.py` |
| Pitch angles | `flux.py:182-289` | |
| Kappa physics | `physics/kappa.py` | |
| Kappa fitting (CPU) | `kappa.py` | |
| Kappa fitting (GPU) | `kappa_torch.py` | `utils/optimization.py` |
| Loss-cone physics | `model.py` | |
| Loss-cone fitting (CPU) | `flux.py:291-891` | |
| Loss-cone fitting (GPU) | `model_torch.py` | `utils/optimization.py` |
| Coordinates | `potential_mapper/coordinates.py` | `utils/spice_ops.py`, `utils/geometry.py` |
| Pipeline | `potential_mapper/pipeline.py` | `batch.py`, `parallel_batch.py` |
| Temporal fitting | `temporal/coefficients.py` | `temporal/basis.py` |
| Temporal reconstruction | `temporal/reconstruction.py` | |

---

*Document generated from codebase analysis. Last updated: 2025-12-30.*
