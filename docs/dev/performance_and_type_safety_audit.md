# Performance & Type-Safety Audit

Exploratory audit of hotpath performance and of how Python's type system is
(and could be) used to protect scientific correctness. Findings are ranked for
implementation value; nothing here is a commitment to ship every item.

Date: 2026-07-23

---

## 1. Performance

### What is already in good shape

| Pattern | Where | Why keep it |
|---------|-------|-------------|
| Batched torch loss-cone fitting | `src/losscone/torch/fitter.py` | Chunks stay GPU-resident across DE iterations |
| Precomputed log(data) for Halekas chi2 | `src/losscone/torch/chi2.py` | Avoids redundant logs every evaluation |
| Vectorized geometry / intersections | `src/utils/geometry.py`, `src/potential_mapper/pipeline.py` | NumPy batch ops, not row loops |
| Temporal basis `LinearOperator` | `src/temporal/basis.py` | Avoids dense `(N, K·n_coeffs)` materialization |
| Numba kappa magnitude core | `src/physics/kappa.py` | Pint stays off the inner flux path |
| Atomic / resumable I/O | `src/potential_mapper/npz_io.py`, `src/data_acquisition.py` | Safe caches and downloads |

SIMD in the usual sense (explicit AVX kernels) is largely already covered by
NumPy / PyTorch / Numba on the numeric cores. The remaining wins are mostly
**removing Python loops and redundant allocations around those cores**, not
rewriting floating-point kernels by hand.

### High impact

#### 1.1 Per-spectrum Kappa prep rebuilds DataFrames (`pipeline` hotpath)

- **Where:** `src/potential_mapper/pipeline.py` (`_prepare_kappa_batch_data`),
  `src/kappa.py` (`Kappa._prepare_data`)
- **Issue:** For every `spec_no`, the batch path constructs a full `Kappa`
  object: pandas filter/copy, `PitchAngle` rebuild, and
  `np.flatnonzero(spec_values == spec_value)`.
- **Fix:** Slice contiguous spectrum runs once from precomputed column arrays;
  vectorize omnidirectional flux / density; keep `Kappa` construction for
  scalar/debug paths only.

#### 1.2 CPU spacecraft-potential path copies the full energy column per spectrum

- **Where:** `src/potential_mapper/pipeline.py` (`_spacecraft_potential_per_row`),
  `src/spacecraft_potential.py` (`calculate_potential`)
- **Issue:** Each spectrum backs up `ENERGY_COLUMN` with
  `to_numpy(copy=True)` because `calculate_potential` mutates `er_data`.
  Cost is O(N_spectra × N_rows) plus pandas assignment churn.
- **Fix:** Accept a local corrected-energy array (no mutation). Prefer the
  existing torch path for production batch; treat CPU as fallback.

#### 1.3 CPU loss-cone chunks re-convert pandas every chunk

- **Where:** `src/losscone/cpu.py`, `src/losscone/normalization.py`
- **Issue:** Every ~15-row chunk calls `build_norm2d()` and
  `er_data.data[...].to_numpy(...)` again.
- **Fix:** Cache `energy_all` / `flux_all` / `pitch_all` /
  `spacecraft_potential` on fitter init; reuse `build_norm2d_batch()` on CPU
  blocks the same way the torch path already does.

#### 1.4 Kappa CPU objective rebuilds the response matrix every L-BFGS eval

- **Where:** `src/kappa.py` (`_objective_function_fast`, ~381–388)
- **Issue:** `build_log_energy_response_matrix(...)` depends only on the fixed
  energy grid / config, not on `(κ, θ)`, but is rebuilt on every objective call.
- **Fix:** Cache `_response_matrix`, weights, and `log(data + EPS)` in
  `_prepare_data()` (or lazily once per `use_convolution` setting).

#### 1.5 Torch loss-cone forward materializes large model tensors per LHS/DE step

- **Where:** `src/losscone/torch/forward.py`, `src/losscone/torch/fitter.py`
- **Issue:** Each iteration builds `(N_chunks, n_pop, nE, nPitch)` model plus
  masks/diffs. For modest grids, allocation and bandwidth dominate arithmetic.
- **Fix:** Fuse forward + χ² reduction (accumulate residuals without storing
  the full model), or reuse scratch buffers / safe in-place ops.

#### 1.6 Temporal window partitioning is O(N × windows)

- **Where:** `src/temporal/coefficients.py` (`_partition_into_windows`)
- **Issue:** After sorting, every window builds a full boolean mask over all
  timestamps and copies slices.
- **Fix:** `np.searchsorted` for `[start, end)` indices; O(N log N + W log N).

#### 1.7 Coupled temporal derivative / design matrices blow memory

- **Where:** `src/temporal/coefficients.py`
  (`_build_temporal_derivative_matrix`, `_build_block_diagonal_design`)
- **Issue:** Dense cumulative deltas are O(W²); per-window dense harmonics plus
  `bmat` / block-diagonal sparse copies inflate memory (especially
  `complex128`).
- **Fix:** Prefix sums + direct COO stencils; mirror the existing
  `LinearOperator` pattern in `src/temporal/basis.py`; cache rotation diagonals
  by `(lag, total_angle)`.

#### 1.8 Engineering global stats holds all reconstructed maps at once

- **Where:** `src/engineering/analysis.py`, `src/temporal/reconstruction.py`
- **Issue:** `compute_potential_series` allocates `(n_times, lat, lon)` then
  `abs_maps` for mean/percentiles — a large transient spike.
- **Fix:** Stream map evaluation; online mean / threshold fractions; chunked
  or approximate p95 if exact quantiles are not required.

### Medium impact

| # | Area | Issue | Suggestion |
|---|------|-------|------------|
| 2.1 | Batch NPZ | Row-level arrays repeat spectrum-level fit fields | Normalized cache: row geometry + spectrum keyed fits; `float32` diagnostics where safe |
| 2.2 | Temporal NPZ load | Full arrays cast to float64 before date mask | Epoch index + mask before cast; prefer spectrum-level arrays for temporal work |
| 2.3 | SPICE “batch” helpers | Python loops over scalar `spkpos` / `pxform` | Deduplicate ETs; one geometry call per spectrum when 15 rows share time |
| 2.4 | Environment classification | Per-row Python enum construction | Vectorized int8 masks |
| 2.5 | Torch LHS samples | Regenerated + H2D copied every mega-batch | Cache by `(n_lhs, bounds, seed, device, dtype)` |
| 2.6 | Fit-result expansion | Chunk Python loops filling 15-row slices | `np.repeat(chunk_values, SWEEP_ROWS)` |
| 2.7 | Data acquisition | Eager task + Future lists for all years | Bounded producer/consumer submission |
| 2.8 | SZA / viz helpers | Row-wise SPICE in analysis scripts | Batch sun vectors; vectorize dots; cache repeated timestamps |
| 2.9 | Residual analysis | Dense temporal-spatial design | Same LinearOperator matvec as `fit_temporal_basis` |

### Low impact

- Date filtering via Python string loops in plotting scripts
- `np.load` without context managers in long PNG loops
- `dtype=object` environment label arrays (prefer int codes / fixed-width Unicode)

### Suggested performance adoption order

1. Cache kappa response matrix + weights (small change, clear win on CPU kappa).
2. Vectorize `_prepare_kappa_batch_data` (batch pipeline prep).
3. Stop mutating / copying energy columns in spacecraft-potential CPU path.
4. Temporal `searchsorted` windows + LinearOperator-style coupled design.
5. Torch forward/χ² fusion or scratch reuse (profile first — depends on GPU).
6. NPZ schema normalization (compatibility layer first).

---

## 2. Type safety for calculation correctness

### Current state

**Strongest today — Kappa physics boundary pattern**

`src/physics/kappa.py` is the best model of “types supporting correctness”:

1. `KappaParams` (`frozen`, `slots`) groups coupled quantities with pint types.
2. `__post_init__` canonicalizes units (density → m⁻³, theta → m/s).
3. Reference APIs keep `pint.Quantity` through physics ops.
4. Fast paths cross an **explicit magnitude boundary** via `to_tuple()` /
   `*_magnitude` helpers.
5. Parity tests (`tests/test_kappa_response_parity.py`,
   `tests/test_kappa_torch.py`, `tests/physics/test_kappa.py`,
   `tests/test_losscone_parity.py`) protect duplicated CPU / Torch / Numba math.

That combination — unit-aware reference path + explicit magnitude boundary +
parity tests — is more valuable than annotations alone.

**Also good**

- `FitMethod` / `NormalizationMode` (`StrEnum`) in `src/losscone/types.py`
- `PlasmaEnvironment` (`IntEnum`) in `src/potential_mapper/results.py`
- Runtime shape checks in geometry / some chi2 helpers
- Annotated unit aliases in `src/utils/units.py` (documentation + pint runtime)

**Weak**

- Most scientific arrays are bare `np.ndarray` / `Tensor` — no distinction
  between energy grids, potentials, pitch grids, or frames.
- Optimized / scalar APIs use raw `float` (volts vs eV is convention-only).
- `Annotated[Quantity, unit]` is not enforced by mypy (treats as `Quantity`).
- Mode strings and NPZ keys are often stringly typed.
- Physics invariants sometimes use `assert` (stripped under `python -O`).
- mypy remains advisory (~120 findings backlog as of 2026-07-05).

### Correctness gaps types could catch

| Risk | Example | Why types matter |
|------|---------|------------------|
| Volts vs eV magnitudes | `spacecraft_potential.py`, loss-cone `U` / energy grids | Numerically equal for e± but semantically distinct; raw floats swap silently |
| Magnetic ratio convention | `losscone/model.py` documents `B_sc / B_Moon`; `potential_mapper/results.py` docstring says the inverse | Semantic NewType / named field would force one convention |
| Coordinate frames | SPICE helpers return IAU_MOON vs ECLIPJ2000 as plain `ndarray` | Phantom frame types catch wrong-direction transforms |
| Array shapes | `(E,)`, `(E,P)`, `(N,E,P)`, `(N,3)`, spherical-harmonic coeffs | Shape typing / boundary validators |
| NPZ schemas | String keys like `rows_projected_potential` | `TypedDict` + runtime schema check |
| Row alignment | `PotentialResults` fills missing arrays but does not assert equal lengths | `__post_init__` length/dtype checks |

### Kappa as the template (and its remaining gaps)

Keep:

- pint at the reference boundary
- magnitude-only fast path with a named escape hatch
- parity tests for every optimized duplicate

Tighten:

- Replace physics `assert` (`κ > 1.5`, `θ > 0`) with always-on `ValueError`
- Distinguish “sigma in log-theta” from `SpeedType` in uncertainty fields
- Add `NewType` magnitudes for `theta_to_temperature_ev` /
  `omnidirectional_flux_magnitude` argument order

### Opportunities ranked (correctness value vs cost)

| Phase | Change | Value | Cost |
|-------|--------|-------|------|
| 1 | `Literal` for fit method, dtype names, flux stats | High | Low |
| 1 | `numpy.typing.NDArray` aliases (`Float64Array`, `EnergyGridEV`, …) | Med–High | Low |
| 2 | `NewType` scalars for magnitude APIs (`EnergyEV`, `PotentialV`, `ThetaMPerS`, …) | High | Low–Med |
| 2 | Always-on `ValueError` for physics invariants | High | Low |
| 3 | Boundary shape validators + `PotentialResults` length checks | High | Med |
| 3 | `TypedDict` + runtime validators for NPZ payloads | Med–High | Med |
| 4 | `jaxtyping` pilot on `losscone/model.py` + `chi2.py` | Very high locally | Med–High |
| 5 | Phantom frame types at SPICE / geometry boundaries | Very high locally | High |
| — | Big-bang strict mypy / rewrite everything with units | Low ROI | Very high |

### Example patterns that fit this codebase

**Semantic magnitudes (fast path, zero runtime cost under mypy):**

```python
from typing import NewType

EnergyEV = NewType("EnergyEV", float)
PotentialV = NewType("PotentialV", float)
ThetaMPerS = NewType("ThetaMPerS", float)
DensityM3 = NewType("DensityM3", float)
KappaValue = NewType("KappaValue", float)

def temperature_ev_to_theta(
    temperature_ev: EnergyEV,
    kappa: KappaValue,
) -> ThetaMPerS: ...
```

**Config literals:**

```python
from typing import Literal

FitMethodName = Literal["halekas", "lillis"]
TorchDTypeName = Literal["auto", "float16", "float32", "float64"]
```

**Boundary shape check (runtime, tests + production):**

```python
def require_shape(
    name: str,
    arr: np.ndarray,
    ndim: int,
    trailing: tuple[int, ...] = (),
) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got {arr.shape}")
    if trailing and arr.shape[-len(trailing):] != trailing:
        raise ValueError(f"{name} must end with {trailing}, got {arr.shape}")
    return arr
```

**Frame phantoms (SPICE boundary only):**

```python
from dataclasses import dataclass
from typing import Generic, TypeVar
from numpy.typing import NDArray
import numpy as np

Frame = TypeVar("Frame")

class IAUMoon: ...
class EclipJ2000: ...

@dataclass(frozen=True)
class Vec3(Generic[Frame]):
    values: NDArray[np.float64]  # (..., 3)
```

### Suggested type-safety adoption path

1. **Annotate without behavior change:** `Literal`, `NDArray` aliases, tighten
   return types on hot APIs.
2. **Semantic scalars on magnitude APIs:** start in `physics/kappa.py`,
   `physics/charging.py`, `spacecraft_potential.py`, `losscone/model.py`.
3. **Always-on validation at boundaries:** results dataclasses, loss-cone
   model/chi2, NPZ loaders, temporal coefficient load.
4. **Shape-typing pilot:** one subsystem (`losscone` model + chi2) with
   optional runtime checks in tests only.
5. **Frame phantoms:** wrap SPICE I/O in `spice_ops` / coordinates / geometry;
   do not wrap every internal array.
6. **mypy:** keep full-tree advisory; enforce stricter checking only on
   converted modules to avoid fighting the backlog.

---

## 3. Bottom line

**Performance:** The numeric cores are already largely vectorized / GPU /
Numba. The biggest remaining costs are Python orchestration around them —
per-spectrum pandas/`Kappa` rebuilds, mutating energy backups, uncached kappa
response matrices, O(NW) temporal windows, and dense temporal / engineering
map materialization. Prefer structural fixes over hand-written SIMD.

**Type safety:** Kappa already demonstrates the right architecture: pint at
the reference boundary, explicit magnitude escape hatches, and parity tests.
Extend that pattern with low-cost `Literal` / `NewType` / `NDArray` aliases and
boundary validators before investing in `jaxtyping` or frame phantom types.
Do not rely on static types alone for duplicated math — keep parity tests as
the correctness backstop.

### Near-term candidates if implementing next

1. Cache kappa response matrix in `_objective_function_fast`
2. Fix / unify `bs_over_bm` documentation (and ideally a named type or alias)
   so model and results agree on `B_sc / B_Moon`
3. Replace physics `assert` invariants with `ValueError`
4. Add `Literal` annotations for fit-method / dtype config
5. Vectorize `_prepare_kappa_batch_data`
