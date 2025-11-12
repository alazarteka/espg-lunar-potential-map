# `src.temporal` Reference Guide

This document explains why we carved the temporal spherical-harmonic tooling into a proper package, what lives inside, and how to use it from both the CLI and regular Python code. Share it with anyone reviewing the reorg or trying to hook into the new APIs.

---

## Motivation

Before this refactor, every temporal CLI (`temporal_harmonic_coefficients.py`, `temporal_harmonics_animate.py`, `temporal_harmonics_analysis.py`, etc.) maintained its own copy of:

- NPZ-loading logic for the coefficient bundles.
- Reconstruction code that rebuilt global maps using `sph_harm`.
- Plot-specific helpers (cell edges, color scaling, timestamp formatting).

Any bugfix (e.g., switching to `sph_harm_y` for argument-order safety) had to be applied in five places, and it was impossible to reuse those routines from notebooks or tests without copy/paste.

The new `src.temporal` package centralizes this functionality so that:

- Every CLI shares one implementation of the fitter, loaders, and reconstruction helpers.
- The logic is importable from pytest/notebooks/services without shelling out.
- Changes are tested once and automatically propagate to every visualization.

---

## Package Layout

```
src/temporal/
├── __init__.py           # Re-export common entry points
├── coefficients.py       # Fitting pipeline + CLI entry point
├── dataset.py            # TemporalDataset dataclass + loader
└── reconstruction.py     # Map reconstruction & visualization helpers
```

### `coefficients.py`

- `TimeWindow` / `HarmonicCoefficients` dataclasses encapsulate each time slice.
- `_discover_npz`, `_load_all_data`, `_partition_into_windows`, and `_fit_*` functions implement the coupled/independent solutions.
- `compute_temporal_harmonics(...)` is the programmatic API for fitting.
- `save_temporal_coefficients(...)` writes the standard NPZ bundle.
- `parse_args()` / `main()` expose the CLI used by `python -m src.temporal.coefficients`.

### `dataset.py`

- `TemporalDataset` dataclass wraps `times`, `coeffs`, `n_samples`, `spatial_coverage`, and `rms_residuals`.
- `load_temporal_coefficients(path: Path)` reads an NPZ bundle and returns a `TemporalDataset` instead of a raw dict.

### `reconstruction.py`

- `reconstruct_global_map(...)`: build a single lat/lon grid from coefficients.
- `compute_potential_series(...)`: stack reconstructions for every time slice.
- `compute_cell_edges(...)` / `compute_color_limits(...)`: helpers for images/animations.
- `format_timestamp(...)`: consistent human-readable labels (minute precision).

### `__init__.py`

Re-exports the most common functions/classes so callers can simply do:

```python
from src.temporal import (
    compute_temporal_harmonics,
    load_temporal_coefficients,
    compute_potential_series,
    reconstruct_global_map,
)
```

---

## Command-Line Usage

### Fit Coefficients

```bash
UV_CACHE_DIR=.uv-cache uv run python -m src.temporal.coefficients \
  --start 1998-04-01 \
  --end 1998-04-30 \
  --lmax 10 \
  --window-hours 24 \
  --regularize-l2 1000 \
  --temporal-lambda 1e-2 \
  --output data/temporal_coeffs_apr1998.npz \
  --log-level INFO
```

The CLI flags are identical to the previous `scripts/dev/temporal_harmonic_coefficients.py`. The script now simply imports `src.temporal.coefficients` and invokes `main()`.

### Analyze, Animate, Compare

Every temporal CLI (`temporal_harmonics_animate`, `temporal_harmonics_analysis`, `temporal_harmonics_interactive_map`, `temporal_harmonics_azimuthal_modes`, `temporal_harmonics_l_curve`, and `temporal_harmonics_smoothness`) already imports `src.temporal`. No CLI flags changed; they just benefit from the shared implementation (and you can piggyback on the same APIs from your own scripts).

---

## Programmatic Usage Examples

### Load an Existing Bundle

```python
from pathlib import Path
from src.temporal import load_temporal_coefficients

dataset = load_temporal_coefficients(Path("data/temporal_coeffs_apr1998.npz"))
print(dataset.times.shape)    # (N_windows,)
print(dataset.lmax)           # e.g., 10
print(dataset.coeffs.shape)   # (N_windows, (lmax+1)^2)
```

### Reconstruct Maps

```python
from src.temporal import compute_potential_series, format_timestamp

lats, lons, maps = compute_potential_series(
    dataset.coeffs,
    dataset.lmax,
    lat_steps=181,
    lon_steps=361,
)
print(f"{format_timestamp(dataset.times[0])}: {maps[0].min():.1f} to {maps[0].max():.1f} V")
```

### Compute Temporal Roughness

```python
import numpy as np
from src.temporal import load_temporal_coefficients

dataset = load_temporal_coefficients(Path("data/temporal_coeffs_apr1998.npz"))
diffs = np.diff(dataset.coeffs, axis=0)
roughness = np.mean(np.linalg.norm(diffs, axis=1))
print(f"Mean temporal roughness: {roughness:.2f} V")
```

### Build a Custom Plot (Matplotlib)

```python
import matplotlib.pyplot as plt
from src.temporal import load_temporal_coefficients, reconstruct_global_map

dataset = load_temporal_coefficients(Path("data/temporal_coeffs_apr1998.npz"))
lat, lon, potential = reconstruct_global_map(dataset.coeffs[0], dataset.lmax)

plt.imshow(
    potential,
    extent=(lon.min(), lon.max(), lat.min(), lat.max()),
    origin="lower",
    cmap="RdBu_r",
)
plt.colorbar(label="Φ_surface (V)")
plt.title(f"Snapshot at {dataset.times[0]}")
plt.show()
```

---

## Testing & Extensibility

- The new package is regular Python code under `src/`, so it is importable from tests (add coverage as needed, e.g., verifying `compute_color_limits` behavior).
- If you need new shared helpers (smoothing/interpolation, filtering by `m`, etc.), implement them in `src.temporal` and import them in the CLIs. That way both programmatic users and CLI consumers stay in sync.

---

## Migration Checklist

1. Use `python -m src.temporal.coefficients …` for all coefficient fitting tasks.
2. Prefer `load_temporal_coefficients` over manual `np.load`.
3. Pull reconstruction/visualization helpers from `src.temporal.reconstruction`.
4. Contribute any new temporal utilities directly to `src.temporal` (not `scripts/`), then wire the CLI to the shared helper.

Following this pattern keeps every visualization, analysis notebook, and CLI in lockstep while making the temporal stack far easier to test and reason about.
