# Temporal Harmonics

Fits time-dependent spherical harmonic coefficients a_lm(t) to Lunar Prospector
ER surface-potential measurements. Lunar Prospector's instantaneous spatial
coverage is small relative to the surface, so recovering a_lm(t) requires
resolving joint space-time variation from data that cannot identify it: this
tool is the identifiability / sampling-limits analysis behind the paper's
negative result that **a global spatiotemporal lunar surface-potential map
cannot be recovered from LP ER data** — it is not a working map-reconstruction
deliverable. Per-measurement loss-cone inversion and the aggregate/per-site
statistics in `src.engineering` (see [engineering.md](engineering.md)) remain
valid and are unaffected by this limitation.

## Usage

```bash
uv run python -m src.temporal --start YYYY-MM-DD --end YYYY-MM-DD --output FILE [OPTIONS]
```

## Required Arguments

| Flag | Type | Description |
|------|------|-------------|
| `--start` | YYYY-MM-DD | Start date (inclusive) |
| `--end` | YYYY-MM-DD | End date (inclusive) |
| `--output` | path | Output NPZ file for coefficients |

## Optional Arguments

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--cache-dir` | path | `artifacts/potential_cache` | Directory with potential NPZ files |
| `--require-u-identifiable` | flag | False | Only use measurements whose U-surface fit passes identifiability QC (requires `rows_u_is_identifiable_lhs_dchi2red_0p001` in cache NPZs) |
| `--lmax` | int | 5 | Maximum spherical harmonic degree |
| `--fit-mode` | str | `window` | Fitting mode: `window` or `basis` |
| `--temporal-basis` | str | `constant,synodic` | Comma-separated basis (basis mode only) |
| `--window-hours` | float | 24.0 | Temporal window duration (hours) |
| `--window-stride` | float | None | Stride for overlapping windows |
| `--l2-penalty` | float | 0.0 | Ridge penalty (spatial regularization) |
| `--temporal-lambda` | float | 0.0 | Temporal continuity regularization |
| `--max-lag` | int | 1 | Max lag for multi-scale temporal regularization |
| `--decay-factor` | float | 0.5 | Weight decay per lag step |
| `--min-samples` | int | 100 | Minimum measurements per window |
| `--min-coverage` | float | 0.1 | Minimum spatial coverage (0-1) |
| `--co-rotate` | flag | False | Use solar co-rotating frame |
| `--rotation-period-days` | float | 29.53 | Rotation period (synodic month) |
| `--spatial-weight-exponent` | float | None | Degree-weighted damping |
| `-v`, `--verbose` | flag | False | Enable DEBUG logging |

## What It Does

1. Loads potential NPZ files from the cache directory
2. Filters measurements to the specified date range
3. **Window mode**: partitions data into temporal windows (default: 24h)
4. **Basis mode**: fits temporal basis functions across the full range
5. Fits spherical harmonic coefficients up to degree `lmax`
6. Optionally applies temporal smoothing between windows
7. Saves coefficients to compressed NPZ file

The fitted coefficients are the artifact this identifiability analysis
operates on — the `--min-coverage`, `--min-samples`, `--l2-penalty`, and
`--temporal-lambda` knobs exist to probe how much regularization is needed
to make the joint space-time fit tractable, which is itself evidence of the
underdetermination the paper reports. Use `--require-u-identifiable` to
additionally exclude measurements where the fit is entangled with an
unidentifiable spacecraft potential.

## Fit Modes

- **window**: Independent fits per time window (default).
- **basis**: Fits coefficients using a temporal basis (e.g., constant + synodic).

## Output Format

The output NPZ contains:
- `times` – datetime64 midpoints of each window
- `lmax` – maximum spherical harmonic degree
- `coeffs` – complex array of shape `(n_windows, (lmax+1)²)`
- `n_samples` – measurement count per window
- `spatial_coverage` – coverage fraction per window
- `rms_residuals` – RMS fit residual per window

## Examples

```bash
# Basic: daily coefficients for January 1998
uv run python -m src.temporal \
    --start 1998-01-01 \
    --end 1998-01-31 \
    --output artifacts/jan98_harmonics.npz

# Higher resolution with temporal smoothing
uv run python -m src.temporal \
    --start 1998-01-01 \
    --end 1998-02-28 \
    --lmax 8 \
    --temporal-lambda 1e-2 \
    --output artifacts/harmonics_smooth.npz

# Solar co-rotating frame (synodic period)
uv run python -m src.temporal \
    --start 1998-01-01 \
    --end 1998-03-31 \
    --co-rotate \
    --output artifacts/harmonics_corot.npz

# Overlapping 12-hour windows with 6-hour stride
uv run python -m src.temporal \
    --start 1998-01-01 \
    --end 1998-01-07 \
    --window-hours 12 \
    --window-stride 6 \
    --output artifacts/harmonics_overlap.npz
```

## Loading Results

```python
import numpy as np

data = np.load("artifacts/jan98_harmonics.npz")

times = data["times"]
coeffs = data["coeffs"]  # Complex coefficients
lmax = int(data["lmax"])

print(f"Windows: {len(times)}, lmax: {lmax}, coeffs per window: {(lmax+1)**2}")
```

## Prerequisites

Requires potential cache files from `src.potential_mapper.batch`:

```bash
# First: generate potential cache
uv run python -m src.potential_mapper.batch --year 1998 --month 1

# Then: compute harmonics
uv run python -m src.temporal --start 1998-01-01 --end 1998-01-31 --output out.npz
```
