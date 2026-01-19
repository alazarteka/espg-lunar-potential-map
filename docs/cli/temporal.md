# Temporal Harmonics

Computes time-dependent spherical harmonic coefficients a_lm(t) for lunar surface potential.

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
| `--lmax` | int | 5 | Maximum spherical harmonic degree |
| `--window-hours` | float | 24.0 | Temporal window duration (hours) |
| `--window-stride` | float | None | Stride for overlapping windows |
| `--l2-penalty` | float | 0.0 | Ridge penalty (spatial regularization) |
| `--temporal-lambda` | float | 0.0 | Temporal continuity regularization |
| `--min-samples` | int | 100 | Minimum measurements per window |
| `--min-coverage` | float | 0.1 | Minimum spatial coverage (0-1) |
| `--co-rotate` | flag | False | Use solar co-rotating frame |
| `--rotation-period-days` | float | 29.53 | Rotation period (synodic month) |
| `--spatial-weight-exponent` | float | None | Degree-weighted damping |
| `-v`, `--verbose` | flag | False | Enable DEBUG logging |

## What It Does

1. Loads potential NPZ files from the cache directory
2. Filters measurements to the specified date range
3. Partitions data into temporal windows (default: 24h)
4. For each window, fits spherical harmonic coefficients up to degree `lmax`
5. Optionally applies temporal smoothing between windows
6. Saves coefficients to compressed NPZ file

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
