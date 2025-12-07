# Potential Mapper (Batch Mode)

High-performance batch processing for lunar surface potential mapping with parallel fitting.

## Usage

```bash
uv run python -m src.potential_mapper.batch [OPTIONS]
```

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--output-dir` | path | `artifacts/potential_cache` | Directory for output NPZ files |
| `--year` | int | None | Filter to specific year |
| `--month` | int | None | Filter to specific month (1-12) |
| `--day` | int | None | Filter to specific day (1-31) |
| `--no-parallel` | flag | False | Disable parallel fitting (sequential mode) |
| `--overwrite` | flag | False | Overwrite existing output file |
| `-v`, `--verbose` | flag | False | Enable DEBUG-level logging |

## Differences from Standard Mode

| Aspect | `potential_mapper` | `potential_mapper.batch` |
|--------|-------------------|-------------------------|
| Fitting | Sequential per-file | Parallel across spectra |
| Data loading | Per-file | Merged into single dataset |
| Output | In-memory | Compressed NPZ cache |
| Speed | Slower | Faster (multiprocessing) |
| Use case | Interactive/debugging | Large-scale processing |

## Output Files

Output filename is auto-generated based on date filters:

- `potential_batch_1998.npz` – if `--year 1998`
- `potential_batch_1998_06.npz` – if `--year 1998 --month 6`
- `potential_batch_1998_06_15.npz` – if `--year 1998 --month 6 --day 15`
- `potential_batch_all.npz` – if no date filters

### NPZ Contents

**Row-level arrays** (one entry per ER measurement row):
- `rows_spec_no` – spectrum number
- `rows_utc`, `rows_time` – timestamps
- `rows_spacecraft_latitude`, `rows_spacecraft_longitude`
- `rows_projection_latitude`, `rows_projection_longitude`
- `rows_spacecraft_potential`, `rows_projected_potential`
- `rows_spacecraft_in_sun`, `rows_projection_in_sun`

**Spectrum-level arrays** (aggregated per spectrum):
- `spec_spec_no` – unique spectrum numbers
- `spec_time_start`, `spec_time_end` – time range
- `spec_has_fit` – whether a valid fit was obtained
- `spec_row_count` – rows per spectrum

## Examples

```bash
# Process an entire year with parallel fitting
uv run python -m src.potential_mapper.batch --year 1998

# Process a month, overwriting existing cache
uv run python -m src.potential_mapper.batch --year 1998 --month 6 --overwrite

# Sequential mode (for debugging or memory-constrained systems)
uv run python -m src.potential_mapper.batch --year 1998 --month 1 --no-parallel

# Custom output directory
uv run python -m src.potential_mapper.batch --year 1998 --output-dir ./my_cache

# Verbose logging
uv run python -m src.potential_mapper.batch --year 1998 --month 1 -v
```

## Loading Results

```python
import numpy as np

data = np.load("artifacts/potential_cache/potential_batch_1998_06.npz")

# Access row-level data
latitudes = data["rows_projection_latitude"]
longitudes = data["rows_projection_longitude"]
potentials = data["rows_projected_potential"]

# Filter to valid fits
valid = np.isfinite(potentials)
print(f"Valid fits: {valid.sum()} / {len(valid)}")
```

## Performance

- Uses `multiprocessing` with CPU count - 1 workers
- BLAS/LAPACK threading is disabled for deterministic results
- Progress bars show overall and per-spectrum progress

## Prerequisites

Run `src.data_acquisition` first to download required data files.
