# Potential Mapper (Batch Mode)

High-performance batch processing for lunar surface potential mapping.

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
| `--parallel` | flag | False | Legacy CPU parallel fitting (deprecated; falls back to sequential) |
| `--fast` | flag | False | Use PyTorch-accelerated fitter (GPU/CPU) |
| `--losscone-fit-method` | str | None | Loss-cone fitter (`halekas` or `lillis`, defaults to config) |
| `--overwrite` | flag | False | Overwrite existing output file |
| `-v`, `--verbose` | flag | False | Enable DEBUG-level logging |

## Differences from Standard Mode

| Aspect | `potential_mapper` | `potential_mapper.batch` |
|--------|-------------------|-------------------------|
| Fitting | Sequential per-file | Merged dataset; GPU via `--fast` |
| Data loading | Per-file | Merged into single dataset |
| Output | In-memory | Compressed NPZ cache |
| Speed | Slower | Faster with `--fast` |
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
- `rows_bs_over_bm`, `rows_beam_amp`, `rows_fit_chi2`
- `rows_spacecraft_in_sun`, `rows_projection_in_sun`

**Spectrum-level arrays** (aggregated per spectrum):
- `spec_spec_no` – unique spectrum numbers
- `spec_time_start`, `spec_time_end` – time range
- `spec_has_fit` – whether a valid fit was obtained
- `spec_row_count` – rows per spectrum

## Examples

```bash
# Process a month, overwriting existing cache
uv run python -m src.potential_mapper.batch --year 1998 --month 6 --overwrite

# Sequential mode (default)
uv run python -m src.potential_mapper.batch --year 1998 --month 1

# Custom output directory
uv run python -m src.potential_mapper.batch --year 1998 --output-dir ./my_cache

# Verbose logging
uv run python -m src.potential_mapper.batch --year 1998 --month 1 -v

# Use Lillis masked linear chi2 fitting
uv run python -m src.potential_mapper.batch --year 1998 --losscone-fit-method lillis
```

## Loading Results

```python
import numpy as np

data = np.load("artifacts/potential_cache/potential_batch_1998_06.npz")

# Access row-level data
latitudes = data["rows_projection_latitude"]
longitudes = data["rows_projection_longitude"]
potentials = data["rows_projected_potential"]

# Filter to higher-quality fits (example threshold)
chi2 = data["rows_fit_chi2"]
valid = np.isfinite(potentials) & np.isfinite(chi2) & (chi2 <= 6.57e5)
print(f"Valid fits: {valid.sum()} / {len(valid)}")
```

## Performance

- CPU parallel fitting is deprecated; use `--fast` for acceleration
- BLAS/LAPACK threading is disabled for deterministic results
- Progress bars show overall and per-spectrum progress

## GPU Acceleration

Enable GPU acceleration with the `--fast` flag for significant speedups (~160-1000x):

```bash
# GPU-accelerated batch processing
uv run python -m src.potential_mapper.batch --fast --year 1998 --month 4
```

**Requirements:**
- Install GPU extra: `uv sync --extra gpu` (or `--extra gpu-legacy` for GTX 10xx)

**Auto-detection:**
- **dtype**: Uses float16 on modern GPUs (Volta+), float32 on older GPUs or CPU
- **batch_size**: Automatically calculated from available VRAM

**Performance tips:**
- Always use `--fast` when a GPU is available
- For profiling, see `scripts/profiling/gpu_batch_sweep.py`

## Prerequisites

Run `src.data_acquisition` first to download required data files.
