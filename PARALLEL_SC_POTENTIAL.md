# Parallel Spacecraft Potential Implementation

## Summary

Implemented parallel processing for spacecraft potential calculation, addressing the major performance bottleneck identified in the pipeline.

## Performance Impact

### Before Parallelization (April 1998, 444k rows):
- **Total runtime**: 2660s (44.3m)
  - Coordinates: 71s (3%)
  - **SC Potential: 1890s (71%)** ← Major bottleneck
  - Surface Fitting: 683s (26%)

### After Parallelization (Expected):
- **SC Potential**: ~300-400s (est. 5-6x speedup with 7 workers)
- **Total runtime**: ~1100-1200s (18-20m)
- **Overall improvement**: ~60% faster

## Implementation Details

### Key Components

1. **`spacecraft_potential_worker()`** (`pipeline.py:32-65`)
   - Worker function for multiprocessing pool
   - Creates isolated ERData for each spectrum
   - Handles exceptions gracefully
   - Returns (spec_no, row_indices, potential_value)

2. **`_spacecraft_potential_per_row_parallel()`** (`pipeline.py:150-223`)
   - Orchestrates parallel execution
   - Groups data by spec_no
   - Manages worker pool with SPICE initialization
   - Uses `imap_unordered` for memory efficiency
   - Progress tracking with tqdm

3. **`_init_worker_spice()`** (`pipeline.py:26-29`)
   - Initializes SPICE kernels in each worker process
   - Ensures thread-safety for SPICE operations

### Design Decisions

**Thread Safety:**
- Uses multiprocessing (not threading) for true parallelism
- Each worker initializes its own SPICE kernel pool
- DataFrame copies ensure no shared mutable state

**Data Isolation:**
- `group_df.copy()` creates independent DataFrame for each worker
- `ERData.from_dataframe()` creates isolated ERData instance
- `calculate_potential()` mutations are worker-local

**Load Balancing:**
- Adaptive chunksize: `max(1, len(tasks) // (num_workers * 4))`
- `imap_unordered` for better memory efficiency vs `map`
- Workers = `cpu_count - 1` to leave one core for system

**Error Handling:**
- Try/except in `process_merged_data()` with fallback to sequential
- Worker exceptions logged but don't crash entire batch
- Failed spectra return None and are skipped

## Integration

### Updated Functions

**`process_merged_data(er_data, *, use_parallel=False)`**
- Added parallel/sequential branching for SC potential
- Fallback to sequential on `PermissionError` or `OSError`
- Updated docstring to reflect parallel capabilities

**`batch.py`**
- Already had `use_parallel` parameter
- Now applies to both SC and surface potential

### Backward Compatibility

- Sequential version preserved as `_spacecraft_potential_per_row()`
- Default behavior unchanged (`use_parallel=False`)
- Existing tests continue to pass
- `process_lp_file()` wrapper unchanged

## Testing

### Correctness

Parallel and sequential produce identical results:
- Same NaN masks
- Floating-point values match within `rtol=1e-9`
- All PotentialResults fields identical

### Benchmark Scripts

1. **`scripts/dev/test_parallel_sc_potential.py`**
   - Full correctness validation
   - Compares all output fields
   - Runs on single day

2. **`scripts/dev/bench_sc_parallel.py`**
   - Focused SC potential benchmark
   - Measures sequential vs parallel
   - Reports speedup and time saved

## Usage

### Command Line

```bash
# Parallel (default for batch.py)
uv run python -m src.potential_mapper.batch --year 1998 --month 4

# Sequential (for comparison)
uv run python -m src.potential_mapper.batch --year 1998 --month 4 --no-parallel
```

### Programmatic

```python
from src.potential_mapper import pipeline

# Load data
er_data = pipeline.load_all_data(files)

# Parallel processing
results = pipeline.process_merged_data(er_data, use_parallel=True)

# Sequential processing (fallback or testing)
results = pipeline.process_merged_data(er_data, use_parallel=False)
```

## Performance Characteristics

### Scaling

- **Bottleneck**: Individual spectrum κ-fitting
- **Parallelism**: Spectrum-level (977 spectra/day, ~30k spectra/month)
- **Worker utilization**: High (minimal overhead)
- **Memory overhead**: Moderate (isolated DataFrames per worker)

### When to Use Parallel

- ✅ **Use parallel**: Multi-day or month-long datasets (>50k rows)
- ✅ **Use parallel**: Systems with 4+ CPU cores
- ⚠️ **Consider sequential**: Single day (<15k rows) - overhead may not be worth it
- ⚠️ **Consider sequential**: Memory-constrained systems

## Known Limitations

1. **Multiprocessing overhead**: Small datasets (<1000 spectra) may not benefit
2. **SPICE initialization**: Each worker loads kernels (one-time cost)
3. **Memory**: Each worker holds ~1/N of dataset in memory
4. **Non-deterministic ordering**: Results collected out-of-order (doesn't affect correctness)

## Future Optimizations

Potential further improvements (not implemented):

1. **Dynamic worker scaling**: Adjust worker count based on dataset size
2. **Shared SPICE pool**: Investigate shared-memory SPICE if thread-safe
3. **Prefetching**: Overlap data loading with computation
4. **Hybrid approach**: Parallel dayside, sequential nightside (nightside is slower)
5. **Caching**: Memoize geometry calculations across spectra

## References

- Original sequential implementation: `pipeline.py:96-147`
- Spacecraft potential physics: `src/spacecraft_potential.py`
- Performance discussion: Issue #TBD
