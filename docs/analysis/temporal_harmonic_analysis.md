# Time-Dependent Spherical Harmonic Analysis of Lunar Surface Potential

## Scientific Goal

Represent lunar surface potential as a time-dependent spherical harmonic expansion:

$$U(\phi, \theta, t) = \sum_{l=0}^{l_{\text{max}}} \sum_{m=-l}^{l} a_{lm}(t) \, Y_{lm}(\phi, \theta)$$

where:
- **U(φ, θ, t)**: Surface potential at latitude φ, longitude θ, time t
- **Y_lm(φ, θ)**: Spherical harmonic basis functions of degree l, order m
- **a_lm(t)**: Time-dependent complex coefficients to be determined

This approach captures both:
1. **Spatial structure**: Through spherical harmonic basis (degree l controls spatial resolution)
2. **Temporal evolution**: Through time-dependent coefficients a_lm(t)

## Implementation

### 1. Compute Temporal Coefficients

Use `scripts/dev/temporal_harmonic_coefficients.py` to discretize time into windows and fit coefficients:

```bash
uv run python scripts/dev/temporal_harmonic_coefficients.py \
  --start 1998-01-01 \
  --end 1998-12-31 \
  --lmax 5 \
  --window-hours 24.0 \
  --min-coverage 0.15 \
  --regularize-l2 1e-3 \
  --output data/temporal_coefficients_1998.npz
```

**Key Parameters:**
- `--lmax`: Maximum spherical harmonic degree (lmax=5 → 36 coefficients per window)
- `--window-hours`: Temporal resolution (24h = daily snapshots)
- `--min-coverage`: Minimum global coverage fraction (0.15 = 15% of 10°×10° bins)
- `--regularize-l2`: Ridge penalty to prevent overfitting to sparse data

**Algorithm:**

1. **Load all measurements** in date range with (lat, lon, potential) tuples
2. **Partition into time windows** of specified duration
3. **For each window:**
   - Check spatial coverage (reject if < min_coverage)
   - Build design matrix X where X_ij = Y_lm(φ_i, θ_i)
   - Solve weighted least squares: (X^T X + λI) a = X^T Φ
   - Store coefficients a_lm with metadata (n_samples, coverage, RMS)
4. **Save to NPZ** with structure:
   - `times`: datetime64 array of window midpoints
   - `coeffs`: complex array of shape (n_windows, n_coeffs)
   - `rms_residuals`: fit quality per window

### 2. Analyze Coefficient Evolution

Use `scripts/analysis/temporal_harmonics_analysis.py` to visualize a_lm(t):

```bash
uv run python scripts/analysis/temporal_harmonics_analysis.py \
  --input data/temporal_coefficients_1998.npz \
  --output-dir plots/temporal_harmonics \
  --snapshot-times 0 50 100
```

**Generates:**
- **Coefficient time series**: Re[a_lm(t)] and Im[a_lm(t)] vs. time for key modes
- **Fit quality evolution**: RMS residuals, sample counts, spatial coverage over time
- **Power spectrum**: Average |a_lm| by degree l (shows dominant spatial scales)
- **Snapshot maps**: Reconstructed U(φ, θ) at selected times

### 3. Reconstruct Potential at Arbitrary (φ, θ, t)

Once you have a_lm(t), reconstruct surface potential:

```python
def reconstruct_potential(lat_deg, lon_deg, time_index, coeffs, lmax):
    """Evaluate U(φ, θ, t) at specific location and time."""
    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)
    colat = np.pi/2 - lat_rad
    
    potential = 0.0
    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            Y_lm = sph_harm(m, l, lon_rad, colat)
            potential += np.real(coeffs[time_index, idx] * Y_lm)
            idx += 1
    
    return potential
```

## Key Improvements Over Static Fitting

### Previous Approach (potential_cache_reader.py)
❌ Merged all measurements across entire date range  
❌ Fitted single coefficient set → time-averaged field  
❌ Mixed sunlit/shadow conditions  
❌ No temporal resolution  

### New Approach (temporal_harmonic_coefficients.py)
✅ Separate fits per time window → a_lm(t) time series  
✅ Preserves temporal evolution  
✅ Quality filtering per window (coverage, sample count)  
✅ Enables correlation analysis with external drivers (solar wind, terminator crossing)  

## Scientific Questions Enabled

With a_lm(t), you can now investigate:

1. **Monopole evolution**: Does the global average potential (a_00(t)) vary with solar activity?
2. **Day-night asymmetry**: Track dipole components (a_1m(t)) across terminator crossings
3. **Polar vs. equatorial**: Examine zonal harmonics (m=0) for latitude-dependent processes
4. **Periodicities**: Fourier transform a_lm(t) to find dominant timescales (lunar day, solar rotation)
5. **Event response**: Correlate coefficient changes with known plasma events

## Validation & Quality Control

### Spatial Coverage Check
Each window must have measurements distributed globally (not just along a single orbit track):
- `min_coverage=0.15`: At least 15% of 10°×10° bins populated
- Polar orbits naturally provide good coverage after ~few hours

### Temporal Sampling
- `window_hours=24`: Daily snapshots for 1 year → ~365 coefficient sets
- Too short: Poor spatial coverage per window
- Too long: Temporal averaging loses variability

### Regularization
- `l2_penalty=1e-3`: Prevents high-degree modes from fitting noise
- Cross-validation: Try multiple λ values, check RMS vs. coefficient norm

### Residual Analysis
Check `rms_residuals` array:
- Systematic trends → missing physics (e.g., local topography)
- Sudden spikes → data quality issues or real transient events

## Example Workflow

```bash
# 1. Compute temporal coefficients for 1998
uv run python scripts/dev/temporal_harmonic_coefficients.py \
  --start 1998-01-01 --end 1998-12-31 \
  --lmax 5 --window-hours 24 \
  --min-coverage 0.15 --regularize-l2 1e-3 \
  --output data/temporal_coefficients_1998.npz

# 2. Analyze and visualize
uv run python scripts/analysis/temporal_harmonics_analysis.py \
  --input data/temporal_coefficients_1998.npz \
  --output-dir plots/temporal_1998

# 3. Compare different years
for year in 1998 1999; do
  uv run python scripts/dev/temporal_harmonic_coefficients.py \
    --start ${year}-01-01 --end ${year}-12-31 \
    --lmax 5 --window-hours 24 \
    --output data/temporal_coefficients_${year}.npz
done
```

## Next Steps

1. **Run on full dataset**: Process all available years (1998-1999 based on your data/)
2. **Coefficient interpolation**: Fit smooth functions (splines, Fourier series) to a_lm(t)
3. **Physical interpretation**: Correlate dominant modes with known processes:
   - Solar zenith angle dependence
   - Spacecraft orbital parameters
   - Plasma environment measurements
4. **Uncertainty quantification**: Bootstrap resampling to estimate σ[a_lm(t)]
5. **Predictive model**: Use learned a_lm(t) to interpolate/extrapolate to unmeasured times

## Technical Notes

### Spherical Harmonic Convention
- Uses `scipy.special.sph_harm` (or `sph_harm_y` in SciPy ≥1.15)
- Condon-Shortley phase included in Y_lm
- Real-valued coefficients for m=0, complex pairs for |m|>0

### Storage Efficiency
- Compressed NPZ format: ~1 MB per 1000 windows with lmax=5
- Complex128 for coefficients (necessary for m≠0 terms)
- Datetime64[ns] for timestamps (nanosecond precision)

### Computational Cost
- Per-window fitting: O(N × n_coeffs²) where N = samples per window
- Typical: 1000 samples, lmax=5 (36 coeffs) → ~1 ms per window
- Full year (365 windows): ~1 second + I/O overhead

## References

For spherical harmonic analysis of planetary fields:
- Whaler & Purucker (2005): Spherical harmonic analysis of Mars
- Anderson et al. (2012): Global mapping of lunar magnetic fields
- Halekas et al. (2011): Lunar Prospector plasma observations
