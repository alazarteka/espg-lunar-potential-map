# Loss Cone Fitting - Normalization Experiments

## Objective
Match the Halekas et al. (2008) loss cone fitting results for validating our implementation.

## Reference Case from Halekas Paper

**Source**: Figure 5, page 8 of Halekas et al. (2008) JGR paper

**Exact quote from caption**:
> "Normalized energy pitch angle distribution measured at 14:45 UT on 29 April 1999, with best fit synthetic distribution (on right) from automated surface potential determination. The best fit uses the spacecraft potential USC = 11 V and finds that the lunar surface potential UM = −160 V and the magnetic field ratio BS/BM = 0.975."

**Expected results**:
- Spacecraft potential: USC = +11 V (positive, per caption)
- Lunar surface potential: UM = −160 V
- Magnetic field ratio: BS/BM = 0.975
- Timestamp: 14:45 UT on 29 April 1999 (UTC: 925397048.0)

**Our test case**:
- File: `data/1999/091_120APR/3D990429.TAB`
- Spectrum number: 653
- Timestamp: 925397048.0 (matches!)

## Experiments (caption-correct USC = +11 V)

All runs use spec_no=653 from `data/1999/091_120APR/3D990429.TAB`. Target: UM = −160 V, BS/BM = 0.975, USC = +11 V.

| Experiment | Normalization | USC [V] | U_surface [V] | BS/BM | beam_amp | χ² |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Per-energy (ratio) | None | -802.5 | 0.999 | 13.875 | 19502 |
| 2 | Global | None | -236.6 | 1.200 | 1.071 | 51877 |
| 3 | Global | -11 | -82.5 | 0.983 | 11.875 | 57739 |
| 4 | Global | +11 (expected sign) | -247.3 | 1.404 | 0.900 | 51697 |
| 5 | Per-energy (ratio) | -11 | -982.6 | 1.115 | 9.217 | 17069 |
| 6 | Global | -11 (same as 3) | -82.5 | 0.983 | 11.875 | 57739 |

Key takeaways:
- Expected setup (global, USC=+11) does not hit the target; U_surface is ~−247 V and BS/BM ~1.404.
- Best BS/BM (~0.983) occurs with the wrong sign (USC=−11) and global normalization, but U_surface is far off.
- Per-energy (“ratio”) normalization drives U_surface very negative and remains incompatible.
- Bottom line: correcting the USC sign alone does not reconcile our fits with Figure 5; likely culprits include spectrum mismatch, energy calibration, beam handling, or normalization details beyond max/ratio.

## CONCLUSIONS (updated with caption USC = +11 V)

- None of the configurations reproduce the Figure 5 target (UM = −160 V, BS/BM = 0.975) when using USC = +11 V.
- Global normalization retains energy structure better than per-energy, but with USC = +11 V it overshoots BS/BM (~1.4) and U_surface (~−247 V).
- A deeper mismatch remains (possible spectrum mismatch, energy calibration, beam handling differences, or normalization details beyond the max/ratio tested here).

## Updated quick check with corrected caption sign (USC = +11 V)

Halekas-style settings (ratio normalization, USC=+11 V, fixed beam_amp=1):

```
uv run python scripts/plots/plot_losscone_fit_paper.py \
  --input data/1999/091_120APR/3D990429.TAB \
  --spec-no 653 \
  --output scratch/losscone_paper_mode.png \
  --paper-mode
```

Result:
```
U_surface = -1138.6 V
Bs/Bm    = 1.137
beam_amp = 1.000 (fixed)
χ²       = 16892
```

The mismatch to the target persists even with USC = +11 V.

## Appendix: legacy runs under misread USC = −11 V (traceability only)

These pre-correction runs assumed USC = −11 V. They motivated the global-normalization change but were based on the wrong caption sign.

1) Per-energy, no SC: U=-802.5 V, Bs/Bm=0.999, beam=13.875, χ²=19502  
2) Global, no SC: U=-236.6 V, Bs/Bm=1.200, beam=1.071, χ²=51877  
3) Global, USC=-11: U=-82.5 V, Bs/Bm=0.983, beam=11.875, χ²=57739  
4) Global, USC=+11: U=-247.3 V, Bs/Bm=1.404, beam=0.900, χ²=51697  
5) Per-energy, USC=-11: U=-982.6 V, Bs/Bm=1.115, beam=9.217, χ²=17069  
6) Global, USC=-11 with reversed energy sign: U=-247.3 V, Bs/Bm=1.404, beam=0.900, χ²=51697

## Code Changes (historical)

### `src/flux.py::build_norm2d()`
Changed from per-energy to global normalization:

**Before**:
```python
norm2d = np.vstack([
    self._get_normalized_flux(energy_bin, measurement_chunk)
    for energy_bin in range(config.SWEEP_ROWS)
])
```

**After**:
```python
# Extract raw flux data
flux_2d = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[s:e]
pitches_2d = self.pitch_angle.pitch_angles[s:e]
incident_mask = pitches_2d < 90

# Global normalization
global_norm = float(np.nanmax(flux_2d[incident_mask]))
norm2d = flux_2d / global_norm
```

### `scripts/plots/plot_losscone_fit_paper.py`
Added spacecraft potential correction:

```python
# Create spacecraft potential array (constant -11V per Halekas et al.)
spacecraft_potential = np.full(len(er_data.data), -11.0)

# Create fitter with spacecraft potential correction
fitter = LossConeFitter(er_data, str(theta_file), pitch_angle, spacecraft_potential)
```

## References

Halekas, J. S., G. T. Delory, R. P. Lin, T. J. Stubbs, and W. M. Farrell (2008),
Lunar Prospector observations of the electrostatic potential of the lunar surface
and its response to incident currents, J. Geophys. Res., 113, A09102,
doi:10.1029/2008JA013194.
