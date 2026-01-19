# Diagnostics Tools

Data quality analysis and beam detection tools for lunar surface potential estimation.

## Physics Background

### Secondary Electron Beams

When the lunar surface charges to a negative potential relative to the spacecraft, secondary electrons emitted from the surface are accelerated toward the spacecraft. These appear as excess flux at high pitch angles (150-180°) in loss-cone normalized data.

The beam energy corresponds to the potential difference:

```
ΔU = U_spacecraft - U_surface
```

Since U_surface is typically negative (surface charges negative) and U_spacecraft is small positive (~10V), the beam energy approximately equals |U_surface|.

### Detection Strategy

Beam detection works by:
1. Normalizing flux data by the loss cone model (removes ambient plasma signature)
2. Identifying peaks in the high-pitch-angle band (150-180°)
3. Validating peaks have energy neighbors (filters noise)

---

## Command-Line Scripts

All scripts are in `scripts/diagnostics/`.

### losscone_peak_scan.py

Scan ER files for beam signatures in loss-cone normalized data.

```bash
# Scan a single file for beams
uv run python scripts/diagnostics/losscone_peak_scan.py data/1999/091_120APR/3D990429.TAB

# List all detected peaks
uv run python scripts/diagnostics/losscone_peak_scan.py data/1999/091_120APR/3D990429.TAB --list-peaks
```

### view_norm2d.py

Visualize normalized flux grids for specific sweeps. Useful for understanding what the detection algorithm sees.

```bash
# View a specific sweep
uv run python scripts/diagnostics/view_norm2d.py data/1999/091_120APR/3D990429.TAB --spec-no 653

# Show the high-pitch-angle detection band
uv run python scripts/diagnostics/view_norm2d.py data/1999/091_120APR/3D990429.TAB --spec-no 653 --show-band

# Compare raw vs filtered views (row MAD clip + pitch smoothing)
uv run python scripts/diagnostics/view_norm2d.py data/1999/091_120APR/3D990429.TAB --spec-no 653 --row-mad 3 --smooth-pitch 5 --compare --filter-report
```

### beam_label_studio.py

Blind browser-based labeling tool for **pre-rendered** sweep images (e.g., from
`view_norm2d.py`). It reads a manifest `selection.json` and writes your labels
to a separate JSON file. The manifest’s truth labels (e.g. `human_label`) are
**not shown** during labeling.

Requires the `diagnostics` extra: `uv sync --extra diagnostics`.

```bash
# Serve the labeling UI (writes labels_user.json next to the manifest by default)
uv run python scripts/diagnostics/beam_label_studio.py \
  --manifest artifacts/diagnostics/beam_goldset_2026-01-19/selection.json

# Resume / write to an explicit output path
uv run python scripts/diagnostics/beam_label_studio.py \
  --manifest artifacts/diagnostics/beam_goldset_2026-01-19/selection.json \
  --output artifacts/diagnostics/beam_goldset_2026-01-19/labels_al.json

# After labeling: generate a markdown comparison report
uv run python scripts/diagnostics/beam_label_studio.py \
  --manifest artifacts/diagnostics/beam_goldset_2026-01-19/selection.json \
  --output artifacts/diagnostics/beam_goldset_2026-01-19/labels_al.json \
  --report
```

### beam_detection_survey.py

Sample files across the dataset to measure detection rates and temporal trends.

```bash
# Survey detection rates (4 samples per month)
uv run python scripts/diagnostics/beam_detection_survey.py --samples-per-month 4

# More samples for better statistics
uv run python scripts/diagnostics/beam_detection_survey.py --samples-per-month 10
```

**Survey results** (as of 2026-01):
- Overall beam detection rate: ~23% of valid sweeps
- Range by month: 14-35%
- Higher rates observed in May 1998, May 1999

### beam_potential_estimate.py

Estimate lunar surface potential directly from beam peak energy. Bypasses the complex loss cone fitter for a simpler, more robust estimate.

```bash
# Basic usage
uv run python scripts/diagnostics/beam_potential_estimate.py data/1999/091_120APR/3D990429.TAB

# With custom spacecraft potential (default is 10V)
uv run python scripts/diagnostics/beam_potential_estimate.py data/1999/091_120APR/3D990429.TAB --u-spacecraft 15

# Output per-sweep estimates
uv run python scripts/diagnostics/beam_potential_estimate.py data/1999/091_120APR/3D990429.TAB --verbose
```

**Key finding**: Beam-based estimates typically yield physically reasonable values (median ~-55V) compared to loss cone fitter extremes (-600 to -900V in some cases).

### beam_losscone_crossval.py

Cross-validate beam detection estimates against loss cone fitted U_surface. Uses Latin Hypercube sampling to explore the parameter space.

```bash
# Run cross-validation
uv run python scripts/diagnostics/beam_losscone_crossval.py data/1999/091_120APR/3D990429.TAB

# Detailed comparison output
uv run python scripts/diagnostics/beam_losscone_crossval.py data/1999/091_120APR/3D990429.TAB --verbose
```

**Key insight**: The loss cone fitter's chi² landscape is often flat, leading to multiple local minima. Beam-based validation helps identify when the fitter finds physically unreasonable solutions.

---

## Interactive Browser Tools

These tools require the `diagnostics` extra: `uv sync --extra diagnostics`

### losscone_studio.py

Interactive browser-based loss cone viewer. Visualizes raw flux, normalized flux, and model fits with parameter sliders.

```bash
uv run python scripts/diagnostics/losscone_studio.py data/1998/060_090MAR/3D980323.TAB
```

Opens a Panel/Bokeh app in your browser with:
- Raw log-flux heatmap
- Normalized flux heatmap
- Loss cone boundary overlay
- Parameter sliders (U_surface, Bs/Bm, beam amplitude, etc.)

**Beam Filter**: Toggle "Beam-detected only" to navigate only through sweeps where beam signatures were detected. The app uses the same detection thresholds as `losscone_peak_scan.py` (min_peak=2.0, min_neighbor=1.5, etc.). When enabled, the chunk slider shows filtered index, and metrics display the actual chunk number and spec_no.

### losscone_orbit_studio.py

Multi-panel orbit diagnostics with flux correction and SPICE geometry.

```bash
uv run python scripts/diagnostics/losscone_orbit_studio.py data/1999/091_120APR/3D990429.TAB
```

Features:
- Orbit timeline with spacecraft-moon geometry
- Sun illumination status
- Spacecraft potential series
- Energy-resolved flux panels

---

## Detection Thresholds

Tuned to filter noise while keeping real beams:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `min_peak` | 2.0 | Peak normalized flux must exceed this |
| `min_neighbor` | 1.5 | At least one adjacent energy bin must exceed this |
| `min_band_points` | 5 | Minimum valid pitch bins in the 150-180° band |
| `energy_min` | 20 eV | Minimum energy included in detection |
| `energy_max` | 500 eV | Maximum energy included in detection |
| `high_energy_floor` | 400 eV | High-energy deficit check lower bound |
| `high_energy_ratio_max` | 0.5 | High-energy mean must be ≤ this fraction of peak |
| `peak_width_max` | 4 bins | Max contiguous bins above half-peak |
| `contiguity_min_bins` | 3 bins | Min contiguous pitch bins above threshold |
| `contiguity_min_value` | 1.5 | Threshold used for pitch contiguity |

These thresholds were determined empirically through the beam detection survey.
The energy window defaults to 20–500 eV; override via `losscone_peak_scan.py --energy-min/--energy-max` if needed.

### Heuristic Update Notes (2026-01-19)

Added two lightweight checks to the beam detector:
- **High-energy deficit**: after the peak, the high-pitch band should drop at
  high energies (default floor 400 eV, or 2x peak energy). This matches the
  expected beam signature (localized low-energy excess).
- **Peak width constraint**: limits the number of contiguous energy bins above
  half-peak (default max 4) to avoid overly broad enhancements.
- **Pitch contiguity**: requires a short contiguous run of pitch bins above a
  threshold in the high-pitch band (default min 3 bins at ≥1.5).

**Empirical review (5 files, 20 samples)**:
- Files sampled: `3D980503`, `3D980523`, `3D990323`, `3D990429`, `3D990505`
- Detection rates by file: 31.9%, 21.0%, 11.2%, 31.2%, 22.4%
- Eyeball vs detector: 9/10 detected samples looked beam-like; 4/10 non-detected
  samples were ambiguous or weak beams (low-contrast/broad bands).

These findings suggest the detector is conservative. A future "soft beam" mode
may be warranted for low-contrast or broad beams, but it is not yet implemented.

---

## Python API

The `src.diagnostics` module provides programmatic access:

```python
from src.diagnostics import (
    LossConeSession,
    compute_loss_cone_boundary,
    interpolate_to_regular_grid,
)

# Load a file for interactive analysis
session = LossConeSession("data/1999/091_120APR/3D990429.TAB")

# Get data for a specific sweep
chunk = session.get_chunk_data(chunk_idx=0)

# Compute loss cone boundary for given parameters
boundary = compute_loss_cone_boundary(energies, u_surface=-50.0, bs_over_bm=0.8)

# Interpolate irregular data to regular grid
grid = interpolate_to_regular_grid(energies, pitches, flux, n_pitch_bins=36)
```

### LossConeSession

Main class for loading and analyzing ER flux files:
- Loads flux data and groups by spectrum number
- Provides chunk-based access for sweep-by-sweep analysis
- Integrates with SPICE for geometry calculations

### Helper Functions

- `compute_loss_cone_boundary()`: Calculate the pitch angle boundary for given surface potential
- `interpolate_to_regular_grid()`: Resample irregular pitch angle data to uniform grid
