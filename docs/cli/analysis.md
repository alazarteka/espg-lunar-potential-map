# Analysis & Plotting Scripts

Exploratory and publication scripts that consume the pipeline's cached NPZ output
(`artifacts/potential_cache/`) and temporal-coefficient bundles. Every script
supports `--help`; run them with `uv run python scripts/<dir>/<name>.py --help`.

- **`scripts/analysis/`** — reusable, shareable analysis and exploration plots.
- **`scripts/plots/`** — publication ("paper") figures, styled via
  `src.visualization`.

## `scripts/analysis/`

### Surface potential maps (from cached NPZ)
| Script | Purpose |
|--------|---------|
| `potential_map_matplotlib_sphere.py` | 3D lunar-sphere render of cached potentials over a date range (Matplotlib). |
| `potential_map_plotly_static.py` | Interactive Plotly globe of cached potentials, with optional animation / MP4 export. |
| `potential_map_plotly_terminator.py` | Cached potentials with day/night shading from SPICE geometry. |

### Time series
| Script | Purpose |
|--------|---------|
| `potential_timeseries_plotly.py` | Potential time series vs. angular distance to the subsolar point (Plotly). |
| `potential_timeseries_daily_png.py` | Per-day time-series PNGs from monthly cache NPZ files. |
| `spacecraft_potential_series.py` | Spacecraft potential estimates over a single day. |

### Temporal spherical-harmonic reconstruction
| Script | Purpose |
|--------|---------|
| `temporal_cv.py` | Cross-validate the temporal spherical-harmonic basis. |
| `temporal_cv_grid.py` | Grid search over `(lmax, λ)` for the temporal basis via CV. |
| `temporal_harmonics_analysis.py` | Analyze harmonic-coefficient evolution over time. |
| `temporal_harmonics_l_curve.py` | L-curve for the temporal-regularization strength `λ`. |
| `temporal_harmonics_interactive_map.py` | Interactive browser map of reconstructed potentials. |
| `temporal_harmonics_animate.py` | Animate hemispheric + global reconstructions over time (consumes `python -m src.temporal` bundles). |
| `temporal_harmonics_animate_terminator.py` | As above, with a lunar-terminator overlay. |

### Day-level exploration
| Script | Purpose |
|--------|---------|
| `electron_density_temperature.py` | Electron density and temperature for a selected day. |
| `energy_pitch_explorer.py` | Interactive energy–pitch explorer for a day's spectra. |
| `flux_pitch_plot.py` | Flux vs. energy and pitch for one spectrum. |

### Terminator charge
| Script | Purpose |
|--------|---------|
| `potential_terminator_charge.py` | Estimate surface charge density across the terminator from cached NPZ rows. |
| `potential_charge_report_md.py` | Convert terminator-charge JSON reports into Markdown summaries (consumes the above). |

### Statistics
| Script | Purpose |
|--------|---------|
| `compute_regime_stats.py` | Plasma-regime statistics (paper Table 1). |
| `residual_analysis.py` | Model residual (U_obs − U_model) breakdown. |
| `summarize_daily_measurements.py` | Aggregate sunlit/shaded measurement statistics over a date range. |

## `scripts/plots/` (publication figures)

| Script | Figure |
|--------|--------|
| `plot_harmonic_reconstruction_multiday.py` | 2×3 grid of spherical-harmonic reconstructions from monthly data. |
| `plot_harmonic_reconstruction_paper.py` | Single spherical-harmonic reconstruction map. |
| `plot_losscone_fit_paper.py` | Measured vs. model loss-cone fit. |
| `plot_measurements_global.py` | Global surface-measurement projection. |
| `plot_measurements_hemispheric.py` | Hemispheric surface-measurement projection. |
| `plot_terminator_profile_paper.py` | Terminator potential profile. |
| `plot_timeseries_potential.py` | Potential time series at a specific lunar-surface location. |

See also [diagnostics.md](diagnostics.md) for beam-detection / loss-cone diagnostic
tools and [visualization.md](../visualization/visualization.md) for the shared plot
styling.
