# Pipeline Overview

High-level map of the ESPG lunar potential pipeline.

## End-to-End Flow

```
ER flux (.TAB)
  -> load + validate (ERData)
  -> coordinate transforms (SCD -> ECLIPJ2000 -> IAU_MOON)
  -> B-field line intersection (surface footpoint)
  -> spacecraft potential U_sc (day/night branches)
  -> surface potential U_surface (loss-cone fitting, per measurement)
  -> NPZ cache (batch)
  -> temporal harmonics a_lm(t) (identifiability / sampling-limits test)
  -> engineering products (per-measurement statistics)
```

The temporal harmonics step fits spherical-harmonic coefficients jointly in
space and time, but that joint fit is the paper's identifiability test, not a
mapping deliverable: Lunar Prospector's instantaneous spatial coverage is too
sparse to constrain the joint space-time reconstruction, so a global
spatiotemporal surface-potential map is not recoverable from LP ER data (this
is further entangled with the spacecraft's own floating potential). The
per-measurement loss-cone potential estimate and the aggregate/per-site
measurement statistics remain valid outputs.

## Forward vs Inverse Steps

**Forward (deterministic):**
- Coordinate transforms and geometry
- Loss-cone forward model (given parameters)
- Harmonic reconstruction from coefficients

**Inverse (fitting):**
- Kappa distribution fit to spectra
- Spacecraft potential (current balance / JU inversion)
- Surface potential loss-cone fit
- Harmonic coefficient fitting

## Primary Modules

- Data ingest: `src/losscone/cpu.py` (shim: `src/flux.py`), `src/data_acquisition.py`
- Geometry + SPICE: `src/potential_mapper/coordinates.py`, `src/utils/spice_ops.py`
- Spacecraft potential: `src/spacecraft_potential.py`
- Loss-cone model + fitter: `src/losscone/model.py`, `src/losscone/cpu.py` (shims: `src/model.py`, `src/flux.py`)
- GPU loss-cone fitting: `src/losscone/torch/` (shim: `src/losscone_torch.py`, legacy: `src/model_torch.py`)
- Batch cache: `src/potential_mapper/batch.py`
- Temporal harmonics: `src/temporal/`
- Engineering outputs: `src/engineering/`

## Recommended Entry Points

- Interactive map (small slices): `uv run python -m src.potential_mapper`
- Batch cache: `uv run python -m src.potential_mapper.batch --year 1998 --month 6`
- Harmonics (identifiability / sampling-limits test): `uv run python -m src.temporal --start 1998-01-01 --end 1998-01-31 --output out.npz`
- Engineering: `uv run python -m src.engineering out.npz`
