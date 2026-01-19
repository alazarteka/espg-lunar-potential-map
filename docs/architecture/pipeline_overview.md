# Pipeline Overview

High-level map of the ESPG lunar potential pipeline. This replaces the older
`Calculation.md` and `WALKTHROUGH.md` (now archived) with a concise summary.

## End-to-End Flow

```
ER flux (.TAB)
  -> load + validate (ERData)
  -> coordinate transforms (SCD -> J2000 -> IAU_MOON)
  -> B-field line intersection (surface footpoint)
  -> spacecraft potential U_sc (day/night branches)
  -> surface potential U_surface (loss-cone fitting)
  -> NPZ cache (batch)
  -> temporal harmonics a_lm(t)
  -> global maps + engineering products
```

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

- Data ingest: `src/flux.py`, `src/data_acquisition.py`
- Geometry + SPICE: `src/potential_mapper/coordinates.py`, `src/utils/spice_ops.py`
- Spacecraft potential: `src/spacecraft_potential.py`
- Loss-cone model + fitter: `src/model.py`, `src/flux.py`
- Batch cache: `src/potential_mapper/batch.py`
- Temporal harmonics: `src/temporal/`
- Engineering outputs: `src/engineering/`

## Recommended Entry Points

- Interactive map (small slices): `uv run python -m src.potential_mapper`
- Batch cache: `uv run python -m src.potential_mapper.batch --year 1998 --month 6`
- Harmonics: `uv run python -m src.temporal --start 1998-01-01 --end 1998-01-31 --output out.npz`
- Engineering: `uv run python -m src.engineering out.npz`

## Archived Deep Dives

- `docs/archive/legacy/Calculation.md`
- `docs/archive/legacy/WALKTHROUGH.md`
