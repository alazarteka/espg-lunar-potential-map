# Lunar Surface Potential Mapping Pipeline

This note distills the current implementation strategy for reconstructing a
surface potential distribution from Lunar Prospector Electron Reflectometer
(ER) observations. It is intentionally verbose so that another model (or
human) can reason about the assumptions, data flow, and known limitations
without spelunking the repository.

The document is organized as follows:

1. **Scientific intent** – what physical quantity we want to estimate and why.
2. **Data products and preprocessing** – which raw tables are consumed and how
   they are prepared for modeling.
3. **Geometry pipeline** – coordinate transforms, magnetic field tracing, and
   illumination logic.
4. **Spacecraft potential estimator** – branch logic for sunlit vs. shaded
   cases, parameterizations, and fitting procedures.
5. **Surface potential fitter** – how loss-cone spectra map to potentials on
   the surface.
6. **Cache layout (`data/potential_cache`)** – contents of the NPZ rows that
   down-stream tooling consumes.
7. **Known issues and open questions** – why certain metrics look odd (e.g.,
   high projection failure rates, binary spacecraft potentials).

Throughout, the references in backticks point to the implementation files.

---

## 1. Scientific Intent

The goal is to build a *global, time-resolved* map of the electric potential
at the lunar surface, using the ER instrument aboard Lunar Prospector (LP).
Physically, this requires a model that connects the in-situ distribution of
electrons measured in orbit to the potential that would appear on the surface
when the same magnetic field line intersects the regolith. This potential is
strongly modulated by illumination (photoemission→positive potentials on the
dayside, current balance with sparse plasma on the nightside).

The project aims to:

* Derive the spacecraft floating potential (`Φ_sc`) per ER spectrum so we can
  remove instrument charging biases from the energy spectra.
* Trace the magnetic field measured in the ER frame back to the lunar surface
  and compute the location (latitude/longitude) of the intersection.
* Fit a loss-cone model to derive the surface potential difference between the
  spacecraft and the footprint (`ΔΦ = Φ_surface − Φ_sc`).
* Accumulate these per-row measurements into caches that can be plotted, used
  for statistics, or fed to higher-level inversion schemes.

Everything is implemented in **Python 3.12** using the UV-managed virtualenv,
NumPy, SciPy, SPICE kernels via `spiceypy`, and Matplotlib/Plotly for
visualization.

---

## 2. Data Products and Preprocessing

The ER instrument produces tab-delimited spectra (`*_EXT_TAB`) with columns
for UTC timestamps, spectrum numbers, energy bins, fluxes, and supporting
quantities. The pipeline entry point `src/potential_mapper/pipeline.py`
discovers these files via `DataLoader.discover_flux_files(...)`.

Supporting inputs:

* **Attitude tables** – RA/Dec of the spin axis, used to build rotation
  matrices from the spacecraft coordinates to inertial frames.
* **SPICE kernels** – planetary positions, transformation matrices, and Sun
  vectors accessible through helper functions in `src.utils.spice_ops`.
* **Theta file** – needed by the `LossConeFitter` to map energy sweeps.

The ER data are ingested into a `FluxData`/`ERData` object (`src/flux.py`)
which wraps the Pandas table but exposes typed accessors. Each row in the ER
table becomes one “measurement point” in later stages.

---

## 3. Geometry Pipeline

All geometric calculations are orchestrated by `CoordinateCalculator`
(`src/potential_mapper/coordinates.py`):

1. **UTC→ET conversion**: Each ER row’s timestamp is converted with
   `spice.str2et`. If this fails, the row is marked invalid and skipped.
2. **Spacecraft position (`lp_positions`)**: `get_lp_position_wrt_moon(et)`
   returns the LP location in the IAU_MOON frame (km).
3. **Sun vectors**:
   * `lp_vectors_to_sun`: spacecraft→Sun vectors in the lunar-fixed frame.
   * `moon_vectors_to_sun`: Moon-center→Sun vectors, also in IAU_MOON.
4. **Attitude**: `get_current_ra_dec` interpolates RA/Dec vs. ET; we convert to
   unit vectors.
5. **Rotations**: `build_scd_to_j2000(...)` and
   `get_j2000_iau_moon_transform_matrix(et)` yield matrices for transforming
   vectors from the spacecraft coordinates to IAU_MOON.

All of these per-row arrays are stored in `CoordinateArrays`.

### Magnetic Field Projection

`project_magnetic_fields(flux_data, coordinate_arrays)`:

```python
magnetic_field = flux_data.data[config.MAG_COLS].to_numpy(dtype=float)
finite_mask = np.isfinite(magnetic_field).all(axis=1)
norms = np.linalg.norm(magnetic_field, axis=1)
valid = finite_mask & (norms > 0.0)
unit_magnetic_field = np.full_like(magnetic_field, np.nan)
unit_magnetic_field[valid] = magnetic_field[valid] / norms[valid, None]

projected_magnetic_field = np.einsum(
    "nij,nj->ni", coordinate_arrays.scd_to_iau_moon_mats, unit_magnetic_field
)
```

Rows with zero/invalid fields are logged and produce NaN vectors.

### Field-Line Tracing

`find_surface_intersection(...)` performs a ray–sphere intersection while
assuming the magnetic field direction is the direction to follow:

```python
points, mask = get_intersections_or_none_batch(
    pos=coordinate_arrays.lp_positions,
    direction=projected_magnetic_field,
    radius=config.LUNAR_RADIUS,
)
```

`get_intersections_or_none_batch` (in `src/utils/geometry.py`) solves the
quadratic equation for |p + t·v| = R and retains the **smallest positive** root.
Hence, an intersection exists only when the projected field line points toward
the Moon. If `v` aims outward (dot product with position > 0), the discriminant
or the positive-root test fail and `mask` is `False`; the corresponding entries
in `points` remain NaN. This yields the high “projection failure” percentages
observed in the data (~65% of rows per month).

### Derived Quantities

Once we have `points`:

* Spacecraft latitude/longitude are computed directly from
  `coord_arrays.lp_positions`.
* Projection latitude/longitude are computed from `points[mask]`; entries where
  `mask` is `False` stay NaN.
* Illumination flags:
  * `spacecraft_in_sun`: shoot a ray from the LP position toward the Sun and
    test if it intersects the Moon (shadow) using
    `get_intersections_or_none_batch`.
  * `projection_in_sun`: compare the surface normal `n_hat` at the footprint to
    the `moon_to_sun_hat` vector; dot > 0 ⇒ sunlit.

---

## 4. Spacecraft Potential Estimator (`src/spacecraft_potential.py`)

Each ER spectrum (`spec_no`) needs a single floating potential so that energy
bins can be corrected upstream of the loss-cone fit. The estimator branches on
illumination using a simple ray-casting test:

```python
lp_position = get_lp_position_wrt_moon(et)
lp_vector_to_sun = get_lp_vector_to_sun_in_lunar_frame(et)
intersection = get_intersection_or_none(lp_position, lp_vector_to_sun, R_moon)
is_day = intersection is None
```

### Daylight Branch

1. Fit κ parameters to the uncorrected spectrum via `Kappa.fit()`.
2. Compute the photoemission current density `J_target` using
   `electron_current_density_magnitude`.
3. Invert the JU curve with `U_from_J` (bounded between 0 and +150 V) to get an
   initial potential.
4. Shift the energy grid by `−U` (because the electrons gained that potential
   en route to the detector) and refit the spectrum.
5. Recompute `J_target` with the refitted parameters and invert again to obtain
   the final positive potential (typically +10 to +30 V).

This value is applied uniformly to every row with the same `spec_no`.

### Nightside Branch

1. Fit κ parameters to the raw spectrum.
2. Construct a logarithmic energy grid [E_min, E_max] with `n_steps`.
3. Define the current balance function
   `F(U) = Ji(U) + Jsee(U) − Je(U)`:
   * `Je`: ambient electron collection, integrates the κ distribution above the
     barrier.
   * `Jsee`: Sternglass secondary-electron emission tied to the electrons that
     still reach the surface.
   * `Ji`: orbital-motion-limited ion current.
4. Use `scipy.optimize.brentq` with brackets
   `[spacecraft_potential_low, spacecraft_potential_high]` (defaults −1500 V to
   0 V, expanding the low bracket if needed) to find a root. This yields a
   negative potential (often −50 to −70 V).
5. Map the κ θ-parameter to the ambient temperature at the solved U so the loss
   cone sees a consistent plasma.

There is no smoothing between branches: as soon as `is_day` flips, the
potential snaps to whichever branch applies.

---

## 5. Surface Potential (Loss-Cone) Fitter

Once `spacecraft_potential` is known per row, we want the additional drop
between the spacecraft and the surface footprint. This is handled by
`LossConeFitter` (`src/flux.py`), which iterates over 15-row “chunks”:

```python
fitter = LossConeFitter(er_data, theta_file, spacecraft_potential=sc_potential)
fit_mat = fitter.fit_surface_potential()  # returns matrix columns:
# [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
r = np.full(n_rows, np.nan)
for U_surface, *_ , chunk_idx in fit_mat:
    if U_surface and chi2 ok:
        r[s:e] = U_surface
```

Rows inherit the same `U_surface` within each chunk if the fit converges and
passes a `chi2` threshold; otherwise they remain NaN. `U_surface` is stored in
the cache as `rows_projected_potential` and represents the potential difference
between the surface and the spacecraft, so the total surface potential is
`Φ_surface = Φ_sc + ΔΦ`.

---

## 6. Potential Cache Layout

Every processed day produces a `3DYYMMDD.npz` under
`data/potential_cache/YYYY/DDD_RANGE`. Each archive stores row-level arrays:

| Key                              | Shape       | Description                                                  |
| -------------------------------- | ----------- | ------------------------------------------------------------ |
| `rows_utc`                       | (N,) str    | ISO timestamps per ER row                                    |
| `rows_spacecraft_latitude`/`lon` | (N,) float  | LP coordinates in degrees                                    |
| `rows_projection_latitude`/`lon` | (N,) float  | Footprint coordinates; NaN when projection failed            |
| `rows_spacecraft_potential`      | (N,) float  | Potential per row (copied from spectrum-level estimate)      |
| `rows_projected_potential`       | (N,) float  | ΔΦ per row (chunk-level fit)                                 |
| `rows_spacecraft_in_sun`         | (N,) bool   | LP illumination flag                                          |
| `rows_projection_in_sun`         | (N,) bool   | Footprint illumination flag                                   |
| `rows_spec_no`                    | (N,) int    | Spectrum numbers per row                                      |
| `spec_*` arrays                   | (~N/15,)    | Spectrum-level metadata (fit status, etc.)                    |

Downstream scripts (e.g., `scripts/analysis/plot_daily_measurements.py`) load
these NPZ files, filter out rows where lat/lon are NaN, and render the
potentials in hemispheric/global views. Time-series tools fetch the
spacecraft/surface potentials over arbitrary date windows.

---

## 7. Known Issues / Observations

1. **Projection failure rate** – About 65% of rows per month have NaN footprint
   coordinates because the magnetic field ray is only followed forward; the
   code never tries the opposite direction. This means the cache contains a
   large number of rows with valid `Φ_sc` and `ΔΦ` but unusable geometry. Plot
   scripts drop those rows.

2. **Binary spacecraft potentials** – Because we branch strictly on sunlight
   with no penumbra model or smoothing, the time series exhibits square
   transitions between day (~+20 V) and night (~−60 V). The paper being
   replicated appears to interpolate between these states, possibly by blending
   currents using solar zenith angle or by post-processing the spectrum-level
   potentials.

3. **Sunlit footprint scarcity** – Even though the spacecraft is sunlit ~70% of
   the time, only ~17% of footprints are sunlit. There are zero rows where the
   spacecraft is in eclipse but the footprint is sunlit because the illumination
   logic effectively enforces `projection_in_sun ⇒ spacecraft_in_sun`. This
   is a geometric limitation rather than a data-availability problem.

4. **Loss-cone fits inherit chunk artifacts** – Each 15-row group shares the
   same `ΔΦ`. If a chunk spans regions with different surface conditions, the
   fit averages them, and neighboring chunks can show discontinuities.

5. **Potential caching includes invalid geometry** – `rows_projected_potential`
   is populated even when `projection_latitude/longitude` are NaN. Consumers
   must mask on `np.isfinite` before interpreting the surface maps.

---

## 8. Example Workflow (Pseudocode)

```python
from pathlib import Path
from src.potential_mapper.pipeline import process_lp_file

flux_path = Path("data/ER/1998/3D980323_EXT_TAB")
results = process_lp_file(flux_path)

# Results is a PotentialResults dataclass
lat, lon = results.projection_latitude, results.projection_longitude
phi_sc = results.spacecraft_potential
delta_phi = results.projected_potential
mask = np.isfinite(lat) & np.isfinite(lon)

surface_phi = phi_sc + delta_phi
surface_phi_valid = surface_phi[mask]
lat_valid = lat[mask]
lon_valid = lon[mask]
```

```python
# Writing to cache (simplified)
np.savez(
    "data/potential_cache/1998/060_090MAR/3D980323.npz",
    rows_projection_latitude=lat,
    rows_projection_longitude=lon,
    rows_spacecraft_potential=phi_sc,
    rows_projected_potential=delta_phi,
    rows_spacecraft_in_sun=results.spacecraft_in_sun,
    rows_projection_in_sun=results.projection_in_sun,
    # plus UTC, spectrum metadata, etc.
)
```

---

## 9. Intent for Future Questions / Enhancements

When asking another model for help, key prompts might include:

* **Improving projection coverage** – Evaluate strategies for tracing the
  magnetic field in both directions, or using field models to estimate
  footpoints when the measured direction points outward.
* **Smoothing spacecraft potentials** – Devise a way to transition smoothly
  between day/night branches, perhaps blending the current balance with a
  penumbral weighting or applying time-domain filters.
* **Validating loss-cone fits** – Assess whether chunking biases the surface
  potential distribution and propose adaptive chunk sizes or regularization.
* **Illumination agreement** – Compare cached `projection_in_sun` to direct
  solar zenith angle calculations to quantify errors in the simple dot test.

This document should provide enough context for such discussions without
requiring the assistant to navigate the entire codebase.
