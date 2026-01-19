# Coordinate Frames & SPICE

This pipeline transforms ER measurements from spacecraft coordinates into a
lunar body-fixed frame for surface mapping.

## Frames Used

- **SCD**: Lunar Prospector spacecraft frame (instrument data is native here)
- **J2000**: Inertial frame used by SPICE
- **IAU_MOON**: Moon body-fixed frame for surface coordinates

Frame chain used per measurement row:

```
SCD (instrument vectors)
  -> J2000 (inertial)
  -> IAU_MOON (body-fixed)
```

## Inputs

- SPICE kernels in `data/spice_kernels/`
  - LP ephemeris kernels (`lp_ask_*.bsp`)
  - Generic kernels (`latest_leapseconds.tls`, `pck00011.tpc`)
  - LP ephemeris emulator (`lpephemu.bsp`)
- Attitude table `data/attitude.tab` (spin axis RA/Dec)

## Core Modules

- `src/potential_mapper/coordinates.py`
  - Builds rotation matrices per row
  - Transforms vectors SCD -> J2000 -> IAU_MOON
- `src/utils/attitude.py`
  - Parses attitude table (spin axis per time)
- `src/utils/spice_ops.py`
  - SPICE time conversions and Moon/Sun vectors
- `src/potential_mapper/spice.py`
  - Loads SPICE kernels and caches handles

## Outputs

- Spacecraft position in IAU_MOON (km)
- Sun vector in IAU_MOON
- Per-row rotation matrices for B-field and velocity transforms
- Surface footpoints from B-field ray-sphere intersection

## SPICE Geometry (Positions and Sun Vectors)

The pipeline uses SPICE `spkpos` calls in J2000 and then rotates into IAU_MOON:

- LP position relative to Moon: `spkpos(LP, et, "J2000", ..., MOON)`
- Sun position relative to Moon: `spkpos(SUN, et, "J2000", ..., MOON)`
- Sun position relative to LP: `spkpos(SUN, et, "J2000", ..., LP)`

Each vector is rotated with the same `pxform("J2000", "IAU_MOON", et)` matrix.

## Time Handling

- ER rows contain UTC timestamps as strings
- SPICE uses ET; conversions happen in `src/utils/spice_ops.py`
- Attitude data is interpolated to match ER timestamps (per-row rotation)

## Transform Construction (How the Matrices Are Built)

The SCD→J2000 transform is built from the spacecraft spin axis and Sun direction
at the measurement time:

- Convert attitude RA/Dec to a unit spin axis **ẑ** in J2000.
- Compute Sun direction **ŝ** (LP→Sun) in J2000.
- Project **ŝ** onto the plane orthogonal to **ẑ** to get **x̂**:

```
x̂ = normalize(ŝ - (ŝ·ẑ) ẑ)
```

- Define **ŷ** as the right-handed cross product:

```
ŷ = ẑ × x̂
```

The rotation matrix `M_scd_to_j2000` is formed with `[x̂ ŷ ẑ]` as columns.

The J2000→IAU_MOON transform is computed directly from SPICE:

```
M_j2000_to_iau = pxform("J2000", "IAU_MOON", et)
```

Finally, the composite SCD→IAU_MOON transform is:

```
M_scd_to_iau = M_j2000_to_iau · M_scd_to_j2000
```

Vectors are transformed by matrix-vector multiplication per row. For example,
unit magnetic field vectors in SCD become body-fixed vectors in IAU_MOON.

## Surface Footpoints

After projecting B-field unit vectors into IAU_MOON, the pipeline computes
ray–sphere intersections from the LP position along ±B. This yields:

- A surface footpoint (lat/lon) where the field line intersects the Moon
- A polarity flag (+1 for +B, -1 for -B) used by diagnostics

## Practical Checks

- If the Sun vector is nearly parallel to the spin axis, the SCD X-axis
  projection becomes ill-defined and the transform is set to NaN.
- If SPICE kernels are not loaded, frame conversions will fail early
- Attitude table coverage must include the ER timestamps
- Surface footpoints assume a spherical Moon (R = 1737.4 km)
- A zero or invalid B-field row should be masked out before transforming
- Rows with invalid time or geometry are explicitly set to NaN and filtered
  downstream (fitters rely on these masks).

## Notes

- All SPICE kernels must be loaded once at startup via
  `src/potential_mapper/spice.py::load_spice_files()`.
- Attitude data is interpolated to match ER timestamps.
