# Coordinate Frames & SPICE

This pipeline transforms ER measurements from spacecraft coordinates into a
lunar body-fixed frame for surface mapping.

## Frames Used

- **SCD**: Lunar Prospector despun-spacecraft frame (instrument data is native here)
- **ECLIPJ2000**: Mean ecliptic and equinox of J2000; the LP attitude frame
- **J2000**: Mean equator and equinox of J2000; a distinct inertial frame
- **IAU_MOON**: Moon body-fixed frame for surface coordinates

Frame chain used per measurement row:

```
SCD (instrument vectors)
  -> ECLIPJ2000 (attitude inertial frame)
  -> IAU_MOON (body-fixed)
```

## Inputs

- SPICE kernels in `data/spice_kernels/`
  - LP ephemeris kernels (`lp_ask_*.bsp`)
  - Generic kernels (`latest_leapseconds.tls`, `pck00011.tpc`)
  - LP ephemeris emulator (`lpephemu.bsp`)
- Attitude table `data/attitude.tab` (spin-axis RA/Dec in ECLIPJ2000)

## Core Modules

- `src/potential_mapper/coordinates.py`
  - Builds rotation matrices per row
  - Transforms vectors SCD -> ECLIPJ2000 -> IAU_MOON
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

`spkpos` returns ephemeris vectors directly in the requested output frame. The
pipeline asks for each vector in the frame where it will be used:

- LP position relative to Moon:
  `spkpos(LP, et, "IAU_MOON", "NONE", MOON)`
- Sun position relative to Moon for surface illumination:
  `spkpos(SUN, et, "IAU_MOON", "NONE", MOON)`
- Sun position relative to LP for spacecraft illumination:
  `spkpos(SUN, et, "IAU_MOON", "NONE", LP)`
- Sun position relative to LP for defining SCD:
  `spkpos(SUN, et, "ECLIPJ2000", "NONE", LP)`

This avoids manually rotating ephemeris vectors through an intermediate frame.

## Time Handling

- ER rows contain UTC timestamps as strings
- SPICE uses ET; conversions happen in `src/utils/spice_ops.py`
- Attitude data is interpolated to match ER timestamps (per-row rotation)

## Transform Construction (How the Matrices Are Built)

The SCD→ECLIPJ2000 transform is built from the spacecraft spin axis and Sun
direction at the measurement time. Both defining vectors must be represented in
the same frame:

- Convert the PDS attitude RA/Dec to a unit spin axis **ẑ** in ECLIPJ2000.
- Request the Sun direction **ŝ** (LP→Sun) in ECLIPJ2000.
- Project **ŝ** onto the plane orthogonal to **ẑ** to get **x̂**:

```
x̂ = normalize(ŝ - (ŝ·ẑ) ẑ)
```

- Define **ŷ** as the right-handed cross product:

```
ŷ = ẑ × x̂
```

The rotation matrix `M_scd_to_eclip` is formed with `[x̂ ŷ ẑ]` as columns.

The ECLIPJ2000→IAU_MOON transform is computed directly from SPICE:

```
M_eclip_to_iau = pxform("ECLIPJ2000", "IAU_MOON", et)
```

Finally, the composite SCD→IAU_MOON transform is:

```
M_scd_to_iau = M_eclip_to_iau · M_scd_to_eclip
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

## External Validation

The corrected production path was checked against the official PDS Level-2
220 eV footprint product for June 1998. Among 346 records matched to a local
3-D MAG row within three seconds, 336 had a surface connection in both the
calculation and comparison set. Great-circle separation from the published
footprint was:

- median: 0.488 degrees;
- 90th percentile: 2.115 degrees;
- 95th percentile: 2.938 degrees.

The focused integration test in `tests/test_coordinate_calculator.py` anchors
one published footprint to within one degree. The broader residual includes
non-identical product cadence, the PDS attitude uncertainty, maneuver/nutation
intervals, and the spherical-Moon approximation.

## Notes

- All SPICE kernels must be loaded once at startup via
  `src/potential_mapper/spice.py::load_spice_files()`.
- Attitude data is interpolated to match ER timestamps.
- PDS attitude provenance:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ENG-6-ATTITUDE-V1.0>
- NAIF frame definitions:
  <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html>
- PDS Level-2 ER footprint product:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ER-4-ELECTRON-DATA-V1.0>
