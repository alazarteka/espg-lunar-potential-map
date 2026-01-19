# Coordinate Frames & SPICE

This pipeline transforms ER measurements from spacecraft coordinates into a
lunar body-fixed frame for surface mapping.

## Frames Used

- **SCD**: Lunar Prospector spacecraft frame (instrument data is native here)
- **J2000**: Inertial frame used by SPICE
- **IAU_MOON**: Moon body-fixed frame for surface coordinates

## Inputs

- SPICE kernels in `data/spice_kernels/`
- Attitude table `data/attitude.tab` (spin axis RA/Dec)

## Core Modules

- `src/potential_mapper/coordinates.py`
  - Builds rotation matrices per row
  - Transforms vectors SCD -> J2000 -> IAU_MOON
- `src/utils/attitude.py`
  - Parses attitude table (spin axis per time)
- `src/utils/spice_ops.py`
  - SPICE time conversions and Moon/Sun vectors

## Outputs

- Spacecraft position in IAU_MOON (km)
- Sun vector in IAU_MOON
- Per-row rotation matrices for B-field and velocity transforms
- Surface footpoints from B-field ray-sphere intersection

## Notes

- All SPICE kernels must be loaded once at startup via
  `src/potential_mapper/spice.py::load_spice_files()`.
- Attitude data is interpolated to match ER timestamps.
