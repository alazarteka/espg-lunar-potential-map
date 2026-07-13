# Protocol Amendment 01: Coordinate-Frame Sensitivity

**Frozen:** 2026-07-13, after the protocol commit and one-day extraction timing
check, but before any rank, singular-spectrum, witness, injection, or recovery
outcome.  
**Reason:** Static code inspection performed while preparing the required
geometry parity test revealed a reference-frame mismatch in both the submitted
and current production coordinate path.

## Discovery

`get_lp_vector_to_sun_in_lunar_frame_batch()` explicitly rotates the LP-to-Sun
vector from J2000 into `IAU_MOON`. `CoordinateCalculator` then supplies that
vector to `build_scd_to_j2000()`, whose two vector arguments are explicitly
documented as J2000. The attitude-derived spin axis remains J2000. Projecting one
vector onto the plane perpendicular to a vector expressed in another frame is
not a valid coordinate operation.

This mismatch is present in submitted commit `f0cab5f` and current source. It is
separate from the next-sample attitude lookup fixed by `ac1f03c`.

## Prospective Amendment

No primary criterion changes. Canonical geometry remains frame-consistent:

1. attitude spin axis in J2000;
2. LP-to-Sun vector in J2000;
3. SCD-to-J2000 construction;
4. J2000-to-IAU_MOON rotation of the MAG direction;
5. lunar surface intersection in `IAU_MOON`.

Three sensitivities are now explicitly separated:

- `legacy_attitude`: next-sample attitude with frame-consistent Sun geometry;
- `frame_mixed`: corrected interpolated attitude with the production frame
  mismatch;
- `submitted_legacy`: next-sample attitude plus the production frame mismatch.

The existing configuration's `legacy_sensitivity_attitude` therefore remains an
attitude-only contrast. This amendment adds the two frame-mismatch contrasts.
All four variants use identical raw rows, SPICE kernels, intersection code, and
downstream operator diagnostics.

## Interpretation

The frame-mixed variants are provenance and impact sensitivities, not competing
physical coordinate systems. A large difference would require a separate
production correction and reassessment of submitted map locations. It does not
alter the information-barrier design or permit fitted-potential inputs.

