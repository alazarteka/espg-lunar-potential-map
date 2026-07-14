# Post-Outcome Coordinate Audit: Attitude Reference Frame

**Recorded:** 2026-07-14, after Experiment 008 was completed and its result
manifest frozen.  
**Effect on frozen result:** Experiment 008's `canonical` geometry is not the
final corrected geometry. Its numerical outcomes are provisional pending replay.

## Finding

The official PDS attitude product defines spacecraft spin-axis right ascension
and declination in the Mean-of-J2000 ecliptic frame, represented in SPICE as
`ECLIPJ2000`. The production source and Experiment 008 converted those angles to
Cartesian vectors and interpreted the components as equatorial `J2000`.

Experiment 008 prospectively identified and corrected a separate error: the
production path supplied an `IAU_MOON` Sun vector to a builder expecting a J2000
vector. Its corrected path put the Sun vector in J2000, but it retained the
incorrect J2000 interpretation of the attitude vector. Thus its two SCD-defining
vectors shared a numerical frame, but not the attitude product's physical frame.

## Corrected Contract

The production correction constructs both the spin axis and LP-to-Sun vector in
`ECLIPJ2000`, builds the SCD-to-ECLIPJ2000 basis, and then applies
`pxform("ECLIPJ2000", "IAU_MOON", et)`. Ephemeris positions needed in
`IAU_MOON` are requested directly in that frame with `spkpos`.

## Independent Data Check

A diagnostic comparison used the official PDS Level-2 220 eV footprint table
for June 1998. Of 346 records matched to a local 3-D MAG row within three
seconds, the corrected production implementation connected 336. Its angular
separation from the published footprint had median `0.488 deg`, p90
`2.115 deg`, and p95 `2.938 deg`.

For the same matched comparison performed before the source correction:

| Geometry interpretation | Median | p95 |
| --- | ---: | ---: |
| Submitted/production frame mix with equatorial attitude | 7.759 deg | 14.256 deg |
| Experiment 008 canonical with equatorial attitude | 3.941 deg | 9.274 deg |
| ECLIPJ2000 attitude and Sun | 0.488 deg | 2.938 deg |

The products are not exact cadence-identical inputs, so this is an external
validation diagnostic rather than an equality test. The metadata establishes
the reference-frame correction independently of these residuals.

## Consequence

The original Experiment 008 files and `result-manifest.csv` remain unchanged as
a reproducible record of the completed analysis. No old cache is silently
relabelled. The geometry census, operator caches, rank spectra, witnesses,
injections, blocked tests, sensitivities, figures, supplement, and final outcome
must be regenerated from the corrected production geometry before Outcome C can
again be called certified.

The sparse-orbit argument suggests the qualitative daily degree-5 result will
survive, but that is a hypothesis, not a substitute for replay.

## Resolution

The isolated [ECLIPJ2000 replay](replays/eclipj2000-v1/README.md) subsequently
regenerated all geometry-dependent caches, primary and extended analyses,
figures, synthesis products, and the LaTeX supplement without altering the
original outputs. The replay processed all 560 files without failure and
retained 389,317 corrected connected sweeps. The unchanged decision tree again
returned Outcome C, with exactly the same passing independent-window classes.
The corrected replay therefore supersedes the provisional geometry result for
scientific interpretation while preserving the original run as provenance.

## Sources

- PDS attitude product:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ENG-6-ATTITUDE-V1.0>
- NAIF frame definitions:
  <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html>
- PDS Level-2 ER product:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ER-4-ELECTRON-DATA-V1.0>
