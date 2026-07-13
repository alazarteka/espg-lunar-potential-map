# Geometry Implementation Check

**Status:** Complete before mission-wide operator outcomes  
**Raw fixture:** `data/1998/001_031JAN/3D980116.TAB`  
**Machine-readable result:** `implementation-check-one-day.json`

## Purpose

The protocol required a one-day geometry validation before the full extraction.
Static inspection had revealed that the submitted and current production path
mixes an `IAU_MOON` Sun vector with a J2000 spin vector when constructing the
nominal SCD-to-J2000 rotation. Protocol Amendment 01 froze canonical and legacy
variants before this impact result was opened.

## Production Parity

The experiment's `frame_mixed` variant reproduces the current production path:

- 9,180 valid projected rows in each implementation;
- zero valid-mask disagreements;
- zero polarity disagreements;
- maximum latitude disagreement `2.27e-13 deg`;
- maximum longitude disagreement `5.12e-13 deg`.

This confirms that the sensitivity reproduces the production operation rather
than an inferred approximation to it.

## One-Day Impact

For rows connected under both canonical and comparison geometry:

| Comparison | Median shift | 95th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Frame-consistent vs current frame-mixed | 47.82 km | 194.77 km | 419.17 km |
| Frame-consistent vs next-sample attitude only | 1.06 km | 9.76 km | 85.26 km |
| Frame-consistent vs submitted combined legacy | 47.95 km | 194.28 km | 417.74 km |

The coordinate-frame mismatch dominates the attitude interpolation effect on
this day. It also changes which rows are surface-connected, so an analysis that
conditions only on common rows would understate the total effect.

## Interpretation

This is a material geometry finding, not an identifiability verdict. It implies
that submitted map footprints, connectivity, polarity-dependent production
fits, and illumination assignments require a mission-wide impact assessment.
It does not automatically invalidate Experiments 001--005, whose benchmark
fitting scripts generally construct pitch angles directly from ER/MAG data and
do not use this coordinate path.

Canonical Experiment 008 results use frame-consistent geometry. The full
operator audit will report both current-production and submitted-legacy
sensitivities rather than silently replacing their provenance.
