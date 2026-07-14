# Experiment 008 ECLIPJ2000 Replay Results

## Decision

**Outcome C remains certified: conditional/coarse recovery.**

Correcting the Lunar Prospector attitude interpretation from equatorial
`J2000` to `ECLIPJ2000` changes individual footprints and the connected-sweep
cohort, but it does not change the frozen scientific decision. Independent
24-hour degree-5 fields remain non-recoverable under every primary and
sensitivity gate. Prospectively declared coarse, static, and compact shared
temporal classes remain conditionally recoverable.

This is a geometry-only resolution result. It neither validates the potential
estimator nor resolves the additive spacecraft-potential gauge.

## Provenance and Geometry

- All 560 raw files processed with zero failures.
- The current input manifest is byte-for-byte identical to the frozen input
  manifest.
- 559,321 exact 15-row sweeps were found; 551,637 passed structural UTC/MAG
  checks.
- Corrected `ECLIPJ2000` geometry retained 389,317 connected sweeps and
  366,715 complete 15-row sweeps.
- The original Experiment 008 equatorial geometry retained 399,643 connected
  sweeps and 376,920 complete sweeps.
- The corrected connected cohort is smaller by 10,326 sweeps, or 2.58%.

The one-day implementation fixture contains 10,056 connected production rows.
The replay extractor and production `CoordinateCalculator` have identical
validity and polarity masks; maximum coordinate disagreement is below
`1.1e-12` degrees. On common rows, corrected footprints differ from the old
equatorial interpretation by 67.0 km median and 194.8 km p95. The frame error
was scientifically material even though the identifiability decision proves
robust to it.

As an additional provenance check, the new `equatorial` variant is bit-for-bit
identical to the original Experiment 008 canonical arrays. Sweep identities,
timestamps, structural masks, and frozen inputs are unchanged.

## Primary Daily Model

The submitted class has 36 independent degree-5 coefficients in each 24-hour
window. All 560 daily matrices are algebraically full rank, but all 560 fail
the physical stability gate and all 560 fail the 25 V recovery gate.

Weakest-mode sampled-RMS/global-RMS visibility is:

| Statistic | Corrected replay | Original run |
| --- | ---: | ---: |
| Minimum | `1.148e-6` | `1.085e-6` |
| Median | `4.449e-5` | `4.519e-5` |
| 90th percentile | `1.872e-4` | `1.87e-4` |
| Maximum | `3.739e-4` | `4.050e-4` |

The median visibility changes by only -1.56%. The median daily condition
number is 47,266, and seven days exceed `1e6`. Formal rank therefore remains a
poor proxy for physical recoverability.

The covariance gate predicts a median unregularized error of 151 times the
100 V truth amplitude at 25 V sweep noise. The explicit 32-replicate trials
give comparable errors and `0/560` passing windows. Across the 17,920
broadband 25 V realizations:

| Estimator | Median relative RMS error | 90th percentile |
| --- | ---: | ---: |
| Unregularized | 144.1 | 1076.9 |
| TSVD `1e-3` | 3.46 | 6.68 |
| Ridge `1e-3` | 0.847 | 1.157 |
| Ridge `1` | 0.862 | 0.938 |
| Ridge `1000` | 0.999 | 1.000 |

Regularization limits variance by suppressing unresolved field components; it
does not recover them. The strongest ridge is effectively a zero-field
estimator.

## Recoverability Boundary

The set of classes passing the unchanged class-level gates is exactly the same
as in the original experiment:

| Window | Passing independent spatial classes |
| --- | --- |
| 1 h | None |
| 6 h | None |
| 24 h | None |
| 72 h | Degree 1 |
| 168 h | Degrees 1, 2, and 3 |
| 708.734112 h | Degrees 1, 2, 3, 5, and 10 |

At 24 hours, degree 1 reaches 80.7% stable windows but misses the frozen 90%
class gate. At 72 hours, degree 2 recovers synthetic coefficients but remains
unstable under the independent visibility/rank criteria. These disagreements
are why the protocol required both stability and recovery.

Whole-mission static fields pass through degree 20. Their condition numbers
range from 1.37 to 3.75, and degree-20 median recovery error remains below 1%.
The constant-plus-synodic degree-10 operator is `363/363` rank with condition
number 3.13 and median relative recovery error 0.00832. Removing all monopole
terms leaves `360/360` rank with nearly identical behavior.

## Sensitivities and Blocked Recovery

Every daily sensitivity has zero stable and zero recovery-passing windows:

- canonical sweep means versus midpoint rows;
- connected versus complete 15-row sweeps;
- interpolated versus legacy next-sample attitude;
- corrected `ECLIPJ2000` versus the old equatorial interpretation;
- corrected versus submitted frame-mixed geometry;
- native versus exact paired cohorts;
- body-fixed versus solar co-rotating longitude;
- with versus without monopole terms;
- sweep observations versus both row-duplication assumptions.

Solar co-rotation produces the largest improvement, raising median visibility
from `4.449e-5` to `1.442e-4`, but remains almost 700 times below the 0.1 gate.
The empirical unit-column diagnostic reports median condition number 28,173,
yet its selected mode has maximum physical visibility only `4.752e-4`.

All 34 shared static/synodic blocked fits remain full rank. Static degree-5
blocked global median errors are 0.263-0.315 V; shared synodic errors are
0.480-0.533 V. In contrast, daily degree-5 blocked fits have median global
errors of 29.1 kV for contiguous-time holdouts and 17.7 kV for longitude
holdouts, while held-out sampled errors are only 22.1 V and 9.7 V. Good sampled
residuals can coexist with a catastrophically unconstrained global field.

Localized 15-30 degree caps give the same warning. Median global errors span
100 V to 15.7 kV and p90 errors reach 161 kV, even when sampled residuals are
small.

## Numerical Certification

- The primary and duplicate grid paths agree in stability and recovery
  fractions exactly; median visibility differs by `1.25e-17`.
- The constant block of the temporal cache agrees with the spatial basis to
  `2.67e-15`; direct sampled temporal evaluation agrees exactly.
- Representative nominal 100 V witnesses have area-weighted grid RMS
  `99.99746 V`.
- Explicit covariance medians agree with analytic predictions to typical
  percent-level Monte Carlo variation.
- The corrected static degree-20 and synodic degree-10 condition numbers are
  slightly better than the original values (3.75 versus 3.88, and 3.13 versus
  3.20).

## Scientific Meaning

The corrected result is stronger than merely repeating the old calculation.
The coordinate change moves footprints by scientifically material distances,
removes 2.58% of the old canonical connected cohort, and changes individual
operators. Nevertheless, the same resolution boundary emerges under the same
prospectively frozen gates.

The defensible claim is therefore:

> Lunar Prospector ER sampling cannot recover independent daily degree-5
> global fields. Geometry supports only substantially longer temporal support
> or compact shared temporal models, conditional on those models and on later
> validation of the measurement estimator.

The passing classes do not establish an absolute lunar potential. Unknown
spacecraft potential remains exactly confounded with the spatial monopole and
matching temporal terms. Real model discrepancy can also alias into static or
synodic fields. These limits must accompany any use of the recoverable classes.

