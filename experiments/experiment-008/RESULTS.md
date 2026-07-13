# Experiment 008 Results

## Decision

**Outcome C: conditional/coarse recovery.**

The submitted independent 24-hour, degree-5 field class is geometrically
non-recoverable under every frozen gate and sensitivity. Several prospectively
declared lower-dimensional or longer-timescale classes are recoverable under
the same optimistic geometry-only assumptions.

This is a resolution-boundary result. It does **not** show that count or
potential inference is impossible, and it does **not** validate an absolute
lunar surface-potential map.

## Data and Geometry

- All 560 raw files processed with zero file-level failures.
- 559,321 exact 15-row sweeps were found.
- 551,637 passed the structural UTC/MAG rule.
- 399,643 had at least one canonical frame-consistent surface connection.
- 376,920 had all 15 canonical rows connected.
- The geometry and degree-10 basis caches replayed deterministically and are
  bound to the tracked analysis freeze by SHA-256.

The one-day production parity check also found a pre-existing coordinate-frame
error: the submitted/current path passes an IAU_MOON Sun vector into a builder
that expects J2000. On the fixture, corrected and production footprints differ
by 47.8 km median, 194.8 km at the 95th percentile, and 419 km maximum. This is
material to any map location, but the identifiability conclusion survives both
native and paired corrected/legacy geometry cohorts.

## Why Algebraic Rank Was Misleading

Every one of the 560 primary daily matrices has formal rank `36/36`. That fact
is not enough. Under the audited physical field metric

\[
\widetilde A=A G^{-1/2}/\sqrt{n}, \qquad G=I/(4\pi),
\]

the singular values are sampled-RMS/global-RMS visibility ratios. The daily
weakest visibility has:

- minimum: `1.09e-6`;
- median: `4.52e-5`;
- 90th percentile: `1.87e-4`;
- maximum: `4.05e-4`.

Thus, a weakest-mode field with exactly 100 V global area RMS produces only
`0.000109` to `0.0405` V sampled RMS. Direct witness replay agrees with the
singular values to numerical precision. All 560 days are unstable and all 560
fail 25 V recovery.

The median daily condition number is `4.71e4`; 11 days exceed `1e6`. More
importantly, the physical visibility gate fails even on days whose algebraic
condition number is below `1e6`.

## Recovery and Regularization

The covariance gate predicts a median daily unregularized error of about 156
times a 100 V truth at 25 V sweep noise. The explicit 32-replicate trials give
a nearly identical median-of-window-medians of 157, with `0/560` windows
passing.

Across all 17,920 broadband 25 V realizations:

| Method | Median relative RMS error | 90th percentile |
| --- | ---: | ---: |
| Unregularized | 148.4 | 1087.7 |
| TSVD `1e-3` | 3.44 | 6.67 |
| Ridge `1e-3` | 0.846 | 1.14 |
| Ridge `1` | 0.864 | 0.940 |
| Ridge `1000` | 0.999 | 1.000 |

Regularization bounds the output by suppressing unresolved modes; it does not
recover the truth. The strongest ridge is effectively a zero-map estimator.
Equal-power degree shells and explicit weakest-mode truths reach the same
conclusion.

## Recoverability Boundary

The independent-window grid identifies a sharp time/complexity boundary:

| Window | Passing spatial classes |
| --- | --- |
| 1 h | None |
| 6 h | None |
| 24 h | None; degree 1 reaches only 77.7% stable windows |
| 72 h | Degree 1 |
| 168 h | Degrees 1, 2, and 3 |
| 708.734 h | Degrees 1, 2, 3, 5, and 10 by the 90% class gate |

For the synodic-length cells, 19 of 20 degree-5 and degree-10 windows pass; the
single short epoch-anchored edge window fails. This should be described as a
full-window result, not proof that an arbitrarily short edge interval is
recoverable.

Whole-mission static operators are exceptionally stable through degree 20:
condition numbers range from 1.39 to 3.88 and minimum visibility from 0.872 to
0.726. Median 25 V recovery error remains below 1% even at degree 20.

The exact shared constant-plus-synodic operator is similarly stable. At degree
10 it has 363/363 rank, condition number 3.20, minimum visibility 0.726, and
median relative recovery error 0.00827. Removing all monopole terms leaves
360/360 rank and nearly identical diagnostics.

## Blocked Synthetic Recovery

All 34 shared-model blocked fits remain full rank:

- static degree 5: maximum condition 4.17, global median errors 0.261-0.313 V;
- constant-plus-synodic degree 5: maximum condition 4.40, global median errors
  0.478-0.530 V;
- every mission-tail, longitude-sector, and interleaved seven-day test remains
  stable under matched-model 25 V synthetic noise.

Within daily degree-5 models, held-out operators also remain formally full rank
but global errors are enormous: the median is about 31.9 kV for contiguous-time
holdouts and 17.7 kV for longitude sectors. Median held-out sampled errors are
only 21.4 and 9.1 V, respectively. Good local residuals can therefore coexist
with a catastrophically unconstrained global field.

## Sensitivity Results

Every daily sensitivity has `0/560` stable and `0/560` recovery-passing
windows:

- sweep mean versus midpoint row;
- all connected versus complete 15-row sweeps;
- corrected versus legacy attitude;
- canonical versus frame-mixed and submitted legacy geometry;
- native populations versus exact common-sweep cohorts;
- body-fixed versus solar co-rotating longitude;
- with versus without all `l=0` terms;
- one sweep observation versus total-weight row footprints;
- the invalid independent-row false-precision assumption.

Solar co-rotation gives the largest improvement, raising median visibility
from `4.52e-5` to `1.43e-4`, but that remains roughly 700 times below the 0.1
gate. The production frame mismatch generally makes conditioning worse rather
than manufacturing the canonical failure.

The original empirical unit-column normalization lowers the median apparent
condition number from 47,125 to 29,843, but the physical visibility of the mode
it selects remains only `4.64e-5` median (`5.65e-4` maximum). It does not rescue
the daily model and remains a secondary collinearity diagnostic.

Localized cap trials provide an out-of-basis warning. Narrow 15-degree caps
have an irreducible degree-5 approximation bias of 80.1 V at 100 V truth RMS;
30-degree caps have 41.1 V bias. Daily fits can achieve sampled residuals near
zero while median global errors range from 100 V to 14.8 kV and 90th-percentile
errors can exceed 170 kV.

## Scientific Interpretation

The actual LP geometry contains abundant mission-integrated spatial
information and can identify compact shared temporal models. It does not
contain enough simultaneous spatial information to support independent daily
degree-5 global maps. Selecting only successful potential fits would remove
rows from this optimistic superset and cannot improve its structural
information.

The manuscript should therefore avoid both extremes:

- **Too strong:** “LP ER data can never constrain any global spatial field.”
- **Defensible:** “Independent daily degree-5 global fields are not recoverable
  from LP ER sampling; recoverability emerges only after imposing substantially
  longer temporal support or a compact shared temporal model.”

Any passing class remains conditional on its declared temporal model. Real
unmodelled variability can alias into a static or synodic reconstruction.

## Limits

Experiment 008 deliberately uses no fitted potential values. It cannot validate
the loss-cone estimator, noise independence, the 25 V scale, spacecraft-
potential correction, or environmental interpretation. An unknown additive
spacecraft potential is exactly confounded with the monopole and matching
temporal terms. Contrast-only geometry is identifiable for the passing classes,
but absolute lunar surface potential is not established.

The next scientific step is not to resume production daily maps. It is to carry
forward the passing static/shared/coarse operators only after the measurement
experiments establish defensible sweep-level potential values, uncertainties,
and temporal model-discrepancy checks.
