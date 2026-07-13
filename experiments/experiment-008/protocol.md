# Experiment 008 Protocol

**Status:** Frozen before geometry extraction or scientific outcomes  
**Branch:** `experiments`  
**Experiment class:** Geometry-only identifiability and sampling-limits audit

## 1. Scientific Question

Does the corrected Lunar Prospector ER sampling operator contain enough
independent information to recover the submitted joint spatial-temporal
spherical-harmonic field, and are any coarser field classes recoverable?

This is a test of sampling and parameterization, not of observed potential
values. The primary analysis deliberately grants the inverse problem every
structurally valid, surface-connected sweep. It therefore supplies an optimistic
upper bound on information available after any fit-quality selection.

## 2. Competing Outcomes

### Outcome A: submitted model recoverable

The primary 24-hour, degree-5 model is stable in at least 90% of eligible
windows, has no material near-null witness, and passes the frozen noisy
injection gates without relying on regularization.

### Outcome B: submitted model non-recoverable

At least 50% of eligible primary windows are unstable, a material near-null
witness exists, and at least 50% fail frozen noisy recovery. Regularization may
produce a map but cannot convert this outcome into recovery.

### Outcome C: conditional/coarse recovery

Outcome B holds for the submitted model, while at least one prospectively
defined static or coarse model passes the same stability and recovery criteria.
The conclusion must name the recoverable model class and must not call it an
instantaneous global map.

### Outcome D: inconclusive

The evidence does not satisfy A, B, or C, or provenance/geometry validation
fails.

## 3. Inputs and Information Barrier

Allowed inputs are:

- all local `data/1998/**/3D*.TAB` and `data/1999/**/3D*.TAB` files;
- `data/attitude.tab`;
- the mission SPICE kernels listed in the frozen input manifest;
- tracked coordinate, SPICE, geometry, and harmonic helper code.

Forbidden inputs are all fitted potential values, fit success flags, U-width
flags, beam/loss-cone estimates, and environment classes derived from fitted
products. Code in this experiment must not discover or load
`artifacts/potential_cache`.

The manifest records file path, byte size, modification time, and SHA-256.

## 4. Population and Observation Unit

Raw files are processed independently in lexical path order. A sweep is one
contiguous `spec_no` group with exactly 15 rows. A sweep is structurally valid
when every row has a finite UTC and finite nonzero MAG vector. It is included in
the optimistic primary population when at least one row has a finite corrected
surface intersection. All exclusion counts are reported.

No potential-fit selection is applied. Rows within an included sweep are not
independent observations. The canonical operator row is the arithmetic mean of
the valid row-level basis vectors. Sensitivities use the row nearest the sweep
midpoint and complete-geometry sweeps only.

## 5. Geometry

Canonical geometry uses the corrected circular RA interpolation and linear
declination interpolation introduced by commit `ac1f03c`, J2000-to-IAU_MOON
SPICE transforms, ER-frame MAG projection, and the existing ray-sphere surface
intersection convention.

The experiment stores row UTC, projected latitude/longitude, connection
polarity, and validity only. Flux and fitted quantities are neither stored nor
used. Legacy next-sample attitude lookup is recomputed solely as a frozen
sensitivity.

## 6. Observation Operator

A real orthonormal spherical-harmonic basis is used, with exactly
`(lmax + 1)^2` real coefficients. The implementation is tested against the
repository's complex basis under the real-field conjugacy condition.

For temporal basis models,

\[
A_{i,q,k}=n_i^{-1}\sum_j T_q(t_{ij})Y_k(x_{ij}).
\]

For independent windows, a sweep is assigned by midpoint UTC but its row-level
spatial basis is averaged within that window. Temporal columns are reported in
their physical scaling and after unit-L2 column normalization. Rank is computed
on the normalized operator so units cannot manufacture ill-conditioning.

## 7. Frozen Model Grid

The primary submitted-model audit is independent 24-hour windows at `lmax=5`.

Window sensitivities cross:

- `lmax`: 1, 2, 3, 5, 10;
- duration: 1, 6, 24, 72, 168, and 708.734112 hours.

Static whole-mission controls use `lmax`: 1, 2, 3, 5, 10, and 20.

Whole-mission temporal-basis controls use `constant` and
`constant+synodic` at `lmax`: 1, 2, 3, 5, and 10. The synodic period is
29.530588 days and expands to cosine and sine columns.

## 8. Stability Diagnostics

For each operator, report:

- row and parameter counts;
- exact numerical rank using `max(n, p) * eps * s_max`;
- singular-value spectrum;
- effective ranks at relative thresholds `1e-3`, `1e-6`, and `1e-9`;
- 2-norm condition number;
- diagonal and trace summaries of the truncated-SVD resolution matrix;
- leverage distribution;
- smallest-mode visibility.

An operator is **unstable** when any of the following holds:

- exact rank is below parameter count;
- condition number is at least `1e6`;
- effective rank at relative threshold `1e-6` is below parameter count;
- the smallest-mode sampled/global RMS visibility is at most `0.1`.

The null witness scales the smallest right-singular vector to 100 V global RMS
on a fixed 2-degree equal-angle grid and reports its sampled RMS.

## 9. Frozen Injection and Recovery

The configuration fixes all seeds before geometry outcomes. Primary injections
are random within-basis real coefficient fields with degree power proportional
to `(l + 1)^-2`, scaled to 100 V global RMS. Secondary injections are smooth
localized caps outside the fitted basis.

Noise is independent zero-mean Gaussian with standard deviation 10, 25, 50,
and 100 V. Each model/noise cell uses 32 fixed replicates. Recoveries use:

1. unregularized SVD least squares;
2. frozen relative truncated-SVD thresholds `1e-3`, `1e-6`, and `1e-9`;
3. ridge sensitivities `1e-6`, `1e-3`, `1`, and `1e3` after column
   normalization.

Regularized results are diagnostics, not evidence of unregularized
identifiability.

A model passes noisy recovery at 25 V only when unregularized median global RMS
error is below 25% of truth RMS and its 90th percentile is below 50%. A window
with fewer rows than parameters is an attempted failure and remains in the
denominator.

## 10. Blocked Validation

Synthetic observations are generated before splitting. Frozen blocked tests
are:

- train on the first 75% of mission UTC and test on the final 25%;
- twelve leave-one-30-degree-longitude-sector-out tests;
- four interleaved seven-day-block folds, keeping complete sweeps together.

Report train and test sampled RMS plus global reconstruction RMS. Random
row-level splitting is prohibited.

## 11. Sensitivities

The following are frozen before outcomes:

- corrected interpolation versus legacy next-sample attitude;
- sweep-averaged versus midpoint-row geometry;
- all connected sweeps versus complete 15-row connected sweeps;
- canonical one-row-per-sweep versus legacy 15-row duplication;
- inertial longitude versus solar co-rotating longitude;
- unregularized versus frozen regularization grid.

## 12. Staged Execution and Stop Rules

1. Freeze protocol, configuration, and input manifest.
2. Validate one synthetic fixture and one raw day without computing a scientific
   rank verdict.
3. Build the complete geometry-only cache and census.
4. Run the primary submitted-model audit.
5. Regardless of primary outcome, complete the static/coarse grid so Outcome C
   remains testable.
6. Complete witnesses, injections, blocked validation, and sensitivities.

Stop with Outcome D if input hashes change, geometry parity fails, fewer than
90% of raw files can be processed, or numerical replay is nondeterministic.

## 13. Resource Policy

Geometry extraction uses at most four CPU workers and no GPU. BLAS thread counts
are fixed to one per worker. Analysis streams or uses the experiment-generated
geometry cache; it never launches nonlinear fits. Large caches remain ignored
under `artifacts/experiment-008/`.

## 14. Interpretation Limits

This experiment can establish structural or noise-relative recoverability of a
declared field class under actual sampling. It cannot validate any measured
potential, spacecraft-potential correction, loss-cone model, beam model, or
environmental interpretation. A geometry-only failure is a strong upper-bound
result; geometry-only success still requires later validation with defensible
measurements and selection.

