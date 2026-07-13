# Protocol Amendment 03: Independent Numerical Audit

**Status:** Frozen after two independent protocol/code audits and before any
scientific singular spectrum or recovery outcome  
**Date:** 2026-07-13  
**Supersedes:** Conflicting metric, witness, validation, and recovery details in
the original protocol and Amendment 02

## 1. Why This Amendment Was Necessary

Two independent reviewers were asked to try to falsify the experiment before
the operator was opened. They identified four outcome-determinative problems:

1. empirical unit-column normalization can hide physically weak columns;
2. the absolute-potential gauge is confounded with spacecraft potential;
3. 32 noisy replicates provide a weak estimate of a 90th percentile;
4. blocked extrapolation is not defined for independent-window coefficients.

No rank, condition number, visibility, witness, or injection result had been
computed. The corrections below are therefore prospective.

## 2. Estimand and Gauge

The canonical operator remains the optimistic submitted representation: each
sweep supplies an oracle-known scalar spacecraft-relative potential and all
harmonics through the declared `lmax`, including `l=0`. This is an upper bound
on geometric information, not evidence that such a scalar has been measured
without error.

An unknown additive spacecraft potential is exactly confounded with the
spatial monopole and its matching temporal terms. The experiment therefore
also reports a contrast-only operator with all `l=0` columns removed. An
augmented unknown-offset model is recorded analytically as rank deficient in
those confounded columns. No Outcome A or C may be described as recovery of
absolute lunar surface potential; at most it establishes conditional design
identifiability of spacecraft-relative fields or spatial contrasts.

## 3. Primary Physical Metric

For real orthonormal static harmonics, the surface-area mean-square metric is

\[
G_x=I/(4\pi), \qquad \|f\|_{\mathrm{global}}^2=c^T G_xc.
\]

For temporal models, `G = G_t tensor G_x`, where `G_t` is the analytic uniform
continuous-time average of the declared temporal basis over the half-open
mission interval from the earliest complete-sweep UTC to one nanosecond beyond
the latest. Independent-window fields use `G_x`.

The primary direct decomposition is

\[
\widetilde A = A G^{-1/2}/\sqrt{n}.
\]

Its singular values are sampled-RMS/global-RMS visibility ratios. Its right
vectors map back to physical coefficients through `G^{-1/2}`. Rank,
condition number, the `0.1` visibility gate, witnesses, and recovery errors use
this metric. This supersedes empirical unit-column normalization as the primary
metric. Unit-L2 column normalization remains a labeled secondary collinearity
diagnostic because it is part of the original protocol.

The original unweighted equal-angle grid is retained only as a legacy replay.
Analytic area RMS is primary; an area-weighted 2-degree grid must agree to
discretization error.

## 4. Exact Operator and Null Spaces

Sweep midpoint uses all 15 structurally valid UTC rows, even when only a subset
projects to the surface. Temporal rows are exactly

\[
A_{i,q,k}=n_i^{-1}\sum_j T_q(t_{ij})Y_k(x_{ij}),
\]

not `T(midpoint) * mean(Y)`. The real basis must pass a field-and-norm parity
test against the repository complex basis.

Underdetermined spectra contain `p-n` explicit zeros. A representative actual
right-null vector is stored and verified by direct multiplication; missing
witness calculations are errors, not zero visibility. When the weakest modes
are clustered, the report includes the near-null subspace dimension and
projector diagnostics rather than over-interpreting one orientation.

## 5. Stable Numerical Decomposition

Windowed operators use direct float64 SVD. Large whole-mission operators use
chunked tall-skinny QR: each chunk is reduced to `R`, stacked `R` factors are
reduced again, and the final small `R` receives a direct SVD. Gram matrices are
not used for rank or the frozen `1e-9` effective-rank threshold. Leverage is
streamed in a second pass where required.

## 6. Executable Outcome Tree

Eligible windows are the nonempty half-open Unix-epoch-anchored intervals
specified in Amendment 02, including the final partial interval. Empty
scheduled intervals and their fraction are reported separately. Nonempty
underdetermined windows are attempted failures.

A model class has a **material near-null witness** when at least 50% of its
eligible windows have physical visibility at most `0.1`; a whole-mission model
has one operator. A windowed model passes recovery when at least 90% of eligible
windows pass the recovery gate. The decision order is:

1. **D** if a provenance, geometry, replay, or numerical stop rule fails.
2. **C** if all B criteria hold for the primary model and at least one frozen
   coarse/static class has at least 90% stable windows, at least 90% recovery-
   passing windows, and no material near-null witness.
3. **B** if at least 50% of primary windows are unstable, at least 50% fail
   recovery, and a material near-null witness exists.
4. **A** if at least 90% of primary windows are stable, at least 90% pass
   recovery, and no material near-null witness exists.
5. **D** otherwise.

Outcome C has precedence over B and means only conditional design
identifiability of the named coarse field class.

## 7. Recovery Gate and Challenges

The original 32 PCG64 injection replicates at all four noise levels remain
frozen implementation and regularization checks. NumPy's linear quantile is
used. Broadband degree power follows Amendment 02. Additional checks use
equal-total-power degree shells and the weakest metric singular subspace.

For an unregularized full-rank linear estimator with iid Gaussian sweep noise,
estimation error is independent of injected truth. The outcome recovery gate
therefore uses the exact SVD covariance evaluated with 16,384 deterministic
PCG64DXSM standard-normal draws per parameter count at 25 V. This stabilizes
the median and 90th percentile without pretending that 32 draws estimate the
tail precisely. A rank-deficient operator automatically fails. Any material
disagreement between covariance prediction and the 32 explicit injections is
a numerical stop-rule failure.

Localized out-of-basis Gaussian caps are descriptive only. Centers are
`(lat, lon) = (0,0), (60,0), (-60,180)` degrees and angular FWHM values are 15
and 30 degrees. Error is decomposed into best-basis approximation bias and
estimation error; cap results cannot decide A, B, or C.

## 8. Blocked Synthetic Recovery

Mission-tail and interleaved seven-day holdouts apply only to shared-parameter
static and constant-plus-synodic models. They are descriptive synthetic
recovery tests with no A/B/C gate. Independent 24-hour models instead receive
within-window contiguous-time and 30-degree-longitude-sector holdouts when both
train and test have data. Normalization and metric transforms are fit from the
declared global metric, never recomputed from test observations.

Longitude sectors are half-open intervals starting at -180 degrees. Seven-day
blocks are Unix-epoch anchored. A sweep is assigned wholly by its all-row UTC
midpoint and circular-mean connected longitude.

## 9. Paired Sensitivities and Row Duplication

Geometry variants are reported both on native connected populations and on
their paired common-sweep cohort. Mean versus midpoint and complete-sweep
sensitivities also report paired cohorts. The legacy 15-row representation is
evaluated with one shared sweep-level noise realization and total sweep weight
one; independent row noise is shown only as the invalid false-precision case.

"Inertial longitude" in the original protocol is corrected to body-fixed
`IAU_MOON` longitude. The solar co-rotating sensitivity otherwise follows
Amendment 02.

## 10. Structural and Provenance Clarifications

Extraction's finite MAG norm bound `1e-9 < |B| < 1e3` is the operative
structural-validity rule. It excludes numerical zero and gross sentinel/outlier
vectors before unit normalization. The census reports its exclusions.

The analysis freeze hashes the original protocol/configuration, all
amendments, manifest, geometry census, geometry cache, operator code, tests,
and injection specification. The original `protocol-freeze.json` remains
immutable provenance rather than being rewritten.
