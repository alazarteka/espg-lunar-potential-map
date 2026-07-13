# Protocol Amendment 02: Numerical Conventions

**Status:** Frozen after geometry certification and before any scientific rank
or recovery result  
**Date:** 2026-07-13  
**Reason:** Resolve implementation choices left implicit by the original frozen
protocol without changing its scientific question or thresholds.

## 1. Independent Windows

All independent windows are non-overlapping half-open intervals anchored to
the Unix epoch. A sweep is assigned using its corrected-geometry midpoint UTC.
Every nonempty interval between the first and last retained sweep is eligible.
Empty intervals are reported but are not attempted operators. A nonempty
interval with fewer rows than parameters is an attempted unstable and failed-
recovery window, as required by the original protocol.

The primary 24-hour intervals therefore coincide with UTC calendar days.

## 2. Real Harmonic Convention

Columns are degree-major and order-minor, with orders from `-l` through `+l`.
Starting from SciPy's orthonormal complex harmonics including the Condon-
Shortley phase, the real columns are

\[
R_{l,m}=\begin{cases}
\sqrt{2}(-1)^{|m|}\operatorname{Im}Y_{l,|m|}, & m<0,\\
Y_{l,0}, & m=0,\\
\sqrt{2}(-1)^m\operatorname{Re}Y_{l,m}, & m>0.
\end{cases}
\]

The convention is unit tested against direct SciPy evaluations and numerical
spherical orthonormality.

## 3. Global RMS and Witnesses

The frozen 2-degree equal-angle grid is evaluated at latitudes including both
poles and longitudes on `[-180, 180)`. Its primary RMS is the unweighted RMS of
all grid samples, exactly as implied by "equal-angle grid" in the protocol.
Area-weighted RMS is an interpretation sensitivity and cannot replace the
frozen metric.

For underdetermined operators, exact nullity is already established by the row
bound and null visibility is zero. Explicit null vectors are constructed only
for representative witnesses, avoiding thousands of scientifically redundant
full-nullspace decompositions.

## 4. Injection Power and Seeds

For degree `l`, the total expected coefficient power is proportional to
`(l + 1)^-2`. Each of the `2l + 1` real coefficients therefore receives an
independent standard-normal draw scaled by

\[
[(l+1)^2(2l+1)]^{-1/2}.
\]

Every realization is subsequently scaled to the frozen 100 V global RMS.
Seeds are SHA-256 derivations of the frozen namespace plus model identifier,
window start, noise level, replicate, and injection class. Results must not
depend on process ordering or Python's randomized hash.

A window passes recovery when its 32 unregularized errors at 25 V satisfy both
frozen median and 90th-percentile gates. A windowed model class passes recovery
when at least 90% of its eligible windows pass, matching the frozen Outcome A
stability fraction. Outcome B retains its separately frozen 50% failed-window
gate. A whole-mission static model has one operator and therefore either passes
or fails.

## 5. Blocked Validation Scope

Blocked validation is run at `lmax=5` for both frozen whole-mission temporal
models, using 32 within-basis realizations at the frozen recovery-gate noise of
25 V. Truth and fitted temporal classes match in this structural test. Each
sweep is assigned wholly by midpoint time and circular-mean corrected
longitude. These tests diagnose extrapolation under actual sampling blocks;
they are not evidence that the real potential obeys either temporal class.

## 6. Solar Co-rotating Sensitivity

The co-rotating longitude is

\[
\lambda_{\mathrm{corot}}=\lambda-
2\pi(t-t_0)/P_{\mathrm{syn}},
\]

where `t0` is the Unix epoch and `P_syn = 29.530588` days. A constant phase
origin or the opposite sign only rotates real sine/cosine coefficient pairs and
does not alter rank; the convention above is fixed for deterministic replay.

## 7. Numerical Decomposition

Primary and windowed operators use direct double-precision SVD after frozen
unit-L2 column normalization. Whole-mission degree-20 and temporal controls may
use chunked double-precision Gram accumulation followed by symmetric
eigendecomposition. Such controls report that their condition estimates are
Gram-derived; they cannot override the direct-SVD primary verdict.
