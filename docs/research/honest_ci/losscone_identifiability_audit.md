# Hard-bin loss-cone identification audit

**Status:** structural-identification implementation and mission-geometry audit,
2026-07-22.

This note defines and proves the identification results implemented by
`src/losscone/identifiability.py`. It does **not** define a likelihood,
confidence interval, calibrated-flux fit, or raw-count model. The empirical
audit uses archived energies, times, magnetic fields, look directions, and
corrected magnetic polarity. It never classifies a loss cone from measured
flux.

## 1. Estimand and declared model

Write

- \(V_M\) for lunar surface potential relative to the ambient plasma;
- \(S=V_{\rm sc}\) for spacecraft potential relative to the ambient plasma;
- \(\delta=V_M-S\) for the primary, spacecraft-relative estimand.

The sign convention is therefore

\[
\boxed{\delta=V_M-V_{\rm sc}}.
\]

A negatively charged surface ordinarily has \(\delta<0\). The modeled
secondary-electron beam energy is \(-\delta=V_{\rm sc}-V_M\). Both are reported
so the sign cannot be confused with the positive beam-energy convention used by
some existing diagnostics.

For energy row \(i\), the current loss-cone boundary before clipping is

\[
q_i
=\sin^2\alpha_i
=r\left(1+\frac{V_M}{E_i-S}\right)
=r\frac{E_i+\delta}{E_i-S},
\tag{1}
\]

where \(r=B_{\rm sc}/B_M>0\). The hard model clips \(q_i\) to \([0,1]\) and
labels a pitch cell as inside when

\[
p_{ij}\le 180^\circ-\arcsin\sqrt{q_i}.
\tag{2}
\]

Beam amplitude is fixed to zero in this audit. Flux levels and background play
no role in the binary pattern.

## 2. Ideal-boundary identification

### Theorem 2.1: known spacecraft potential

Suppose \(S\) is known, \(r>0\), and two distinct energies \(E_1,E_2\) have
exact, unclipped boundary observations \(q_1,q_2\), with
\(E_i\ne S\). Then \(r\), \(V_M\), and \(\delta\) are identified unless the two
energies have identical reciprocal leverage.

**Proof.** Put \(z_i=(E_i-S)^{-1}\) and \(p=rV_M\). Equation (1) becomes

\[
q_i=r+pz_i.
\]

Because \(z_1\ne z_2\), the two points identify the intercept \(r\) and slope
\(p\). Positivity of \(r\) gives \(V_M=p/r\), and \(\delta=V_M-S\). \(\square\)

### Theorem 2.2: unknown common spacecraft potential

Suppose \(S\) is an unknown scalar common to the sweep and three distinct
energies have exact, unclipped boundary observations. Outside the degenerate
constant-boundary and repeated-energy cases, \(S\), \(r\), \(V_M\), and
\(\delta\) are identified.

**Proof.** With \(p=rV_M\),

\[
q_i=r+\frac{p}{E_i-S}.
\]

For nonzero adjacent boundary differences, define

\[
R=
\frac{q_1-q_2}{q_2-q_3}
\frac{E_3-E_2}{E_2-E_1}
=\frac{E_3-S}{E_1-S}.
\]

If \(R\ne1\), then

\[
S=\frac{E_3-RE_1}{1-R}.
\]

Theorem 2.1 then recovers \(r\) and \(V_M\); subtraction gives \(\delta\).
The constant-boundary case identifies \(V_M=0\) when \(r>0\), but does not
identify \(S\) itself. \(\square\)

These theorems show that the physical equation is not merely an unrestricted
additive energy translation. Cross-energy rational curvature can supply an
internal spacecraft-potential anchor.

The distinct-energy condition is essential. If all available rows have one
energy $E_*$, they repeat only

$$
q_*=r\frac{E_*+\delta}{E_*-S}.
$$

Even with known $S$, one equation cannot separate $r$ and $\delta$. With
energies that are distinct but nearly equal, identification still holds in the
exact ideal model, but the reciprocal-leverage difference tends to zero and
recovery becomes arbitrarily ill-conditioned. Finite pitch bins turn that
ill-conditioning into wider exact plateaus.

## 3. Finite pitch cells and exact plateaus

For an incident-side cell \(p\le90^\circ\), equation (2) is always satisfied.
For a reflected-side cell \(p>90^\circ\), it is equivalent to

\[
q_i\le \sin^2p.
\tag{3}
\]

Thus one finite row supplies a bracket

\[
L_i<q_i\le U_i,
\tag{4}
\]

where \(L_i\) is the largest threshold among outside cells and \(U_i\) is the
smallest threshold among inside cells. If either class is absent, the
corresponding endpoint is 0 or 1. The code retains which lower endpoints are
open, but computes the conservative closed hull.

### Proposition 3.1: hard-bin plateaus

For a fixed finite design and zero beam, the predicted loss-cone array takes
only finitely many binary values. Except at threshold boundaries, every
parameter point has a neighborhood with the same pattern. Consequently the
hard-bin loss-cone component cannot globally point-identify a continuously
varying potential; it identifies an interval or a union of plateau regions.

**Proof.** The pattern changes only when some continuous \(q_i\) crosses one of
the finitely many thresholds \(\sin^2p_{ij}\), or a clipping boundary. Away from
those surfaces every strict inequality is preserved on an open neighborhood.
\(\square\)

Clipping makes the loss worse. If a row has clipped \(q_i=0\) or \(q_i=1\), it
supplies only a one-sided constraint on the underlying expression. When every
fixed-safe row is uninformative, the correct hard-bin identified set is the
full declared potential domain.

## 4. Why cross-row nuisance ties matter

The ideal results depend on a common potential and a tied field ratio across
the 15 sequential energy rows. If a nuisance may vary freely by row, it can
absorb a candidate potential change. For interior \(q_i>0\):

\[
r_i=q_i\frac{E_i-S_i}{E_i+\delta_i},
\]

or, with common \(r\) and candidate \(\delta\),

\[
S_i=E_i-r\frac{E_i+\delta}{q_i}.
\]

Likewise, with common \(r,S\), independently free row potentials satisfy

\[
\delta_i=\frac{q_i(E_i-S)}{r}-E_i.
\]

Where these constructed values lie in the allowed physical domains, they give
exact observational equivalence. Therefore an 80-second sweep needs an
explicit constancy, bounded-drift, or other low-dimensional temporal model.
Exchangeability or robust standard errors cannot supply that missing tie.

## 5. Sharp affine identified interval

Normalize archived row times to \(\tau_i\in[-1/2,1/2]\) and declare

\[
\delta_i=\delta_0+d_\delta\tau_i,
\qquad
S_i=S_0+d_S\tau_i,
\tag{5}
\]

with one common \(r\). Put

\[
b_0=r\delta_0,
\qquad
b_1=rd_\delta.
\]

Then the unclipped boundary is

\[
q_i=
\frac{rE_i+b_0+b_1\tau_i}
     {E_i-S_0-d_S\tau_i}.
\tag{6}
\]

Rows invalid at the oracle truth are fixed out before brackets are formed.
Each scenario then uses only the subset satisfying \(E_i>S_i\) throughout its
entire declared nuisance envelope. Their denominators are therefore positive,
and no candidate can drop an inconvenient row. Substituting a finite-bin
bracket (4) into (6) gives linear inequalities in

\[
x=(r,b_0,b_1,S_0,d_S).
\]

For example, the upper constraint is

\[
rE_i+b_0+b_1\tau_i
+U_iS_0+U_id_S\tau_i
\le U_iE_i.
\tag{7}
\]

Potential and drift bounds are also linear:

\[
r\delta_{\min}\le b_0\le r\delta_{\max},
\qquad
rd_{\min}\le b_1\le rd_{\max}.
\]

The implementation additionally applies the declared computational potential
domain to both affine sweep endpoints. Linearity then keeps every intermediate
row in the domain as well.

### Theorem 5.1: LP characterization

Assume \(0<r_{\min}\le r\le r_{\max}\), fixed-safe positive denominators, and
closed bounded nuisance intervals. The closure of the hard-bin identified set
for \(\delta_0\) is one interval. Its endpoints are the solutions of two linear
programs after the Charnes--Cooper transformation

\[
t=1/r,
\qquad y=tx,
\qquad y_r=1.
\]

**Proof.** Equations (7) and their lower-bound counterparts define a convex
polytope \(Ax\le c\). The transformation gives \(Ay\le ct\), \(y_r=1\), and
\(1/r_{\max}\le t\le1/r_{\min}\). The target is
\(\delta_0=b_0/r=y_{b_0}\), a linear objective. Its minimum and maximum over
the transformed polytope are therefore two linear programs. The continuous
image of the connected feasible polytope under the scalar target is an
interval. \(\square\)

The implementation uses SciPy HiGHS, expands both numerical endpoints outward
by \(10^{-6}\) V, and returns the full audit domain on unresolved numerical
status. An infeasible pattern means incompatibility with the declared scenario;
it is not labeled nonidentification.

The transformation is bijective for $r>0$: every original feasible point maps
to a transformed feasible point and every transformed point with $t>0$ maps
back through $x=y/t$. The two LP optima are therefore not relaxations. They are
the sharp endpoints of the conservative closed hull for precisely the
constant/affine model declared here.

## 6. Standard empirical sensitivity ladder

The mission audit generates oracle patterns over a fixed truth grid and then
computes their equivalence intervals under:

1. known \(r\), known \(S\), no drift;
2. common \(r\in[0.3,1.1]\), known \(S\);
3. common \(r\) and \(S\) within ±5 V;
4. common \(r\) and \(S\) within ±20 V;
5. case 4 plus at most 20 V end-to-end affine drift in both \(\delta\) and \(S\);
6. common \(r\) and the broad implementation envelope
   \(S\in[-1500,150]\) V.

These are sensitivity classes, not source-validated calibration uncertainties.
The broad scenario uses only rows with \(E>150\) eV so a candidate spacecraft
potential cannot make low-energy rows disappear from the objective.

## 7. Interpretation contract

- A narrow oracle interval measures the best resolution permitted by the
  finite archived geometry and hard model.
- Widening under a nuisance scenario quantifies the identifying value of the
  corresponding calibration or stability assumption.
- A full-domain interval is a structural negative result for that declared
  scenario.
- None of these sets has sampling coverage. A true confidence construction
  still requires a complete observable law or valid repeated-cluster moment
  model.
- Unknown energy/pitch response and calibrated anode gain are not included in
  this first audit. Any eventual response envelope can only widen the reported
  hard-bin sets.

For the predeclared acceptance comparison, a scenario is called
*median-resolution-useful* when its mission-wide median closure width is at
most 50 V. The summaries also report the fractions at or below 20, 50, 100,
and 200 V, so this screening convention is not the only available cutoff. It
is a computational reporting rule, not a confidence level or a claim that
50 V is sufficient for every science use.

## 8. Reproduction and empirical findings

The resource-bounded CLI is documented in
[`../../cli/diagnostics.md`](../../cli/diagnostics.md). Large census and
per-pattern tables remain under ignored `artifacts/`; compact mission summaries
and their hashes are tracked in [`results/`](results/).

Reproduce the committed summaries from the repository root with:

```bash
uv run python scripts/diagnostics/losscone_hardbin_identifiability.py census \
  --workers 8
uv run python scripts/diagnostics/losscone_hardbin_identifiability.py sample
uv run python scripts/diagnostics/losscone_hardbin_identifiability.py audit \
  --workers 8
uv run python scripts/diagnostics/losscone_hardbin_identifiability.py summarize
```

The completed census covered all 560 calibrated daily files and 559,321
sweeps. Of these, 399,634 (71.45%) passed the sweep-level structural gate and
had at least one fixed-valid pitch row. There were 152,003 sweeps (27.18%) with
no connected valid row and 7,684 (1.37%) rejected by the existing geometry
structural gate, including one that also had invalid row times. Among usable
sweeps, 376,914 had all 15 rows and 22,720 (5.69%) retained a nonempty partial
row mask. The census found 81 exact energy-grid signatures; two accounted for
399,482 of the 399,634 usable sweeps.

The deterministic sample contains exactly 64 sweeps in each of the 19 mission
months, 1,216 total. It includes the 1998-01-16 reference sweep and the minima
and maxima of every declared geometry-extreme metric. Oracle generation used
120 truths per sweep and six scenarios, producing 875,520 interval results.
All 19 monthly shards were complete, every generating truth lay in its closure,
and there were no infeasible, invalid-design, or unresolved-numerical results.
The pointwise intervals had no nesting violations across scenarios 1--5; the
broad spacecraft-potential branch also contained scenario 4 pointwise.

### 8.1 Mission-wide identified-set widths

All widths below are for the primary estimand \(\delta_0\). The source table
also contains 10th and 99th percentiles and fractions below 20, 100, and 200 V.

| Scenario | Median (V) | 90th percentile (V) | Width at most 50 V | Full domain |
|---|---:|---:|---:|---:|
| Known \(r,S\) | 7.40 | 331.79 | 76.55% | 0.11% |
| Common unknown \(r\), known \(S\) | 13.24 | 1912.65 | 68.97% | 1.49% |
| Common \(r\), \(S\) within ±5 V | 20.38 | 1917.20 | 67.66% | 1.59% |
| Common \(r\), \(S\) within ±20 V | 38.07 | 1930.84 | 58.66% | 1.77% |
| Previous case plus 20 V affine drifts | 54.82 | 1949.11 | 44.70% | 1.93% |
| Broad \(S\in[-1500,150]\) V stress case | 286.63 | 2156.26 | 11.89% | 3.22% |

The oracle model therefore does yield useful but nonzero finite-bin plateaus by
the predeclared median rule: its median closure is 7.40 V, and 66.44% are at
most 20 V wide. It is not uniformly useful. The oracle 90th percentile is
331.79 V; at truth \(\delta=-2000\) V its median is 112.05 V, and clipping
creates roughly 2004 V 90th-percentile widths for the nonnegative truths.
Those tails are deterministic information loss, not optimizer failure.

The first ladder step that destroys median-resolution usefulness is scenario 5,
which admits 20 V total end-to-end affine drift in both \(\delta\) and \(S\):
the median rises from 38.07 to 54.82 V. The implicated missing premise is a
validated within-sweep stability bound tighter than that joint drift class over
the roughly 80-second energy sweep. Because this step introduces both drifts
together, this audit does not claim that geometry alone separates how much of
the widening comes from target-potential drift versus spacecraft-potential
drift. The broad stress result separately shows the need for a common
spacecraft-potential calibration envelope materially narrower than
\([-1500,150]\) V if median hard-bin resolution is to remain useful.

### 8.2 Proven, empirical, assumed, and source-derived statements

- **Proven here:** the ideal algebraic results, plateau result,
  row-unrestricted counterexamples, and exact LP characterization conditional
  on the declared closed hard-bin model.
- **Computed here:** census validity, geometry sampling, and all width/status
  summaries. These are deterministic empirical results for the hashed inputs,
  not sampling estimates.
- **Assumed for sensitivity:** zero beam, the hard threshold, common \(r\), the
  six spacecraft/drift envelopes, and the \([-2500,2000]\) V computational
  domain. The ±5 V, ±20 V, and broad envelopes are not asserted calibration
  confidence statements.
- **Source-derived inputs:** archived energy, UTC, magnetic field, and look
  directions from the 560 calibrated products, plus the experiment-008
  corrected-polarity artifact. Flux is not an audit input. Per-day manifests
  record both source-file hashes; corrected polarity is a derived project
  input, not a raw archive field.

The large census hash is
`013f838580a08991999228723fa9cd30870d351036fbbb012a1601ae77da33b0`;
the sample hash is
`5587110366453b46f42daca1af2593888e9c0d19c04e456347036fab870d7068`;
and the audited implementation hash is
`17734d2dcbac9cb9937ebcebccb52b80e087cc214668764ad45e5cafdfcee83c`.
[`results/summary.json`](results/summary.json) records every monthly shard hash,
the theta-table hash, and hashes of all compact tables.

These are structural identified sets, not confidence intervals. Moving from
this result to an honest CI still requires a justified observable law or valid
repeated-cluster moment model for the actual measured product and its response
uncertainty.
