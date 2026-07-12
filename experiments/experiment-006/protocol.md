# Experiment 006 Preregistration

**Protocol status:** Design draft committed before implementation, support
selection, calibration, or confirmation

**Configuration:** [configuration.json](configuration.json)

## 1. Motivation And Epistemic Status

Experiment 005 Attempt 03 supplied discovery evidence, not confirmation, for a
middle-energy operating regime. Its passing cells were at 150 and 350 eV; its
65 and 800 eV cells failed. The `100-500 eV` candidate window was proposed only
after those held-out outcomes were known. Experiment 006 treats that window as
a new falsifiable hypothesis.

Fresh random seeds alone are inadequate. Experiment 005 also coupled its
synthetic loss-cone boundary to the true beam center, reused the same simulated
normalized observation for both normalization branches, reused four support
spectra across calibration and validation, and evaluated coverage only among a
subset whose low-retention cells could become ineligible. Its experiment-local
loader computed pitch without surface-connection polarity. These limitations
do not erase the successful 150-350 eV result, but they define what the next
validation must repair.

## 2. Claims And Estimands

### 2.1 Primary statistical claim

For the frozen population of single secondary-like beams with true centers
from 100 through 500 eV, amplitudes from 4 through 16 normalized units, pitch
sigmas from 12 through 42 degrees, and the specified supports and discrepancy
envelope, the procedure selects a nontrivial usable subset with calibrated
conditional centroid intervals and controlled in-window false acceptance.

Selection is part of the procedure. The primary report must include both:

1. retention among all attempted injected beams; and
2. accuracy and coverage among retained beams.

No attempted case is silently dropped. Crashes, missing outputs, and numerical
failures remain in the denominator and fail retention.

### 2.2 Physical claim

The directly validated quantity is a pre-response beam centroid `E_beam`. Under
the surface-secondary, surface-connection, collisionless-transport, and
quasi-stationary interpretation,

\[
U_{rel} \simeq E_{emit} - E_{beam} + \delta_{transport}.
\]

The experiment may report `-E_beam` as a zero-emission-energy proxy while
keeping emission and transport terms explicit. It may not call `-E_beam` an
exact potential, infer absolute `U_M`, or use beam agreement to validate the
loss-cone boundary.

## 3. Domains And Geometry

- The observed energy support remains `20-2000 eV` so guard channels constrain
  the continuum.
- The optimizer center bounds remain broad at `20-2000 eV`.
- The science reporting window is the inclusive interval `100-500 eV`.
- A fit is never forced into the reporting window and an interval is never
  clipped at a reporting edge.
- `base_valid` is evaluated independently of whether the fitted point lies in
  the reporting window. This avoids defining coverage only after truncating
  estimates at 100 or 500 eV.
- A false in-window acceptance is a beam-absent or control case that passes all
  QC and interval gates and has a corrected centroid in `100-500 eV`.

Every real support must be surface-connected. Connection polarity orients pitch
so 180 degrees is surface-leaving. Disconnected rows receive invalid pitch and
cannot enter the estimator. At least two response-contributing rows must have
adequate sectors within 22.5 degrees of the surface-leaving axis, in addition
to the existing incident and reflected support minima.

## 4. Outcome-Blind Support Freeze

Select 24 spectra from the Experiment 005 interval, specs 572-969, without
examining normalized flux morphology or any beam fit. Exclude specs 572, 654,
772, and 969 used by Experiment 005, and exclude the known benchmark
neighborhood 652-654 from confirmation.

Selection uses only:

- temporal stratum;
- structural completeness and valid-sector counts;
- surface connection and polarity;
- surface-leaving pitch coverage near 180 degrees;
- native energy alignment relative to the reporting window; and
- within-sweep magnetic-vector variation.

The interval is divided into six fixed temporal strata. Four supports per
stratum are selected by deterministic maximin sampling over standardized field-
line elevation, spacecraft altitude, footprint location, polarity, near-axis
pitch support, incident/reflected sector support, and pitch span. Supports with
field-line elevation below 30 degrees are reserved as grazing challenges and
are not primary support templates. Geometry-ranked supports 1 and 4 go to
calibration in alternating strata, with ranks 2 and 3 used in the others; the
complement goes to confirmation. This gives two supports per split per stratum,
12 calibration supports, and 12 disjoint confirmation supports.

The native energy grid and within-sweep magnetic variation are recorded as
provenance checks. If they are constant across candidates, they do not enter
the diversity score. The resulting manifest, complete candidate table,
selection code, and checksums must be committed before simulator development
may inspect outcome summaries. Halekas spectra may later appear only as labeled
known positive controls.

Before D0 execution, an audit found that the production attitude helper selected
the next attitude record although the coordinate documentation specified
per-row interpolation. Commit `ac1f03c` corrected this to shortest-arc circular
interpolation in right ascension and linear interpolation in declination. D0
uses the corrected, tested convention.

## 5. Frozen Estimator Family

The candidate retains Experiment 005 Attempt 03's scientific separation:

\[
R(E,\alpha)=C(E,\alpha)+A G_E(E;E_b)G_\alpha(\alpha;180^\circ,\sigma_\alpha).
\]

`C` is logistic in pitch, with cone level, contrast, boundary, and width each
quadratic in normalized log energy. It contains no adiabatic voltage equation
and cannot vote independently on `E_b`. The fit uses unweighted logarithmic
residuals, a nominal `FWHM/E = 0.48` response, deterministic multistart, and the
existing amplitude, width, covariance, start-stability, leave-one-energy-out,
and interval-resolution concepts.

Required corrections before freeze are physical and observational, not
outcome-tuned:

- compute surface-connection polarity before pitch;
- generate incident-mean and conjugate normalizations independently from one
  latent sector-level realization;
- integrate simulated response over channel support rather than evaluating
  only a Gaussian at channel centers; and
- record within-sweep time and magnetic variation.

Any estimator-family or QC change made after calibration begins requires a
new calibration namespace. Any change after confirmation begins invalidates
Experiment 006.

## 6. Generator Independence And Discrepancy

The primary generator must not import the estimator's continuum evaluator.
Continuum parameters are statistically independent of beam center. This
prevents the background loss-cone boundary from leaking the injected answer.
A physically coupled `U_rel`/boundary generator is retained only as a labeled
secondary arm.

All simulations begin at latent native-sector flux and derive both
normalizations. The primary and confirmation mixtures include:

- independently implemented smooth logistic continua;
- double-transition or refilled cones;
- asymmetric shoulders;
- smooth energy-pitch fields with correlated residual structure;
- heteroskedastic and energy/pitch-correlated noise;
- nominal and stress response widths, channel integration, small energy-scale
  offsets, log-energy response, and intrinsic/asymmetric beam broadening;
- narrow and broad pitch beams, axis offsets, truncated tails, and unresolved
  double peaks; and
- within-sweep ramps or pulses that can masquerade as spectral structure
  because energy rows are acquired sequentially.

Matched-model noiseless cases are implementation tests only. A separate
semi-synthetic layer injects known beams into frozen held-out real residual
templates. Template construction must be outcome-blind and committed before
confirmation.

## 7. Calibration

No Experiment 005 simulation or correction value is reused. New correction
anchors are `100, 140, 220, 340, 500 eV`, with 600 injections per anchor and
3,000 calibration beams total. Within each anchor, amplitude, pitch width,
noise, continuum, response challenge, and calibration support are balanced.

The first deterministic half at each anchor estimates median signed bias. The
second half estimates finite-sample rank-based 68% and 95% absolute-error
quantiles relative to the frozen Jacobian center scale. Correction is linear in
log energy and may not extrapolate beyond the endpoint anchors. Calibration can
estimate correction coefficients; it cannot change the estimator, target
population, QC, decision gates, or confirmation design.

Before confirmation, commit:

- source and configuration checksums;
- support and residual-template manifests;
- exact-recovery and invariance-test results;
- correction coefficients and calibration-only diagnostics;
- all deterministic seed namespaces;
- the planned CSV schema and decision-report implementation; and
- a statement that no confirmation output has been opened.

## 8. One-Shot Confirmation

### 8.1 Primary beam matrix

Cross six true-center strata with two amplitude strata:

| Center stratum | Amplitude strata | Cases per cell |
| --- | --- | ---: |
| exactly 100 eV | `[4,8)`, `[8,16]` | 400 |
| `(100,160)` log-uniform | `[4,8)`, `[8,16]` | 400 |
| `[160,250)` log-uniform | `[4,8)`, `[8,16]` | 400 |
| `[250,375)` log-uniform | `[4,8)`, `[8,16]` | 400 |
| `[375,500)` log-uniform | `[4,8)`, `[8,16]` | 400 |
| exactly 500 eV | `[4,8)`, `[8,16]` | 400 |

This produces 4,800 primary cases. Pitch sigma, noise, support, continuum, and
response are independently balanced within every cell. Native-channel phase is
derived from the continuously sampled center and fixed support grid, not
assigned independently. The exact endpoint cells test the proposed boundaries;
they are not allowed to become descriptive-only after inspection.

### 8.2 Secondary and control matrices

- 600 weak beams with amplitude `[1,4)` characterize sensitivity.
- 1,200 semi-synthetic beams mirror the 12 primary cells with 100 cases each.
- 3,000 beam-absent controls comprise 1,000 matched-smooth, 1,000
  misspecified-continuum, and 1,000 adversarial non-beam cases.
- 1,000 out-of-window beams comprise 500 below 100 eV and 500 above 500 eV.

Adversarial controls balance single-row spikes, energy-only bumps, pitch-only
enhancements, incident-side features, displaced axes, broad tails, double
features, temporal ramps/pulses, missing channels, and normalization conflicts.
Out-of-window beams are classification controls, not physical nulls.

## 9. Primary Gates

All 12 primary cells must pass. No cell can be omitted or declared ineligible.
For each cell:

1. At least 200 of 400 attempts are `final_valid`.
2. The bootstrap 95% upper confidence bound for median absolute corrected error
   is no greater than `0.10` local response FWHM.
3. The bootstrap 95% upper confidence bound for the 90th percentile absolute
   corrected error is no greater than `0.25` response FWHM.
4. The bootstrap confidence interval for median signed corrected error lies
   inside `[-0.05, +0.05]` response FWHM.
5. Empirical 68% coverage lies in `0.60-0.76`, and its two-sided 95% exact
   binomial interval contains 0.68.
6. Empirical 95% coverage lies in `0.90-0.99`, and its two-sided 95% exact
   binomial interval contains 0.95.
7. Every retained 95% interval has full width no greater than one local
   response FWHM.

The same accuracy and coverage gates are applied to retained primary cases
pooled by confirmatory background family. Retention and errors are also
reported by support, response family, pitch width, noise, and distance from a
native channel center. Simultaneous intervals accompany any cell-specific
scientific interpretation.

For each of the three 1,000-case null families, the Bonferroni-adjusted
one-sided familywise-95% exact binomial upper bound on in-window false
acceptance must be no greater than 5%. With three families and 1,000 cases per
family, this means no more than 35 false acceptances in any family. The
unrestricted any-energy false-fit rate is reported separately.

Out-of-window leakage into a final-valid in-window report is tested separately
below and above the window. Each side has 500 cases and uses a Bonferroni-
adjusted one-sided familywise-95% exact upper bound no greater than 5%, allowing
at most 15 leaked reports per side. Semi-synthetic cells must each retain at
least 50 cases and satisfy the primary point-error and 95% coverage gates; their
68% coverage and discrepancy anatomy are reported and cannot be hidden by the
fully synthetic result.

The 5% false-acceptance gate authorizes only the hand-audited 398-sweep pilot.
It is not a production survey threshold. A one-sided upper bound at or below 1%
is reported as a stronger secondary target.

## 10. Outcome Matrix

| Outcome | Permitted interpretation |
| --- | --- |
| Full pass | Calibrated selective `E_beam` inference over the tested 100-500 eV, amplitude, support, and discrepancy envelope; open the frozen real audit. |
| Core passes, edge fails | Full-window hypothesis fails. Any narrower window is exploratory evidence for a new experiment. |
| Recovery passes, nulls fail | Centroid estimation may work for independently established beams; automated beam detection and the real audit remain closed. |
| Fully synthetic passes, semi-synthetic fails | Simulator-limited success only; no real-data measurement claim. |
| Nulls pass, coverage or bias fails | Candidate may reject backgrounds but cannot report quantitative beam energies or intervals. |
| Weak challenges fail | Primary amplitude claim may still pass; weak-beam absence is uninterpretable and the sensitivity curve must be reported. |
| Any primary-cell failure | Experiment 006 fails; no retrospective subdivision using V1 results. |

Even a full pass does not establish absolute lunar surface potential, correct
spacecraft potential, independent loss-cone corroboration, representativeness
of rejected sweeps, a mission-wide population, or global-map identifiability.

## 11. Real Audit If And Only If V1 Passes

The attempted denominator remains all 398 spectra, specs 572-969, whose 15-row
support overlaps 12:57-22:02 UT on 29 April 1999. The audit reports the complete
selection funnel, every nonexclusive failure reason, corrected `E_beam`
intervals, both normalizations, connection/polarity, time stability, support
metrics, residual atlas, and known benchmark status.

Real data have no known centroid truth. The audit can establish prevalence of
resolved secondary-like features and expose residual failure modes. It cannot
measure real-data bias or convert a selective one-day distribution into a
mission-wide lunar-potential map.

## 12. Deviations

At this commit the protocol is a design, not the final execution freeze. The
support manifest, simulator implementation, exact seed derivation, response
stress values, semi-synthetic template construction, and executable gate code
must be completed and committed before C1. Every later deviation will be dated,
explained, and classified as pre-calibration, pre-confirmation, or invalidating.
