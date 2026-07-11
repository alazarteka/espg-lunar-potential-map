# Experiment 005: One-Day Observable Validation and Selection Pilot

**Status:** Ready; protocol and numerical gates frozen before pilot-wide results

**Branch:** `experiments`

**Depends on:** [Experiment 003](../experiment-003/README.md) and
[Experiment 004](../experiment-004/README.md)

## Scientific Question

Across every attempted LP ER sweep in the hand-vetted 29 April 1999 interval,
when is a response-aware upward-beam voltage measurable, when does the loss-cone
boundary add coherent independent information, and what complete selection
funnel results from requiring those claims to remain distinct?

This is an observable-validation experiment, not yet a lunar-potential survey.

## Rationale and Frozen Observable Hierarchy

Experiment 003 showed that matched estimators can recover simulated parameters,
but the shared forward model fails real-data residual checks. Experiment 004
then showed that the historical joint voltage scale is reproducible only after
instrument-response corrections, while the beam and boundary do not provide a
stable independent two-observable solution on the benchmark distribution.

The pilot therefore freezes this hierarchy before examining day-wide rates:

1. **Primary candidate:** response-aware upward-beam centroid, reported as the
   spacecraft-relative voltage `U_rel = -E_beam`.
2. **Physical-consistency diagnostic:** direct per-energy loss-cone boundaries,
   accepted only when their transition and adiabatic-curve diagnostics pass.
3. **Provenance comparison:** the best-supported historical joint fit from
   Experiment 004, never used to select or overwrite the primary voltage.

No absolute surface potential `U_M` will be claimed in this experiment. Adding
spacecraft potential requires an independently validated `U_sc` estimate and
propagated uncertainty.

## Frozen Interval and Attempted Denominator

- Product: `data/1999/091_120APR/3D990429.TAB`
- Hand-vetted figure-forensics window: 1999-04-29 `12:57-22:02 UT`
- Inclusion rule: every spectrum whose 15-row time support overlaps the window
- Inclusive spectrum range: `spec_no = 572-969`
- Attempted denominator: **398 sweeps**
- First support: spectrum 572, `12:56:08-12:57:18 UT`
- Last support: spectrum 969, `22:01:29-22:02:39 UT`

Spectra 572 and 969 overlap a window boundary and will carry explicit
`partial_window_overlap` flags. A contained-only sensitivity analysis may omit
them, but it cannot replace the primary inclusive denominator.

This interval and denominator were fixed before pilot-wide fit results were
examined. The benchmark neighborhood 652--654 is not a separate denominator.

## Primary Beam Model

For each sweep:

- Use calibrated three-dimensional flux on its native 15-row energy support.
- Compute row-specific pitch from the archived magnetic vector and sector
  geometry. Preserve row timestamps and report any within-sweep field changes.
- Use incident-hemisphere-mean normalization as primary and conjugate-sector
  normalization as a required sensitivity representation.
- Fit the reflected field-aligned excess in energy and pitch with a nonnegative
  two-dimensional Gaussian beam plus one local baseline per energy. Fix the
  physical beam axis at 180 degrees and fit its angular width.
- Convolve beam energy with the documented fractional response
  `FWHM/E = 0.48`; fit angular width rather than forcing two pitch bins.
- Search beam energy center over `20-2000 eV`, beam pitch sigma over `3-50 deg`,
  and nonnegative amplitude. Record all bound contacts.
- Use deterministic multistart optimization and deterministic bootstrap seeds.
- Estimate centroid uncertainty by sector/residual bootstrap on the actual
  energy and pitch support; do not interpret a curvature-only error as calibrated.

The primary reported quantity is `U_rel,beam = -E_beam`. The historical
complete-map objective is not part of this estimate.

## Beam Quality-Control Funnel

A sweep is `beam_valid` only if all of the following are recorded and pass:

1. Complete structural parsing or an explicit missing-row reason.
2. Finite magnetic geometry and sufficient incident/reflected pitch support.
3. Positive field-aligned excess over the fitted local continuum.
4. Beam center and angular width away from their search limits.
5. Nonzero amplitude supported by the bootstrap distribution.
6. Stable optimum across deterministic starts.
7. Centroid uncertainty resolved relative to the effective energy response.
8. Primary and conjugate normalizations agree within one local effective
   energy-response FWHM.
9. No residual pattern indicating that a single cell or masked channel creates
   the centroid.

Every failed gate retains its denominator and one or more machine-readable
reason codes. Thresholds needed to operationalize “resolved,” “stable,” and
“single-cell” must be fixed in `configuration.json` and checked by simulation
before the 398-sweep summary is generated.

## Boundary Diagnostic

For each sweep with enough reflected-side support, fit the Experiment 004 A4
transition independently at each usable energy from `300-10000 eV`:

\[
R(\alpha)=R_{in}+\Delta R\,
\left[1+\exp\left(-\frac{\beta-\alpha}{w}\right)\right]^{-1}.
\]

Cone level, contrast, boundary, and width remain free. Use 100 deterministic
sector bootstraps with a 5.625-degree angular resolution floor. Record nuisance
edges; the strict no-nuisance-edge set is the only set eligible for a physical
curve claim.

A sweep is `beam_boundary_valid` only when it is already `beam_valid` and:

- at least four strict boundary energies are resolved;
- the boundary trend has the polarity and monotonicity expected for a negative
  potential over its supported energy range;
- a free `(U_M, B_S/B_M)` adiabatic fit has neither parameter at a bound;
- its residual structure is compatible with one curve rather than an isolated
  low-energy point;
- boundary and beam voltages agree within their combined empirical uncertainty.

Failure of the boundary gate does not invalidate a well-resolved beam. It
classifies the sweep as `beam_valid` but not `beam_boundary_valid`.

## Historical Joint Comparison

Run the Experiment 004 A3 response/support joint model as a labeled diagnostic.
Report its voltage, field ratio, amplitude, objective, and bound flags. It may
explain historical behavior, but it cannot promote or reject a sweep and cannot
be combined with the beam estimate as though it were independent evidence.

## Calibration Before Day-Wide Summary

Before opening the pilot-wide success rates:

- Inject response-convolved beams into representative real sweep supports over
  energy, width, amplitude, and continuum conditions.
- Verify centroid bias, empirical interval coverage, optimizer-start stability,
  and bound behavior.
- Include beam-absent controls to calibrate false detections.
- Verify the boundary pipeline on coherent synthetic curves and on deliberately
  non-monotonic or refilled controls.
- Freeze all numerical QC thresholds in `configuration.json`.

Calibration failures require a labeled protocol revision and complete rerun;
thresholds may not be chosen to improve the observed day-wide usable fraction.

## Fixed Visual Audit Set

The following 20 spectra are fixed before the pilot run for residual and
classification review:

`572, 597, 622, 647, 652, 653, 654, 672, 697, 722, 747, 772, 797, 822, 847,
872, 897, 922, 947, 969`.

This combines regularly spaced coverage, both window edges, and the complete
benchmark neighborhood. Every major automated failure class must also contribute
examples to the residual atlas; those additions are selected by reason code,
not visual attractiveness.

## Required Selection Funnel

Report counts and percentages, always relative to all 398 attempted sweeps, for:

1. Attempted sweeps.
2. Structurally valid 15-row records.
3. Valid geometry and adequate directional support.
4. Positive candidate beam excess.
5. Numerical convergence and start stability.
6. No beam parameter bounds.
7. Resolved empirical centroid uncertainty.
8. Stable result across normalization.
9. Final `beam_valid` sweeps.
10. Sufficient strict boundary points.
11. Coherent, identifiable adiabatic boundary curve.
12. Beam/boundary agreement.
13. Final `beam_boundary_valid` sweeps.

Also report each nonexclusive failure reason, support-conditional percentages,
and the two-edge contained-only sensitivity. Never report only finite fits.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Frozen protocol, results, deviations, and decision |
| [`configuration.json`](configuration.json) | Frozen model, calibration, optimizer, and QC settings |
| `run.py` | Exact experiment-local execution |
| `selection_funnel.csv` | Attempted-to-accepted counts and reasons |
| `sweep_diagnostics.parquet` | One row per attempted sweep with support and QC |
| `beam_results.csv` | Beam estimates, uncertainty, sensitivities, and bounds |
| `boundary_results.csv` | Transition points and physical-curve diagnostics |
| `calibration_results.csv` | Injection recovery, coverage, and false detections |
| `residual_atlas.pdf` | Fixed audit set and all major failure classes |
| `pilot_summary.png` | Selection and observable summary |
| `runtime.json` | Environment, backend, device, memory, and timing |

Large caches remain under `artifacts/`; checksums and provenance are recorded
here.

## Decision Gate

The response-aware beam estimator advances to a second independent day only if:

- injection tests show acceptable bias and empirical interval behavior over the
  support retained by the QC funnel;
- beam-absent controls do not reveal an unacceptably high false-detection rate;
- benchmark and fixed visual-audit classifications agree with residual evidence;
- accepted beam estimates are not dominated by bounds, one energy row, one
  normalization, or optimizer starts;
- all failure rates and denominators are complete and scientifically legible.

The loss-cone boundary advances as corroborating evidence only for the subset
passing its stricter gate. A low corroborated fraction is a result, not grounds
for relaxing that gate.

Failure of the beam estimator returns the project to its specific response,
continuum, or uncertainty assumption. Successful completion does not authorize
mission-wide population claims until a contrasting day and independent plasma
regime/SZA annotations have also been evaluated.

## Non-Goals

- Mission-wide recomputation or headline medians.
- Treating the published voltage as ground truth.
- Inferring absolute `U_M` without validated `U_sc`.
- Using the historical joint fit as an acceptance criterion.
- Relaxing QC to maximize the usable fraction.
- Temporal spherical-harmonic reconstruction.
