# Scientific Resubmission Audit

**Date:** 2026-07-10

**Status:** Active scientific decision record

**Scope:** Submitted manuscript and code, reviewer comments, adviser rewrite,
resubmission plan, post-submission refactors, current scientific branch, and
representative real-data diagnostics

## Executive Assessment

The project has a publishable scientific core, but the revision should be
narrower and more rigorous than the current plan. The strongest defensible
claim is:

> Lunar Prospector Electron Reflectometer observations conditionally constrain
> an effective surface-to-spacecraft potential drop for a regime- and
> geometry-dependent subset of sweeps, but cannot identify an unconstrained
> global time-varying potential field at the tested resolutions without strong
> external priors.

This preserves the important negative result without claiming that every
possible static, coarse, monthly, or covariate-conditioned lunar potential map
is impossible.

The submitted mission-wide per-sweep statistics are not yet comparably secure.
The reported 59.65% identifiable fraction and the medians of -108 V and -841 V
depend on an empirical optimizer-sampling diagnostic, fixed spacecraft
potential, pooled plasma environments, a data-dependent mask, fixed beam
widths, and fits that frequently terminate on parameter bounds. Those archive
statistics should be considered superseded until the revised estimator and
selection function are validated and rerun.

The recent smooth-boundary and Cash/profile work is useful infrastructure, but
the current implementation does not support a formal 68% confidence-interval
claim. Representative real spectra fail its goodness-of-fit criterion by
orders of magnitude, while nominal zero-width profiles arise from coarse grid
resolution. The full archive should not be recomputed with this method until
the observation model and interval calibration pass explicit stop/go tests.

## Audit Basis

This assessment used:

- The complete submitted 21-page manuscript and both reviewer reports from
  `scratch/next-steps.zip`.
- The adviser material in `scratch/professor_comments.docx`, which is a proposed
  rewrite and writing guide rather than a file containing Word comments or
  tracked changes.
- The working plans in `scratch/resubmission_plan.md` and
  `scratch/resubmission_changes.md`.
- Submission-era tag `lp-er-derived-data-v1.0.0` at `bd21373`; its scientific
  tree was effectively frozen at `f0cab5f`.
- Current branch `losscone-weighted-width` at `cbad014`, compared with `main` at
  `4746c3c`.
- Current source, tests, archived analyses, and the available April 29, 1998
  legacy cache.
- Independent manuscript/reviewer, code-history, statistical inverse-problem,
  and lunar-plasma-physics audits.
- Official Lunar Prospector PDS documentation and the published Halekas et al.
  2008 benchmark case.

No repository files were changed during the underlying scientific audit. The
diagnostic fits described below were read-only and did not produce a tracked
scientific artifact.

## What the Submission Did

### Estimand

The manuscript reports a voltage relative to the spacecraft,

\[
U_{\mathrm{rel}} = U_{\mathrm{surface}} - U_{\mathrm{spacecraft}},
\]

and sets `U_spacecraft = 0` as a reporting reference. This is not an assertion
that the spacecraft physically has zero floating potential. When that
reference is used, the result should be called an effective or fitted
surface-to-spacecraft potential drop, not an absolute surface potential.

If a nonzero spacecraft potential is retained, the manuscript and code must
distinguish clearly among ambient-plasma-referenced lunar potential, spacecraft
potential, and their difference.

### Forward Model

Each 15-energy sweep is normalized using incident-side flux and fit with:

- The adiabatic loss-cone relation.
- A hard loss-cone boundary in the submitted version.
- A fixed normalized floor.
- A surface-emitted Gaussian beam centered at the voltage-dependent energy.
- Free voltage, magnetic-field ratio, and beam amplitude.
- Fixed beam widths of 15 eV and 7.5 degrees.

The beam implementation was more physical than the submitted prose suggested:
its center is tied to `U_spacecraft - U_surface`, and it is suppressed when the
required acceleration is not possible. Calling it a generic nuisance or a
"returning" beam was a manuscript error. Its free amplitude and fixed widths
nevertheless remain important sources of model uncertainty.

### Submitted Fit Statistic

The manuscript's Lillis path uses an unweighted normalized squared residual,
divided by the nominal degrees of freedom, over bins selected by observed
relative flux between 0.07 and 0.79. It is not a variance-weighted reduced
chi-square.

The submitted U-width is:

1. Evaluate 400 Latin-hypercube samples in voltage, field ratio, and beam
   amplitude.
2. Find the best objective value among those samples.
3. Retain samples satisfying `Q <= Q_min,LHS + 0.001`.
4. Define U-width as the maximum retained voltage minus the minimum retained
   voltage.
5. Label fits with U-width below 200 V as identifiable.

The width is referenced to the best LHS point, not the later differential-
evolution optimum. It therefore depends directly on sparse three-dimensional
sample placement. A width of zero can mean that only one LHS point happened to
survive, not that voltage was measured exactly.

### Submitted Headline Results

The paper reports:

- 380,166 finite fitted measurements.
- 226,774, or 59.65%, passing the 200 V U-width gate.
- Median voltage -108 V for the accepted subset.
- Median voltage -841 V for the rejected subset.
- Daily folded-longitude widths with a median of 14.9 degrees.
- Sparse daily longitude occupancy even after three- and seven-day aggregation.

The paper did not report the complete flow from attempted sweeps through valid
geometry, sufficient bins, numerical convergence, adequate fit, finite width,
nuisance-bound status, and final reporting. Those denominators are required for
scientific interpretation.

## Reviewer Assessment

### Reviewer 1

Reviewer 1's objections are primarily about scientific validity obscured by
exposition, not merely poor writing.

- **Pooled environments:** Correct. Plasma regime and illumination affect both
  physical charging and model adequacy. They must be independent conditioning
  variables, not inferred circularly from the same fitted spectrum.
- **Spacecraft potential:** Correct. Retaining it requires quantified model and
  parameter sensitivity; pseudocode and recovery fractions are insufficient.
- **U-width:** Defining it more clearly does not repair its sparse-sampling and
  coverage problems.
- **Beam:** Correct that it is emitted physical information. Its centroid can
  become a valuable cross-check against the loss-cone voltage.
- **Magnetic polarity:** The code already traces both directions and orients the
  measured field Moonward before computing pitch angle. What remains missing is
  an end-to-end real-spectrum polarity or reversal-invariance validation.
- **Mask limits:** The 0.07-0.79 thresholds remain empirically chosen and become
  more problematic when used inside a purported formal likelihood.
- **Fixed beam widths:** Neither 15 eV nor 7.5 degrees has been demonstrated to
  represent the instrument response throughout the fitted energy range.

### Reviewer 2

Most of Reviewer 2's requested corrections are editorial, but three identify
central scientific gaps:

- **Statistical weighting:** The paper must explain the observation and error
  model. The reviewer asked why formal weighting was not used; this did not
  require replacing the empirical statistic with an under-validated formal
  confidence claim.
- **Bimodality:** Medians do not explain the approximately -100 V and near-lower-
  bound populations. Bound censoring, nuisance degeneracy, and model failure
  must be separated from plasma-regime mixtures first.
- **Space-time confounding:** A regularized harmonic surface is not evidence
  that the unsampled field is identified. The conclusion must be tied to time
  window, basis, spatial degree, uncertainty, and matrix conditioning.

Reviewer 2's requested reference, notation, caption, figure, equation, symbol,
and organization corrections should all be made directly. They are separate
from the scientific validation work.

### Adviser Rewrite

The proposed rewrite contains a sign error. Under the manuscript relation

\[
\sin^2 \alpha_c = \frac{B_s}{B_m}\left(1 + \frac{U}{E}\right),
\]

making a negative `U` more negative decreases `alpha_c`; it does not increase
it. In the implemented high-pitch convention, the depleted region narrows or
closes toward 180 degrees at low energy.

The bibliography also mismatches several titles and author lists with the
requested DOIs. In particular:

- `10.1029/2010GL046119` is Poppe et al., *Negative potentials above the
  day-side lunar surface in the terrestrial plasma sheet: Evidence of
  non-monotonic potentials*.
- `10.1002/grl.50135` is Futaana et al., *Remote energetic neutral atom imaging
  of electric potential over a lunar magnetic anomaly*.
- `10.1029/JA088iA01p00096` is Feldman et al., *Electron velocity distributions
  near the Earth's bow shock*.

## What the Original Results Actually Establish

The checked-in April 29, 1998 legacy cache contains 1,031 sweeps, of which 606
have finite fits and 387 pass the submitted U-width gate. Among the 606 finite
fits:

- 117, or 19.3%, are at or very near the -2000 V voltage bound.
- Approximately 45% hit a `B_s/B_m` bound.
- Approximately 19% hit the upper beam-amplitude bound.
- 83 have U-width exactly zero.
- 45 lower-voltage-bound fits nevertheless pass the identifiability gate.

This is direct evidence that the submitted gate admits boundary-pinned and
nuisance-degenerate solutions. The visible extreme mode in Figure 2 is
therefore at least partly numerical or censored until shown otherwise.

The submitted identifiable subset is also selected on signal level, geometry,
voltage, background, field complexity, model adequacy, and optimizer behavior.
It cannot be treated as an unbiased lunar population. Mission-wide medians and
environmental contrasts require a measured selection function and correlated-
sample uncertainty, such as orbit- or day-level block bootstrap.

## Scientifically Material Code History

| Commit | Scientific significance |
| --- | --- |
| `19c0a0d` | Corrected the J-U coefficient by a factor of ten. |
| `cc08925` | Removed spectrum-row cross-validation leakage; reported R2 decreased from about 0.166 to 0.120. |
| `dceb7a2`, `2cbd5ac` | Introduced CPU and Torch Lillis fitting. |
| `d96b46a` | Enabled the manuscript's `U_spacecraft = 0` reporting convention. |
| `d243452`, `b8e347d`, `f0cab5f` | Added empirical U-width, diagnostics, and identifiable-only temporal filtering. |
| `7d5c073` | Fixed a collapsed temporal linear basis and added cache aggregation checks. |
| `c72ef63`-`7c67b53` | Added a sigmoid boundary, count-space Poisson deviance, profile widths, and CPU/Torch routing. |
| `f5297d9` | Added top-energy-channel background estimation and fallback behavior. |
| `9750122` | Added profile-width schema and method-aware temporal key selection. |
| `cbad014` | Enabled the Torch Cash path in batch routing. |

The July module splits, deduplication, GPU environment work, and documentation
changes improve maintainability and reproducibility. They should not be
described as new scientific evidence. Reframing the temporal product as an
identifiability analysis is appropriate, but documentation alone does not
establish rank deficiency or construct a null-space witness.

The configured default fit method also remains Halekas rather than the Lillis
or Cash path used for the manuscript and current experiments. Future artifacts
must record the command, commit, fit method, all fixed parameters, and quality-
control schema needed to reproduce them.

## Audit of the New Cash/Profile Method

### Smooth Boundary

Replacing the hard edge with a sigmoid can reduce pitch-bin aliasing, but it is
a physical model change as well as a numerical one. With `tau = 5 degrees`, the
logistic 10-90% transition is approximately 22 degrees. This may be a plausible
proxy for angular response, but it has not been derived from that response.

Matched-model recovery is insufficient because fixed smoothing can itself
manufacture curvature in an otherwise weakly identified voltage direction.
Validation must include response integration or broad tau-mismatch tests.

### Reconstructed Counts

The current method reconstructs floating count-like values from calibrated
flux, geometric factor, energy, and accumulation time. The PDS product has
already undergone dead-time correction and logarithmic compression/digitization
and retains instrumental or cosmic background. Its uncertainty includes more
than raw independent Poisson counting noise.

The current expectation is also statistically inconsistent:

- Incident normalization is estimated from the same sweep and then treated as
  known, ignoring shared-denominator covariance.
- The data-dependent mask is selected using observed flux.
- The normalization contains the archive background, while a separate additive
  background is introduced in the expected counts, creating a likely double-
  counting problem.
- The normalized forward model already has a fixed floor.
- Beam amplitude and other parameters often lie on bounds, violating ordinary
  Wilks regularity.

The resulting objective is a plug-in Poisson pseudo-likelihood, not a complete
raw-count likelihood.

### Profile Width

The current profile uses approximately:

- 80 voltage values over -2000 to +20 V, or 25.6 V spacing.
- 17 magnetic-ratio values.
- 9 beam-amplitude values.

Nuisance parameters are evaluated on this grid rather than continuously
optimized. One accepted voltage cell can therefore produce a zero-width
"interval." Boundary-truncated intervals are converted to the full 2020 V
span without preserving all of the information needed to distinguish an open
limit from a broad interior interval.

`Delta D <= 1` is asymptotically associated with a nominal 68% interval only
under regular, correctly specified likelihood conditions. Those conditions
are not established here. Until empirical coverage is demonstrated, the
quantity should be called a conditional profile-based identifiability width.

### Real-Data Stop/Go Check

On 80 evenly sampled real sweeps from April 29, 1998:

- Zero passed the configured reduced-deviance threshold of 3.
- Among 51 numerically finite fits, reduced deviance ranged from about 56 to
  4,789, with a median near 917.
- None had reduced deviance below 10; only nine were below 100.
- Most nominally accepted widths were exactly zero because of profile-grid
  quantization.
- Voltage, magnetic ratio, and beam amplitude still showed frequent bound
  saturation.

This is not a threshold-tuning problem. Either the likelihood scale is wrong,
the forward model is incomplete relative to counting precision, or both.
`Delta D = 1` has no demonstrated coverage meaning in this regime.

A full-day GPU batch attempt also exhausted the available 12 GB GPU during the
profile scan. Automatic batch sizing currently does not account for profile-
grid memory. Scientific validation is the primary blocker, but production
recomputation is not yet operational either.

### Published Positive Control

For the Lunar Prospector interval nearest April 29, 1999 at approximately
14:45 UT, Halekas et al. 2008 reported roughly:

- Lunar potential `U_M = -160 V`.
- Spacecraft potential `U_sc = +11 V`.
- Magnetic ratio `B_s/B_m = 0.975` in the paper's convention.

Using the current smooth model with `U_sc = 0`, the Lillis path recovered a
voltage of approximately -157 V and a magnetic ratio of 0.990. With
`U_sc = +11 V`, it recovered an effective relative voltage near -162 V and a
similar ratio. The exact absolute/reference notation must be aligned before
claiming one-to-one agreement, but the published voltage scale is reproduced
well.

The Cash path instead moved to approximately -249 V with `U_sc = 0`, returned a
magnetic ratio near 1.044, had reduced deviance near 1,940, and produced zero
profile width. With `U_sc = +11 V`, it remained near -252 V relative to the
spacecraft and had reduced deviance near 1,421.

This comparison is currently the strongest evidence against treating the new
Cash implementation as an achieved scientific improvement. The smooth Lillis
baseline should be retained while the observation model is reconsidered.

## Physical Interpretation Issues

### Beam and Instrument Response

A fixed 15 eV beam width cannot represent an instrument whose energy resolution
is fractional across a range extending to keV energies. The 7.5-degree pitch
width and 5-degree boundary smoothing also require response-based justification
or sensitivity analysis.

The strongest available internal cross-check is to estimate voltage separately
from:

- The loss-cone boundary.
- The upward-beam energy centroid, convolved with the energy response.

Agreement tests a physical prediction without allowing a free beam amplitude
to absorb all residual structure. Beam-free, fixed-amplitude, and flexible-
amplitude fits should be reported because archived sensitivity work already
shows voltage shifts of order 100 V or more.

### Magnetic Ratio and Polarity

There is a live notation inconsistency: one result description labels
`bs_over_bm` as surface over spacecraft, while the model and bounds use the
spacecraft-to-mirror-surface convention. This must be corrected everywhere.

The field-polarity path appears intentional, but geometry unit tests do not
validate the PDS look-direction versus particle-velocity convention. A known
real sweep containing an upward beam and loss cone should be processed under
both orientations and compared with the published physical interpretation.

High nuisance-bound rates also call for comparison between the fitted magnetic
ratio and an independent magnetic-geometry expectation. Straight local-field
projection is least trustworthy near crustal anomalies, so anomaly maps should
not be added before this validation.

### Spacecraft Potential

The dayside kappa/J-U and nightside current-balance algorithms substantially
predate the current revision. The nightside model assumes simplified electron
and ion populations, quasi-neutral proton ions, `T_i = T_e`, and fixed secondary
electron-emission parameters. It returns one bracketed Brent root without
enumerating roots or propagating input uncertainty.

A shallow current-balance curve implies an unstable or imprecise root; it does
not by itself imply multiple roots. Multiple roots require a demonstrated
nonmonotonic curve and explicit enumeration.

Until independently validated, spacecraft charging should remain a sensitivity
analysis. Results should be repeated for `U_sc = 0`, a nominal estimate, and
plausible perturbations. Any interval must be described as conditional on
spacecraft potential unless that uncertainty is propagated.

### Environment, Illumination, and Wake

The current production labels use spectrum-derived electron temperature and a
shadow flag. This is circular with the spectral inference and incorrectly
equates every shadowed footprint with the lunar wake.

The revision should use independent, conservative classifications:

- Broad solar wind, magnetosheath, magnetotail, and ambiguous categories.
- Continuous sweep-level SZA, including the range across the sweep.
- Spacecraft wake membership based on position relative to plasma flow,
  separately from surface illumination.
- Lobe versus plasma-sheet subdivision only where supported by independent
  in-situ information.

OMNI propagation and boundary models have timing and boundary uncertainty at
the Moon. Ambiguous intervals should remain ambiguous rather than being forced
into high-confidence categories.

### Sweep Support

It is inaccurate to describe one fitted spectrum as a uniform 80-second or
120-km boxcar average. Individual energy/angular accumulations have roughly
2.5-second support, while the 15 energies are acquired sequentially over about
70-80 seconds. The nonlinear fit combines measurements taken at different
positions.

Use one estimate per sweep, with start/end or complete row-level footprint,
SZA, magnetic magnitude, and field-direction dispersion retained as support and
quality metadata. Do not spread the fitted value uniformly along a segment.
Sweeps that straddle materially different physical conditions should be flagged
or excluded.

## Global Reconstruction and Identifiability

The negative reconstruction conclusion is plausible and likely to survive, but
the current evidence is incomplete.

The temporal loader currently consumes row-level coordinates while repeating a
sweep-level potential across as many as 15 energy-row footprints. This inflates
nominal sample size and creates artificial spatial support. Filtering before
grouping can also allow a sweep to leak across train and test partitions.
Correcting this should strengthen the negative result, but it will alter quoted
coverage and conditioning metrics.

Occupied-bin fractions and a smooth regularized solution are not
identifiability diagnostics. The revised paper should show:

- Design-matrix singular values and effective rank before regularization.
- Condition number as a function of temporal window and `l_max`.
- Resolution or leverage maps.
- Regularization paths.
- Injection/recovery using the actual LP sampling geometry.
- Orbit-, time-, and longitude-blocked validation.
- An explicit null-space witness: two physically plausible fields that agree
  along LP tracks within uncertainty but differ elsewhere.

The conclusion should then be scoped as failure to identify an unconstrained
daily or joint space-time field at stated resolution and prior strength. It
should not be stated as a theorem that no coarser or more strongly constrained
climatology can ever be estimated.

## Status Against the Resubmission Plan

| Work item | Status | Scientific assessment |
| --- | --- | --- |
| Smooth boundary | Implemented | Useful, but response calibration and mismatch tests are missing. |
| Multi-start optimization | Not implemented | Optimizer agreement would test numerical stability, not identifiability. |
| Cash objective | Implemented on branch | Observation model is not validated and fails real-data adequacy checks. |
| Profile width | Implemented on branch | Too coarse and uncalibrated for formal confidence language. |
| Rich cache metadata | Not implemented | Endpoints, rail flags, valid bins, support, background, optimizer, and provenance are needed. |
| Synthetic validation | Partial | Formula/parity and matched-model recovery exist; repeated-noise coverage and misspecification tests do not. |
| Independent plasma regimes | Not implemented | Current temperature/shadow labels do not answer Reviewer 1. |
| Continuous SZA and sweep support | Not implemented | Needed for conditioning and selection analysis. |
| Spacecraft-potential sensitivity | Not implemented | Existing algorithm is retained without root or uncertainty analysis. |
| Beam cross-validation | Not implemented | Separate centroid/boundary agreement is a high-value positive control. |
| Full archive recompute | Not performed | Must wait for method stop/go criteria. |
| Formal temporal null-space analysis | Not implemented | Documentation was reframed, but rank and injection evidence are absent. |
| Revised manuscript | Not present in repository | Scientific claims should be frozen before rewriting results. |

## Recommended Scientific Sequence

### 1. Resolve the Observation Model

Before any archive run, verify the telemetry-to-flux transformation, effective
dwell time, compression, dead-time correction, background treatment, and
whether a meaningful raw-count likelihood can be reconstructed.

If raw-level inference is defensible, fit counts with row-specific incident
intensity and background nuisance parameters jointly profiled with voltage,
field ratio, and beam parameters. If it is not, do not force a Poisson
interpretation. Use a quasi-likelihood, robust objective, or empirically
calibrated profile diagnostic and explain why nominal counting precision would
be falsely precise relative to model discrepancy.

### 2. Establish a Per-Sweep Validation Ladder

Require all of the following before archive recomputation:

1. Reproduce published Halekas benchmark intervals and notation.
2. Demonstrate physically structured residual improvement on real spectra.
3. Compare loss-cone and independently estimated beam-centroid voltages.
4. Test polarity reversal, mask limits, normalization, background, tau, beam
   widths, parameter bounds, and multiple optimizer starts.
5. Calibrate interval or diagnostic coverage over repeated noisy simulations,
   including model mismatch and parameter-bound cases.
6. Pass a complete one-day pilot with a reported selection funnel and residual
   atlas.

### 3. Make Profiling Numerically Credible

Replace the coarse nuisance grid with adaptive or continuous profiling around
the optimum. Retain lower and upper endpoints, open limits, disconnected
components, boundary contacts, optimizer state, fit deviance, and all nuisance
rail flags. Treat multiple starts only as an optimizer diagnostic.

### 4. Add Independent Physical Conditioning

Implement broad independent plasma-region labels, continuous SZA, separate
wake geometry, and sweep-support metadata. Report attempted-to-accepted
denominators within every stratum and model the probability of passing quality
control. Use orbit/day block bootstrap or an appropriate hierarchical model for
population summaries.

### 5. Bound Spacecraft-Potential Claims

Scan the full current-balance curve, enumerate roots and slopes, propagate
kappa/SEE/ion assumptions, and compare fast and sequential paths. Unless this
passes validation, keep spacecraft charging in a limitations or sensitivity
section rather than making it a second headline result.

### 6. Formalize the Negative Mapping Result

Change temporal inference to one correlated observation per sweep and then add
singular-spectrum, effective-rank, resolution, null-space, injection, and
blocked-validation results. This is likely the cleanest and strongest primary
contribution.

### 7. Freeze and Recompute

Only after the above choices are frozen should the archive be recomputed. Every
artifact should include method, commit, command, fixed parameters, interval
definition, background method, optimizer state, and complete processing
denominators.

## Recommended Paper Structure

### Headline

1. A formal sampling limitation: multiple spatiotemporal fields are
   observationally indistinguishable along LP tracks without strong priors.
2. Validated sweep-scale constraints on effective potential drop for a defined,
   geometry-qualified subset.
3. The selection function itself: success, failure, uncertainty, and bound
   behavior vary with plasma regime, SZA, signal level, and sweep geometry.

### Demote or Defer

- Mission-wide medians and bimodality until the revised selection function is
  understood.
- SEP validation as a headline result; the current -2000 V bound excludes some
  published extreme events and background is least reliable there.
- Crustal-anomaly overlays until field-line geometry and multivariable
  confounding are controlled.
- Detailed secondary-electron-emission charging physics unless independently
  validated.
- Terminator climatology and new global products before the measurement model
  is stable.
- High-confidence four-way OMNI labels where broad or ambiguous categories are
  more defensible.

## Final Judgment

The original conceptual insight survives scrutiny: selected LP ER sweeps contain
information about an effective surface-to-spacecraft voltage, and LP's sampling
does not support a freely varying global spatiotemporal reconstruction. The
submitted U-width and pooled archive statistics do not yet quantify that
information reliably, and the current Cash/profile implementation is not a
validated replacement.

The best resubmission is therefore an honest, quantitatively demonstrated
account of:

- Where the per-sweep forward model succeeds.
- How model adequacy and selection depend on environment and geometry.
- Which voltage quantity is actually identified.
- How spacecraft charging limits interpretation.
- Why the global inverse problem has a large practical null space.

That is narrower than the current expansion plan, but scientifically stronger
and more responsive to both reviewers.

## Primary External References

- Halekas et al. (2008), *Lunar Prospector observations of the electrostatic
  potential of the lunar surface and its response to incident currents*:
  <https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JA013194>
- NASA PDS LP ER 3-D electron flux product:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ER-3-RDR-3DELEFLUX-80SEC-V1.1>
- NASA PDS LP ER derived electron product:
  <https://pds.nasa.gov/ds-view/pds/viewProfile.jsp?dsid=LP-L-ER-4-ELECTRON-DATA-V1.0>
- Poppe et al. (2011):
  <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2010GL046119>
- Futaana et al. (2013):
  <https://agupubs.onlinelibrary.wiley.com/doi/10.1002/grl.50135>
- Feldman et al. (1983):
  <https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/ja088ia01p00096>
