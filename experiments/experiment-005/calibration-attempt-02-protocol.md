# Calibration Attempt 02 Protocol: Flexible Continuum and Held-Out Validation

**Status:** Frozen before Attempt 02 estimator outputs

**Configuration:** [`configuration-attempt-02.json`](configuration-attempt-02.json)

## Motivation

Attempt 01 detected injected beams without false positives, but its constant
continuum per energy row could not represent pitch-dependent loss-cone
structure. That discrepancy shifted the fitted centroid according to native
energy-grid phase. Within-row residual resampling removed the structured
discrepancy and produced narrow intervals around biased pseudo-solutions.

Attempt 02 changes only the continuum and uncertainty treatment needed to test
that failure mechanism. It does not relax the beam QC thresholds, alter the
instrument response, inspect the 398-sweep pilot, or use the published voltage
as a target.

## Observation Model

For energy row `e` and reflected-sector pitch `alpha`, fit

\[
R_e(\alpha)=r_e+c_e
\left[1+\exp\left(-\frac{\beta_e-\alpha}{w_e}\right)\right]^{-1}
+A\,G_E(E_e;E_b)G_\alpha(\alpha;180^\circ,\sigma_\alpha).
\]

The beam parameters `E_b`, `A`, and `sigma_alpha` are shared across energy rows.
Every row receives independent cone level `r_e`, contrast `c_e`, boundary
`beta_e`, and width `w_e`. These continuum parameters are nuisance terms only:

- they are not constrained by the adiabatic boundary equation;
- they do not share a voltage or magnetic ratio;
- they cannot be interpreted as loss-cone corroboration;
- they exist to prevent broad pitch structure from biasing the localized beam.

The beam energy response remains Gaussian with `FWHM/E = 0.48`; its pitch axis
remains fixed at 180 degrees. The fit uses native reflected sectors from
20--2000 eV and minimizes unweighted logarithmic residuals. Bounds and twelve
deterministic starts are frozen in the Attempt 02 configuration.

## Base Quality Control

Before empirical uncertainty calibration, require:

- finite convergence and covariance scale;
- nonnegative beam amplitude resolved at three Jacobian standard errors;
- beam center and angular width away from their bounds;
- stable centers across near-optimal starts;
- at least two energy rows with positive field-aligned excess;
- center standard error no larger than 0.25 response FWHM;
- agreement of incident-mean and conjugate normalizations within one response
  FWHM;
- leave-one-energy-row-out shift no larger than one response FWHM.

The Jacobian standard error is only an ordering/scale statistic at this stage.
It is not reported as a calibrated confidence interval.

## Independent Calibration and Validation Split

Use the same eight frozen beam cells, four real supports, 200 simulations per
cell, and 500 null controls as Attempt 01. The random draws are divided before
execution:

- Beam replicates `0-99`: empirical-calibration split.
- Beam replicates `100-199`: held-out validation split.
- Null replicates `0-249`: empirical-calibration split.
- Null replicates `250-499`: held-out validation split.

Both splits contain a fixed background-family mixture:

- 50% single broadened logistic transition;
- 25% double-transition/refilling structure;
- 25% asymmetric shoulder structure.

The latter two are deliberately outside the fitted single-logistic continuum
family. This tests whether the empirical correction generalizes across modest
continuum misspecification rather than merely validating a matched simulation.

## Empirical Correction

Using base-QC-retained calibration-split beams only:

1. At each frozen energy anchor (65, 150, 350, and 800 eV), estimate median
   signed centroid error.
2. Subtract the interpolated bias from fitted validation centers. Interpolation
   is linear in log energy and uses fitted, not true, energy.
3. At each anchor, estimate the 68th and 95th percentiles of absolute
   bias-corrected error divided by Jacobian center standard error.
4. Interpolate those scale factors in log energy and form symmetric empirical
   intervals on held-out validation cases.

No validation outcome may alter the correction map, model, QC, or gates. A
failed validation requires another labeled attempt and complete rerun.

## Held-Out Decision Gate

After applying the calibration-split correction, validation cases enter the
science-eligible denominator only when their calibrated 95% interval is no
wider than one local response FWHM. Require:

- at least 200 retained validation beams overall;
- at least 25 retained validation beams for a cell-level gate;
- median corrected absolute error no larger than 0.10 response FWHM;
- empirical 68% coverage in `0.60-0.76`;
- empirical 95% coverage in `0.88-0.99`;
- held-out null false-detection rate no larger than 5%;
- every coverage-eligible validation cell passes all point and interval gates.

Only a full held-out pass opens the frozen 398-sweep real-data pilot. Attempt 01
results remain immutable and are not pooled with Attempt 02.

## Limitations Fixed in Advance

The held-out split tests repeated-noise and nuisance-background generalization
within the frozen simulation family; it is not proof that those simulations
span every real LP plasma distribution. Passing would authorize a one-day
residual audit, not mission-wide voltage claims. The loss-cone boundary remains
a separate conditional diagnostic and is not validated by this beam-continuum
fit.
