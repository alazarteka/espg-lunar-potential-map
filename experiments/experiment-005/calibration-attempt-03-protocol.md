# Calibration Attempt 03 Protocol: Smooth Continuum and Held-Out Validation

**Status:** Frozen before noisy calibration outputs

**Configuration:** [`configuration-attempt-03.json`](configuration-attempt-03.json)

## Parent Failure

Attempt 02 allowed independent logistic continuum parameters at every energy.
The correct noiseless solution existed, but practical multistart fitting found a
competing low-residual basin at 139.3 eV for a true 150 eV beam. Arbitrary
row-to-row nuisance variation could absorb part of the beam energy response.

Attempt 03 reduces that freedom without imposing a loss-cone voltage equation.

## Smooth Nuisance Continuum

The observed reflected distribution remains

\[
R(E,\alpha)=C(E,\alpha)+A\,G_E(E;E_b)
G_\alpha(\alpha;180^\circ,\sigma_\alpha).
\]

The continuum `C` is logistic in pitch. Its cone level, contrast, boundary, and
width are each represented by a bounded quadratic function of normalized log
energy. This yields 12 smooth nuisance coefficients plus the three shared beam
parameters.

The continuum:

- has no adiabatic boundary equation;
- has no shared surface voltage or magnetic ratio;
- may describe broad loss-cone morphology but cannot independently vote on
  beam voltage;
- cannot change arbitrarily from one energy row to the next.

The fit uses the same native reflected sectors, 20--2000 eV domain, logarithmic
objective, response `FWHM/E = 0.48`, beam bounds, and twelve deterministic starts
as the frozen Attempt 02 protocol. Smooth latent coefficients are bounded on
`[-12,12]` before transformation to physical continuum bounds.

## Precalibration Implementation Gate

Before this protocol was frozen, noiseless single-logistic simulations at 65,
150, 350, and 800 eV were fitted with the development implementation. Every
center was recovered within 0.05 local response FWHM. No noisy calibration,
coverage, false-detection, or real-pilot result was inspected.

The exact-recovery test remains automated. Failure after subsequent code changes
invalidates the run.

## Base QC and Empirical Uncertainty

Base QC is unchanged from Attempt 02. Jacobian center error is a scale statistic,
not a confidence interval. It must be finite, sufficiently resolved, and paired
with amplitude significance, multistart stability, normalization agreement,
bound checks, positive excess in at least two energy rows, and leave-one-energy-
row-out stability.

Replicates `0-99` in each beam cell calibrate median signed bias and the 68th and
95th percentiles of bias-corrected absolute error divided by Jacobian center
error. Corrections are anchored at 65, 150, 350, and 800 eV and interpolated in
log energy using fitted energy.

Replicates `100-199` are untouched validation cases. Null controls `0-249` are
calibration-side diagnostics; nulls `250-499` form the held-out false-detection
test.

## Background Misspecification

Both splits contain the same frozen mixture:

- 50% single logistic backgrounds;
- 25% double-transition/refilling backgrounds;
- 25% asymmetric shoulder backgrounds.

Only the first family lies inside the fitted continuum family. Empirical
calibration and held-out validation therefore include modest, explicit model
discrepancy.

## Held-Out Gate

After correction, require:

- at least 200 retained validation beams overall;
- at least 25 retained validation beams for a cell-level gate;
- calibrated 95% interval width no larger than one response FWHM;
- median corrected absolute error no larger than 0.10 response FWHM;
- 68% coverage in `0.60-0.76`;
- 95% coverage in `0.88-0.99`;
- held-out null false-detection rate no larger than 5%;
- every coverage-eligible cell passes all gates.

No validation outcome may alter the model, correction map, selection rules, or
thresholds. Only a complete held-out pass opens the frozen 398-sweep pilot.

## Interpretation Limit

Passing authorizes a one-day beam-observable residual audit. It does not validate
absolute lunar potential, the loss-cone boundary as corroborating evidence, or
mission-wide population claims.
