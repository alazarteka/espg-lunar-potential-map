# C1 Simulator And Calibration Execution Specification

**Status:** Pre-outcome execution draft; freezes only with simulator source and manifest

**Scientific role:** This document converts the Experiment 006 design into an
executable calibration protocol. It resolves implementation choices left open
in `protocol.md` without changing the scientific population or V1 gates.

## Fixed Measurement Conventions

Scientific errors are normalized by the nominal local response width

\[
w_0(E)=0.48E,
\]

evaluated at the true center for simulation diagnostics. A deliberately broad
or narrow response does not relax or tighten the gate denominator. The fitted
Jacobian center standard error remains the uncalibrated interval scale.

Bias correction is defined only on `100-500 eV` and is linear in log energy
between the five anchors. Constant nearest-endpoint extension is permitted only
to execute QC and out-of-window classification. It does not extend the
scientifically validated domain and cannot support a corrected measurement
claim outside `100-500 eV`.

## Deterministic Case Allocation

C1 has 3,000 cases: 600 at each of `100, 140, 220, 340, 500 eV`. At each
anchor, case ordinals `0-299` form the bias split and `300-599` form the
interval split. Every split uses all 12 calibration supports exactly 25 times.

Within each anchor and split, a deterministic constrained-allocation search assigns:

- four background families, 75 cases each;
- four noise levels, 75 cases each;
- five response families, 60 cases each;
- four amplitude quartiles, 75 cases each; and
- four pitch-width quartiles, 75 cases each.

Candidate allocations are derived from separate named seed streams and accepted
only when every pairwise factor table equals the floor or ceiling of its
balanced expectation. Failure to find one is a manifest-generation failure,
not permission to relax balance. Temporal families are assigned exactly
150/75/75 as none/ramp/pulse in every anchor and split. Beam center is fixed
before, and independently of, every continuum draw.

Amplitude is log-uniform on `[4,16]`; pitch sigma is uniform on `[12,42] deg`.
Quartiles stratify those continuous draws without replacing them by point
values. Native-channel phase is uniform over one local logarithmic channel
spacing and shifts the latent beam center relative to fixed native channels.
It does not alter channel calibration and is independent of the response
energy-scale challenge.

## Independent Sector-Level Generator

The simulator may share geometry containers and normalization utilities with
the estimator, but it may not import or call `beam_model.model_values` or the
estimator continuum evaluator. Its continuum equations are implemented in a
separate module and tested by properties, not by source-code equality.

Each case is generated in this order:

1. Load one frozen calibration geometry support and orient pitch so 180 degrees
   is surface-leaving.
2. Draw continuum, beam, response, discrepancy, and temporal latent variables.
3. Integrate the latent spectrum over each native energy channel.
4. Construct one positive native `energy row x sector` flux realization.
5. Apply temporal effects, correlated or heteroskedastic log noise, and masks.
6. Derive incident-hemisphere-mean and nearest-conjugate normalizations from
   that same final latent realization.

No independent noise is added after the normalization branch. Invalid or
nonpositive denominators invalidate the affected normalized samples.

The background families are smooth logistic, double-transition refilling,
asymmetric shoulder, and correlated smooth energy-pitch field. C1 balances all
four. The physically coupled voltage/boundary arm and semi-synthetic residual
arm are not used to estimate C1 coefficients; they remain secondary V1 layers.

The five response families are nominal linear-Gaussian, response-width stress,
energy-scale offset, log-energy Gaussian, and intrinsic/asymmetric broadening.
Their numerical ranges are frozen in `configuration-c1.json`.

## Normalization Algorithms

For each energy row, incident sectors have oriented pitch below 90 degrees and
reflected sectors have pitch from 90 through 180 degrees.

The incident-mean product divides every valid sector by the arithmetic mean of
all positive finite incident-sector fluxes in that row. The conjugate product
sets valid incident samples to one; each reflected sector is divided by the
positive finite incident sector whose pitch is nearest to `180 deg - alpha`.
Ties use the lower native sector index. A row requires at least six valid
incident and six valid reflected sectors before either product can enter a fit.

Both fits use reflected sectors only; incident samples never enter the
objective. Both branches begin with the same native finite/positive mask, but a
branch-specific invalid denominator may remove a sample from that branch. A row
enters either fit only when the common native mask has at least six incident and
six reflected sectors. Bias and intervals are calibrated for the incident-mean
branch only; the conjugate branch is a stability diagnostic. Their separately
fitted centroids and difference are retained.

## Frozen Base QC

C1 uses the Experiment 005 Attempt 03 QC vocabulary without outcome tuning:

- fitted amplitude divided by its CPU-Jacobian standard error at least 3;
- center at least 0.5 nominal response FWHM from optimizer bounds;
- pitch sigma at least 1 degree from its bounds;
- center range among successful starts no more than 1% above the best objective
  no more than 0.25 response FWHM;
- center Jacobian standard error no more than 0.25 response FWHM;
- normalization-center difference no more than 1 response FWHM;
- complete leave-one-energy-row-out refits all succeed and shift the center no
  more than 1 response FWHM;
- at least two rows whose fitted beam component at a reflected sample exceeds
  5% of fitted amplitude;
- at least two contributing rows with two sectors within 22.5 degrees of the
  surface-leaving axis;
- finite covariance with Jacobian rank 15 under NumPy's default SVD rule; and
- rejection when removing one contributing row eliminates every remaining row
  above the 5% beam-component threshold.

Response-FWHM QC thresholds use `0.48 * fitted_center`; simulation error gates
use `0.48 * true_center`. Undefined standard errors and failed branch or
leave-one-row-out fits are QC failures.

Every attempted case remains in retention. A failed fit, invalid covariance,
or undefined normalization is not available for bias or interval estimation.

## Calibration Mathematics

For anchor `a`, the first 300 cases estimate

\[
b_a=\operatorname{median}(\widehat E_i-E_i)
\]

among base-valid cases. Interpolation of `b_a` is linear in log energy. The
second 300 cases are corrected using only this first-half function. For each
base-valid second-half case,

\[
r_i=\frac{|E_i-E_{\mathrm{corr},i}|}{s_{J,i}},
\]

where `s_J` is the unscaled CPU-Jacobian center standard error. Invalid,
nonpositive, or nonfinite `s_J` fails base QC.

Correction solves the monotone calibration equation
`E_hat = E_corrected + b(E_corrected)` on `[100,500] eV` by bracketed root
finding; a missing or nonunique root is invalid. At each anchor, the 68% and
95% multipliers are conservative empirical order statistics with one-based rank
`ceil((n+1)q)`, capped at `n`; no interpolation between sample ranks is used.
This is empirical scale calibration, not a distribution-free conformal
guarantee. The final multiplier function is linear in log energy between
anchors and has no scientific extrapolation.

C1 estimates corrections; it does not itself pass or fail the beam hypothesis.
It must report attempted and retained counts, pre/post-correction bias,
calibration-sample coverage, interval widths, and failures by every allocation
factor. Coverage on the interval half is in-sample and descriptive; V1 supplies
held-out coverage evidence. Fewer than 150 retained cases in either half of any
anchor blocks V1. Fewer than 30 retained cases in any factor group within an
anchor blocks V1.

## Seeds And Information Barrier

Seeds use BLAKE2b-128 over compact canonical JSON encoded as ASCII, for example
`["experiment-006-c1-v1","noise",140.0,"interval",840,596]`, interpreted as
an unsigned little-endian integer. Ordered fields are root, namespace, anchor,
split, global case ID, and support ID. Independent namespaces are used for allocations,
continuum, beam morphology, response, noise, temporal effects, masks, and
diagnostic resampling. The root and namespaces are in
`configuration-c1.json`; the complete case manifest is generated and committed
before C1 outcomes.

Confirmation uses a separately frozen root and unused namespaces. No
confirmation flux, residual morphology, or fit output may be read while C1 is
implemented or evaluated.

## Required Pre-C1 Gates

Before C1 execution:

1. The exact 3,000-row manifest and its checksum are committed.
2. Generator output is positive and deterministic.
3. Continuum latents are statistically and programmatically independent of
   beam center.
4. Channel integration converges under quadrature refinement.
5. Both normalizations reproduce hand-computed examples and differ on a
   nonuniform incident hemisphere.
6. Nominal matched noiseless cases spanning anchors, supports, phase and
   morphology quartiles, and both branches recover injected centers within
   `1e-4` nominal response FWHM using complete CPU multistart.
7. Sector permutation, common flux scaling, and deterministic replay
   invariances pass.
8. Configuration, source, support, and manifest hashes are recorded.

Any failed gate is repaired and rerun before C1. Once the first C1 fit is
generated, changing the simulator, estimator, starts, QC, or allocation requires
a new calibration namespace and a complete C1 rerun.
