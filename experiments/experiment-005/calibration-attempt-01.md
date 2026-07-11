# Calibration Attempt 01: Constant-Continuum Beam Fit

## Decision

**Failed the frozen calibration gate. The 398-sweep pilot was not run.**

This attempt evaluates the committed `3112d02` implementation and the
pre-result protocol in `435a013`. It uses 1,600 beam-present simulations, 500
beam-absent controls, 200 residual-bootstrap replicates per case, and the four
real sweep supports preregistered in `configuration.json`.

## What Was Tested

Each synthetic distribution contains:

- the actual energy and native-sector pitch support of spectrum 572, 654, 772,
  or 969;
- a broadened, partially refilled loss-cone background with randomized field
  ratio, cone floor, and transition width;
- a response-convolved field-aligned beam for beam-present cases;
- multiplicative noise at the frozen level for that calibration cell.

The estimator fits one constant continuum per energy row plus a shared beam
energy, amplitude, and angular width. It then applies deterministic multistart,
within-energy residual bootstrap, normalization agreement, and leave-one-energy-
row-out QC. The null controls contain the same randomized boundary backgrounds
and noise but zero beam amplitude.

## Aggregate Result

- Beam-present attempts: **1,600**.
- QC-retained beam estimates: **898** (`56.1%`).
- Beam-absent controls: **500**.
- False accepted beams: **0** (`0.0%`).
- Calibration cells eligible for coverage gating: **5**.
- Eligible cells passing every gate: **1**.

The minimum retained-simulation and false-detection gates pass. The point and
interval calibration gates do not.

## Cell Results

| Center | Amp | Pitch sigma | Retained | Median abs. error / FWHM | 68% coverage | 95% coverage | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 65 eV | 4 | 20 deg | 150/200 | 0.065 | 0.893 | 0.960 | Fail: 68% overcoverage |
| 65 eV | 1 | 20 deg | 2/200 | 0.173 | 0.500 | 1.000 | Below detection regime |
| 150 eV | 4 | 25 deg | 190/200 | 0.105 | 0.074 | 0.221 | Fail: bias and undercoverage |
| 150 eV | 12 | 25 deg | 200/200 | 0.038 | 0.315 | 0.535 | Fail: undercoverage |
| 350 eV | 4 | 35 deg | 139/200 | 0.062 | 0.849 | 0.986 | Fail: 68% overcoverage |
| 350 eV | 12 | 15 deg | 123/200 | 0.085 | 0.675 | 0.919 | **Pass** |
| 800 eV | 4 | 25 deg | 49/200 | 0.082 | 0.224 | 0.388 | Below coverage count; undercoverage |
| 800 eV | 12 | 40 deg | 45/200 | 0.046 | 0.400 | 0.667 | Below coverage count; undercoverage |

The frozen acceptable ranges are `0.60-0.76` for 68% coverage and `0.88-0.99`
for 95% coverage. The median absolute point-error gate is 0.10 response FWHM.

## Failure Anatomy

The 150 eV moderate-beam cell is the clearest failure. Accepted estimates have
a median signed error of `-7.54 eV`, while their median 68% bootstrap width is
only `5.91 eV` and median 95% width is `11.99 eV`. The bootstrap quantifies
scatter around a biased pseudo-solution and therefore misses the truth in most
replicates. The stronger 150 eV beam reduces median absolute error to `2.71 eV`
but still undercovers severely.

At 65 eV and 350 eV, which lie close to native ER energy rows, the same bootstrap
is conservative at the 68% level. At 800 eV, between the 588 and 1,020 eV rows,
retention is low and intervals undercover. Accepted 800 eV moderate beams come
only from spectrum-654 support and have a median signed error of `-30.7 eV`.

The dominant nonexclusive rejection reasons are:

| Reason | Count | Fraction of 1,600 |
| --- | ---: | ---: |
| Amplitude not resolved | 498 | 31.1% |
| Centroid uncertainty unresolved | 413 | 25.8% |
| Leave-one-energy-row-out instability | 316 | 19.8% |
| Insufficient positive-excess rows | 220 | 13.8% |
| Multistart instability | 128 | 8.0% |
| Pitch-width bound margin | 125 | 7.8% |

## Scientific Interpretation

Beam detection itself is promising: the benchmark fits are stable, more than
half of injected beams survive strict QC, and no randomized loss-cone-only
control produces a false accepted beam. The current uncertainty statement is
not defensible.

The failure arises because one constant continuum per energy row does not
represent the pitch-dependent loss-cone background. That structured background
shifts the beam centroid according to its phase relative to the coarse ER energy
grid and its angular overlap with the beam. Resampling residuals within an energy
row removes rather than propagates that structured discrepancy. It can therefore
produce a precise interval around the wrong effective center.

This is closely analogous to the Experiment 003 result: matched or locally
resampled uncertainty can look precise even when the real forward-model
discrepancy controls the estimate.

## Consequence

Running the real 398-sweep selection funnel now would produce an attractive but
scientifically uncalibrated set of voltages. Per the frozen decision gate, no
pilot-wide acceptance rate or voltage distribution is generated.

A second attempt must change the continuum/discrepancy treatment, not relax QC.
The leading candidate is a beam fit conditional on a flexible per-energy
loss-cone transition, with uncertainty calibrated on an independent simulation
split. Any such attempt requires a new recorded protocol version and a complete
rerun of all beam-present and beam-absent controls before real pilot results are
opened.
