# Calibration Attempt 03 Result: Partial Held-Out Success, Full Gate Failure

## Decision

**Attempt 03 fails the frozen full-domain gate. The 398-sweep pilot was not
run.**

The smooth-continuum estimator was executed at commit `fb044eb` under the
pre-result protocol committed as `cefe44c`. The frozen run completed 1,600
beam-present simulations and 500 null controls in 6,386.6 seconds using eight
workers. Empirical bias and interval scales came only from the calibration
split; the reported gate uses 800 untouched beam simulations and 250 untouched
null controls.

## Aggregate Held-Out Result

- Validation beams attempted: **800**.
- Validation beams surviving base and interval QC: **387** (`48.4%`).
- Minimum retained-count gate: **pass**.
- Validation nulls: **250**.
- False accepted nulls: **15** (`6.0%`).
- Frozen false-detection maximum: `5%`.
- Coverage-eligible cells: **6**.
- Eligible cells passing every gate: **4**.

The full calibration decision is therefore **fail**.

## Cell Results

| Center | Amp | Sigma | Retained | Median corrected error / FWHM | 68% coverage | 95% coverage | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 65 eV | 4 | 20 deg | 48/100 | 0.046 | 0.854 | 0.938 | Fail: 68% overcoverage |
| 65 eV | 1 | 20 deg | 34/100 | 0.202 | 0.088 | 0.706 | Fail: bias and undercoverage |
| 150 eV | 4 | 25 deg | 90/100 | 0.022 | 0.644 | 0.967 | **Pass** |
| 150 eV | 12 | 25 deg | 99/100 | 0.019 | 0.687 | 0.970 | **Pass** |
| 350 eV | 4 | 35 deg | 62/100 | 0.056 | 0.726 | 0.952 | **Pass** |
| 350 eV | 12 | 15 deg | 50/100 | 0.062 | 0.760 | 0.960 | **Pass** |
| 800 eV | 4 | 25 deg | 0/100 | -- | -- | -- | Below identifiable regime |
| 800 eV | 12 | 40 deg | 4/100 | 0.355 | 0.000 | 0.250 | Below count; severe failure |

Attempt 03 fixes the central Attempt 01 failure at 150 eV. Both 150 eV cells
now have small corrected error and calibrated held-out coverage. Both 350 eV
cells also pass. The improvement generalizes across the background discrepancy
mixture.

## Background-Family Check

Pooling retained validation cells by background family gives:

| Background | Retained | 68% coverage | 95% coverage | Median error / FWHM |
| --- | ---: | ---: | ---: | ---: |
| Single logistic | 190 | 0.674 | 0.942 | 0.034 |
| Double transition | 96 | 0.646 | 0.906 | 0.030 |
| Asymmetric shoulder | 101 | 0.624 | 0.931 | 0.039 |

Thus the held-out failure is not driven solely by the deliberately misspecified
backgrounds. The smooth continuum successfully propagates moderate continuum
discrepancy in the retained middle-energy regime.

## Remaining Failure Anatomy

The dominant nonexclusive base-QC failures among 800 validation beams are:

- amplitude not resolved: `165`;
- centroid scale unresolved: `104`;
- leave-one-energy-row-out instability: `88`;
- multistart instability: `77`.

The 800 eV empirical 95% multiplier is `134.55`, compared with `2.00` at 150 eV
and `3.26` at 350 eV. The native 588 and 1,020 eV rows do not provide a stable,
resolved centroid for the frozen 800 eV conditions after nuisance-background
uncertainty is propagated.

Of 15 held-out false detections, most are low-amplitude solutions near 50 eV.
Four lie between approximately 137 and 277 eV. They are spread across all four
supports and all three background families, so no single support or discrepancy
branch explains them.

## Scientific Meaning

The smooth nuisance continuum is a substantive success: it converts the 150--350
eV regime from severe undercoverage to held-out calibrated inference, including
under continuum misspecification. Beam voltage is therefore not generically
unrecoverable.

The result is nevertheless domain-limited. Weak low-energy beams are biased and
800 eV beams are effectively unidentified on this support. The full search also
admits a 6% null false-detection rate, narrowly but genuinely above the frozen
limit.

It would be post hoc to rescue Attempt 03 by restricting its energy domain after
seeing validation. The next defensible hypothesis is a separately preregistered
`100-500 eV` observable window tested with new independent random seeds and
energy anchors. That window contains the Halekas-scale beam while excluding the
two regimes that failed here. Until such a test passes, the real 398-sweep pilot
remains closed.
