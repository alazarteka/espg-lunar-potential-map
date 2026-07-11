# D0 Result: Polarity-Aware Support Freeze

## Decision

**D0 passes.** The calibration and confirmation support sets are frozen before
any Experiment 006 normalization, simulator output, beam fit, or confirmation
seed has been generated.

The selector ran at commit `78f84f3` after the attitude interpolation correction
in commit `ac1f03c`. The SPICE kernel lock passed complete SHA-1 verification.
The candidate table, selected manifest, selector, configuration, source data,
attitude table, theta table, and every loaded kernel have recorded SHA-256
digests in [`support_provenance.json`](support_provenance.json).

## Parent Geometry Correction

The pre-D0 audit found that `get_current_ra_dec_batch()` selected the next
attitude record even though the coordinate-frame documentation specified
interpolation. For the experiment interval, that used the 30 April noon
attitude for measurements acquired after 12:57 UT on 29 April. The source fix
now interpolates right ascension along the shortest circular arc and
declination linearly at every ER timestamp. Focused tests and the complete
repository hook suite passed before support selection.

## Candidate Audit

All `398` requested spectra were present with exactly 15 rows. The native
energy grid was identical in every spectrum; its ordered 15-row SHA-256 is
`a4d78885bfa04cb0bd6644d4624ed986ff9362f4bdea9fb198ee33b1ae70ec34`.
Within-sweep magnetic direction variation was numerically zero, no greater than
`1.3e-6 deg`. Energy alignment and within-sweep MAG variation were therefore
retained as provenance checks but excluded from the diversity score.

Of 398 candidates, `258` passed every primary-support eligibility rule. The
following exclusion reasons are nonexclusive:

| Reason | Spectra |
| --- | ---: |
| Field-line elevation below 30 deg | 135 |
| Incomplete connection or inconsistent polarity | 47 |
| Insufficient near-180-deg support | 42 |
| Insufficient usable guard-energy rows | 36 |
| Previously used or known positive control | 6 |

Every one of six fixed temporal strata retained between 39 and 49 eligible
candidates, well above the required four.

## Frozen Supports

Four maximin-diverse supports were selected in each temporal stratum and split
two/two by alternating geometry rank.

| Stratum | Calibration | Confirmation |
| ---: | --- | --- |
| 0 | 576, 596 | 619, 638 |
| 1 | 641, 684 | 651, 707 |
| 2 | 725, 768 | 716, 748 |
| 3 | 784, 799 | 819, 835 |
| 4 | 842, 864 | 856, 893 |
| 5 | 945, 964 | 909, 939 |

The complete 24-row record is [`support_manifest.csv`](support_manifest.csv).
Across selected supports:

- field-line elevation spans `31.30-83.79 deg`;
- spacecraft altitude spans `25.21-36.57 km`;
- footprint latitude spans `-58.31 to +88.95 deg`;
- polarity counts are 13 positive and 11 negative; and
- response-relevant rows with near-axis support span `2-7`.

Calibration and confirmation mean field-line elevations are 51.11 and 50.23
degrees, mean altitudes are 30.34 and 31.15 km, and mean axis-support row counts
are 4.92 and 5.08. The standardized multifeature split imbalance is
`0.49438211`; no flux-derived adjustment was made to improve it.

## Information Barrier

The selection features are limited to field-line elevation, spacecraft
altitude, footprint location, polarity, pitch support, and structural sector
support. The script does not call either normalization, the smooth continuum,
or a beam estimator. Flux magnitudes enter only through finite-positive sector
validity; no flux amplitude, shape, residual, or beam statistic is computed.

Repeated execution produced byte-identical candidate and manifest files:

- `support_candidates.csv`: `421462509b979e2d4fdfcaa4477198927f2c674fe3904f92657e2758f14d536b`
- `support_manifest.csv`: `1fe28bc3c12b7b91db110b6f6c58bb29e9ed422b882a02cad800e7112bd2bd7a`

## Next Gate

D1 may now use calibration supports for implementation tests, simulator
development, CPU/GPU parity work, and performance tuning. Confirmation support
fluxes, residual templates, seeds, and outcome summaries remain unopened until
the implementation and calibration freeze specified by the protocol.
