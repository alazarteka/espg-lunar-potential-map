# Phase 1: Primary, Window-grid, and Static Results

**Status:** Complete under analysis freeze `fff8d25`  
**Inputs:** 399,643 canonical connected sweeps from all 560 raw files

## Primary 24-hour Degree-5 Audit

All 560 daily operators are algebraically full rank (`36/36`), but all are
physically unstable under the frozen area-RMS metric and all fail 25 V
recovery. This distinction matters: algebraic rank alone would have given a
false impression of recoverability.

- weakest-mode visibility: median `4.52e-5`, range `1.09e-6` to `4.05e-4`;
- condition number: median `4.71e4`, maximum `1.93e6`;
- 25 V median relative global recovery error: range `17.0` to `6373`;
- stable windows: `0/560`;
- recovery-passing windows: `0/560`.

Each stored witness was scaled to exactly 100 V analytic surface-area RMS and
replayed directly. Its sampled RMS equals `100 * visibility` to numerical
precision. Representative sampled RMS values are `0.000109`, `0.00456`, and
`0.0405` V for the weakest, median, and strongest daily windows. The legacy
unweighted equal-angle grid assigns these same modes only about 81 V RMS,
confirming the need for the audited physical metric.

## Temporal-resolution Boundary

The frozen window grid shows a transition rather than universal failure:

- 24-hour degree 1: 77.7% stable, below the 90% class gate;
- 72-hour degree 1: 100% stable and recovery-passing;
- 168-hour degrees 1, 2, and 3: at least 97.5% stable and 98.8% recovery-passing;
- one-synodic-period windows: degrees 1 through 10 pass the 90% class gates,
  with 19 of 20 degree-5/10 windows passing and one short edge window failing.

These are independent piecewise-constant window fields, not a continuously
resolved global history. Passing classes establish conditional design
identifiability only.

## Whole-mission Static Controls

All static controls through degree 20 are strongly identifiable:

| `lmax` | Parameters | Condition | Minimum visibility | Median relative error |
|---:|---:|---:|---:|---:|
| 1 | 4 | 1.39 | 0.872 | 0.00074 |
| 2 | 9 | 1.66 | 0.837 | 0.00120 |
| 3 | 16 | 1.89 | 0.819 | 0.00165 |
| 5 | 36 | 2.27 | 0.799 | 0.00254 |
| 10 | 121 | 2.97 | 0.770 | 0.00475 |
| 20 | 441 | 3.88 | 0.726 | 0.00933 |

The geometry therefore contains abundant mission-integrated spatial
information. It does not contain enough simultaneous spatial information for
the submitted daily degree-5 estimand.

## Interim Outcome

The primary model satisfies all frozen B criteria. Multiple prospectively
declared coarse and static classes satisfy the C criteria, so the interim
classification is **Outcome C**, which has precedence over B. This remains
interim until temporal controls, injections, blocked recovery, gauge checks,
and geometry sensitivities are complete.
