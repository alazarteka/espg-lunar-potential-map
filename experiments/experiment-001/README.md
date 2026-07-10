# Step 1 Findings: Halekas Benchmark and Cash Failure

**Date:** 2026-07-10

**Repository state:** `d039fcc` on `losscone-weighted-width`

**Status:** Initial Step 1 diagnosis complete; Cash stop/go gate failed

## Objective

Step 1 asks two questions:

1. Can the current pipeline reproduce the published Halekas et al. 2008
   benchmark voltage and magnetic ratio on the correct Lunar Prospector sweep?
2. Why does the new Cash fit reject representative real spectra and move the
   fitted voltage away from the published solution?

The controlled experiment is implemented in [`run.py`](run.py). The residual
image is [`residuals.png`](residuals.png).

## Published Target and Exact Sweep

Halekas et al. identify their Figure 5 normalized distribution as measured at
approximately 14:45 UT on April 29, 1999. They report approximately:

- Lunar surface potential relative to ambient plasma: `U_M = -160 V`.
- Spacecraft potential relative to ambient plasma: `U_sc = +11 V`.
- Surface-to-spacecraft voltage: `U_rel = U_M - U_sc = -171 V`.
- Magnetic ratio: `B_S/B_M = 0.975`, where `S` is the spacecraft and `M` is the
  lunar surface.

Primary source:
<https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2008JA013194>

The nearest repository sweep is `spec_no = 653` in
`data/1999/091_120APR/3D990429.TAB`:

- Row time range: 1999-04-29 14:44:08 through 14:45:18 UTC.
- Midpoint: approximately 14:44:43 UTC.
- Offset from nominal 14:45: approximately -17 seconds.
- Internal chunk index after invalid-sweep removal: 650.

Selecting by spectrum number is essential. Two earlier invalid sweeps have been
removed, so spectrum 653 is not chunk 652 or 653.

The production geometry audit finds valid surface connection for all 15 rows,
projection polarity `-1`, and finite pitch angles. The simpler `PitchAngle(er)`
path used by the scratch diagnostic happens to choose the same orientation for
this sweep.

## Potential Convention

The intended code convention is:

\[
U_{\mathrm{rel}} = U_M - U_{sc}.
\]

`U_surface` is intended to represent `U_M` when `U_sc` is supplied. Setting
`U_sc = 0` makes the fitted value numerically equal to an effective
surface-to-spacecraft drop, but does not reproduce the published decomposition.

The published target is therefore `U_M = -160 V` and `U_rel = -171 V`, not a
relative voltage of -160 V.

## Controlled Benchmark Results

All fits below use spectrum 653, `U_sc = +11 V`, the current smooth boundary,
the same pitch geometry, the same 0.07-0.79 mask, deterministic LHS/DE settings,
and current parameter bounds.

| Method | `U_M` [V] | `U_rel` [V] | `B_S/B_M` | Beam amp | Objective |
| --- | ---: | ---: | ---: | ---: | ---: |
| Published | -160 | -171 | 0.975 | -- | -- |
| Smooth Lillis, incident mean | -150.93 | -161.93 | 0.9892 | 3.635 | 0.1209 |
| Smooth Lillis, pairwise ratio | -153.36 | -164.36 | 0.9622 | 0.621 | 0.1264 |
| Production Cash, diagnostic cutoff bypass | -241.74 | -252.74 | 1.0420 | 1.588 | 1420.7 |

The current smooth Lillis model reproduces the published voltage scale to about
7-9 V in relative voltage and brackets the published magnetic ratio across the
two normalization choices. This is good agreement given the approximate
published timestamp, broad energy/angular response, and implementation
differences, but it should not be called exact reproduction.

The pairwise normalization is closer to the reflected/incident construction
described by Halekas. The incident-mean normalization gives the closer relative
voltage, while pairwise normalization gives the closer magnetic ratio. This
normalization sensitivity should remain part of the benchmark uncertainty.

The current Lillis model contains a 5-degree sigmoid and therefore differs from
the hard-boundary model used at submission.

Under normal production behavior, Cash rejects this sweep because reduced
deviance exceeds the configured threshold of 3. The reported Cash parameters
above are diagnostic optimizer outputs, not accepted science results.

## Timestamp Sensitivity

The adjacent sweeps differ substantially:

| Spectrum | UTC range | Smooth Lillis `U_rel` [V] | Production Cash `U_rel` [V] | Cash reduced deviance |
| --- | --- | ---: | ---: | ---: |
| 652 | 14:42:48-14:43:58 | -88.05 | -124.90 | 1590.8 |
| **653** | **14:44:08-14:45:18** | **-161.93** | **-252.74** | **1420.7** |
| 654 | 14:45:28-14:46:38 | -146.60 | -213.50 | 1745.9 |

This confirms that spectrum 653 is the correct primary benchmark and that a
rounded paper timestamp cannot be mapped to an arbitrary neighboring sweep.
Cash is rejected on all three and shifts every estimate more negative than
Lillis.

## Cash Ablations on Spectrum 653

The following variants change one observation-model assumption at a time and
refit all three physical parameters.

| Cash variant | `U_rel` [V] | `B_S/B_M` | Reduced objective | Objective at Lillis solution |
| --- | ---: | ---: | ---: | ---: |
| Production | -252.74 | 1.042 | 1420.7 | 1591.2 |
| No additive background | -252.46 | 1.041 | 1424.8 | 1597.5 |
| Background-subtracted signal scale | -251.60 | 1.041 | 1419.6 | 1590.6 |
| Near-zero normalized floor | -221.60 | 0.743 | 1552.4 | 2054.3 |
| Near-zero floor and no additive background | -192.58 | 0.621 | 1587.0 | 2222.7 |
| One profiled intensity per energy row | -256.71 | 0.971 | 1351.3 | 1565.9 |
| Reflected side of current mask only | -251.88 | 1.041 | 2253.1 | 2636.4 |
| Pairwise normalization | -249.42 | 1.035 | 1151.3 | 1301.9 |
| All finite positive bins | -207.15 | 0.981 | 3469.3 | 4100.8 |

### What These Ablations Rule Out

**Additive background is not the dominant cause.** Removing it or correcting
the incident count scale changes reduced deviance by less than one percent and
leaves the voltage essentially unchanged. The current background algebra still
needs correction before formal inference, but it does not explain this failure.

**The fixed 0.05 floor is not the dominant cause.** Removing the floor worsens
the objective and drives magnetic ratio and beam parameters into different
compromises.

**Plug-in row normalization is not the dominant cause.** Profiling one intensity
per energy row improves the objective by only about five percent relative to
the production Cash optimum. The fit remains catastrophically inconsistent
with a Poisson model.

**Incident-side selection is not the sole cause.** Restricting to reflected bins
makes reduced deviance worse. In the production fit, approximately 71% of raw
deviance already arises on the reflected side.

**This is not one bad channel.** The ten largest per-bin contributions account
for about 22% of production deviance. Removing a handful of outliers cannot make
the model statistically adequate.

**Profile-grid resolution is not causal for rejection.** The profile is only
computed after the point fit passes the reduced-deviance gate. It remains an
important later problem, but it cannot explain the current point-fit failure.

## Residual and Count Diagnostics

For the production Cash fit on spectrum 653:

- Valid masked bins: 776.
- Incident-side fraction of mask: 55.3%.
- Median reconstructed count-like value: approximately 6,044 per bin.
- 90th percentile: approximately 21,291 per bin.
- Maximum: approximately 96,933 per bin.
- Median absolute Pearson residual: 13.4 nominal standard deviations.
- Reflected-side fraction of total deviance: 70.6%.
- Ten largest bin contributions: 22.3% of deviance.

The largest single contribution is at:

- Energy: 12.548 eV.
- Pitch angle: 131.54 degrees.
- Reconstructed observed value: approximately 85,483 counts.
- Cash expected value: approximately 41,081 counts.
- Incident-mean-normalized observation: 2.08.

The mask is based on flux relative to the incident-side maximum, whereas
`norm2d` is divided by the incident-side mean. It can therefore admit bins with
normalized observations substantially above one even when they satisfy the
upper mask limit of 0.79 relative to the incident maximum.

The fitted Cash beam is centered near the inferred voltage drop, about 253 eV,
so it cannot explain a reflected enhancement at 12.5 eV. The residual is not
simply a slightly misplaced loss-cone boundary.

With pairwise normalization, the median absolute Pearson residual falls near
0.12 because incident bins are set to one by construction. Nevertheless, more
than 99.9% of the remaining deviance lies on the reflected side and the total
reduced deviance remains about 1,151. This isolates the principal mismatch to
the reflected distribution rather than the incident normalization.

The residual image shows broad angular and energy-dependent structure absent
from the three-parameter model. Large positive and negative residuals occur
throughout the masked region, particularly at low energy and on the reflected
side. This visual pattern is incompatible with independent Poisson fluctuation
around the current forward model.

## Why Cash Moves the Voltage

At the smooth Lillis solution, the production Cash reduced objective is about
1,591. Cash shifts relative voltage from approximately -162 V to -253 V and
changes the nuisance parameters, but only lowers the objective to about 1,421,
an improvement of roughly 11%.

The large residual field remains. Cash is therefore not finding a statistically
adequate alternative physical solution. It is reweighting model discrepancy by
the reconstructed count scale and moving voltage to reduce the most expensive
high-count residuals.

This is precisely the failure mode expected when an approximate forward model
is assigned raw-count precision: the optimizer sacrifices agreement with the
known physical benchmark to fit structure the model cannot represent.

## Effective Information Scale

For a common exposure multiplier `q`, Poisson deviance scales exactly:

\[
D(qC,q\mu) = qD(C,\mu).
\]

Reducing the benchmark's production value from 1420.7 to roughly one would
require `q` near `7.0e-4`. This would reduce the effective median from about
6,044 reconstructed counts to about 4.3 independent counts per bin.

This does **not** prove that the dwell time or geometric factor is wrong by a
factor of 1,400. It shows that the combination of telemetry processing,
calibration, induced normalization covariance, and forward-model discrepancy
contains vastly less independent information than the current pure-Poisson
likelihood asserts. A modest dwell-time correction cannot explain the failure.

The LP ER telemetry is log-compressed to eight bits and may be pre-scaled before
compression. The calibrated archive also applies dead-time and calibration
processing. Reconstructed floating values are therefore not automatically raw,
independent Poisson observations.

Primary telemetry source:
<https://pds-ppi.igpp.ucla.edu/volume/LP_1201/DOCUMENT/PROJECT/LVL0SIS.PDF>

PDS calibrated collection:
<https://pds-ppi.igpp.ucla.edu/data/lp-er-calibrated/data-3deleflux/>

## Step 1 Decision

### Benchmark

**Pass with qualification.** Current smooth Lillis reproduces the published
benchmark voltage scale and magnetic ratio reasonably well on the correct
sweep. Pairwise versus incident-mean normalization contributes a measurable but
modest systematic difference.

### Current Cash Method

**Fail.** The present Cash objective is not a scientifically usable replacement
for the Lillis baseline. It rejects the benchmark, shifts the voltage about
90 V more negative, and remains orders of magnitude outside nominal Poisson
adequacy after background, floor, mask-side, and row-intensity ablations.

### Dominant Diagnosis

The dominant failure is broad forward-model discrepancy evaluated at an
unsupported Poisson information scale. Background algebra, data-dependent
masking, normalization covariance, response convolution, telemetry compression,
and profile numerics remain real issues, but none of the tested scalar fixes
rescues the likelihood.

## Next Experiment

Do not tune the reduced-deviance cutoff and do not run Cash over the archive.
The next controlled experiment should have two parts.

### A. Verify the Observation Product

For spectrum 653 and a small validation panel:

1. Trace PDS calibrated flux back through dead-time correction, decompression,
   pre-scaling, geometric factor, and channel exposure.
2. Determine whether original integer count intervals or compression bins can
   be recovered.
3. Quantify digitization, dead-time, background, calibration, and geometric-
   factor uncertainty separately from counting noise.
4. Verify whether `2.5 s / N_azimuth` is the correct exposure for every archived
   angular bin and energy acquisition mode.

### B. Compare Two Model-Discrepancy Strategies

Without changing production defaults, compare:

1. A conditional shape likelihood with one profiled intensity per energy row,
   explicit background, and an empirically estimated multiplicative
   overdispersion or fractional-error term.
2. The smooth Lillis/ratio baseline with uncertainty calibrated by realistic
   simulation and orbit/day block resampling rather than nominal Wilks theory.

Evaluate both on spectrum 653, spectra 652 and 654, and a discovery panel
spanning real-data residual quality. The winning approach must preserve the
published benchmark, yield interpretable residuals, and avoid zero-width or
boundary-driven confidence claims before it advances to the one-day pilot.

## Reproduction Commands

```bash
.venv/bin/python experiments/experiment-001/run.py \
  --spec-no 653 --u-spacecraft 11 --brief \
  --plot experiments/experiment-001/residuals.png

.venv/bin/python experiments/experiment-001/run.py \
  --spec-no 652 --u-spacecraft 11 --brief

.venv/bin/python experiments/experiment-001/run.py \
  --spec-no 654 --u-spacecraft 11 --brief
```

For full JSON output, omit `--brief`.

## Limitations of This Initial Diagnostic

- The published timestamp is approximate and the original authors may have
  applied processing choices not fully documented in the paper.
- Current Lillis uses a smooth boundary and is not byte-for-byte submission-era
  Lillis.
- The scratch experiment does not yet reconstruct original telemetry counts.
- The model-response, energy-smearing, angular-smearing, sheath-distortion, and
  sequential-acquisition effects have not yet been added.
- The benchmark is a positive control, not a representative mission sample.
- The profiled-row ablation used zero additive background to preserve an
  analytic amplitude solution; a full conditional model with nonzero background
  requires a convex one-dimensional solve per row.
