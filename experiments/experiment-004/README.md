# Experiment 004: Clean-Room Halekas Reconstruction and Forward-Model Validation

**Status:** Running; Phase A protocol frozen

**Branch:** `experiments`

**Depends on:** [Experiment 003](../experiment-003/README.md)

## Scientific Question

Can a best-supported clean-room reconstruction of the Halekas et al. (2008)
lunar-potential estimator reproduce the published spectrum-653 distribution,
voltage, and magnetic ratio, and which historical ambiguities or modern
response corrections materially control that result?

## Provenance

- Frozen clean-room reconstruction:
  [`provenance/halekas-cleanroom-reconstruction.md`](provenance/halekas-cleanroom-reconstruction.md)
- Snapshot record and checksum: [`provenance/README.md`](provenance/README.md)
- Pre-result scientific interpretation:
  [`reconstruction-implications.md`](reconstruction-implications.md)

These files were frozen before Experiment 004 fitting results were inspected.

## Frozen Positive Control

- Product: `data/1999/091_120APR/3D990429.TAB`
- Spectrum: `spec_no = 653`, internal chunk 650
- UTC rows: 1999-04-29 14:44:08 through 14:45:18
- Published `U_sc`: `+11 V`
- Published `U_M`: `-160 V`
- Published `U_rel = U_M - U_sc`: `-171 V`
- Published `B_S/B_M`: `0.975`

The benchmark voltage gate remains `|U_rel + 171 V| <= 25 V`, but voltage alone
is insufficient for acceptance.

The published estimate is a historical reproducibility target, not physical
ground truth. A stable disagreement admits at least three live explanations:

1. The public reconstruction is missing a consequential implementation detail.
2. The private processing used a different normalization, response, weighting,
   selection, or hand-QC decision.
3. The published point estimate is itself biased or wrong under the physical
   and instrumental conditions of spectrum 653.

Experiment 004 must not tune factors to force the first two explanations or
exclude the third.

## Ordered Experimental Phases

### Phase A: Historical Reconstruction

Build the closest public-evidence implementation before introducing modern
improvements:

- Rebin the 88 sectors into 16 fixed 11.25-degree pitch bins.
- Use all finite positive normalized bins rather than the Lillis mask.
- Compare conjugate-pitch and incident-hemisphere normalization as the remaining
  documented ambiguity.
- Use a hard loss-cone boundary and finite interior floors `0.05` and `0.10`.
- Use a response-limited beam centered at `U_sc - U_M`, approximately one
  effective energy channel wide and confined to the two most field-aligned
  upward pitch bins.
- Compare beam amplitude fixed from an independently detected measured excess
  and profiled/co-fitted on a coarse nuisance grid.
- Search a deterministic two-dimensional `(U_M, B_S/B_M)` grid. Beam-amplitude
  variants are nuisance branches, not an excuse to tune the voltage.
- Evaluate unweighted squared log residual over the complete normalized map.

### Phase B: Observable Decomposition

- Estimate beam-centroid voltage independently after estimating the local
  continuum or subtracting the loss-cone component.
- Estimate voltage from the energy dependence of the loss-cone boundary without
  the beam region.
- Require agreement within the effective energy-bin width on positive controls.
- Diagnose polarity, pitch coverage, low-energy contamination, and field-ratio
  degeneracy separately from optimizer behavior.

### Phase C: Historical Ambiguity Sensitivity

Quantify the effect of choices that remain publicly undocumented:

- Conjugate versus incident-mean normalization.
- Floor `0.05` versus `0.10`.
- Beam amplitude handling.
- Exact pitch-bin aggregation and empty-bin handling.
- Grid spacing and bounds.

### Phase D: Modern Extensions

Only after Phases A-C are interpreted:

- Hard boundary versus instrument-response convolution and physically motivated
  broadening.
- Historical log objective versus conditional-shape inference.
- Raw interval-count inference with response and model discrepancy.
- Per-spectrum spacecraft potential and its uncertainty.
- Geometry-based masks and a preregistered quality-control funnel.

The different objectives may use different defensible data domains. They will
not all be forced through the Lillis mask.

## Phase A Decision Metrics

- Error in `U_rel` and `B_S/B_M` relative to the published benchmark.
- Beam energy-bin and pitch-bin agreement with the measured distribution.
- Loss-cone boundary agreement by energy.
- Signed residual structure over the complete map.
- Parameter-rail and grid-edge contact.
- Stability across normalization, floor, and beam-amplitude ambiguity branches.
- Agreement of independent beam and loss-cone voltage estimates.

## Phase A Decision Gate

The clean-room baseline advances only if:

- At least one evidence-supported branch passes the voltage gate without
  optimizing factor choices against the target.
- The same branch approximately reproduces the published magnetic ratio and
  observed beam/boundary morphology.
- Neighboring ambiguity branches reveal a stable scientific solution rather
  than an isolated tuned combination.
- The optimum is not created by parameter rails, empty-bin conventions, or a
  single anomalous cell.

Failure is retained as a reconstruction result. It does not authorize tuning a
modern model until the failure anatomy is recorded.

## Initial Phase A Execution

The first implementation and benchmark run were completed on 2026-07-11. It
uses ratio-of-binned-flux normalization, fixed 11.25-degree pitch bins, a hard
boundary, a one-energy-bin by two-pitch-bin beam, and the complete positive
normalized map. Eight frozen branches cover two normalizations, two floors,
and two beam-amplitude treatments.

All eight branches selected the same physical grid point:

| Quantity | Clean-room optimum | Published |
| --- | ---: | ---: |
| `U_M` | `-250 V` | `-160 V` |
| `U_rel` | `-261 V` | `-171 V` |
| `B_S/B_M` | `0.910` | `0.975` |

The lowest objective uses conjugate normalization, floor `0.10`, and a
profiled beam amplitude of `2.75`. Its reduced descriptive objective is
`0.2146`; this is not a calibrated reduced chi-square. None of the eight
branches passes the predeclared 25 V benchmark gate, and no optimum contacts a
parameter-grid edge.

At the exact published `(U_M, B_S/B_M)` pair, the best-branch objective is
`25.6%` above its minimum. After profiling only `B_S/B_M`, the `U_M = -160 V`
profile is `Delta objective = 5.33` above the minimum. These differences have
no confidence interpretation because the historical objective has no recovered
variance scale.

The measured field-aligned excess peaks in the 113 eV row and spans the 113 and
196 eV rows; the synthetic beam associated with both the published solution
and the clean-room optimum occupies the 196 eV row. The measured peak pitch bin
is centered at 163.125 degrees, consistent with a beam broader than the
two-bin synthetic component. The residual map also shows broad, graded
reflected-side structure that a binary floor/one model cannot reproduce.

### Initial Decision

Phase A has **not** passed. The result is stable across the frozen top-level
branches, but it does not yet isolate which unrecovered historical detail
causes the voltage shift. Before any modern model is selected, the next
diagnostics are:

1. Reverse the order of conjugate normalization and pitch-bin aggregation.
2. Test sector-support weighting and oversampled integration across each pitch
   bin while retaining a sharp physical boundary.
3. Fit the response-limited beam centroid independently rather than assigning
   it to one nearest energy row.
4. Extract energy-by-energy loss-cone boundaries and compare their voltage to
   the beam-only estimate.
5. Verify the complete-map objective and floor treatment against the rendered
   Figure 5 morphology.

These are reconstruction diagnostics, not post hoc tuning: each corresponds to
an ambiguity already recorded in the frozen provenance document.

## Phase A2: Representation and Response Diagnostics

**Status:** Protocol frozen before Phase A2 results

Phase A2 changes only the operations most directly implicated by the initial
failure. Its primary branches form a `2 x 2 x 2` design:

| Factor | Levels |
| --- | --- |
| Observation construction | bin flux then divide; divide sector pairs then bin ratios |
| Sharp-boundary evaluation | pitch-bin center; average the sharp sector-level model over actual bin support |
| Beam energy response | nearest energy row; Gaussian response with `FWHM/E = 0.48` |

All Phase A2 primary branches fix the cone floor at `0.10`, profile beam
amplitude over the same `[0, 8]` grid, use conjugate normalization, and retain
the Phase A `(U_M, B_S/B_M)` grid. These settings are controls selected from the
initial run, not reoptimized within Phase A2. The original Phase A best branch
is included exactly as one factorial cell.

### Independent Observable Checks

For each observation construction:

- Fit a continuous beam centroid using a nonnegative amplitude, constant local
  baseline, and `FWHM/E = 0.48`, without using the published voltage.
- Fit the sharp support-integrated loss-cone model after excluding beam-affected
  field-aligned cells identified from that response fit.
- Report both spacecraft-relative voltages and their disagreement. Neither
  estimate is forced to equal the published result or the joint-fit result.

### Phase A2 Interpretation Gate

- Agreement of beam, boundary, and joint estimates near the publication
  supports a representation-error explanation.
- A publication-like beam with a more-negative boundary implicates the
  adiabatic boundary model, its weighting, or non-adiabatic broadening.
- Agreement of independent observables away from the publication makes a
  biased published point estimate a serious hypothesis, subject to remaining
  provenance uncertainties.
- Instability across the two observation constructions means the public method
  is not reconstructed well enough to adjudicate the publication.

No Phase A2 objective difference is assigned a confidence interpretation.

### Phase A2 Results

Phase A2 was executed on 2026-07-11. Matching the synthetic support to the
sectors that contributed valid observed flux or ratios was fixed before the
final run.

The two support-integrated, fractional-Gaussian branches agree exactly with
each other despite reversing normalization and aggregation order:

| Observation construction | Boundary | Beam response | `U_rel` | `B_S/B_M` | Beam amp |
| --- | --- | --- | ---: | ---: | ---: |
| Bin flux, then divide | Sector support | Fractional Gaussian | `-161 V` | `0.880` | `8.0` |
| Divide sector pairs, then bin | Sector support | Fractional Gaussian | `-161 V` | `0.880` | `8.0` |

Both pass the 25 V historical voltage gate and differ from the published
`-171 V` by 10 V. The first is the lowest-objective Phase A2 branch, with a
descriptive objective per degree of freedom of `0.1938`. The exact published
`(U_M, B_S/B_M) = (-160 V, 0.975)` pair has an objective `15.1%` above that
branch's minimum. The voltage is reproduced; the published magnetic ratio is
not.

The factorial isolates the response effect:

- Nearest-row beam branches remain at `U_rel = -251` to `-261 V`.
- A fractional beam response moves center-evaluated fits to `-151 V` or
  `-201 V`, depending on observation order.
- Integrating the sharp boundary over matching sector support removes that
  normalization-order disagreement and yields `-161 V` in both cases.

Thus the initial `-261 V` result was primarily a discretized beam-response
artifact, with angular support integration needed for representation stability.
No sigmoid smoothing was required to recover the historical voltage scale.

### Independent Observable Results

The independent checks do not validate the joint solution:

| Observation construction | Beam-only `U_rel` | Boundary-only `U_rel` | Difference |
| --- | ---: | ---: | ---: |
| Bin flux, then divide | `-145.7 V` | `-281 V` | `135.3 V` |
| Divide sector pairs, then bin | `-145.5 V` | `-281 V` | `135.5 V` |

The beam response fit requires amplitude about `10.56`, above the joint fit's
frozen upper bound of `8.0`. Both joint historical-reproduction branches
therefore hit the beam-amplitude rail. The beam-only fit is approximately 25.3
V shallower than the publication, while the boundary-only fit is 110 V more
negative. Neither difference has a formal uncertainty interpretation yet.

### Phase A2 Decision

Phase A2 establishes that the published **voltage scale is reproducible** from
the archived spectrum using evidence-supported angular and energy response
treatment. It does not establish that the published voltage is physically
correct, nor that the two observables are mutually consistent.

The joint result may be a compromise between a shallow beam estimate and a
much more negative sharp-boundary estimate. Remaining structured residuals,
the beam-amplitude rail, and the magnetic-ratio discrepancy prevent Phase A
acceptance. The hypothesis that the published point estimate is biased or
wrong therefore remains live alongside incomplete beam and boundary models.

The next phase must remove the artificial beam-amplitude rail, model the
measured beam's broader angular support, and test boundary extraction against
beam masking, floor/refilling, and non-adiabatic broadening. Those tests must
evaluate beam and boundary consistency, not merely preserve the joint voltage.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Frozen protocol, execution record, and decision |
| `provenance/` | Immutable reconstruction snapshot and checksum |
| `reconstruction-implications.md` | Pre-result scientific interpretation |
| `run.py` | Experiment-local historical reconstruction |
| `phase_a_results.csv` | Per-branch fit and morphology diagnostics |
| `phase_a_profiles.csv` | Field-ratio-profiled voltage objectives |
| `phase_a_comparison.png` | Measured/model/residual benchmark comparison |
| `phase_a2.py` | Frozen representation/response diagnostic |
| `phase_a2_results.csv` | Phase A2 factorial fits |
| `phase_a2_profiles.csv` | Phase A2 voltage profiles |
| `beam_boundary_crosscheck.csv` | Independent beam and boundary estimates |
| `phase_a2_comparison.png` | Best Phase A2 measured/model/residual comparison |
| `beam_boundary_crosscheck.csv` | Independent observable estimates |
| `supplement/main.tex` | Mathematical methods, results, and limitations |

## Non-Goals

- Mission-wide recomputation.
- Choosing a production observation model in Phase A.
- Treating exact reproduction of one scalar voltage as validation.
- Recovering undocumented private IDL implementation choices by tuning.
- Crustal-anomaly or plasma-regime population science.
