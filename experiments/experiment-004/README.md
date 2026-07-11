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
| `beam_boundary_crosscheck.csv` | Independent observable estimates |
| `supplement/main.tex` | Mathematical methods, results, and limitations |

## Non-Goals

- Mission-wide recomputation.
- Choosing a production observation model in Phase A.
- Treating exact reproduction of one scalar voltage as validation.
- Recovering undocumented private IDL implementation choices by tuning.
- Crustal-anomaly or plasma-regime population science.
