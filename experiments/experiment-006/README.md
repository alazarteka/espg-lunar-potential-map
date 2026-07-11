# Experiment 006: Restricted-Domain Beam-Proxy Validation

**Status:** Planned; design recorded before implementation or simulation

**Branch:** `experiments`

**Parent commit:** `41632376f0928249800dfa157da8ed82f1cfad60`

**Depends on:** [Experiment 004](../experiment-004/README.md) and
[Experiment 005](../experiment-005/README.md)

## Scientific Question

For surface-connected Lunar Prospector Electron Reflectometer sweeps with
adequate directional support, can the frozen smooth-continuum procedure identify
a usable subset of single, upward, secondary-like beams whose pre-response
centroids lie in a `100-500 eV` reporting window, with calibrated conditional
uncertainty and controlled false acceptance?

The primary measured quantity is the beam centroid `E_beam`. Its physical
interpretation is

\[
E_{beam} \simeq E_{emit} - U_{rel} + \delta_{transport},
\qquad U_{rel}=U_M-U_{sc},
\]

when energies are in electron-volts and potentials in volts. Therefore
`-E_beam` is a zero-emission-energy spacecraft-relative proxy, not an exact
absolute lunar potential. Experiment 006 does not validate `U_sc`, `U_M`, the
loss-cone boundary, or a global map.

## Why This Is A New Experiment

Experiment 005 discovered, after opening held-out results, that its 150 and
350 eV cells passed while 65 and 800 eV cells failed. The resulting `100-500
eV` window is therefore an outcome-derived hypothesis. Experiment 005 cannot
confirm it by retrospective filtering.

Experiment 006 starts over with:

- no reused Experiment 005 simulations, correction factors, seeds, or
  confirmatory support spectra;
- continuously sampled and new fixed energy locations, including edge tests;
- beam difficulty balanced independently of energy;
- a continuum generator whose parameters do not encode the injected beam
  energy in the primary arm;
- paired normalizations derived from one latent sector-level observation;
- surface-connection polarity, with 180 degrees explicitly oriented as
  surface-leaving;
- response, morphology, temporal-confounding, and out-of-window challenges;
- powered null-family gates based on binomial upper confidence bounds; and
- retention gates, so difficult cells cannot disappear from the decision.

The full preregistration is in [protocol.md](protocol.md). The machine-readable
design is in [configuration.json](configuration.json), and the scientific
rationale is developed in [supplement/main.tex](supplement/main.tex).

## Stages And Information Barriers

1. **D0: Support and geometry freeze.** Divide the interval into six fixed time
   strata and select four maximin-diverse supports per stratum using only
   structural completeness, connection, footprint geometry, pitch support, and
   magnetic provenance. Assign two per stratum to each split, giving 12
   calibration and 12 disjoint confirmation supports before fitting any beam
   simulation.
2. **D1: Implementation verification.** Verify polarity, paired normalization,
   response integration, simulator independence, deterministic seeds, and
   noiseless exact recovery. These are code gates, not scientific results.
3. **C1: Fresh calibration.** Learn a new bias and interval correction using
   only calibration supports and calibration seeds. No Experiment 005
   correction is imported.
4. **Freeze:** Commit code, support manifest, correction map, thresholds, seed
   namespaces, checksums, and an executable decision report before confirmation.
5. **V1: One-shot confirmation.** Run all primary beams, weak challenges,
   semi-synthetic cases, nulls, and classification controls without inspecting
   partial summaries.
6. **P1: Gated real audit.** Open the frozen 398-sweep audit only if every
   primary confirmation gate passes. A pass opens an observable/residual audit,
   not production mapping.

Any method, threshold, window, support, or correction change after V1 begins
invalidates V1. A revised method requires unused random draws and a new numbered
experiment.

## Primary Confirmatory Population

The primary estimand is selective: performance among cases passing all frozen
quality and interval-resolution rules. Retention is a co-primary property.

- True center: `100-500 eV`.
- Normalized beam amplitude: `[4,16]`.
- Beam pitch sigma: `12-42 deg`.
- Frozen continuum, noise, response, morphology, and LP-support envelope.
- Six center strata crossed with two amplitude strata.
- `400` cases per cell, or `4,800` primary confirmation beams.

An additional `600` weak-beam cases with amplitude `[1,4)` estimate the
detection boundary but cannot rescue or fail the primary amplitude claim.

## Controls And Challenges

- `3,000` beam-absent controls: 1,000 each from matched-smooth,
  misspecified-continuum, and adversarial non-beam families.
- `1,000` out-of-window beams split below 100 eV and above 500 eV, testing
  leakage into the reporting window.
- `1,200` semi-synthetic injections over held-out real residual templates.
- Explicit single-row spikes, energy-only bumps, pitch-only enhancements,
  incident-side features, axis offsets, double peaks, temporal ramps/pulses,
  missing channels, and normalization disagreement.

The optimizer retains a broad search domain. The `100-500 eV` interval is a
scientific reporting rule, not an optimizer boundary, and confidence intervals
are never clipped to it.

## Decision In Brief

A full pass requires every primary cell to meet retention, point-error,
tail-error, signed-bias, interval-width, and coverage gates; every background
family to preserve pooled calibration; all three null families to have a
Bonferroni-adjusted one-sided 95% false-acceptance upper bound no greater than
5%; and the semi-synthetic and out-of-window challenges to pass their frozen
gates. For 1,000 nulls per family, the frozen null rule allows at most 35 false
in-window acceptances per family.

Insufficient retention is failure, not ineligibility. A core-only or edge-only
success does not validate the full window. Agreement with the Halekas example
is a known positive-control observation and is not confirmatory evidence.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Experiment status, rationale, and decision summary |
| `protocol.md` | Full preregistration and claim matrix |
| `configuration.json` | Machine-readable design and gates |
| `design-review.md` | Independent statistical and physical design critique |
| `support_manifest.csv` | Frozen calibration/confirmation support assignment |
| `support_candidates.csv` | Outcome-blind metrics and exclusions for all 398 specs |
| `support_provenance.json` | Input, kernel, code, and output checksums |
| `support_selection.py` | Outcome-blind support selection |
| `beam.py` | Polarity-aware frozen estimator |
| `simulate.py` | Independent sector-level generators and controls |
| `calibrate.py` | Fresh calibration only |
| `confirm.py` | One-shot confirmatory runner and decision report |
| `supplement/main.tex` | Mini-paper updated from expectation to result |

No calibration, confirmation, or real-pilot output exists at this design
commit.
