# Experiment 004: Instrument-Response and Forward-Model Sensitivity

**Status:** Blocked by Experiment 003

**Planned branch:** `experiment/004-instrument-response-sensitivity`

**Depends on:** [Experiment 003](../experiment-003/README.md)

## Scientific Question

How much do inferred voltage and identifiability depend on fixed smoothing,
beam-shape, angular-response, and energy-response assumptions in the loss-cone
forward model?

## Planned Factors

- Hard boundary versus multiple sigmoid `tau` values.
- Response-convolved boundary where instrument information permits.
- Fixed 15 eV beam width versus fractional energy-response width.
- Multiple beam pitch widths or angular-response convolution.
- Beam-free, fixed-amplitude, and fitted-amplitude models.
- Incident-mean versus pairwise reflected/incident normalization.
- Current data-dependent mask versus predeclared geometry-based alternatives.
- Nominal versus perturbed spacecraft potential.

## Physical Cross-Checks

- Fit loss-cone boundary and upward-beam centroid separately, then compare their
  inferred voltage drops.
- Validate magnetic polarity and look-direction convention on spectrum 653 and
  additional known upward-beam cases.
- Compare fitted `B_S/B_M` with independent magnetic-geometry expectations where
  feasible.
- Record field-direction variability and avoid interpreting straight-field
  projections near crustal anomalies without a separate geometry check.

## Experimental Design

Use the observation/uncertainty model selected by Experiment 003. Freeze a
factorial or sensitivity design before fitting. Include the Halekas benchmark,
the Experiment 003 confirmation panel, and deliberately difficult spectra.

Report systematic voltage shifts, not only changes in fit objective. Separate
measurement uncertainty from forward-model sensitivity.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Protocol, results, and decision |
| `sensitivity_manifest.yaml` | Frozen factor levels and combinations |
| `run.py` | Reproducible sensitivity study |
| `results.csv` | Per-spectrum parameter and diagnostic shifts |
| `beam_crosscheck.csv` | Independent beam/loss-cone voltage comparison |
| `sensitivity.png` | Primary systematic-sensitivity figure |

## Decision Gate

Freeze production forward-model settings only after:

- Benchmark recovery is stable across defensible response assumptions.
- Dominant systematic shifts are quantified.
- Beam and boundary estimates are physically consistent on positive controls.
- Polarity behavior is validated end to end.
- Fixed parameters are justified or their uncertainty is propagated.

## Non-Goals

- Optimizing settings to reproduce one published voltage exactly.
- Crustal-anomaly science.
- Full archive recomputation.
- Treating a smoother model as more identifiable without mismatch tests.
