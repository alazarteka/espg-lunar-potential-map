# Experiment 003: Observation and Uncertainty Model Comparison

**Status:** Blocked by Experiment 002

**Planned branch:** `experiment/003-observation-model-comparison`

**Depends on:** [Experiment 001](../experiment-001/README.md) and
[Experiment 002](../experiment-002/README.md)

## Scientific Question

Which observation and uncertainty model best preserves known loss-cone physics,
fits real LP ER spectra without false precision, and yields empirically
calibrated voltage identifiability?

## Frozen Comparison

Candidate models will be finalized from Experiment 002 before looking at the
confirmation panel. At minimum compare:

1. A conditional shape model with one profiled intensity per energy row,
   explicit background, and justified multiplicative overdispersion or
   fractional model discrepancy.
2. Smooth Lillis/ratio fitting with uncertainty calibrated by simulation and
   block resampling rather than nominal Wilks theory.
3. A count-level likelihood only if Experiment 002 supports it.

A constant quasi-Poisson scale alone is not an adequate candidate because it
widens intervals without changing the biased Cash point estimate.

## Data Split

### Benchmarks

- Spectra 652, 653, and 654 from April 29, 1999.
- Published spectrum 653 is the primary positive control.

### Discovery Panel

Freeze seven additional sweeps spanning existing residual/deviance quantiles,
count levels, voltage estimates, and nuisance-bound behavior. Model choices may
be diagnosed on this panel.

### Confirmation Panel

Freeze 50 additional sweeps before tuning. Do not inspect their candidate-model
results until model forms and tuning rules are fixed.

## Required Comparisons

- Benchmark voltage and magnetic-ratio recovery.
- Signed residual structure by energy and pitch.
- Parameter-bound and optimizer-failure rates.
- Sensitivity to normalization and mask definitions.
- Stability across optimizer seeds.
- Interval/profile resolution and boundary behavior.
- Repeated-noise coverage under matched and misspecified simulations.
- Effective information scale and residual overdispersion.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Preregistered comparison and decision record |
| `panel_manifest.csv` | Frozen benchmark, discovery, and confirmation splits |
| `run.py` | Reproducible candidate comparison |
| `results.csv` | Per-sweep, per-model fit and diagnostic output |
| `residual_atlas.pdf` | Successful and failed real-spectrum residuals |
| `coverage.csv` | Simulation coverage and calibration results |
| `comparison.png` | Primary model-comparison figure |

## Decision Gate

Advance one method only if it:

- Preserves the Halekas benchmark within a predeclared tolerance.
- Gives interpretable, non-structured residuals at its claimed uncertainty
  scale, or explicitly models remaining discrepancy.
- Avoids systematic voltage and nuisance-bound artifacts.
- Produces resolved intervals or diagnostics with demonstrated empirical
  calibration.
- Generalizes from discovery to confirmation spectra.

If no candidate passes, revise the forward model before uncertainty work.

## Non-Goals

- Mission-wide population statistics.
- Plasma-regime interpretation.
- Choosing the final 200 V or any other identifiability gate.
- Treating optimizer agreement as physical identifiability.
