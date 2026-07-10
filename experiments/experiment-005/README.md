# Experiment 005: One-Day Scientific Validation Pilot

**Status:** Blocked by Experiments 003 and 004

**Branch:** `experiments`

**Depends on:** [Experiment 003](../experiment-003/README.md) and
[Experiment 004](../experiment-004/README.md)

## Scientific Question

Does the frozen retrieval method remain numerically operational, statistically
adequate, and physically interpretable across every attempted sweep of one
representative Lunar Prospector day?

## Day Selection

The primary candidate is April 29, 1999 because it contains the Halekas
benchmark. Before execution, document whether this day adequately spans count
levels, connectivity, illumination, and residual quality or whether a second
contrasting day is required.

Day selection must be fixed before examining pilot-wide success rates.

## Frozen Method

The pilot must use the exact observation model, forward model, bounds,
normalization, mask, optimizer, uncertainty calculation, and quality-control
rules accepted from Experiments 003 and 004. Any post hoc change invalidates the
pilot gate and requires a rerun under a new recorded configuration.

## Required Selection Funnel

Report counts and percentages for:

1. Attempted sweeps.
2. Structurally valid 15-row records.
3. Valid magnetic geometry and surface connection.
4. Sufficient usable bins.
5. Numerical convergence.
6. Adequate residual or goodness-of-fit behavior.
7. Resolved uncertainty/identifiability diagnostic.
8. No disqualifying voltage, magnetic-ratio, or beam bound contact.
9. Final science-usable sweeps.

Never report only the denominator of finite fits.

## Required Diagnostics

- Per-sweep residual adequacy and signed-residual summaries.
- Voltage, field-ratio, and beam-amplitude distributions with bound flags.
- Optimizer-seed stability on a fixed subset.
- Uncertainty/width distribution with endpoint and edge flags.
- Selection probability versus count level, geometry, and observable quality.
- Sweep-level support: row times, start/end footprint, SZA range, magnetic-field
  magnitude and direction variation where available.
- CPU/GPU parity on a fixed sample if the production run uses Torch.
- Runtime and peak-memory measurement, including profile evaluation.

Produce a residual atlas containing preregistered quantiles and every major
failure class rather than only attractive examples.

## Planned Artifacts

| Artifact | Purpose |
| --- | --- |
| `README.md` | Frozen pilot protocol, results, and decision |
| `configuration.json` | Complete scientific and optimizer configuration |
| `run.sh` or `run.py` | Exact reproducible invocation |
| `selection_funnel.csv` | Attempted-to-usable counts and reasons |
| `sweep_diagnostics.parquet` | One row per sweep with fit/QC/support metadata |
| `residual_atlas.pdf` | Fixed successful and failed cases |
| `pilot_summary.png` | Primary pilot figure |
| `runtime.json` | Backend, device, memory, and timing |

Large raw cache products should remain under `artifacts/` with checksums and
provenance recorded here.

## Decision Gate

Advance to independent plasma-regime/SZA annotation and archive recomputation
only if:

- The pipeline completes within documented resource limits.
- Residual adequacy is interpretable under the selected observation model.
- Benchmark and positive controls remain stable in batch processing.
- Failure and nuisance-bound rates are scientifically acceptable and fully
  reported.
- Uncertainty diagnostics are resolved rather than grid-quantized or dominated
  by parameter edges.
- CPU/GPU results agree within predeclared numerical tolerances.

Failure is a useful outcome: preserve the failure anatomy and return to the
specific upstream model assumption implicated by the diagnostics.

## Non-Goals

- Mission-wide recomputation.
- Plasma-regime causal interpretation before independent labels exist.
- Headline lunar potential medians.
- Temporal spherical-harmonic reconstruction.
- Changing method settings to maximize the usable fraction.
