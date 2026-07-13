# Experiment 008: Geometry-Only Map Identifiability

## Status

**Protocol freeze in progress. No scientific outcomes have been computed.**

## Question

Given the actual Lunar Prospector sampling schedule and corrected surface
projection geometry, which spatial and spatiotemporal potential fields are
identifiable without using any fitted potential values?

This experiment tests the observation operator behind the manuscript's map
non-recoverability claim. It does not fit a loss cone, estimate a beam voltage,
or use the precomputed potential archive.

## Outcomes

- **A -- recoverable:** the submitted spatiotemporal model is stably
  identifiable and recovers frozen injections.
- **B -- non-recoverable:** the submitted model is effectively rank deficient,
  admits scientifically material near-null maps, and fails frozen recovery
  gates.
- **C -- conditionally recoverable:** the submitted dynamic global model fails,
  but a prospectively defined static or coarse model passes.
- **D -- inconclusive:** input, geometry, or numerical evidence is insufficient
  for A, B, or C.

The full decision rules are frozen in [protocol.md](protocol.md) and
[configuration.json](configuration.json).

[Protocol Amendment 01](protocol-amendment-01.md) prospectively separates an
attitude-only sensitivity from a newly discovered production coordinate-frame
mismatch. It was frozen before any operator or recovery outcome.

## Information Barrier

The experiment may read raw Level-1 ER/MAG files, attitude, SPICE kernels, and
tracked source code. It must not read fitted potential values, fit acceptance,
U-width QC, or scientific columns from `artifacts/potential_cache`.

Using every structurally valid and surface-connected sweep is an optimistic
sampling bound. If that superset cannot identify the model, selecting fewer
successful fits cannot add information.

## Observation Unit

One 15-row energy sweep is one potential observation. Its design row is the
mean of the valid row-level spatiotemporal basis evaluations:

\[
A_{i,q,\ell m}=\frac{1}{n_i}\sum_{j\in i}
T_q(t_{ij})Y_{\ell m}(\theta_{ij},\phi_{ij}).
\]

The legacy representation that repeats one fitted potential over 15 row
footprints is retained only as a sensitivity diagnostic.

## Planned Record

- `protocol.md`: frozen scientific protocol and decision rules.
- `configuration.json`: machine-readable frozen choices.
- `input-manifest.csv`: raw and auxiliary input provenance.
- `geometry.py`: raw-data geometry extraction.
- `operator.py`: real harmonic and temporal observation operators.
- `run.py`: phased analysis runner.
- `test_*.py`: focused correctness and information-barrier tests.
- `results/`: compact machine-readable outcomes and figures.
- `supplement/main.tex`: full mathematical and scientific analysis.

Large geometry caches remain under `artifacts/experiment-008/` and are not
tracked.
