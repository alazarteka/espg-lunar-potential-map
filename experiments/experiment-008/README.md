# Experiment 008: Geometry-Only Map Identifiability

## Status

**Complete. Outcome C: the submitted daily degree-5 model fails, while frozen
coarse, static, and shared-synodic field classes are conditionally recoverable.**

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

[Protocol Amendment 02](protocol-amendment-02.md) freezes window anchoring,
real-basis conventions, injection normalization, recovery aggregation, blocked
validation scope, and numerical decomposition choices. It was also frozen
before any operator or recovery outcome.

[Protocol Amendment 03](protocol-amendment-03.md) records the corrections from
two independent pre-outcome audits. It makes a physical area-RMS metric primary,
states the spacecraft-potential gauge limitation, stabilizes the recovery gate,
and makes the decision tree executable. It supersedes conflicting numerical
details in the earlier protocol while all scientific spectra remained unopened.

The [one-day geometry implementation check](geometry-implementation-check.md)
then reproduced the production frame-mixed path to machine precision and found
that frame-consistent footprints shifted by a median 48 km on that fixture.
The tracked
[geometry census](geometry-census.csv) records zero failures across all 560 raw
files. The ignored merged cache contains 559,321 exact 15-row sweeps, including
399,643 with at least one canonical connection and 376,920 with all 15.

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

## Result

All 560 primary daily operators are algebraically full rank, but all have
physical weakest-mode visibility below `4.1e-4` and all fail the 25 V recovery
gate. Recoverability begins at 72 hours for degree 1, at seven days for degrees
1-3, and is strong for whole-mission static and compact shared-synodic models.

See [RESULTS.md](RESULTS.md), [phase-1-results.md](phase-1-results.md), the
[machine-readable results](results/final-summary.json), and the
[LaTeX supplement](supplement/main.tex).

## Record

- `protocol.md`: frozen scientific protocol and decision rules.
- `configuration.json`: machine-readable frozen choices.
- `input-manifest.csv`: raw and auxiliary input provenance.
- `geometry.py`: raw-data geometry extraction.
- `observation_operator.py`: real harmonic and temporal observation operators.
- `run_analysis.py`: primary, grid, and static runner.
- `extended_analysis.py`: temporal, sensitivity, injection, cap, and blocked
  runner.
- `test_*.py`: focused correctness and information-barrier tests.
- `results/`: compact machine-readable outcomes and figures.
- `supplement/main.tex`: full mathematical and scientific analysis.

Large geometry caches remain under `artifacts/experiment-008/` and are not
tracked.

## Reproduction

From the repository root, with BLAS held to the four-thread experiment budget:

```bash
export OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
.venv/bin/python experiments/experiment-008/geometry.py extract --workers 4
.venv/bin/python experiments/experiment-008/geometry.py merge
.venv/bin/python experiments/experiment-008/run_analysis.py prepare
.venv/bin/python experiments/experiment-008/run_analysis.py primary
.venv/bin/python experiments/experiment-008/run_analysis.py grid
.venv/bin/python experiments/experiment-008/run_analysis.py static
.venv/bin/python experiments/experiment-008/extended_analysis.py prepare-temporal
.venv/bin/python experiments/experiment-008/extended_analysis.py temporal
.venv/bin/python experiments/experiment-008/extended_analysis.py sensitivities
.venv/bin/python experiments/experiment-008/extended_analysis.py column-normalization
.venv/bin/python experiments/experiment-008/extended_analysis.py injections
.venv/bin/python experiments/experiment-008/extended_analysis.py caps
.venv/bin/python experiments/experiment-008/extended_analysis.py blocked
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
  .venv/bin/python experiments/experiment-008/plot_results.py
```
