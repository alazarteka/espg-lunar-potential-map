# G2 Global Differential-Evolution Protocol

## Decision Context

The frozen C1 CPU estimator is scientifically certified, but one complete case
takes approximately 460 CPU-seconds. Four courteous low-priority workers
completed 339 of 3,000 cases in 20.6 hours, implying an unacceptable remaining
runtime of roughly one week on the shared host.

The earlier D1 projected-LM candidate optimizer is rejected and remains
rejected. G2 instead defines a new global estimator requiring fresh C1
calibration. It does not claim numerical equivalence to the old twelve-start
local estimator.

## Information Barrier

G2 begins after 339 C1 cases were computed, but no fitted centers, errors, QC
outcomes, factor outcomes, or calibration summaries were inspected. Only
checkpoint row counts and runtime were read. Those partial results are
quarantined from G2 implementation and certification. Confirmation flux
remains unopened.

The revealed D1 failure summary may motivate the algorithmic class but may not
be used for threshold tuning. G2 is frozen and tested on a newly generated,
implementation-only corpus before either D1 trajectories or partial C1
outcomes may be used as a final audit.

## Estimator Invariance

G2 prospectively changes the estimator before any G2 optimizer outcomes are
opened:

1. Construct the same 12 observation-derived points as population seeds.
2. Initialize four independent 128-member populations per fitting problem,
   completing each with a separately scrambled Sobol design.
3. Run bounded float64 differential evolution independently on all islands.
4. Exchange two elite members every 40 generations without merging island
   convergence state.
5. CPU-polish all four island winners with the frozen SciPy float64 solver.
6. Select the canonical winner using the same finite-success and objective
   rules as `fit_reference.py`.
7. Compute the final Jacobian, rank, covariance, standard errors, and all QC on
   CPU exactly as frozen.
8. Apply the same procedure to both normalizations and every leave-one-energy-
   row-out fit.

Any nonfinite GPU result, incomplete trajectory set, bound-handling failure, or
GPU runtime failure falls back to the complete 12-start CPU reference for that
fit. Fallbacks are retained and reported, never silently dropped.

## GPU Solver

The DE implementation uses distinct parent indices, per-fit and per-island
random streams, per-island convergence state, bounded mutation/crossover, and
retained generation diagnostics. Residuals use the already certified float64
Torch forward model. Cases, leave-one-row-out variants, islands, and population
members may be batched; batching must not couple random streams or stopping.

## Prospective Certification Corpus

The G2 corpus contains 192 newly seeded cases, disjoint from D1 and C1 seeds:

- 48 matched-model cases spanning the five C1 anchors and reporting edges;
- 72 independently generated C1-like discrepancy cases;
- 48 cases spanning all 12 calibration geometry supports;
- 24 null, weak, boundary, and deliberately difficult controls.

Every island contains the 12 frozen observation-derived seeds. The corpus
generator, seeds, allocation, configuration, and hashes are committed before
GPU or CPU fit outcomes are opened.

## Gates

G2 may execute remaining C1 cases only if all gates pass:

- exact recovery on matched identifiable implementation cases;
- deterministic replay and batch-order invariance;
- agreement of independent island winners after CPU polishing on identifiable
  cases, or explicit retained multimodality diagnostics;
- no worse objective than the old complete CPU multistart reference on the
  newly frozen comparison corpus;
- finite final covariance and stable QC behavior on identifiable cases;
- deterministic replay and batch-order invariance;
- complete retention of four island winners per fit;
- no confirmation flux access and no partial-C1 outcome access;
- projected completion of all 3,000 C1 cases in at most 24 wall-clock hours on
  four TITAN Xp GPUs, including CPU polishing and QC.

A miss in any correctness gate rejects G2 for C1. Because this is a new
estimator, all 3,000 C1 cases must be recomputed; the 339 old-estimator
checkpoints are provenance only. Performance failure permits implementation
optimization only while correctness outputs remain sealed.

## C1 Resumption

After certification, all 3,000 cases run through G2. The result records island
agreement, backend, fallback, source hashes, device, and timing for every case.
The 339 old-estimator checkpoints remain excluded provenance artifacts.
