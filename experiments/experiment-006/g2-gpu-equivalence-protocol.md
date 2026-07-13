# G2 Complete GPU-Equivalence Protocol

## Decision Context

The frozen C1 CPU estimator is scientifically certified, but one complete case
takes approximately 460 CPU-seconds. Four courteous low-priority workers
completed 339 of 3,000 cases in 20.6 hours, implying an unacceptable remaining
runtime of roughly one week on the shared host.

The earlier D1 projected-LM candidate optimizer is rejected and remains
rejected. G2 does not revive candidate pruning, top-K selection, or the claim
that the D1 backend was equivalent to complete SciPy multistart.

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

G2 changes execution, not the C1 estimator definition:

1. Construct the same 12 observation-derived starts.
2. Advance every start independently with a float64 bounded
   trust-region-reflective GPU solver; no trajectory may be pruned.
3. CPU-polish all 12 GPU endpoints with the frozen SciPy float64 solver.
4. Select the canonical winner using the same finite-success and objective
   rules as `fit_reference.py`.
5. Compute the final Jacobian, rank, covariance, standard errors, and all QC on
   CPU exactly as frozen.
6. Apply the same procedure to both normalizations and every leave-one-energy-
   row-out fit.

Any nonfinite GPU result, incomplete trajectory set, bound-handling failure, or
GPU runtime failure falls back to the complete 12-start CPU reference for that
fit. Fallbacks are retained and reported, never silently dropped.

## GPU Solver

The new solver follows SciPy's bounded trust-region-reflective structure rather
than projected Levenberg--Marquardt. It must include Coleman--Li scaling,
strictly feasible iterates, trust-radius adaptation, and selection among the
constrained trust-region, reflected, and constrained-gradient steps. Residuals
and Jacobians use the already certified float64 Torch forward model.

All cases, starts, derivative directions, and leave-one-row-out variants may be
batched. Batching must not couple trajectory state or stopping criteria.

## Prospective Certification Corpus

The G2 corpus contains 192 newly seeded cases, disjoint from D1 and C1 seeds:

- 48 matched-model cases spanning the five C1 anchors and reporting edges;
- 72 independently generated C1-like discrepancy cases;
- 48 cases spanning all 12 calibration geometry supports;
- 24 null, weak, boundary, and deliberately difficult controls.

Every case uses all 12 frozen starts. The corpus generator, seeds, allocation,
configuration, and hashes are committed before GPU or CPU fit outcomes are
opened.

## Gates

G2 may execute remaining C1 cases only if all gates pass:

- 100% final hybrid-versus-complete-CPU winner-basin agreement;
- 100% final QC classification and reason-code agreement;
- final objective relative difference at most `1e-10`;
- identifiable-center difference at most `1e-6` nominal response FWHM;
- final model log-RMS difference at most `1e-10`;
- identical bound-contact classification and Jacobian rank;
- deterministic replay and batch-order invariance;
- complete retention of 12 trajectories per fit;
- no confirmation flux access and no partial-C1 outcome access;
- projected completion of all 3,000 C1 cases in at most 24 wall-clock hours on
  four TITAN Xp GPUs, including CPU polishing and QC.

A miss in any correctness gate rejects G2 for C1. Performance failure permits
implementation optimization only while correctness outputs remain sealed;
changing scientific tolerances or the estimator is not permitted.

## C1 Resumption

After certification, the 339 CPU checkpoints remain valid because they were
computed by the frozen reference estimator. Remaining cases run through G2.
The merged result records backend, fallback, source hashes, device, and timing
for every case. A deterministic overlap audit recomputes at least one completed
CPU case per anchor with G2 before calibration summaries are opened.
