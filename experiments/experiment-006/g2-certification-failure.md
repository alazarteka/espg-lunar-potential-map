# G2 Certification Early Failure

**Decision:** Reject the frozen four-island differential-evolution estimator;
do not run C1 with this backend.

**Implementation commit:** `af67db5`

**Information barrier:** No partial-C1 scientific outcomes or confirmation flux
were opened. The frozen G2 certification corpus was opened prospectively.

## What Passed

All 48 canonical matched cases completed before the discrepancy arms were
opened. They recovered the injected centers, produced deterministic replay,
and demonstrated that the GPU forward model, DE implementation, and CPU polish
can work together on identifiable matched problems.

## Decisive Failures

The run was stopped as soon as retained diagnostics established that the
prospective correctness gate could not pass.

- Case 48 (`discrepancy`) had a valid old 12-start CPU result and a valid G2
  result, but G2's final objective was 0.8713% worse (`93.8860` versus
  `93.0751`). Three of four island polishes converged and one exhausted its
  budget. The polished island centers agreed closely, indicating population
  collapse into a reproducible but inferior basin rather than random numerical
  instability. The final Jacobian rank was 9, not 15.
- Case 96 (`discrepancy`) had SciPy status 0 for all 12 old starts and all four
  G2 island winners. Status 0 means the frozen `max_nfev=2000` budget was
  exhausted. This is an unavailable/non-identifiable reference case and should
  be retained as a failed observation, not treated as a software exception.

The certification runner originally aborted on unavailable trajectories. Commit
`af67db5` corrected the diagnostic plumbing to retain status counts; it did not
change the estimator or any scientific threshold.

## Interpretation

Differential evolution is fast on the four GPUs and exact on matched cases, but
the frozen four-island design does not preserve the basin reached by every old
observation-derived local trajectory under model discrepancy. Seeding the
population with those starts is insufficient: selection and migration can
concentrate all islands around a basin whose unpolished objective is attractive
but whose final CPU-polished objective is worse than the old estimator's.

The performance benchmark therefore remains valid as an engineering result,
but the estimator fails the prospective scientific gate. Tuning migration,
population, or selection against these revealed cases and then reusing the
same certification corpus would be invalid.

## Consequence

Serious GPU acceleration must preserve the twelve local trajectories rather
than replacing them with population winners. The next defensible backend is a
batched bounded trust-region implementation, or a hybrid that keeps each of the
twelve original trajectories as a separately protected local solve. It needs a
newly frozen implementation and certification corpus before C1 resumes.
