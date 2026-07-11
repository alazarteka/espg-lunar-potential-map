# Experiment 006 Independent Design Review

Three independent review roles examined Experiment 005 and the proposed
restricted-domain confirmation: a statistical calibration reviewer, a plasma
and instrument-physics reviewer, and a hostile journal reviewer focused on
post-hoc rescue. This record summarizes their convergent findings and the
design decisions adopted before implementation.

## Shared Criticisms

All reviewers agreed that `100-500 eV` is legitimate as a new hypothesis but
not as a retrospective Experiment 005 conclusion. They independently identified
the following requirements:

- fresh seeds are insufficient without disjoint supports and a new correction;
- the optimizer must search outside the reporting window;
- every mandatory stratum needs a retention gate;
- false-positive control must use a powered binomial upper bound;
- confirmation must include new energies between and near prior anchors;
- calibration and confirmation require an information barrier; and
- a pass validates a conditional beam centroid, not absolute lunar potential.

## Statistical Findings Adopted

The Experiment 005 continuum generator used the injected beam center to set its
loss-cone boundary. That allowed indirect information about the target to enter
through the background. Experiment 006 makes continuum and beam center
independent in its primary arm and relegates physical coupling to a secondary
test.

Experiment 005 calibrated and validated on the same four real supports and the
same generator families. Experiment 006 uses outcome-blind, disjoint support
sets and includes independently generated and semi-synthetic discrepancies.

Coverage in Experiment 005 was conditional on passing QC, but low-retention
cells could become ineligible. Experiment 006 keeps conditional coverage as the
estimand while making retention co-primary. Every primary cell must pass.

The earlier `15/250 = 6%` null result was compared as a point estimate. Its
one-sided 95% upper bound is much larger. Experiment 006 uses 1,000 nulls in
each of three families and a Bonferroni-adjusted exact upper-bound gate. The
resulting frozen maximum is 35 false in-window acceptances per family.

## Physical And Instrument Findings Adopted

The Experiment 005 local loader constructed pitch without supplying
surface-connection polarity. Experiment 006 requires SPICE-derived connection
and polarity so 180 degrees has the intended surface-leaving meaning.

Experiment 005 passed the same simulated normalized observation as both its
primary and sensitivity normalization, making normalization agreement automatic.
Experiment 006 simulates one latent native-sector realization and independently
derives incident-mean and conjugate representations.

The old simulation evaluated a symmetric response at channel centers. The new
design requires channel integration and challenges response width, energy
scale, log-energy shape, intrinsic broadening, and asymmetric morphology.

Because LP energy rows are sequential in time, temporal variation can imitate
an energy-local feature. Temporal ramps and pulses are explicit non-beam
controls rather than an unmentioned limitation.

Finally, the beam relation includes secondary emission and transport terms:

\[
U_{rel} \simeq E_{emit} - E_{beam} + \delta_{transport}.
\]

The design therefore validates `E_beam` directly and treats `-E_beam` only as a
zero-emission-energy proxy.

## Deliberate Compromise

One reviewer recommended a 1% false-positive upper bound because a 5% detector
could produce many false candidates in 398 mostly-null sweeps. The adopted
primary gate retains a familywise 5% upper bound because passing Experiment 006
opens only a hand-audited one-day residual study, not production or a survey.
The 1% upper bound is retained as a reported stronger target. Any later
automated population analysis requires a separate, stricter selection-error
design.

## Conditions For Credibility

The confirmation is credible only if the proposed window can genuinely lose.
Accordingly:

- edge failure rejects the full window;
- null failure keeps the real audit closed;
- semi-synthetic failure limits the result to simulation;
- low retention cannot avoid coverage evaluation;
- the known Halekas match is not confirmation; and
- an outcome-driven narrower window becomes a hypothesis for a new experiment,
  not a salvaged Experiment 006 pass.
