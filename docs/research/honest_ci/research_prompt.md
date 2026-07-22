# Prompt for GPT Pro

I have attached `lp-er-honest-ci-research-bundle.zip`. Unpack it and treat the
directory `lp-er-honest-ci-research-bundle/` as the case record for this task.

Act as a senior mathematical statistician with expertise in marked point
processes, interval-censored and quantized count data, semiparametric inference,
partial identification, inverse problems, distributionally robust inference,
and scientific instrument response. Produce a research-grade mathematical
answer, not a brainstorming list and not an implementation patch.

## Outcome

Determine whether Lunar Prospector Electron Reflectometer observations can
support a nontrivial, honest frequentist confidence set for spacecraft-relative
lunar surface potential `U` under scientifically defensible assumptions weaker
than an ordinary independent Poisson count model. Locate the identification and
coverage boundary as assumptions are weakened. For every class in which useful
inference is possible, derive the observation-appropriate estimation or testing
objective and a computationally realizable confidence-set construction. For
every class in which it is not possible, prove the obstruction and state the
smallest additional information or restriction that would remove it.

The final answer must be sufficiently formal that another statistician could
check the definitions and proofs, and sufficiently concrete that the project
could implement and simulate the recommended procedure without inventing
missing assumptions.

## Evidence protocol

Read `README.md` and `KNOWN_FACTS_AND_OPEN_QUESTIONS.md` first. Then audit, at
minimum:

1. `current_branch/docs/architecture/er_measurement_contract.md`;
2. `primary_sources/magerfmt.txt` and `primary_sources/lvl0sis.txt`, checking
   their PDFs wherever equations, tables, or layout matter;
3. all four `primary_sources/data-*-desc.txt` files;
4. `current_branch/src/losscone/level0.py`, `raw_calibration.py`, their tests,
   and the relevant data-loader/forward-model code;
5. PR 14's `docs/physics/profile_likelihood_ci.md`,
   `src/losscone/quasi_likelihood.py`, `profile_ci.py`, `response_folded.py`,
   `regime_gate.py`, confidence-set code, scripts, and coverage tests;
6. the reference labels/data when needed to verify a concrete claim or design a
   diagnostic.

Use this evidence hierarchy: primary archive/instrument sources first;
calibrated product descriptions second; directly reproducible binary/data and
code behavior third; repository interpretation fourth. Report material
conflicts. Distinguish a fact stated by a source, an inference you make, and an
assumption you propose. Cite bundle paths and page/section/table or line/function
locations closely enough for an auditor to find the support.

Do not assume that a `TBS` item exists merely because the instrument may once
have used it. Treat it as unavailable unless it is present in the bundle. Do not
silently infer missing response coefficients from rounded calibrated flux.

## Core mathematical setup to refine, not merely repeat

Build a layered probability model. At minimum distinguish:

- a latent marked incident-electron process `X` over time, energy, angle/anode,
  and any physically relevant marks;
- a physical loss-cone/spacecraft-potential forward component indexed by target
  `U` and physical nuisance `beta`;
- instrument state `S` and response/loss nuisance `eta`, including exposure,
  angular and energy response, efficiency, masks, rejection, background, dead
  time, FIFO loss, prescale, accumulation arithmetic, timing, and selection;
- the accepted integer accumulator `K` before telemetry compression;
- the observed 8-bit code `C = q_S(K)`;
- the downstream calibrated flux `Y`, when that rather than `C` is declared the
  observation.

For a code `c = 16e+m`, the documented lower endpoint is

```text
L(c) = m                              if e=0,
L(c) = (16+m) 2^(e-1)                if e>0.
```

For nonterminal, unprescaled codes, truncation ordinarily implies
`K in I(c)={L(c),...,L(c+1)-1}`. Prescale-by-eight and terminal saturation need
state-specific treatment. Verify the exact scope and edge cases from the
sources. Any valid code-space model must use interval probabilities, for
example

```text
Pr(C=c | U,beta,eta,S)
  = sum_{k in I(c,S)} Pr(K=k | U,beta,eta,S),
```

or the appropriate dependent multivariate analogue. Never substitute `L(c)` as
if it were an exact count.

Define the exact estimand. If the physical forward model only identifies a
spacecraft-relative shift, a set, or an equivalence class, say so. Make the
statistical unit and dependence structure explicit: the 15 energy rows of one
3-D sweep are sequential components of a correlated sweep/visit, and the 88
angular cells may share normalization, state, and losses.

For every proposed model class `M`, define honest coverage precisely. A template
is

```text
inf_{U in Uset} inf_{P in P_U} inf_{eta in H(U,P)}
    Pr_{U,P,eta}{U in C_alpha(O)} >= 1-alpha,
```

with `O` exactly equal to the observables used. State whether the guarantee is
finite-sample, uniform asymptotic, pointwise asymptotic, conditional, or only a
sensitivity statement. Do not call pointwise asymptotics or simulation at a few
nuisance values “honest” without qualification.

## Assumption lattice

Construct a partial order of nested or otherwise comparable assumption classes.
Do not claim a unique “weakest model” when classes are incomparable. Include at
least the following, modifying them when the instrument physics requires it:

1. independent Poisson accepted counts with known response/state;
2. independent Poisson with bounded or estimable response nuisance;
3. conditional Poisson or Cox/mixed-Poisson processes;
4. Poisson-binomial/thinning models with heterogeneous efficiencies;
5. negative-binomial or other explicit overdispersion families;
6. renewal/refractory/dead-time models and models with finite-capacity/FIFO
   losses;
7. clustered, finite-range-dependent, or sweep-level random-effect models;
8. martingale or counting-process models specified by compensator plus bounded
   jumps/variance conditions;
9. exchangeability assumptions, carefully distinguishing what finite and
   infinite exchangeability imply;
10. estimating-equation or conditional-mean models with variance bounds;
11. moment-equality and moment-inequality models;
12. contamination or distributionally robust neighborhoods around a nominal
    observation law;
13. a nearly unrestricted law constrained only by code support and the known
    deterministic compressor.

For each class, tabulate:

- scientific meaning and every substantive assumption;
- observable law and nuisance set;
- whether `U` is point identified, partially identified, or not identified;
- whether a nontrivial honest confidence set is possible;
- finite-sample versus asymptotic status;
- derived optimization/test target;
- computational burden;
- which missing calibration/state inputs are required;
- diagnostics that could falsify or stress the class using the bundled data.

## Identification must precede optimization

Give explicit definitions of observational equivalence, point identification,
the identified set, and uniform distinguishability for this experiment. Analyze
both code-space inference and calibrated-flux inference.

Prove positive identification results where justified. Also construct explicit
counterexamples: whenever possible, exhibit `U_1 != U_2` and nuisance/law pairs
that induce exactly the same distribution of the declared observable. In
particular, analyze whether arbitrary unknown angle/energy response, unknown
monotone calibration, unrestricted mixing/overdispersion, unknown selection, or
unbounded contamination can absorb the loss-cone change caused by `U`.

Do not stop at “there are many unknowns.” State and prove an observational-
equivalence theorem or proposition. If the proof needs a positivity, rank,
shape, monotonicity, smoothness, bounded-distortion, anchor-cell, repeated-
state, or calibration restriction, state it exactly. Name a minimal or near-
minimal sufficient restriction that breaks each important equivalence.

Separate these questions:

- identification of `U` when the complete observable distribution is known;
- estimability from finitely many sweeps;
- existence of a uniformly honest, nontrivial confidence set;
- practical power/width under the Lunar Prospector design.

## Derive the inferential objectives

For every viable class, derive the objective from the observation model or
valid estimating restrictions. Do not begin with log-MSE and retrofit a story.

At minimum:

- derive the exact interval-code likelihood for independent Poisson `K` using
  stable Poisson-CDF/survival-function differences, including zero, wide bins,
  terminal saturation, and prescale state;
- derive the corresponding likelihood/integral or computable representation
  for at least one defensible overdispersed or mixed-Poisson class;
- explain what dependence does to factorization and give a valid composite,
  conditional, cluster, or full likelihood only when its calibration is proved;
- derive an estimating-equation/GMM or empirical-likelihood route under stated
  mean/covariance conditions;
- derive a moment-inequality or partial-identification route when response is
  bounded rather than known;
- formulate a minimax or distributionally robust test when the nuisance is a
  set of laws;
- characterize when a calibrated-flux quasi-likelihood is legitimate, what its
  sandwich/cluster covariance must estimate, and what is merely a working
  criterion.

Explain invariance to angular-cell ordering and simplex-coordinate choice. If a
criterion drops one normalized cell, show the conditions under which the
result is invariant, or repair it using a full-rank contrast, generalized
inverse, logistic coordinates, or another proved construction.

For each criterion distinguish:

1. the estimator or optimizer;
2. the test statistic;
3. nuisance treatment;
4. critical-value calibration;
5. the resulting coverage theorem.

## Confidence sets by test inversion

Give at least one complete confidence-set construction for each recommended
viable route. Define a level-`alpha` test for every composite null `H_0:U=u`,
then invert it. Address nuisance by an explicitly justified method such as
least-favorable supremum p-values, Berger-Boos adjustment, conditional
elimination, profile tests with uniform calibration, subsampling, multiplier or
cluster bootstrap under proved conditions, or finite-sample concentration.

If using a bootstrap, specify exactly what is resampled or simulated and make it
reproduce quantization, response uncertainty, preprocessing, missingness,
selection/regime gates, dependence, and nuisance refitting. State a theorem
with conditions for its validity. Do not invoke Wilks, chi-square cutoffs, or a
parametric bootstrap solely by analogy.

Analyze boundary parameters, weak identification, disconnected confidence
sets, empty sets, grid discretization, Monte Carlo error, nuisance values near
nonidentification, and selection performed on the same data. Include a method
for conservative numerical certification: tolerances, tail-probability bounds,
and how to ensure optimizer failure cannot spuriously narrow a set.

If only a partial-identification result is defensible, distinguish a confidence
set for `U`, a confidence region covering the identified set, and a confidence
set for the true parameter under maintained assumptions. Use the correct target
and prove its coverage statement.

## Required theorem-level deliverables

State and prove, or prove impossible and replace with the sharpest correct
claim, all of the following:

1. a quantized-observation lemma mapping code intervals to probabilities,
   including prescale/saturation qualifications;
2. an identification theorem under a clearly stated known-response baseline;
3. a nonidentification theorem under a materially weaker response or stochastic
   class, with an explicit observationally equivalent construction;
4. a partial-identification proposition under bounded response/calibration
   uncertainty, with a characterization or computable outer approximation of
   the identified set;
5. a confidence-set coverage theorem for the primary recommended method;
6. a nuisance/least-favorable calibration theorem or valid conservative bound;
7. a theorem or counterexample governing bootstrap validity after
   quantization, nuisance refitting, and selection;
8. an invariance result or counterexample for normalized/compositional angular
   data and the choice of dropped cell;
9. a result showing exactly when log-MSE approximates a legitimate likelihood
   or estimating criterion, including the approximation regime and remainder,
   or a counterexample showing why it is inappropriate here;
10. an impossibility/triviality result for the broadest nearly unrestricted
    class;
11. a proposition naming the additional TBS/calibration information sufficient
    to make code-space inference operational.

Use complete proofs, not “standard arguments apply,” unless you state the
external theorem precisely, verify every condition in this setting, and explain
the adaptation. Mark conjectures as conjectures. Do not manufacture a theorem
when a regularity condition cannot be defended.

## Computational realization

For the primary recommended method and the best fallback method, provide:

- mathematical pseudocode from raw observable through confidence-set output;
- stable numerical formulas for interval probabilities and log differences;
- nuisance parameterization and constraints;
- optimization strategy, initialization, global-versus-local guarantees, and
  failure handling;
- test-inversion/grid refinement strategy;
- Monte Carlo or resampling calibration with reproducible random-number use;
- complexity in sweeps, cells, grid points, nuisance dimension, and simulations;
- a minimal simulation design that tests coverage at ordinary, boundary,
  weakly identified, misspecified, and least-favorable cases;
- a concrete audit plan against the included 1998 and 1999 reference data.

You may run read-only analyses on the included data and code. If you do, include
the exact script or pseudocode, random seed, and distinguish empirical output
from proof. Absence of a detected failure is not a coverage theorem.

## Audit PR 14 without anchoring on it

Treat PR 14 as one candidate implementation. Determine which parts, if any,
survive the formal analysis. Explicitly audit the review leads in
`KNOWN_FACTS_AND_OPEN_QUESTIONS.md`, plus model-to-CLI routing, missing sentinels,
shared normalization, preprocessing/selection in bootstrap, nuisance coverage,
and common-random-number use in comparisons. Do not let existing code define
the mathematical answer.

Conclude with a disposition for each major PR component: retain, repair,
relabel as heuristic, replace, or reject, with mathematical reasons.

## Required final structure

Organize the research memo so it contains all of the following, combining
adjacent sections only when nothing is lost:

1. executive conclusion;
2. audited facts and unresolved source gaps;
3. notation, probability space, estimand, and observation operators;
4. assumption lattice and comparison table;
5. identification analysis and proofs;
6. impossibility and observational-equivalence proofs;
7. interval-code likelihood derivations;
8. weaker stochastic models and their tractability;
9. partial-identification construction;
10. derived optimization/testing objectives;
11. confidence-set algorithms and coverage proofs;
12. nuisance, dependence, selection, and calibration treatment;
13. numerical implementation and certification;
14. simulation/coverage-validation plan;
15. empirical diagnostics using the bundle;
16. PR 14 audit and disposition;
17. exact additional information needed from the TBS/calibration package;
18. recommended staged research/implementation plan;
19. theorem/proposition/lemma index;
20. assumption ledger separating source facts, proposed assumptions, and
    unverified conjectures;
21. limitations and claims the project must not make;
22. final decision matrix.

The final decision matrix must assign each assumption class one of:

- **A — point identified with a computable honest confidence set**;
- **B — partially identified with a computable honest outer confidence set**;
- **C — nontrivial honest inference impossible over this class**;
- **D — unresolved because a specifically named source fact or mathematical
  result is missing**.

For A or B, state the exact recommended objective and coverage guarantee. For C,
name the theorem and give the smallest practical assumption strengthening that
moves it toward A or B. For D, name exactly what would resolve it.

## Non-negotiable standards

- Do not conflate incident arrivals, accepted accumulator values, compressed
  codes, calibrated flux, and reconstructed pseudo-counts.
- Do not treat decompressed lower endpoints as exact counts.
- Do not assume independent Poisson sampling merely because an average rate and
  a counter exist.
- Do not average over unknown frequentist nuisance without naming a mixing law
  or prior; keep any Bayesian analysis clearly separate as sensitivity analysis.
- Do not claim that robust standard errors fix nonidentification.
- Do not claim that optimizer curvature or profile width alone is a confidence
  interval.
- Do not claim exact coverage from asymptotics or a finite simulation grid.
- Do not hide missingness, regime gates, or selection inside a fit mask.
- Do not make the result artificially positive. A rigorous impossibility or
  partial-identification result is a successful outcome.
- Do not make the result artificially negative. Exploit genuine conditioning,
  invariance, repeated structure, bounds, or test inversion when the sources
  and mathematics justify them.

## Completion condition

Stop only when the memo supplies the definitions, proofs/counterexamples,
algorithms, coverage statements, evidence citations, PR audit, and decision
matrix needed to answer the outcome above. Before finalizing, perform an
adversarial self-check: try to construct two distinct `U` values with
observationally equivalent nuisance laws for every claimed positive result, and
try to locate an unstated Poisson, independence, calibration, selection, or
regularity assumption in every coverage proof. Repair the result or narrow the
claim when the attack succeeds.

If response length prevents a complete memo, do not compress proofs into vague
claims. End at a section boundary with `CONTINUATION REQUIRED: next section ...`
and retain the same notation and theorem numbering in the continuation.

