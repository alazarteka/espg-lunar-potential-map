# Codex reconciliation of the honest-CI research exchange

**Date:** 2026-07-22

**Branch reviewed:** `codex/fresh-experiments` at `3cb1ce2`

**Draft PR reviewed:** PR 14 head `e698047`

**Disposition:** preserve the research; do not treat PR 14 as an honest
confidence-set implementation.

## Executive decision

The research exercise produced a defensible central conclusion:

> The currently available Lunar Prospector material does not justify a
> nontrivial honest frequentist confidence set for per-sweep
> spacecraft-relative surface potential.

This is a current-evidence result, not a proof of permanent impossibility. The
scientifically intended raw-code and calibrated-flux experiments remain class
**D**: named observation-law, calibration, and response premises are missing.
Deliberately broader classes that admit arbitrary cell response, arbitrary
outcome-dependent selection, arbitrary vector mixing, or only deterministic
code support are class **C** by explicit observational-equivalence arguments.

The original memo was mathematically useful but not safe to adopt. The referee
correctly identified failures in its observable sigma-algebra, treatment of
endogenous state, LP identification claim, marginal-versus-joint sharpness,
vector calibration argument, selection construction, mixing-law restriction,
regularity claims, numerical certificate direction, sequential Monte Carlo,
bootstrap claim, and calibration uncertainty. The revised memo materially
repairs all sixteen findings and provides a sound conditional statistical
framework.

## Results that survive

1. Incident arrivals, accepted events, the onboard accumulator `K`, compressed
   code `C`, calibrated flux `Y`, and reconstructed pseudo-counts are different
   random objects. A decompressed lower endpoint is not an observed count.
2. Ordinary nonterminal codes define integer inverse intervals. Under a fully
   specified accepted-count law, inference must integrate probability over
   those intervals rather than substitute their lower endpoints.
3. Quantization alone does not destroy identification. For example, under a
   maintained Poisson marginal law, the zero-code probability is strictly
   monotone in its mean.
4. Weakening independent Poisson sampling is mathematically possible through
   specified shared-effect Cox, negative-binomial, renewal/dead-time,
   Poisson-binomial, cluster-moment, or bounded-response models.
5. Weakening the stochastic law does not compensate for an undefined
   instrument observation operator. Every useful likelihood or moment method
   still requires response/state/selection restrictions connecting `U` to the
   observable law or expectation set.
6. Composite-null confidence sets require test inversion. Profiling nuisance
   inside a statistic must be followed by least-favourable, Berger-Boos, or
   otherwise proved calibration over the generating-null nuisance.
7. A fixed-budget Monte Carlo upper certificate can preserve finite-sample
   coverage. Repeated inspection of ordinary fixed-look binomial intervals
   cannot.
8. Under only bounded observable features and genuinely independent replicated
   clusters, simultaneous concentration bounds can yield an honest outer set.
   This is a partial-identification route, not a per-sweep solution unless a
   common/regression/local-homogeneity target and expectation envelope are
   scientifically justified.

## Source-aware corrections to the revised memo

### Energy resolution

The memo describes PR 14's `de_over_e=0.5` as inconsistent with the calibrated
product's stated 25% energy resolution. That is incomplete. The analyzer's
intrinsic resolution was approximately `0.25`, while adjacent onboard energy
bins were summed, giving the operational 15-channel data an effective
resolution of approximately `0.5`. Halekas et al. state both values directly:

<https://ntrs.nasa.gov/api/citations/20140010167/downloads/20140010167.pdf>

Therefore `0.5` is defensible as an effective-width magnitude. PR 14 still does
not have evidence for its uniform top-hat response with equally weighted
quadrature nodes. The unsupported response shape, rather than the numerical
width alone, is the defect.

### Physical translation nuisance

The revised memo proves a correct conditional theorem for models of the form
`f(E - (U + theta))` with unrestricted additive `theta`, but it does not
establish that this nuisance class describes the repository's actual loss-cone
forward model. The current unclipped boundary has the form

\[
x(E)=r\left(1+\frac{U}{E-U_{\rm sc}}\right).
\]

With known `U_sc`, put `z=1/(E-U_sc)`. Then `x(E)=r+(rU)z`. Two distinct
unclipped energy channels identify the intercept `r` and slope `rU`, and hence
identify `U` when `r>0`. Pitch binning, clipping, response uncertainty, and an
unknown `U_sc` can defeat or weaken this result, but the generic additive-shift
counterexample does not prove that they do.

An independent plasma-energy anchor may be required for some enriched physical
classes. It has not been proved necessary for the actual repository model. A
model-specific global identification analysis must replace that application of
the generic theorem.

### Classification of PR 14

PR 14 does not define a complete stochastic experiment with a coverage theorem.
That makes its honest-confidence interpretation unsupported, normally class
**D**, rather than automatically class **C**. Class C applies only after an
explicit broader model class is declared and an observational-equivalence or
misspecified-target result is proved within it.

The practical conclusion is unchanged: PR 14 is not an honest confidence-set
implementation. Its output is a working-model profile or sensitivity set.

### Reproducibility and citations

The adversarial referee explicitly did not receive the evidence bundle, source
PDFs, repository, scripts, or PR implementation. It is an independent
mathematical review, not an independent source audit.

The revised memo refers to an `audit/` directory that was not included in the
delivered attachments. It also retains incorrect source paths such as
`pr14/src/losscone/pipeline.py`; the relevant implementation is under
`pr14/src/potential_mapper/`.

Independent checks in the working repository did reproduce the decisive
empirical facts:

- `M9801523.B` contains 7,047 records and yields 165 accepted structural
  format-1/2 sweeps;
- their code range is 0-153 and their zero-code fraction is
  `0.4586363636363636`;
- `M9911914.B` carries version-5 observations, which the current parser does
  not support, and yields no accepted sweep;
- PR 14's positivity mask accepts the `9.999E-99` missing sentinel;
- its dropped-cell diagonal quasi-criterion is order-dependent;
- the batch CLI exposes only `halekas` and `lillis`, while the enum advertises
  `quasi_likelihood`; and
- non-Lillis CPU/Torch routing falls through existing legacy behavior rather
  than invoking the new profile method.

## Consequence for PR 14

PR 14 should not merge under an honest-confidence-set claim. A defensible
diagnostic-only revision can retain selected ideas after these changes:

| Component | Disposition |
|---|---|
| Measurement-layer distinctions | Retain |
| Response-folded mean | Retain only as an explicitly assumed experimental approximation |
| Disconnected/full level-set representation | Retain concept; make storage dynamic and continuum handling conservative |
| Missing sentinel handling | Fix immediately |
| Dropped-cell diagonal criterion | Replace with tangent-space full covariance or another permutation-invariant criterion |
| Profile surface | Retain as a working-model sensitivity diagnostic |
| Plug-in bootstrap | Do not label finite-sample honest; replace only after a complete observation law and nuisance calibration exist |
| Regime gates | Retain as diagnostics; treat data-dependent gates as selection |
| CLI routing and provenance | Repair before claiming an exposed method |
| Coverage tests | Relabel as smoke tests and replace before any coverage claim |
| Numerical failures | Make every unresolved failure retain/widen rather than exclude |

A clean development path is to split the work into a diagnostic/bug-fix PR now
and a later inference PR only after the observation class and coverage theorem
are established.

## Correct optimization target

At the present research stage, the primary target is identification or
compatibility, not log-mean-square error and not a point estimate of `U`.

For a defined observable-law class, analyze separation:

\[
\Delta_\epsilon=
\inf_{\substack{|u-v|\ge\epsilon\\
\nu\in\mathcal H_u,\ \xi\in\mathcal H_v}}
d\!\left(P_{u,\nu}^{O},P_{v,\xi}^{O}\right).
\]

For bounded response or moment restrictions, solve a feasibility problem:

\[
\rho(u)=
\inf_{\nu\in\mathcal H_u}
\max_r\{\text{violation of observable moment restrictions}\}.
\]

Only after that analysis succeeds should the estimator use either:

- the exact interval-code negative log likelihood for a complete joint law; or
- a minimax moment-inequality criterion for a validated expectation envelope.

The existing log-MSE and PR 14 quasi-NLL remain exploratory profile surfaces.
They have no confidence-set semantics without additional assumptions and
calibration.

## Research handoff

The next recommended task is a focused global identification audit of the
actual loss-cone forward model and its nuisance symmetries. It should determine
which combinations of energy channels, pitch resolution, field ratio,
spacecraft potential, beam/background terms, and tied response uncertainty
identify `U`, leave a proper identified set, or create exact plateaus. That
result should define the smallest calibration envelope needed before further
likelihood work.

No implementation of that next stage is included in this documentation commit.
