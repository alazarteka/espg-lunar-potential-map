# Adversarial referee report

This audit is based on the attached memorandum alone. The referenced case bundle, source PDFs, repository, scripts, and PR 14 implementation were not attached, so the memorandum’s source-fact and reproducibility assertions could not be independently verified. Line references below refer to the [line-numbered memorandum](sandbox:/mnt/data/lp_er_honest_ci_research_memo_numbered.txt).

## Concise theorem/claim audit

| Audit issue                           | Finding                                                                                                                                                                                                                                                    |                  Severity | Referee disposition                                                                                                                 |
| ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------: | ----------------------------------------------------------------------------------------------------------------------------------- |
| 1. Observable-data sigma-algebra      | (S) is defined to contain latent state and then placed inside the observable (O). Retention indicators and the sampling frame are not included in the core definition.                                                                                     |                 **Fatal** | The operative likelihood is not defined until observed and latent state are separated.                                              |
| 2. Meaning of “count”                 | The memorandum correctly distinguishes incident arrivals, accepted accumulator values (K), compressed codes (C), and calibrated flux (Y). But the actual accumulator arithmetic is missing, so no current count law is operational.                        | **Fatal operational gap** | The conceptual distinction survives; the application does not.                                                                      |
| 3. Conditioning on row totals         | The conditional multinomial calculation is correct only for exact independent Poisson counts and an observed exact total. The memorandum correctly rejects compressed or product totals as substitutes.                                                    |              **Survives** | No substantive correction needed beyond normalizing (p_a) if it is not already a composition.                                       |
| 4. Unknown response versus (U)        | Arbitrary cellwise response or background reproduces any change in (U).                                                                                                                                                                                    |                 **Fatal** | An exact nonidentification certificate exists.                                                                                      |
| 5. Point identification               | Identification is assumed through “disjoint profiled law sets”; it is never established for a source-backed Lunar Prospector nuisance class. Physical nuisance can also absorb an energy shift.                                                            |                 **Fatal** | There is no point-identified current target.                                                                                        |
| 6. Silent Poisson independence        | The main likelihood section states independence explicitly. However, Proposition 9.1 calls a set based only on Poisson marginals “sharp,” although joint dependence is not specified.                                                                      |                 **Major** | The proposition is sharp only if the joint law is determined by the means or arbitrary compatible dependence is explicitly allowed. |
| 7. Unrestricted nuisance distribution | Arbitrary vector Cox mixing is handled correctly. Unrestricted shared-scalar mixing is not: compactness alone does not prevent an all-zero degeneracy or nonparametric overprofiling.                                                                      |                 **Major** | A finite-dimensional or otherwise identifying mixing class is needed.                                                               |
| 8. Supremum/infimum ordering          | The abstract least-favorable ordering is correct: profile inside the statistic, then take a supremum over the generating null nuisance in the p-value.                                                                                                     |              **Survives** | The operational approximation must be an upper certificate for this supremum.                                                       |
| 9. Bootstrap calibration              | No uniform bootstrap theorem is proved. The proposed sequential Monte Carlo procedure uses fixed-sample binomial bounds under optional stopping.                                                                                                           |    **Fatal for coverage** | Use a fixed simulation budget, alpha spending, or an anytime-valid confidence sequence.                                             |
| 10. Wilks/normal/(\chi^2) calibration | The memorandum expressly disclaims these calibrations at boundaries, under weak identification, and for discrete selected observations.                                                                                                                    |              **Survives** | No defect found in the disclaimer.                                                                                                  |
| 11. Discrete interval observations    | Nonrandomized upper-tail p-values are conservative and valid. Randomization is optional if exact nominal size rather than size at most (\alpha) is required.                                                                                               |              **Survives** | “Exact” should consistently mean finite-sample valid, not exact equality of size.                                                   |
| 12. Optimizer versus model            | The conservative lower-LR statistic is defensible if calibrated as its own statistic. Two other optimization claims are wrong: a zero lower bound does not certify feasibility, and a zero lower bound on law distance does not certify nonidentification. |                 **Major** | Correct the direction of the numerical certificates.                                                                                |
| 13. Coverage type                     | The least-favorable theorem is finite-sample and uniform over a declared nuisance class. The GMM theorem merely assumes the needed uniform size statement. No theorem applies to the current LP experiment.                                                | **Fatal operational gap** | Current claims are conditional mathematical templates, not established coverage.                                                    |
| 14. Instrument uncertainty            | The memo says uncertainty must be bounded or included in an honest nuisance region, but supplies neither. Estimated calibration uncertainty is not propagated.                                                                                             |                 **Fatal** | A fixed validated envelope or a joint calibration-data procedure is required.                                                       |
| 15. Simulation and identification     | The plan mainly tests self-model calibration. Its proposed identification certificate has the optimization direction reversed.                                                                                                                             |                 **Major** | Identification requires an analytic equality witness or a certified positive separation bound, not ordinary coverage simulation.    |

---

## Nonidentification certificate

The memorandum’s response-absorption argument is correct and decisive for the broad current class.

Take any two distinct values, for example

[
U_1=-100\ \mathrm V,\qquad U_2=-50\ \mathrm V.
]

Let

[
f^{(r)}_j=f_j(U_r,\beta_r)\ge 0,\qquad r=1,2,
]

be the ideal physical cell signals. Choose any vector (m\in(0,\infty)^d). For each (r,j), define

[
(a_j^{(r)},b_j^{(r)})=
\begin{cases}
\left(m_j/f^{(r)}_j,0\right),&f^{(r)}_j>0,[3pt]
(0,m_j),&f^{(r)}_j=0.
\end{cases}
]

Then under both parameter values,

[
b_j^{(r)}+a_j^{(r)}f_j(U_r,\beta_r)=m_j.
]

Now let the accepted accumulators have, for example,

[
K_j\overset{\mathrm{ind}}{\sim}\operatorname{Pois}(m_j)
]

under both mechanisms. Use the same decoded state, compressor, terminal rule, ground map, and fixed selection mechanism. The complete (K) laws coincide, hence

[
\operatorname{Law}_{U_1,\nu_1}(C)
=================================

\operatorname{Law}_{U_2,\nu_2}(C),
]

and any deterministic downstream (Y=G(C,S,Z)) also has the same law.

Thus

[
\mathcal P^O_{U_1}\cap\mathcal P^O_{U_2}\ne\varnothing.
]

This is an exact nonidentification certificate, not a local-rank warning or an asymptotic approximation.

There is also a separate physical-nuisance certificate that does not rely on unknown instrument response. Suppose the physical signal has the form

[
f_j(U,\theta)
=============

\int R_j(E),g{E-(U+\theta)},dE,
]

where (\theta) is an unknown plasma energy-location nuisance. Choosing

[
\theta_2=\theta_1+U_1-U_2
]

gives

[
f_j(U_1,\theta_1)=f_j(U_2,\theta_2)
]

for every cell. Unless the physical nuisance class rules out this translation confounding, known detector response alone does not identify (U).

The weakest repair is a source-backed response and physical-nuisance class whose profiled observable-law sets are proved disjoint. That is a substantial strengthening of the scientific assumptions. Merely imposing smoothness, using robust standard errors, or obtaining a well-behaved optimizer does not suffice.

---

# Detailed findings

## 1. The observable-data sigma-algebra is internally inconsistent

**Exact claims.**

> “Let (S_i) be all observed or latent state relevant to sweep (i)…”
> — lines 153–154

and

> “The declared observable (O) must be one of: (O=(C,S,Z))… (O=(Y,S,Z))… (O=(C,Y,S,Z))…”
> — lines 200–204

A latent component cannot simultaneously be an element of the observed random object. The memorandum never formally defines a sigma-algebra such as

[
\mathscr O
==========

\sigma!\left(
N_{\rm opportunities},
R_i,,
R_iC_i,,
R_iY_i,,
S_i^{\rm obs},,
Z_i^{\rm obs},,
\text{timestamps and record metadata}
\right).
]

This matters because a likelihood conditional on an unobserved state is not an observable-data likelihood. Integrating over latent state, conditioning on observed state, and taking a least-favorable supremum over state alternatives are different experiments.

**Counterexample.** Let (U,S,C\in{0,1}), let the latent state (S\sim\operatorname{Bernoulli}(1/2)), and let

[
C=U\mathbin{\mathrm{xor}}S.
]

If (S) is observed, (U=C\mathbin{\mathrm{xor}}S) is perfectly identified. If (S) is latent, then

[
C\mid U=0\sim\operatorname{Bernoulli}(1/2),
\qquad
C\mid U=1\sim\operatorname{Bernoulli}(1/2),
]

so (U) is completely nonidentified. Writing (O=(C,S)) chooses the first experiment even when the state is actually latent.

Retention is equally important. If rejected sweeps, opportunity counts, or retention indicators are recorded, they belong to the observable sigma-algebra. If they are unavailable and inference is conditional on retention, that conditioning must be part of the theorem.

**Severity:** fatal.

**Weakest repair:** split

[
S=(S^{\rm obs},S^{\rm lat}),\qquad
Z=(Z^{\rm obs},Z^{\rm lat}),
]

define the exact sampling frame and observable sigma-algebra, and either integrate over or maximize over latent components according to a stated model. Include (R) and the number of observable opportunities whenever available.

**Assumption effect:** the notational split itself does not strengthen assumptions. Declaring state observed, ancillary, or exogenous when it is not would materially strengthen them.

---

## 2. Conditioning on decoded state can itself be invalid

**Exact claim.**

> “Conditional on decoded (S_i), (K_{ij}) independent Poisson…”
> — line 264

Decoded state is not necessarily an exogenous design variable. Automatic prescaling, rate-triggered masks, dead-time status, FIFO status, and some validity flags can be functions of the same or adjacent count process. Conditioning on such a state changes the distribution of the accumulator.

**Counterexample.** Let

[
K\sim\operatorname{Pois}(\mu),\qquad
S=1{K>0}.
]

Then

[
K\mid S=1
]

has a zero-truncated Poisson distribution, not a Poisson distribution. A likelihood that treats (S) as fixed and still uses an ordinary Poisson law is wrong.

The same issue arises if a prescale flag is triggered by a previous high-rate sample correlated with the current sweep. Conditional independence given the flag does not follow merely because the flag is telemetered.

**Severity:** fatal for any positive model using endogenous state as an ordinary conditioning variable.

**Weakest repair:** model the joint law

[
P_{u,\nu}(S,K,C,R)
]

or restrict conditional inference to state variables established to be externally commanded or otherwise ancillary after conditioning on observed covariates.

**Assumption effect:** declaring state exogenous is a substantive strengthening. Joint modeling is not, but requires more instrument information.

---

## 3. The count taxonomy is conceptually sound, but no operational count law exists

**Exact claim.**

> “(K_{ij}) is the accepted integer accumulator immediately before telemetry compression. It is not the number of incident arrivals unless the response/loss model proves that equality.”
> — lines 155–164

This is one of the memorandum’s stronger distinctions. The relevant levels are:

[
N^{\rm incident}
\longrightarrow
N^{\rm detected}
\longrightarrow
N^{\rm accepted}
\longrightarrow
K^{\rm accumulated}
\longrightarrow
C^{\rm compressed}
\longrightarrow
Y^{\rm calibrated}.
]

The observed archive contains (C) or (Y), not (K). The memorandum correctly refuses to call (L(C)) an observed count.

The operational problem remains that it also states that the exact onboard arithmetic, prescaling, overflow, summing order, and loss chain are absent. An event count can be Poisson while the register after digital operations is not.

**Counterexample.** If incident accepted events satisfy (N\sim\operatorname{Pois}(\mu)) but the onboard register stores

[
K=\left\lfloor \frac{N}{8}\right\rfloor,
]

then (K) is not (\operatorname{Pois}(\mu/8)). A Poisson interval likelihood for (K) would change the procedure, not approximate the stated model harmlessly.

**Severity:** fatal operational gap, though not an internal conceptual error.

**Weakest repair:** recover or validate the complete accumulator arithmetic and accepted-event chain, then derive the law of (K) from the attempted-event process.

**Assumption effect:** exact recovery adds knowledge. Replacing recovery with a guessed stochastic model strengthens scientific assumptions.

---

## 4. Point identification is assumed rather than established

**Exact claims.**

> “Point identified iff the profiled mean/code-law sets for distinct (u) are disjoint…”
> — line 264

and

> “Suppose … the profiled law sets for separated (u) are disjoint. Then … [the construction] yields an operational finite-sample honest confidence set.”
> — lines 2055–2063

These are conditional criteria, not an identification result for Lunar Prospector. “Suppose the profiled law sets are disjoint” assumes the central property that must be proved.

No source-backed response set (\mathcal H_u) is supplied, and no global separation calculation is completed for the physical nuisance (\beta). Local rank, even if established, would not prove global point identification.

The response-absorption certificate above shows exact nonidentification under the broad current class. The additive energy-location example shows that physical nuisance can also absorb (U).

**Severity:** fatal.

**Weakest repair:** specify a scientifically defensible nuisance set and prove, for every (u_1\ne u_2),

[
\mathcal P^O_{u_1}\cap\mathcal P^O_{u_2}=\varnothing.
]

For useful uniform inference, a positive separation condition is needed for separated values:

[
\inf_{\substack{|u-v|\ge\varepsilon\
P\in\mathcal P_u^O,\ Q\in\mathcal P_v^O}}
d_{\rm TV}(P,Q)>0.
]

**Assumption effect:** this necessarily strengthens the response and physical-model assumptions relative to the currently defensible broad class.

---

## 5. Proposition 9.1 is not a sharp identified-set result under Poisson marginals alone

**Exact claims.**

> “Suppose repeated observations in a fixed state identify a vector of Poisson means…”
> — line 985

and

> “The sharp identified set under model (9.1) is…”
> — lines 994–1003

The section is titled “bounded response under interval-coded Poisson marginals.” Equality of marginal means is not sufficient for equality of the joint observable law unless the joint law is determined by those means or the model explicitly permits every compatible dependence law.

**Counterexample.** Use identity coding for simplicity and let both cells have Poisson mean one.

Under (u_1), let

[
K_1=K_2=Z,\qquad Z\sim\operatorname{Pois}(1).
]

Under (u_2), let

[
K_1,K_2\overset{\rm ind}{\sim}\operatorname{Pois}(1).
]

Both models have the same vector of Poisson marginals and the same mean vector ((1,1)). Therefore the feasibility equations in (9.2) retain both. But the joint laws differ:

[
P_{u_1}(K_1=K_2)=1,
\qquad
P_{u_2}(K_1=K_2)<1.
]

Thus the set in (9.2) need not be sharp for the full observable law.

**Severity:** major.

**Weakest repair:** either:

1. assume independent Poisson counts or another joint law completely determined by (\mu);
2. include all dependence parameters and joint-law restrictions in the feasibility definition; or
3. relabel (9.2) as an outer identified set based on marginal information.

**Assumption effect:** option 1 materially strengthens stochastic assumptions. Option 3 does not, but weakens the conclusion from “sharp” to “outer.”

---

## 6. The monotone-calibration theorem is scalar but is applied to a vector product

**Exact claims.**

> “Let latent scalar observations (Z_r)… Suppose the observed value is (Y=h_r(Z_r))…”
> — lines 563–571

and

> “The obstruction applies directly to continuously valued calibrated products…”
> — line 575

The quantile-transform proof is correct for a scalar continuous observation. It does not automatically extend to a 1320-dimensional calibrated product under separate monotone cell calibrations. Coordinatewise monotone maps preserve the copula.

**Counterexample.** Let (X\sim\operatorname{Unif}(0,1)). Under (u_1),

[
Z^{(1)}=(X,X),
]

while under (u_2),

[
Z^{(2)}=(X,1-X).
]

Every scalar marginal is uniform under both laws. Marginal quantile maps therefore provide no distinction. But the first vector is perfectly comonotone and the second perfectly countermonotone. Strictly increasing coordinatewise calibrations cannot make their joint laws equal.

**Severity:** major.

**Weakest repair:** restrict Theorem 6.3 and its conclusion to scalar or marginal identification. To obtain joint observational equivalence, assume equal copulas, allow an arbitrary joint measurable transport, or permit a sufficiently broad randomized calibration kernel.

**Assumption effect:** allowing arbitrary joint or randomized calibration substantially enlarges the nuisance class. Narrowing the theorem to marginals does not strengthen assumptions but weakens the nonidentification claim.

---

## 7. The selection-equivalence proof omits observable retention

**Exact claims.**

> “The selected law is proportional to (\pi_u(o)p_u(o)).”
> — line 579

and

> “the rule ‘retain iff (C=0)’ gives the same degenerate retained law for every (u). Therefore unrestricted selection destroys identification.”
> — lines 591–595

The argument establishes equality of the conditional distribution of retained values. It does not establish equality of the full observable law if the retention indicator, number of attempted sweeps, or number of rejected observations is observed.

**Counterexample.** Suppose

[
P_{u_1}(C=0)=0.9,\qquad
P_{u_2}(C=0)=0.1,
]

and retain exactly when (C=0). Conditional on retention, the observed value is (C=0) under both parameters. But for (n) known opportunities,

[
\sum_{i=1}^n R_i
\sim
\operatorname{Binomial}(n,0.9)
]

under (u_1) and

[
\sum_{i=1}^n R_i
\sim
\operatorname{Binomial}(n,0.1)
]

under (u_2). The full observable distributions are not equal.

The general conclusion can be repaired. If (q/p_u) is bounded for both values, choose a common retention probability (r>0) and set

[
\pi_u(o)=r,\frac{q(o)}{p_u(o)}.
]

Then

[
P_u(R=1,O\in do)=r,q(do),
\qquad
P_u(R=0)=1-r,
]

which matches the complete ((R,RO)) law.

**Severity:** major.

**Weakest repair:** either define the experiment explicitly as conditional on (R=1), or modify the equivalence construction to match both retained values and retention probability.

**Assumption effect:** the corrected construction does not require stronger scientific assumptions, but conditional inference changes the experiment and its interpretation.

---

## 8. “Compact” shared mixing does not prevent degeneracy or overprofiling

**Exact claims.**

> “Shared scalar Cox/mixed Poisson … mixing family specified or compact…”
> — line 267

and

> “A shared scalar random effect (A_i) does not permit this cellwise construction. If it multiplies every cell … angular shape can identify (u).”
> — line 557

A specified Gamma family with bounded shape is a legitimate model. A merely compact set of mixing distributions is not sufficient.

**Counterexample 1: common all-zero law.** Let the nuisance class contain the point mass

[
H=\delta_0.
]

Then (A=0) almost surely, all accepted counts are zero, and the observable distribution is identical for every (U). The singleton ({\delta_0}) is compact.

**Counterexample 2: unrestricted profiling.** With one sweep and mixing distribution (H),

[
P_{u,H}(C=c)
============

\int g_u(a;c),dH(a).
]

If (H) is unrestricted, then

[
\sup_H P_{u,H}(C=c)
===================

\sup_{a\ge0}g_u(a;c),
]

because a point mass at a maximizing (a) attains the supremum. The “distributional nuisance” has become a data-fitting amplitude. If each sweep is allowed its own (H_i), this occurs independently sweep by sweep.

Exact counts can eliminate a common amplitude by conditioning on an exact total, but the memorandum correctly notes that the relevant latent total is not observed after componentwise compression. Consequently, that identification argument cannot simply be transferred to the coded data.

**Severity:** major.

**Weakest repair:** use a specified finite-dimensional mixing family, or a common mixing distribution subject to identifying restrictions, support bounded away from zero, uniform moment/tail constraints, and a proved injectivity result for the coded joint law.

**Assumption effect:** this materially strengthens the stochastic model.

---

## 9. Several stated identification regularity claims are mathematically too strong

### 9.1 The nuisance-tangent condition is not necessary for local identification

**Exact claim.**

> “A weaker necessary condition for local identification of (u) after profiling is (\partial_u\mu\notin\operatorname{span}(D_\nu\mu)).”
> — line 443

This is a first-order regularity condition, not a necessary condition for local identification.

**Counterexample.** With no nuisance, let

[
\mu(u)=1+u^3
]

near (u=0). The map is one-to-one, hence (u) is locally identified. But

[
\partial_u\mu(0)=0,
]

which lies in the zero nuisance tangent space.

**Severity:** repairable.

**Weakest repair:** say that the condition is necessary for regular first-order identification or nonzero efficient information, not for identification itself.

**Assumption effect:** no stronger scientific assumption is needed unless regular asymptotic inference is claimed.

### 9.2 Finitely many threshold equations need not have finitely many crossings

**Exact claim.**

> “the resulting mean vector is piecewise constant in (U), with changes only at the finite set of threshold/clipping solutions.”
> — lines 459–468

A finite number of equations does not imply a finite number of solutions.

**Counterexample.** On ([0,1]), define

[
h(U)=
\begin{cases}
e^{-1/U^2}\sin(1/U),&U>0,\
0,&U=0.
\end{cases}
]

The single equation (h(U)=0) has infinitely many solutions accumulating at zero.

**Severity:** repairable.

**Weakest repair:** add that every threshold or clipping equation has finitely many crossings on the parameter domain, as would follow from suitable monotonicity or analyticity assumptions.

**Assumption effect:** this is an additional regularity assumption on the forward map.

### 9.3 Compactness must apply to the feasible nuisance graph

**Exact claim.**

> “If (\mathcal U\times\mathcal H) is compact … and its profiled law sets are disjoint … then (\Delta(\varepsilon)>0).”
> — lines 403–430

Because the memorandum elsewhere permits (\mathcal H_u) to depend on (u), the relevant set is

[
\Gamma={(u,\nu):\nu\in\mathcal H_u}.
]

It is (\Gamma), not an ambient product, that must be compact or at least closed in a compact ambient space.

**Counterexample.** Let (u\in{0,1}). Under (u=0), permit only a Bernoulli success probability zero. Under (u=1), permit every success probability in ((0,1]). The profiled law sets are disjoint, but the second set approaches the first in total variation, so their minimum separation is zero.

**Severity:** repairable but relevant to uniformity.

**Weakest repair:** assume the feasible graph is compact and restrict the minimization in the proof to feasible parameter pairs.

**Assumption effect:** this imposes a closedness restriction on the nuisance class.

---

## 10. The abstract composite-null nuisance ordering is correct

The memorandum’s ordering in Theorem 11.1 survives independent derivation.

Let

[
\varphi_u(O)=1{u\notin\mathcal C_\alpha(O)}.
]

Finite-sample honest coverage is equivalent to

[
\sup_{\nu\in\mathcal H_u}
E_{u,\nu}\varphi_u(O)
\le\alpha
\qquad\text{for every }u.
\tag{A}
]

For a high-is-bad statistic (T_u), define the simple-null tail probability

[
p_{u,\nu}(o)
============

P_{u,\nu}{T_u(O^*)\ge T_u(o)}.
]

The valid composite-null p-value is

[
p_u^{\rm LF}(o)
===============

\sup_{\nu\in\mathcal H_u}p_{u,\nu}(o).
\tag{B}
]

For the true nuisance (\nu_0),

[
p_u^{\rm LF}(O)\ge p_{u,\nu_0}(O).
]

Since the nonrandomized tail p-value is super-uniform,

[
P_{u,\nu_0}{p_{u,\nu_0}(O)\le\alpha}\le\alpha.
]

Therefore

[
P_{u,\nu_0}{p_u^{\rm LF}(O)\le\alpha}
\le
P_{u,\nu_0}{p_{u,\nu_0}(O)\le\alpha}
\le\alpha.
]

Inverting the test gives

[
\mathcal C_\alpha(O)
====================

{u:p_u^{\rm LF}(O)>\alpha}.
]

This proves (A).

An infimum over nuisance p-values is wrong. For a simple counterexample, let the null nuisance be (\nu\in{0,1}), with a high-is-bad statistic satisfying

[
T=0\ \text{a.s. under }\nu=0,
\qquad
T=-1\ \text{a.s. under }\nu=1.
]

When (\nu=0) is true, the observed statistic is zero. Its true p-value is one, but the p-value computed under (\nu=1) is zero. Hence

[
\inf_\nu p_\nu=0
]

and the true null would be rejected with probability one.

There are two distinct optimization levels:

[
T_u(o)
======

2\left[
\sup_{v,\xi}\ell(v,\xi;o)
-------------------------

\sup_{\eta\in\mathcal H_u}\ell(u,\eta;o)
\right]
]

profiles nuisance and alternatives **inside the statistic**. The composite-null calibration then takes the **supremum over the generating null nuisance outside the statistic** as in (B). Similarly, an infimum over nuisance inside a best-fitting moment discrepancy is permissible, but the null rejection probability still requires an outer nuisance supremum.

For a robust neighborhood, the correct order is

[
p_u^{\rm rob}(o)
================

\sup_{\nu\in\mathcal H_u}
\sup_{P\in\mathcal N_{u,\nu}}
P{T_u(O^*)\ge T_u(o)}.
]

No minimization over null laws is compatible with uniform size control.

---

## 11. The discrete-test treatment is valid but necessarily conservative

**Exact claims.**

> “atoms only make it conservative”
> — line 1392

and

> “the test … has size at most (\alpha)”
> — lines 1443–1446

This is correct. Interval coding produces a discrete statistic distribution. The upper-tail p-value

[
P(T^*\ge T_{\rm obs})
]

is super-uniform and gives coverage of at least (1-\alpha).

Randomization is not required for honest coverage. To obtain exact nominal size under a fully specified simple null, one could use

[
p^{\rm rand}
============

P(T^*>T_{\rm obs})
+
V,P(T^*=T_{\rm obs}),
\qquad
V\sim\operatorname{Unif}(0,1),
]

with the auxiliary randomization included in the experiment. For a composite null, the least-favorable supremum can still introduce conservatism.

The wording should therefore distinguish:

* exact finite-sample validity: size at most (\alpha);
* exact level: size equal to (\alpha), generally requiring randomization;
* asymptotic calibration.

No substantive repair is required to the theorem.

---

## 12. The numerical feasibility and identification certificates have the wrong direction

### 12.1 A zero lower bound does not certify feasibility

**Exact claim.**

> “retain (u) iff a certified lower bound on (\rho(u)) is zero within declared arithmetic tolerance.”
> — lines 1026–1036

Here

[
\rho(u)
=======

\inf_\nu\max_j
\frac{|\mu_j-b_j-a_jf_j(u,\beta)|}{s_j}
\ge0.
]

A certified lower bound (L(u)=0) is compatible with every value (\rho(u)\ge0), including (\rho(u)=1). It cannot establish (\rho(u)=0).

For conservative outer-set computation:

* a feasible nuisance witness or certified upper bound near zero establishes compatibility;
* a strictly positive certified lower bound establishes incompatibility;
* an unresolved zero lower bound requires retention, but only as an outer approximation.

**Severity:** major.

**Weakest repair:** replace “iff” with: exclude (u) only when a positive lower bound on (\rho(u)) is proved; otherwise retain it and label the result a numerical outer set.

**Assumption effect:** none; this is a correction to optimization logic.

### 12.2 A zero lower bound on law distance does not establish nonidentification

**Exact claim.**

> “If the certified lower bound is zero for separated values, narrow the claim to B/C.”
> — lines 1890–1896

Law distances are nonnegative. A lower bound of zero is usually the default uninformative result. It does not imply that the true infimum is zero.

To establish identification, one needs a certified strictly positive lower bound:

[
\inf_{\nu_1,\nu_2}
d(P_{u_1,\nu_1},P_{u_2,\nu_2})
\ge\delta>0.
]

To establish observational equivalence, one needs an exact feasible equality construction or a certified upper bound of zero, which in exact arithmetic means an equality witness.

**Severity:** major.

**Weakest repair:** reverse the certificate logic and distinguish an equivalence witness from failure to prove separation.

**Assumption effect:** none.

---

## 13. Sequential Monte Carlo invalidates the stated binomial guarantee

**Exact claims.**

> “A one-sided Clopper–Pearson upper bound (U_B)… Allocate (\sum_B\delta_B\le\delta_{\rm MC}).”
> — lines 1754–1764

and

> “Use sequential batches and stop only when a one-sided confidence bound proves…”
> — lines 1816–1826

A fixed-sample Clopper–Pearson bound is not an anytime-valid confidence sequence. Repeatedly inspecting it and stopping when it crosses the decision threshold increases the probability of a false certificate.

**Concrete counterexample.** Let the true Bernoulli tail probability be (p=0.05). At sample sizes

[
n=20,30,\ldots,3000,
]

compute the ordinary one-sided 99% Clopper–Pearson upper bound and stop if it falls below 0.05. Exact binomial recursion gives

[
P_{p=0.05}
\left{
\exists n\le3000:
U_n^{\rm CP}<0.05
\right}
\approx0.08153,
]

not 0.01. Repeated fixed-look guarantees do not remain 99% simultaneous guarantees.

The same issue affects the validation plan’s stopping when a Clopper–Pearson interval reaches a target width.

Adaptive nuisance-box refinement creates another multiplicity layer. Allocating (\sum_B\delta_B) over a fixed finite cover is valid; reusing a nominal (\delta_B) for newly generated boxes or repeated looks is not.

**Severity:** fatal for the claimed operational finite-sample numerical certification.

**Weakest repair:** use one of:

1. a fixed, predeclared simulation count for every decision;
2. an alpha-spending schedule over all looks and adaptively generated boxes;
3. an anytime-valid binomial confidence sequence or e-process;
4. a predeclared countable refinement tree with a summable failure budget.

All simulation randomness must be included in the coverage probability.

**Assumption effect:** none on the scientific model; this changes the computational calibration.

---

## 14. The bootstrap/GMM theorem does not establish uniform calibration

**Exact claim.**

> “Suppose …
> [
> \limsup_{n\to\infty}\sup_{u,\nu}
> P_{u,\nu}{T_n(u)>c_{n,\alpha}(u)}\le\alpha.
> ]
> Then … [the set] is uniformly asymptotically honest.”
> — lines 1503–1516

This theorem is logically correct but nearly tautological: it assumes the uniform size property needed for uniform coverage. It is an inversion lemma, not a proof that a cluster bootstrap, multiplier bootstrap, or empirical-likelihood calibration has that property.

The subsequent list of mixing, moment, covariance, eigenvalue, and equicontinuity conditions is not a theorem for the proposed LP statistics. In particular:

* the cluster definition may be chosen using empirical autocorrelation;
* the response and calibration are estimated or unresolved;
* nuisance can approach weak-identification boundaries;
* covariance rank can change with sparse codes;
* preprocessing and model-class choice may be data-dependent.

A bootstrap proved for each fixed ((u,\nu)) would establish only pointwise asymptotic validity. It would not justify the supremum over the declared nuisance class.

**Severity:** fatal if the GMM/bootstrap route is presented as currently calibrated; otherwise major incompleteness.

**Weakest repair:** state a precise triangular-array experiment and prove a uniform bootstrap approximation for the actual profiled statistic, including selection, clustering, covariance-rank changes, and nuisance boundaries. Alternatively, use the finite-sample bounded-feature construction.

**Assumption effect:** a uniform bootstrap theorem would require materially stronger mixing, moment, smoothness, covariance, and nuisance-compactness assumptions.

---

## 15. Instrument uncertainty is demanded but not incorporated

**Exact claims.**

> “Uncertainty envelope: calibration covariance or defensible bounds tied across cells/states…”
> — line 2047

and

> “Construct an honest nuisance set (B_u) with error (\gamma), or use all (\mathcal H_u).”
> — lines 1722–1724

The memorandum correctly rejects prior averaging as a frequentist default. But it never constructs either:

1. a fixed deterministic response envelope known to contain the true instrument response; or
2. a frequentist confidence region for response parameters obtained from calibration data.

If calibration bounds are themselves estimated and then treated as fixed, the coverage proof omits their estimation error. If the bounds are engineering judgments, coverage is conditional on the event that those judgments contain the truth—not an unconditional frequentist statement.

Berger–Boos requires

[
\inf_{\nu}
P_{u,\nu}{\nu\in B_u(O)}\ge1-\gamma.
]

Merely calling a set “validated” or “defensible” does not establish this property.

**Severity:** fatal for current operational inference.

**Weakest repair:** either:

* predeclare a deterministic source-backed envelope accepted as a maintained scientific assumption; or
* include independent calibration data (D_{\rm cal}) in the experiment and construct a simultaneous nuisance region satisfying a proved coverage statement, then apply Berger–Boos or joint test inversion.

Model discrepancy not represented by finite calibration parameters must receive a separate robust envelope.

**Assumption effect:** a deterministic envelope strengthens the maintained scientific assumptions. A calibration-data confidence region adds sampling information but still requires a model relating calibration conditions to flight conditions.

---

## 16. The simulation plan cannot establish identification

**Exact claims.**

> “Simulation is an implementation audit, not a substitute for Theorems…”
> — line 1847

This statement is correct.

However:

> “Every claimed A model should include a numerical search for distinct (u_1,u_2)… If the certified lower bound is zero … narrow the claim…”
> — lines 1888–1896

has the certificate direction reversed, as discussed above.

Ordinary self-model coverage simulation answers:

[
\text{Does the implemented procedure approximately attain its nominal size
when data are generated from the same implemented model?}
]

It does not answer:

[
\text{Is the implemented model an adequate description of the instrument?}
]

and it does not establish:

[
\mathcal P_{u_1}^O\cap\mathcal P_{u_2}^O=\varnothing.
]

A finite scenario panel cannot establish global identification or uniform coverage between panel points. Model-misspecification simulations are useful sensitivity analyses, but they do not prove that the declared uncertainty class contains reality.

**Severity:** major.

**Weakest repair:** separate three validation layers:

1. deterministic implementation tests against known probabilities;
2. exact or conservatively bounded optimization certificates for identification and nuisance suprema;
3. empirical falsification and misspecification diagnostics on held-out instrument data.

Identification must be established by an analytic argument, an equality witness showing failure, or a certified global positive distance bound.

**Assumption effect:** none for the logical correction. A positive identification proof will usually require a narrower scientific nuisance class.

---

# Claims that survive the audit

Several important parts of the memorandum are mathematically sound.

First, the distinction between incident events, accepted accumulators, compressed codes, and calibrated products is correct. In particular, treating (L(C)) or a rounded inverse of calibrated flux as an observed count is invalid.

Second, the row-total argument is correct. If exact counts satisfy

[
K_a\overset{\rm ind}{\sim}\operatorname{Pois}(\rho p_a),
\qquad
\sum_a p_a=1,
]

then

[
(K_1,\ldots,K_A)\mid \sum_aK_a=N
\sim\operatorname{Multinomial}(N,p).
]

The result does not apply when the total is unobserved, when component counts are interval-coded, or when dead time, FIFO loss, normalization, or shared response destroys the independent-Poisson representation. The memorandum correctly rejects a sum of decompressed lower endpoints as the conditioning total.

Third, the least-favorable supremum ordering and Berger–Boos construction are correct at the abstract theorem level.

Fourth, no Wilks, Wald, or (\chi^2_1) cutoff is justified by the memorandum. Its express refusal to make those substitutions is appropriate.

Fifth, discrete interval coding does not require randomized tests for honest coverage. Conservative exact-tail tests are valid.

Sixth, the lower-bounded likelihood-ratio statistic in Section 13 can be a valid conservative statistic if the same complete algorithm is applied to observed and simulated data and its own null distribution is calibrated. The defects arise in the subsequent sequential Monte Carlo certification, not from using a smaller statistic itself.

---

# Coverage classification

The memorandum contains three different levels of statements:

1. **Abstract finite-sample theorem.** Theorem 11.1 is finite-sample and uniform over the nuisance class actually included in the supremum, assuming exact observable laws and exact p-values.

2. **Conditional numerical theorem.** The proposed numerical route would inherit finite-sample coverage only if all likelihood, nuisance-cover, total-variation, optimization, continuum, and Monte Carlo errors were controlled from the conservative side. The current sequential procedure does not meet that condition.

3. **Asymptotic route.** Theorem 11.5 does not prove uniform bootstrap validity; it assumes the required uniform rejection bound. No LP-specific uniform result is established.

For the present Lunar Prospector experiment, neither finite-sample, pointwise-asymptotic, nor uniform-asymptotic coverage has been established, because the observation law and calibration/response nuisance class are unresolved.

A full-domain confidence set could still be honest under nonidentification. What is unavailable is a justified **nontrivial** per-sweep confidence set.

# Final decision

**Reject because the target is not identified.**

This decision applies to the current Lunar Prospector experiment under the broad nuisance class supported by the attached record. Distinct (U) values can be paired with distinct response or physical nuisance mechanisms that induce exactly the same observable distribution. The memorandum’s abstract least-favorable inversion theorem is retainable, but it becomes relevant only after the observable sigma-algebra is corrected, the response and physical nuisance class is materially restricted, global identification is proved, instrument uncertainty is incorporated, and the sequential numerical calibration is replaced by a genuinely simultaneous procedure.

