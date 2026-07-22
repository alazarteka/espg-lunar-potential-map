# Honest frequentist confidence sets for Lunar Prospector ER surface-potential inference

**Research memorandum**  
**Case record:** `lp-er-honest-ci-research-bundle/`  
**Date:** 2026-07-22  
**Target:** spacecraft-relative lunar surface potential, denoted \(U\)

## Citation and claim conventions

A source citation has the form `[Bundle: path, lines x–y; PDF p. z]`. Page numbers refer to the page shown in the corresponding bundled PDF, checked against rendered pages when equations, tables, or layout matter. A statement marked **Source fact** is stated by a bundled source. A statement marked **Reproducible finding** is output of a read-only script supplied with this memorandum. A statement marked **Inference** is a deduction from those facts. A statement marked **Proposed assumption** is not asserted by the case record and must be adopted explicitly before using the associated theorem.

The bundle manifest was verified before analysis with `sha256sum -c MANIFEST.sha256`; every entry passed.

---

# 1. Executive conclusion

The answer has three levels.

**First, the present bundle does not yet support an operational, nontrivial, honest per-sweep confidence set for \(U\) from either raw 3-D telemetry codes or archived calibrated flux.** For code-space inference, the deterministic compression map is sufficiently documented for ordinary nonterminal codes, but the state/response package required to evaluate \(P(C\mid U,\beta,\eta,S)\) is not present: the effective-anode efficiency table and exact onboard arithmetic are explicitly `TBS`; the applicable state, event-time, exposure, response, masks, loss process, terminal behavior, and 1999 Digital Subcom version-5 layout are unresolved. This is a **D** result—unresolved for specifically named missing inputs—not a proof that code-space inference is impossible in principle. For calibrated flux, the public product description documents a many-stage, state-dependent transformation from compressed counts to flux, with dead-time masking, retained background, geometric-factor conversion, and no spacecraft-potential correction. The bundle supplies neither a complete conditional law for that product nor a defensible uncertainty set for the missing calibration. Under the broad class that permits arbitrary cellwise calibration/response or outcome-dependent selection, \(U\) is exactly nonidentified and any uniformly honest set is necessarily trivial. That is a **C** result.

**Second, weaker-than-independent-Poisson classes do permit useful honest inference once the observation operator is fixed or bounded.** Quantization is not the fundamental obstruction. If the state-dependent inverse code intervals and the physical/instrument mean map are known and injective after nuisance profiling, then:

* independent Poisson accepted counts admit an exact interval-code likelihood;
* a specified shared-gamma Cox model admits a one-dimensional integral likelihood and preserves angular-shape information while allowing overdispersion and sweep-level dependence;
* negative-binomial, Poisson-binomial, known renewal/dead-time, or known finite-capacity Markov loss models admit computable interval probabilities;
* repeated independent or suitably mixing sweep clusters admit finite-sample concentration-based moment sets, or uniformly asymptotic cluster-GMM sets under explicit uniform CLT and covariance conditions, without assuming within-sweep independence;
* bounded response/calibration uncertainty yields a computable partially identified set and an honest outer confidence region.

The primary recommended procedure, once the named response/state inputs are supplied, is **composite-null test inversion in code space**, using the exact interval probability law, least-favorable or Berger–Boos nuisance treatment, and exact or conservatively certified Monte Carlo calibration of the entire preprocessing/selection/quantization/refitting pipeline. Its finite-sample coverage is

\[
\inf_{u\in\mathcal U}\inf_{\nu\in\mathcal H_u}
P_{u,\nu}\{u\in\mathcal C_\alpha(O)\}\ge 1-\alpha,
\]

provided the null simulator/evaluator is correct and the numerical supremum and Monte Carlo tail are certified conservatively. The best fallback under much weaker stochastic assumptions is a **bounded-feature moment-inequality outer set across genuinely replicated sweep/visit clusters**, with finite-sample Hoeffding/Bonferroni coverage. It requires a scientifically justified response envelope and a common-\(U\), regression, or local-homogeneity design; a single arbitrary sweep does not supply replication.

**Third, the boundary is sharp.** Exact observational-equivalence constructions show that no nontrivial uniformly honest inference is possible when any of the following is unrestricted: cell-by-cell energy/angular efficiency, vector-valued Cox mixing, outcome-dependent selection, cellwise monotone/randomized calibration, or contamination large enough to make null law neighborhoods overlap. In the support-plus-compressor-only class, the all-zero code law is admissible for every \(U\); any honest confidence set has expected measure at least \((1-\alpha)\) times the full parameter-space measure, and a nonrandomized procedure must return the full domain on that common law. Allowing negative-binomial dispersion to approach zero similarly destroys uniform distinguishability: every finite-mean model converges to the all-zero law.

PR 14 does not satisfy an honest-coverage claim. Its useful components are the explicit “calibrated-flux quasi-likelihood” label, recognition that confidence sets may be disconnected or full-domain, and separation of some regime-gate reasons. The current criterion, response folding, plug-in bootstrap, selection handling, nuisance calibration, confidence-set discretization, tests, and CLI routing require repair or replacement. In particular, the missing sentinel `9.999E-99` is treated as valid positive flux; the dropped-cell diagonal simplex criterion is order-dependent; the bootstrap does not reproduce the criterion’s stated covariance, preprocessing, selection, or nuisance uncertainty; `bootstrap_n=0` silently uses a cutoff of one; and the bundled “coverage” test uses twelve trials with an override cutoff of 25 and accepts coverage above 0.5. Those are diagnostic facts, not a theorem.

The final decision matrix is in Section 22. In brief: theoretically specified known-response count-process classes are **A**; bounded response or moment classes are **B**; unrestricted response/mixing/selection/support classes are **C**; and the currently available Lunar Prospector code-space path is **D** until the exact items in Section 17 are recovered or replaced by a validated uncertainty-bounded emulator.

---

# 2. Audited facts and unresolved source gaps

## 2.1 Instrument and archive facts

1. **Source fact—observational geometry and timing.** ER was spin-synchronous; the energy supply swept 32 times per spin, two sweeps were summed, fixed-angle samples from a half-spin were combined from 128 oversampled angular bins into 88 angular bins, and the retrace energy was discarded, leaving 15 energies. A real-time 3-D distribution is an \(88\times15\) array accumulated during one half-spin and read out once per 16 spins. `[Bundle: primary_sources/magerfmt.txt, lines 23–62; magerfmt.pdf, pp. 1–3]` The 1320 code bytes are ordered as 15 groups of 88 angles in ascending energy-step order. `[Bundle: primary_sources/magerfmt.txt, lines 551–560; magerfmt.pdf, p. 16]` The 3-D half-spin is the interval immediately preceding the computed accumulation end time \(T_{\rm Acc}\). `[Bundle: primary_sources/magerfmt.txt, lines 680–697; magerfmt.pdf, p. 20]`

2. **Inference—statistical unit.** The 15 rows are sequential components of one sweep, not 15 independent experiments. The angular cells share the same accumulation, response state, total-rate environment, and possible FIFO/dead-time losses. This agrees with the repository measurement contract. `[Bundle: current_branch/docs/architecture/er_measurement_contract.md, lines 308–320]`

3. **Source fact—lossy compressor.** All ER counter data are compressed to eight bits; the format is a four-bit exponent and four-bit hidden-bit mantissa; omitted mantissa bits are truncated, not rounded; and the tabulated decompressed value is the minimum accumulator value capable of producing that code. `[Bundle: primary_sources/magerfmt.txt, lines 326–328 and 964–971; magerfmt.pdf, pp. 10 and 27]` For \(c=16e+m\),

\[
L(c)=\begin{cases}
m,&e=0,\\
(16+m)2^{e-1},&e>0.
\end{cases}
\]

For an ordinary nonterminal unprescaled code, \(K\in\{L(c),\ldots,L(c+1)-1\}\). The table reaches code `FF`, whose documented lower endpoint is 507,904, but the document does not supply a state-specific terminal upper bound or an explicit overflow/saturation rule sufficient for a terminal-bin likelihood. `[Bundle: primary_sources/magerfmt.txt, lines 964–991; magerfmt.pdf, p. 27]`

4. **Source fact—scope of prescaling.** The document states that *pitch-angle data (not cuts)* may be prescaled, automatically or by command, in eight energy bands; the trigger is a prior pitch-angle sample above 524,287, and the affected values are divided by eight before compression. `[Bundle: primary_sources/magerfmt.txt, lines 342–351; magerfmt.pdf, p. 10]` The header description narrows this to the longer real-time pitch-angle accumulations and explicitly says high-time-resolution and cut data are not prescaled. `[Bundle: primary_sources/magerfmt.txt, lines 503–531; magerfmt.pdf, p. 15]`

5. **Inference—3-D prescale qualification.** The cited passages do not state that the \(88\times15\) 3-D array is prescaled by eight. A general observation operator should allow state-specific prescale where independently documented, but the 3-D code model must not apply pitch-angle prescale bits by analogy. This is a material qualification to the generic interval formula in the orientation memo.

6. **Source fact—diagnostic counters and losses.** The “Total Counts” header counter excludes masked-anode events and FIFO-overrun losses and is divided by 16 before compression. The “Rate Counter” further differs through pulse-height rejection, approximately 300 ns processing dead time, FIFO loss, and masks; it is prescaled in hardware and software. `[Bundle: primary_sources/magerfmt.txt, lines 532–550; magerfmt.pdf, pp. 15–16]` These counters are not exact sums of the 3-D cells and cannot be substituted as a row total without a proved state-specific relationship.

7. **Source fact—response gaps.** The full-instrument geometric factor is described approximately as \(0.07E_{\rm keV}\;\mathrm{cm^2\,sr\,keV}\), spread over 256 effective look directions, with significant anode-to-anode variation due to ADC differential nonlinearity and MCP gain. The anode efficiency table \(F\) is explicitly `TBS`; exact onboard phase-space arithmetic and rounding are referred to an “Appendix TBS.” `[Bundle: primary_sources/magerfmt.txt, lines 698–721; magerfmt.pdf, p. 20]` The introduction also points to a detailed instrument/operation description as `<TBS>`. `[Bundle: primary_sources/magerfmt.txt, lines 4–8; magerfmt.pdf, p. 1]`

8. **Source fact—Level-0 time and merge process.** The archived merged stream chooses between real-time and delayed records by matching timestamps and signal-to-noise, defaulting to real time in specified ties; delayed VCDU/ERT values are corrected by subtracting 3200 s; three selection/status flags are appended. `[Bundle: primary_sources/lvl0sis.txt, lines 492–523; lvl0sis.pdf, pp. 13–14]` A record is 472 bytes; MAG/ER occupies bytes 287–454; Earth Receive Time occupies bytes 457–462 and is the station receive time of the first sync bit, not the electron event time. `[Bundle: primary_sources/lvl0sis.txt, lines 524–626; lvl0sis.pdf, pp. 13–15]`

## 2.2 Calibrated-product facts

9. **Source fact—declared product.** The public 3-D product consists of 88 approximately equal solid-angle bins and 15 energy channels, with stated energy resolution \(\Delta E/E=25\%\); a spectrum is represented by 15 consecutive table records. `[Bundle: primary_sources/data-3deleflux-desc.txt, lines 19–48 and 149–166]` The rows are not simultaneous: the look azimuth changes as the energy sweep proceeds. `[Bundle: primary_sources/data-3deleflux-desc.txt, lines 41–48]`

10. **Source fact—processing chain.** The product description says processing decompresses raw counts, uses accumulation times depending on energy and solid angle, corrects for dead time using approximately \(0.3\,\mu\mathrm{s}\) and \(1/(1-RT)\), masks values when that factor exceeds 1.25, retains an approximately 10 counts/s cosmic-ray/noise background, and divides by \(0.02\;\mathrm{cm^2\,sr}\) and channel-center energy to obtain differential flux. `[Bundle: primary_sources/data-3deleflux-desc.txt, lines 80–114]` The low- and high-resolution product descriptions repeat this chain. `[Bundle: primary_sources/data-lowresflux-desc.txt, lines 69–98; primary_sources/data-highresflux-desc.txt, lines 69–100]`

11. **Source fact—scientific cautions.** Sunlight can contaminate the distribution; spacecraft charging shifts incoming energies and can bend low-energy trajectories; no correction for spacecraft-potential effects is applied to the archived product. `[Bundle: primary_sources/data-3deleflux-desc.txt, lines 256–281]`

12. **Source fact—reported uncertainty is incomplete.** The electron-product uncertainty description includes Poisson counting and digitization noise but explicitly excludes uncertainty in the absolute geometric factor, whose estimate depends on electrostatic optics, grids, and MCP efficiency. `[Bundle: primary_sources/data-electron-desc.txt, lines 104–112]` It does not provide the covariance or state-specific response coefficients needed for a likelihood.

13. **Source fact—missing-value sentinel.** The 1999 detached label defines `9.999E-99` as a missing constant, and a spectrum is 15 records. Its per-row timestamps are incremented from a spectrum timestamp using an approximate spin rate. `[Bundle: reference_data/1999_pr_day/3D990429.LBL, lines 27–40, 75–98, 117–120, and 156–165]`

## 2.3 Repository behavior and reproducible checks

14. **Source fact—current code is a structural extractor.** `decompress_er_counter` implements \(L(c)\), and the iterator assembles supported Digital Subcom versions into \(15\times88\) code arrays while rejecting incomplete, duplicate, mixed, out-of-order, or unsupported sequences. The module explicitly disclaims calibration, event-time reconstruction, exposure, dead-time, and quality interpretation. `[Bundle: current_branch/src/losscone/level0.py, module lines 1–6; function decompress_er_counter, lines 126–140; function iter_level0_er3d_sweeps, lines 160 onward; current_branch/tests/test_level0.py, lines 85–174]`

15. **Source fact—calibrated pseudo-counts are reconstructed, not observed.** The data loader can convert calibrated flux to rounded “counts” using fixed accumulation/geometric constants. `[Bundle: current_branch/src/losscone/er_data.py, lines 119–180; current_branch/src/losscone/config.py, relevant count-column constants around line 136]` This is downstream algebra and does not recover \(K\).

16. **Source fact—current loss-cone implementation.** The hard loss-cone boundary uses a clipped expression of the form \(\sin^2\alpha_c=r\{1+U_{\rm surface}/(E-U_{\rm spacecraft})\}\), evaluates a finite pitch grid, and notes a TODO for physically invalid \(E<U_{\rm spacecraft}\). `[Bundle: current_branch/src/losscone/model.py, lines 42–70 and 118–229]` The model’s normalized-flux preprocessing shares row normalizers, inducing dependence among angular coordinates. `[Bundle: current_branch/src/losscone/normalization.py, relevant row-normalization functions]`

17. **Source fact—raw/calibrated association is explicitly forensic.** `raw_calibration.py` states that it ranks time-near raw/code and calibrated/flux candidates from response-independent angular-shape evidence, does not reconstruct detector counts or calibrated flux, and must not be used as a likelihood input. It reads only complete 15-row calibrated sweeps; its score decompresses codes only to lower endpoints, separately centers and scales the log angular pattern within each energy row, excludes the documented positive missing sentinel, and therefore cannot determine absolute response or energy-dependent exposure. Its time coordinate is ERT minus calibrated sweep-end time and is explicitly not event-time correspondence. `[Bundle: current_branch/src/losscone/raw_calibration.py, lines 1–6, 23, 51–75, 78–121, and 124–176]` The associated tests verify the documented merged-record ERT byte offsets, scale-free rowwise scoring, and preference for a shape-matching candidate over a time-nearer decoy; they do not validate a physical latency or count/flux calibration. `[Bundle: current_branch/tests/test_raw_calibration.py, lines 19–87]`

18. **Reproducible finding—1998 raw data.** Script `lp_er_audit/audit_reference.py` read `reference_data/1998_anchor/M9801523.B` and found 7,047 records and 165 accepted complete format-1/2 3-D sweeps, matching the repository contract’s accepted count. Observed codes ranged from 0 to 153; no `FF` code occurred; the lower-endpoint range was 0–6,400; 45.8636% of cells had code zero. Code 153 corresponds to the ordinary interval \([6400,6655]\), not count 6400. The first accepted ERT was 1998-01-15 23:44:36.266 UTC and the last 1998-01-16 03:36:16.320 UTC. `[Reproducible script/output: /mnt/data/lp_er_audit/audit_reference.py; reference_audit.json; no random sampling]`

19. **Reproducible finding—1999 decoder gap.** The same structural parser yielded no 3-D sweep from `M9911914.B`. A direct frame audit found Digital Subcom version 5 in 455 relevant observations. The instrument document maps post-May-1998 “format 3” to version code 3, but the bundle contains no verified statement mapping version 5 to a 3-D byte layout. `[Bundle: primary_sources/magerfmt.txt, lines 451–496; magerfmt.pdf, pp. 14–15]` Therefore “zero parsed sweeps” means “unsupported state,” not “no 3-D telemetry.”

20. **Reproducible finding—calibrated missingness.** The 1998 calibrated file has 1,032 complete 15-row spectra and 1,320 sentinel cells, comprising one all-sentinel spectrum whose timestamps are 1970 after parsing. The 1999 file has 1,055 spectra and 2,658 sentinel cells: two all-sentinel spectra plus 18 sparse sentinel cells. PR 14’s `finite & flux>0` mask retains every positive sentinel. `[Reproducible script/output: /mnt/data/lp_er_audit/audit_reference.py and sentinel_patterns.py; reference_audit.json and sentinel_patterns.txt]`

21. **Reproducible finding—coarse association.** The 1998 raw/calibrated shape-matching audit produced 2,084 time-near pairs; the best observed scale-free angular correlation was 0.842 and the dominant ERT-minus-calibrated-sweep-end offset was approximately \(-63.716\) s. This reproduces the repository contract’s coarse association but is not an event-time reconstruction. `[Bundle: current_branch/docs/architecture/er_measurement_contract.md, lines 234–266; reproducible output: /mnt/data/lp_er_audit/reference_audit.json]`

## 2.4 Material conflicts and gaps

* **Energy response.** The calibrated-product description states \(\Delta E/E=25\%\), while PR 14 defaults to `de_over_e=0.5` and uniform quadrature over that width. `[Bundle: primary_sources/data-3deleflux-desc.txt, lines 19–26; pr14/src/losscone/quasi_likelihood.py, lines 30–46; pr14/src/losscone/response_folded.py, lines 39–63]` The bundle does not contain a state-specific response kernel that resolves this into a probability law. A width parameter is not a response matrix.

* **Geometric factors.** The telemetry document’s approximate full-instrument \(0.07E_{\rm keV}\) factor and the calibrated product’s \(0.02\;\mathrm{cm^2\,sr}\) conversion serve different levels of the chain and are not interchangeable. `[Bundle: primary_sources/magerfmt.txt, lines 698–704; primary_sources/data-3deleflux-desc.txt, lines 101–114; current_branch/docs/architecture/er_measurement_contract.md, lines 220–228]`

* **Prescale.** The documented prescale is for specified pitch-angle accumulations, not established for 3-D cells. A 3-D likelihood cannot use the expanded \(8L(c)\) interval without a 3-D state source.

* **Terminal code.** The lower endpoint is documented; the terminal upper endpoint, counter width, overflow behavior, and whether `FF` is saturation or simply the last representable logarithmic bin are not specified well enough for an exact terminal likelihood.

* **State/version.** The current parser’s version-1/2 support is adequate for the 1998 anchor but not a mission-wide decoder. Version 5 must be mapped from a commanding/interface or later format document before the 1999 raw record can enter inference.

* **Timing.** ERT and calibrated row timestamps do not by themselves identify each cell’s physical integration window. The header, spin period, phase, merge selection, and latency must be decoded and validated.

* **Response/loss.** Effective exposures, anode efficiencies, masks, background, dead-time, FIFO, pulse-height rejection, selection, and their uncertainty/covariance are not available at the cell/state level.

These are not minor nuisance omissions. Sections 5–6 prove that unconstrained versions of these unknowns can exactly absorb a change in \(U\).

---

# 3. Notation, probability space, estimand, and observation operators

## 3.1 Probability space and indices

Let \((\Omega,\mathcal F,P)\) support all incident particles, instrument randomness/state, telemetry, ground processing, and any sampling/selection mechanism. A cluster \(i\) is one 3-D sweep or, when dependence persists, a prespecified visit containing several sweeps. Within a sweep, \(e\in\{1,\ldots,15\}\) indexes sequential energy rows and \(a\in\{1,\ldots,88\}\) indexes 3-D angular cells; write \(j=(e,a)\), \(d=1320\). The physical integration region for cell \(j\) in cluster \(i\) is \(A_{ij}\subset\mathbb R_t\times\mathbb R_+\times\mathbb S^2\times\mathcal M\), where the last component contains marks such as anode, pulse height, trajectory class, and event-processor state.

## 3.2 Latent incident process

Let

\[
X_i=\sum_n \delta_{(T_{in},E_{in},\Omega_{in},M_{in})}
\]

be a marked incident-electron random measure at a declared physical reference surface, preferably the instrument aperture. It need not be Poisson. Its law belongs to a physical class \(\mathfrak X(U,\beta)\), where \(U\) is the target and \(\beta\) contains plasma distribution, magnetic mirror ratio, beam, background environment, and other physical nuisance.

The physical forward operator maps the incident law to an ideal cell-specific intensity or set of moments. A generic first-moment representation is

\[
\lambda_{ij}(U,\beta)=
\int_{A_{ij}} f_{U,\beta}(t,E,\omega,m)\,dt\,dE\,d\omega\,dm,
\]

but no Poisson assumption is implicit in this notation.

## 3.3 Instrument state and accepted accumulator

Let \(S_i\) be all observed or latent state relevant to sweep \(i\): software/format version, command state, spin phase and period, energy table, masks, prescale, gain, merge flags, timing, and quality flags. Let \(\eta\) contain response/loss nuisance: exposure, energy/angular response, anode efficiency, geometric factor, background, pulse-height rejection, dead time, FIFO, accumulation arithmetic, saturation, and selection.

The instrument maps \(X_i\) to the integer vector

\[
K_i=(K_{ij}:j=1,\ldots,d)\in\mathbb N_0^d,
\]

where \(K_{ij}\) is the accepted integer accumulator immediately before telemetry compression. It is not the number of incident arrivals unless the response/loss model proves that equality. A fully specified count-process model supplies

\[
p_{K,i}(k\mid U,\beta,\eta,S_i).
\]

## 3.4 Code operator

The telemetry code is deterministic conditional on state:

\[
C_i=q_{S_i}(K_i)\in\{0,\ldots,255\}^d.
\]

For a scalar code define the inverse set

\[
I_s(c)=\{k\in\mathbb N_0:q_s(k)=c\}.
\]

For an ordinary nonterminal unprescaled cell, \(I_s(c)=\{L(c),\ldots,L(c+1)-1\}\). Prescale and terminal behavior are state-specific, not generic. The exact code law is the push-forward

\[
P(C_i=c\mid U,\beta,\eta,S_i)
=\sum_{k\in I_{S_i}(c)}p_{K,i}(k\mid U,\beta,\eta,S_i),
\]

where \(I_s(c)=\prod_jI_{s,j}(c_j)\) only when the compressor acts componentwise. This sum does not factor unless the \(K_{ij}\) law factors.

## 3.5 Calibrated flux operator

Let \(Z_i\) denote downstream ancillary inputs and ground state. The archived calibrated output is

\[
Y_i=G(C_i,S_i,Z_i;\gamma,R_i),
\]

where \(\gamma\) are calibration parameters and \(R_i\) may include rounding, missingness, and masks. The public description establishes that \(G\) includes decompression, rate/exposure conversion, dead-time correction/masking, retained background, and geometric/energy normalization; it does not provide a complete deterministic implementation or stochastic law. If the observation is declared to be \(Y\), the likelihood must be the push-forward distribution of this full operator. A Gaussian or lognormal criterion may be used only as an explicitly assumed working law or estimating restriction.

The declared observable \(O\) must be one of:

* \(O=(C,S,Z)\), when raw code state is decoded;
* \(O=(Y,S,Z)\), when only calibrated flux is used;
* \(O=(C,Y,S,Z)\), only after a validated association identifies them as the same physical sweep and specifies their timing relationship.

No theorem below treats \(L(C)\) or a rounded inverse of \(Y\) as observed \(K\).

## 3.6 Estimand

Fix a sign convention before analysis. Let

\[
U=\Phi_{\rm surface}-\Phi_{\rm spacecraft}
\]

in volts, interpreted through the electron-energy convention used by the forward model. Equivalently, the barrier often denoted \(D=\Phi_{\rm spacecraft}-\Phi_{\rm surface}\) is \(-U\). The estimand is the value—or identified set of values—of this **relative shift** entering the loss-cone/trajectory operator. If the physical map depends only on \(\Phi_{\rm surface}-\Phi_{\rm spacecraft}\), the absolute pair \((\Phi_{\rm surface},\Phi_{\rm spacecraft})\) is identified only up to common translation:

\[
(\Phi_{\rm surface},\Phi_{\rm spacecraft})
\sim(\Phi_{\rm surface}+h,\Phi_{\rm spacecraft}+h).
\]

Claims about absolute lunar potential require an independent spacecraft-potential reference and are outside the identified target considered here.

## 3.7 Model classes and honest coverage

For a model class \(\mathcal M\), let

\[
\mathcal P_u^O=\{\operatorname{Law}_{u,\nu}(O):\nu\in\mathcal H_u,\ (u,\nu)\in\mathcal M\}.
\]

A confidence set \(\mathcal C_\alpha(O)\subseteq\mathcal U\) is **finite-sample honest over \(\mathcal M\)** if

\[
\inf_{u\in\mathcal U}\inf_{\nu\in\mathcal H_u}
P_{u,\nu}\{u\in\mathcal C_\alpha(O)\}\ge1-\alpha.
\tag{3.1}
\]

It is **uniformly asymptotically honest** for a sequence of experiments if the liminf of the left side is at least \(1-\alpha\). Pointwise asymptotic validity fixes \((u,\nu)\) before taking the limit and is weaker; it will be labeled as such. A sensitivity interval obtained by varying nuisance values without a coverage theorem is not called a confidence set.

---

# 4. Assumption lattice and comparison tables

There is no unique “weakest” stochastic model. Response knowledge, arrival dependence, loss dynamics, replication, selection, and contamination are separate axes. For example, a known nonparalyzable-dead-time model is neither a subset nor a superset of a shared-gamma Cox model; finite-range dependence is incomparable with infinite exchangeability. The useful partial order is coordinatewise:

\[
\begin{aligned}
&\text{known response/state}\subset\text{finite-dimensional or bounded response}\subset\text{unrestricted response},\\
&\text{fully specified law}\subset\text{specified family}\subset\text{moments/compensator only}\subset\text{support only},\\
&\text{predeclared/known selection}\subset\text{bounded/MAR selection}\subset\text{unrestricted outcome-dependent selection},\\
&\epsilon=0\subset\text{fixed small robust neighborhood}\subset\text{unbounded contamination}.
\end{aligned}
\]

A weakening in any axis can move an otherwise identifiable model from A to B or C.

## 4.1 Identification, objective, and coverage status

| Class | Substantive assumptions and observable law | Identification of \(U\) | Honest inference and status | Derived objective |
|---|---|---|---|---|
| M1. Independent Poisson, known state/response | Conditional on decoded \(S_i\), \(K_{ij}\) independent Poisson with known mean map \(\mu_{ij}(u,\beta)\); exact code intervals known; selection fixed | Point identified iff the profiled mean/code-law sets for distinct \(u\) are disjoint; local rank required | **A mathematically**: finite-sample least-favorable test inversion. **D in current bundle** because response/state is incomplete | Exact interval-code log likelihood; composite-null LR or score statistic |
| M2a. Poisson with compact estimable response nuisance | Same count law; \(\eta\) finite-dimensional/compact, tied across cells/states, with injective profiled law | Point ID if \(\{P_{u,\nu}\}\) are disjoint and uniformly separated | **A** with least-favorable or Berger–Boos calibration; may be wide/full near weak ID | Profile or minimax interval-code likelihood; nuisance suprema |
| M2b. Poisson with bounded response/calibration | \(a_j,b_j\) and response kernels lie in validated coupled bounds | Usually partial ID; point ID only if bounds still separate \(u\) | **B**: exact/finite-sample outer region for identified set; **D currently** because no bounds are supplied | Feasibility over mean/code-probability bands; moment inequalities or robust LR |
| M3a. Shared scalar Cox/mixed Poisson | \(K_{ij}\mid A_i\) independent Poisson\((A_i\mu_{ij})\); mixing family specified or compact, common within sweep | Shape can point-identify \(u\) if response and profiled mixed law are injective | **A** by full integral likelihood and LF inversion; asymptotic cluster methods also possible | Gamma-mixed interval likelihood or conditional/full cluster likelihood |
| M3b. Arbitrary vector Cox mixing | Cell-specific random intensity vector unrestricted and may depend on \(u\) through nuisance | Not identified: the random intensity law can be held fixed for every \(u\) | **C** | No nontrivial objective has uniform identification; require shared/low-rank mixing or known mixing law |
| M4. Poisson-binomial / heterogeneous thinning | Known or bounded number of trials/arrival opportunities and event-specific efficiencies; dependence across trials specified | Point ID with known efficiencies and rank; partial with bounds; none with arbitrary cellwise thinning | **A/B/C** according to response restriction | Interval sums of Poisson-binomial PMF computed by DP/FFT; robust feasibility under bounds |
| M5. Negative-binomial or explicit overdispersion family | Mean map fixed; dispersion \(r\) known or bounded below and profiled on compact set | Point ID if mixed code laws separate; uniform ID fails as \(r\downarrow0\) | **A** for \(r\ge r_{\min}>0\) with LF inversion; **C** for unrestricted \(r\downarrow0\) if useful uniform inference is required | Negative-binomial interval CDF differences; profile over \(r\) |
| M6a. Known renewal/refractory/dead-time/FIFO model | Arrival renewal law and non/paralyzable dead time or finite-state queue dynamics specified; state/exposure known | Point ID if accepted-count/code law injective | **A** with full transition/renewal likelihood and LF inversion | Renewal-tail differences or Markov-additive matrix/PGF rectangle probabilities |
| M6b. Bounded or unrestricted loss process | Known loss envelope gives accepted-mean/code bounds; unrestricted cell-specific loss may mimic \(U\) | Partial ID under bounds; non-ID if arbitrary | **B** under validated bounds; **C** if unrestricted; **D currently** for actual bounds/state | Moment/code-probability inequalities or minimax likelihood |
| M7. Clustered, finite-range dependent, sweep random effects | Sweeps/visits are independent or mixing clusters; arbitrary within-cluster dependence; common \(u\) or specified regression; bounded moments/covariance | Mean/moment point ID if rank; otherwise partial | **A** finite-sample with bounded-feature concentration; **A** uniformly asymptotic GMM under a uniform CLT; a single arbitrary sweep is **C** | Cluster estimating equations, GMM/empirical likelihood, or finite-sample moment box |
| M8. Martingale/counting-process restrictions | Observable counting path or interval observations have known compensator family; bounded jumps and predictable variance/tail bounds | Point/partial ID through compensator moments if rank | **A/B** using martingale concentration/test inversion; **D for LP** because the path, compensator, and terminal bounds are not available | Martingale score, predictable quadratic-variation statistic, concentration bound |
| M9. Exchangeability | Finite exchangeability only, or infinite exchangeability with unrestricted de Finetti mixing | Finite exchangeability alone does not identify a law beyond symmetry; unrestricted mixing can absorb \(u\) | **C** unless a parametric/shared mixing or moment restriction is added | Permutation inference only for a sharp invariant null; not a likelihood for \(U\) by itself |
| M10. Conditional-mean estimating equations with variance bounds | \(E\{m_i(u,\nu)\}=0\), independent/mixing clusters, rank, bounded covariance/eigenvalues; no full distribution | Point ID if zero set singleton; partial otherwise | **A** finite-sample if features bounded and concentration used; otherwise uniformly asymptotic only under explicit uniform CLT | GMM, Anderson–Rubin-type infimum statistic, cluster multiplier/subsampling |
| M11. Moment equalities/inequalities | Response and code interval imply \(E h(O)\in\mathcal M(u)\); no parametric count law | Usually partial ID; point ID if moment sets disjoint | **B**, or **A** if singleton and separated, using simultaneous moment bounds | Set-membership feasibility; generalized moment-selection/minimax statistic |
| M12a. Fixed distributionally robust neighborhood | \(P\) lies in a known TV, Hellinger, KL, Wasserstein, or contamination ball around a nominal observation law; radius below separation | Point/partial ID depending whether neighborhoods overlap | **A/B** with minimax tests and calibrated radius | \(\inf\)-power/\(\sup\)-size robust test; convex dual where available |
| M12b. Unbounded contamination or overlapping neighborhoods | Contamination radius unrestricted, or \(\mathcal N(P_u)\cap\mathcal N(P_v)\ne\varnothing\) | Not uniformly distinguishable | **C** | Smallest repair: radius smaller than half the relevant TV separation, plus response restrictions |
| M13. Support plus known deterministic compressor only | Any law on supported code arrays; no response, moment, selection, or replication restriction | Not identified; common all-zero law for every \(u\) | **C**, with a sharp triviality lower bound | No informative test; an honest set must be essentially full on common laws |

## 4.2 Computational burden, required inputs, and diagnostics

| Class | Computational burden | Missing calibration/state inputs | Bundle-based falsification or stress diagnostic |
|---|---|---|---|
| M1/M2 | \(O(nd)\) per likelihood evaluation plus nuisance optimization; Poisson CDF tails | Exact intervals, state/version, cell exposure/response, masks/loss/background, event timing | Code histograms by state; zero/tail frequencies; 3-D versus aggregate products; response-residual maps |
| M3 | Shared-gamma: one quadrature dimension per cluster/evaluation; arbitrary mixing intractable without structure | Same as M1 plus mixing-family validation | Cross-cell covariance and overdispersion across repeated stable-state sweeps; row-total/shape dependence |
| M4 | DP \(O(Nd)\), FFT \(O(N\log N)\), or saddlepoint approximations | Trial/arrival opportunity model and heterogeneous efficiencies | Variance-to-mean and upper-support diagnostics; efficiency-linked heterogeneity |
| M5 | Similar to M1 using NB CDFs; nuisance dispersion profile | Response plus validated dispersion lower bound/state relation | Zero inflation and tail ratios across comparable sweeps; stability of estimated \(r\) |
| M6 | Renewal tails cheap for simple dead time; finite FIFO can require matrix exponentials/PGFs | Exact dead time, exposure, queue depth/service, rejection/mask state, input-arrival model | Total/rate counter discrepancies; saturation/dead-time strata; rate dependence of residuals |
| M7/M10 | \(O(nR^2)\) covariance/GMM; bootstrap \(O(BnR)\) plus refits | Repeated stable-state design, response-derived moments, clustering/mixing scale | Sweep autocorrelation, block-length sensitivity, covariance eigenvalues, leave-visit-out analysis |
| M8 | Low if bounded martingale features observed; otherwise unavailable | Event/accepted-count path or certified compensator and variance bounds | Residual martingale plots, predictable-variation checks, time-rescaling where timestamps exist |
| M9 | Permutations can be cheap; not identifying by itself | Justification of exchangeable units and an identifying invariant statistic | Order/epoch effects, state imbalance, finite-population constraints |
| M11 | Feasibility/convex optimization over \(u,\nu\); may be mixed-integer for hard masks | Numerical response/calibration bounds and terminal upper bounds | Bound sensitivity; active inequalities; cells driving exclusion; held-out aggregate checks |
| M12 | Robust optimization, often convex for fixed \(u\), costly with nuisance and simulation | Metric/radius tied to calibration or validation evidence | Radius stress curve; adversarial response perturbations; least-favorable law diagnostics |
| M13 | Trivial computationally | All scientifically identifying information is absent | All-zero/common-law construction is already decisive |

The classification applies to a declared experiment. Pooling 1,032 daily spectra does not automatically create \(n=1{,}032\) replicates for one per-sweep \(U\): a common-parameter, regression, or local-stationarity assumption is substantive and must be recorded.

---

# 5. Identification analysis and proofs

## 5.1 Definitions

For two parameter/nuisance pairs \((u,\nu)\) and \((v,\xi)\), write

\[
(u,\nu)\equiv_O(v,\xi)
\quad\Longleftrightarrow\quad
\operatorname{Law}_{u,\nu}(O)=\operatorname{Law}_{v,\xi}(O).
\]

The values \(u\) and \(v\) are **observationally equivalent over \(\mathcal M\)** if \(\mathcal P_u^O\cap\mathcal P_v^O\ne\varnothing\). Given a complete observable law \(Q\), the identified set is

\[
\Theta_I(Q)=\{u\in\mathcal U:Q\in\mathcal P_u^O\}.
\tag{5.1}
\]

Point identification means \(\Theta_I(Q)=\{u_0\}\) for every admissible true law \(Q\). Partial identification means the set can contain more than one value but is smaller than \(\mathcal U\). Nonidentification in the strongest sense occurs when a common law \(Q\) belongs to \(\mathcal P_u^O\) for every \(u\).

For a metric \(d_{\rm TV}\), the experiment is **uniformly distinguishable at separation \(\varepsilon>0\)** if

\[
\Delta(\varepsilon)=
\inf_{|u-v|\ge\varepsilon}
\inf_{P\in\mathcal P_u^O,Q\in\mathcal P_v^O}
 d_{\rm TV}(P,Q)>0.
\tag{5.2}
\]

Point identification does not imply (5.2). The negative-binomial boundary in Proposition 5.8 below is an explicit counterexample.

## 5.2 Quantized-observation lemma

**Lemma 5.1 (push-forward through the ER compressor).**  
Fix a state \(s\), and let \(q_s:\mathbb N_0^d\to\{0,\ldots,255\}^d\) be the deterministic compressor, including any state-specific prescale, saturation, and component mapping. For every probability mass function \(p_K\),

\[
P(C=c\mid s)=\sum_{k:q_s(k)=c}p_K(k\mid s).
\tag{5.3}
\]

If compression is componentwise, the inverse set is a rectangle \(I_s(c)=\prod_jI_{s,j}(c_j)\), but the probability in (5.3) factors only if the corresponding \(K_j\) are independent.

For the documented ordinary, unprescaled, nonterminal scalar code \(c<255\),

\[
I(c)=\{L(c),L(c)+1,\ldots,L(c+1)-1\}.
\tag{5.4}
\]

If a documented state first applies integer division \(\lfloor K/d\rfloor\) with \(d=8\), then for a nonterminal code

\[
I_d(c)=\{dL(c),dL(c)+1,\ldots,dL(c+1)-1\}.
\tag{5.5}
\]

For a terminal code, the inverse is \(\{L(255),\ldots,K_{\max}(s)\}\) only if a finite state-specific counter maximum and saturation rule are known; it is \(\{L(255),L(255)+1,\ldots\}\) if the code is documented to collect all larger values; and it is otherwise not identified from the supplied compressor table.

**Proof.** Because \(C=q_s(K)\), the event \(\{C=c\}\) is exactly the disjoint union \(\bigcup_{k:q_s(k)=c}\{K=k\}\). Countable additivity gives (5.3). The source states that untransmitted bits are truncated and that a tabulated value is the minimum count producing a code, with values below the next code producing the same current code. This is exactly (5.4). If \(q_{s,d}(k)=q(\lfloor k/d\rfloor)\), then \(q_{s,d}(k)=c\) iff \(L(c)\le\lfloor k/d\rfloor\le L(c+1)-1\), which is equivalent to \(dL(c)\le k\le dL(c+1)-1\), proving (5.5). The terminal statements are definitions of the inverse image under the three possible documented terminal rules. No formula for a nonterminal next code can establish which terminal rule applies. \(\square\)

**Instrument qualification.** Equation (5.5) is valid only for an accumulator proved to use that prescale. The bundle documents the factor-eight state for specified pitch-angle accumulations, not for the 3-D array. `[Bundle: primary_sources/magerfmt.txt, lines 342–351 and 503–531]`

**Corollary 5.1.1 (lower endpoints do not identify accepted counts or their mean).**  
Suppose some observed code \(c\) has \(|I_s(c)|>1\). There are two conditional laws for \(K\) with the same code law but different conditional means: put all mass at \(\min I_s(c)\) in one law and at \(\max I_s(c)\) in the other. If a terminal inverse set is unbounded and has positive code probability, the conditional mean of \(K\) can be made arbitrarily large, or infinite, without changing the code law.

**Proof.** Both point masses map deterministically to \(c\), but their means differ. For an unbounded terminal inverse, assign its conditional mass to an arbitrarily large point, or to a heavy-tailed distribution with infinite mean, while retaining the same terminal code probability. \(\square\)

This corollary rules out treating \(L(C)\) as exact \(K\), and it rules out estimating \(E K\) from code support alone.

## 5.3 Identification under a known-response Poisson baseline

Let \(\nu=(\beta,\eta)\), and suppose for the moment that the state and response are known except for \(\nu\). For cell \(j\), let \(d_j\ge1\) denote a documented integer prescale divisor, with \(d_j=1\) ordinarily.

**Theorem 5.2 (identification from interval-coded Poisson marginals).**  
Assume:

1. conditional on \((u,\nu,S=s)\), each \(K_j\) has marginal distribution \(\operatorname{Pois}(\mu_j(u,\nu,s))\), with \(0<\mu_j<\infty\);
2. code zero has inverse \(\{0,\ldots,d_j-1\}\) for cell \(j\);
3. the complete code distribution is known.

Then every \(\mu_j\) is identified from the marginal code law. Specifically,

\[
P(C_j=0)=F_{\operatorname{Pois}(\mu_j)}(d_j-1),
\tag{5.6}
\]

and the right side is strictly decreasing from 1 to 0 as \(\mu_j\) ranges from 0 to \(\infty\).

If, in addition, the \(K_j\) are independent conditional on \((u,\nu,s)\), define the profiled mean set

\[
\mathcal M_u(s)=\{\mu(u,\nu,s):\nu\in\mathcal H_u\}\subset(0,\infty)^d.
\tag{5.7}
\]

Then \(u\) is point identified from the code law if and only if \(\mathcal M_u(s)\cap\mathcal M_v(s)=\varnothing\) for every \(v\ne u\).

If \(\mathcal U\times\mathcal H\) is compact, \((u,\nu)\mapsto P_{u,\nu}^C\) is continuous, and its profiled law sets are disjoint for all \(|u-v|\ge\varepsilon\), then \(\Delta(\varepsilon)>0\).

**Proof.** Let \(m=d_j-1\). The Poisson CDF is

\[
F_m(\mu)=e^{-\mu}\sum_{k=0}^m\frac{\mu^k}{k!}.
\]

Differentiating the finite sum gives

\[
F_m'(\mu)
=-e^{-\mu}\sum_{k=0}^m\frac{\mu^k}{k!}
+e^{-\mu}\sum_{k=1}^m\frac{k\mu^{k-1}}{k!}
=-e^{-\mu}\frac{\mu^m}{m!}<0
\]

for \(\mu>0\). Also \(F_m(0)=1\) and \(F_m(\mu)\to0\). Thus (5.6) has a unique inverse and identifies each mean.

Under conditional independence the complete latent law, and hence the code law, is determined by \(\mu\). If two profiled mean sets intersect, choose \(\nu,\xi\) giving the same mean vector; the independent Poisson laws and their code push-forwards are identical, so \(u\) is not identified. Conversely, equality of code laws implies equality of each marginal zero probability, hence equality of every mean. Disjoint mean sets therefore imply distinct \(u\).

For the final claim, the set

\[
A_\varepsilon=\{(u,\nu,v,\xi):|u-v|\ge\varepsilon\}
\]

is compact. On the finite code sample space, total variation is one half the finite sum of absolute differences of probability masses, hence is continuous in the law. Its value is strictly positive everywhere on \(A_\varepsilon\) by disjointness. A continuous positive function on a compact set has a positive minimum. \(\square\)

**Corollary 5.2.1 (local rank condition).**  
Suppose \(\mu(u,\nu)\) is continuously differentiable and the Jacobian

\[
D_{(u,\nu)}\mu=[\partial_u\mu,\,D_\nu\mu]
\]

has full column rank at \((u_0,\nu_0)\). Then the parameter pair, and hence \(u\), is locally identified from the complete code law.

**Proof.** Full column rank means a square subvector of \(\mu\) has nonsingular derivative with respect to \((u,\nu)\). The inverse-function theorem makes that subvector locally one-to-one. Theorem 5.2 identifies the subvector from the code law. \(\square\)

A weaker necessary condition for local identification of \(u\) after profiling is \(\partial_u\mu\notin\operatorname{span}(D_\nu\mu)\). If the derivative lies in the nuisance tangent, first-order changes in \(u\) can be absorbed by nuisance changes and weak identification should be expected.

## 5.4 Quantization does not by itself destroy identification

Theorem 5.2 uses only the probability of code zero. Thus wide upper bins reduce Fisher information and practical power, but ordinary interval quantization does not automatically erase point identification. Conversely, identification of a mean from a complete code law does not imply useful finite-sample precision: if almost all probability lies in one code, distributions for separated means can be very close.

The same monotonicity argument extends to many specified mixed-Poisson families. If \(K\mid A\sim\operatorname{Pois}(A\mu)\), \(A>0\) almost surely, and the law of \(A\) is known, then

\[
P(C=0)=E\{F_{d-1}(A\mu)\}.
\]

This probability gives a strictly decreasing continuous function of \(\mu\), because differentiation under the integral yields a strictly negative integrand whenever the required moment exists; monotone convergence gives the endpoints. The full multivariate law is still needed to identify dependence nuisance.

## 5.5 A forward-model plateau obstruction

**Proposition 5.3 (finite hard-threshold grids identify only plateaus).**  
Consider a forward model evaluated on finitely many energy and pitch/quadrature nodes. Suppose its \(U\)-dependence enters only through finitely many hard indicators

\[
1\{p_r\le h_r(U,\beta)\},\qquad r=1,\ldots,R,
\]

and clipping at finitely many boundary equations, followed by deterministic finite sums and linear smoothing that does not itself depend continuously on \(U\). For fixed \(\beta\), the resulting mean vector is piecewise constant in \(U\), with changes only at the finite set of threshold/clipping solutions. On every nonempty open interval between adjacent solutions, all \(U\) values are observationally equivalent under any count law determined only by that mean.

**Proof.** For each \(r\), the truth value of the indicator changes only when \(p_r=h_r(U,\beta)\), and a clipping regime changes only at its boundary equation. By assumption the union \(B\) of all such solutions is finite. On a connected component of \(\mathcal U\setminus B\), no indicator or clipping state changes, so every finite sum is constant. A fixed linear map of a constant vector is constant. Any observation law depending on \(U\) only through that mean is therefore identical throughout the component. \(\square\)

**Application to the current repository model.** With beam amplitude zero, the finite pitch-grid hard loss-cone mask in `current_branch/src/losscone/model.py` has exactly this form. Point identification between threshold crossings cannot be claimed from that discretized model, even with perfect counts. A valid repair is one of:

1. integrate a source-backed continuous energy/angular response against a continuous trajectory boundary, producing a mean with nonzero \(U\)-derivative and verified rank;
2. include a physically justified smooth beam/edge component whose parameters do not themselves absorb \(U\);
3. retain the hard model but report the plateau as a partially identified set rather than an optimizer point.

Numerical interpolation between grid points does not repair structural plateaus.

## 5.6 Code-space versus calibrated-flux identification

**Proposition 5.4 (known transformations).**  
Let \(Y=g(C,S)\) with known \(S\).

1. If \(g(\cdot,S)\) is one-to-one on the code support, \(C\) and \(Y\) generate the same sigma-field and have identical identified sets.
2. If \(g\) is many-to-one, \(Y\) is a coarsening; \(d_{\rm TV}(P_u^Y,P_v^Y)\le d_{\rm TV}(P_u^C,P_v^C)\), and identification can be lost.
3. If \(g\) is unknown and allowed to vary with \(u\), cell, or state, identification requires restrictions on the class of maps; monotonicity alone is generally insufficient for continuous calibrated data, as Theorem 6.3 shows.

**Proof.** A known bijection has a measurable inverse, so laws determine one another. For a measurable many-to-one map, total variation contracts under push-forward: for any measurable set \(B\) in the \(Y\)-space, \(|P_u^Y(B)-P_v^Y(B)|=|P_u^C(g^{-1}B)-P_v^C(g^{-1}B)|\le d_{\rm TV}(P_u^C,P_v^C)\); taking the supremum proves the inequality. The third statement follows from the explicit equivalences in Section 6. \(\square\)

For Lunar Prospector \(Y\), the bundle does not establish a known one-to-one \(g\). Dead-time masking, missingness, rounding, response conversion, and possibly lost state make the public product a coarsening with uncertain calibration.

---

# 6. Impossibility and observational-equivalence proofs

## 6.1 Arbitrary response absorbs the loss-cone shape

Let \(f_j(u,\beta)\ge0\) denote the ideal physical cell signal before response. A common response form is

\[
\mu_j=b_j+a_j f_j(u,\beta),
\tag{6.1}
\]

where \(a_j\) combines exposure, efficiency, geometric response, and accepted-event probability, and \(b_j\) is accepted background.

**Theorem 6.1 (cellwise response-absorption nonidentification).**  
Suppose the nuisance class permits arbitrary \(a_j\ge0\) and \(b_j\ge0\) separately for every observed cell, with no cross-cell or cross-state restrictions. For any \(u_1\ne u_2\), any \(\beta_1,\beta_2\), and any chosen vector \(m\in(0,\infty)^d\), there are response nuisances \((a^{(r)},b^{(r)})\), \(r=1,2\), such that

\[
b_j^{(r)}+a_j^{(r)}f_j(u_r,\beta_r)=m_j
\quad\text{for every }j.
\tag{6.2}
\]

Consequently every observation family whose law depends on \((u,\beta,a,b)\) only through \(\mu\)—including independent Poisson, negative-binomial with common fixed dispersion, and many specified count-process models—has observationally equivalent parameter pairs at \(u_1\) and \(u_2\).

**Proof.** For each \(r,j\), if \(f_j(u_r,\beta_r)>0\), choose \(a_j^{(r)}=m_j/f_j(u_r,\beta_r)\) and \(b_j^{(r)}=0\). If \(f_j(u_r,\beta_r)=0\), choose \(a_j^{(r)}=0\) and \(b_j^{(r)}=m_j\). Equation (6.2) follows. Equal mean vectors give equal laws in every family stipulated to depend on the parameters only through that vector, and deterministic compression preserves equality. \(\square\)

If backgrounds are fixed at zero and all relevant \(f_j\) are positive, the simpler construction

\[
a_j^{(2)}=a_j^{(1)}\frac{f_j(u_1,\beta_1)}{f_j(u_2,\beta_2)}
\tag{6.3}
\]

already proves equivalence.

**Near-minimal restrictions that break the construction.** At least one of the following must tie cells together strongly enough that (6.2) is infeasible:

* known response ratios \(a_j/a_k\) up to one row- or sweep-level scale;
* a finite-dimensional response \(a_j=a_j(\eta)\) with \(\partial_u\mu\notin\operatorname{span}D_\eta\mu\);
* anchor cells or calibration exposures whose physical signal is known independently;
* a bounded-distortion set, for example \(|\log a_j-\log a_j^0|\le\delta_j\), narrow enough that the response-feasible sets for distinct \(u\) are separated;
* repeated instrument states with the same response parameters but different, independently known physical illumination, giving an overidentified rank condition.

“Robust standard errors” do not impose any such restriction and therefore do not repair this nonidentification.

## 6.2 Unrestricted intensity mixing

**Theorem 6.2 (arbitrary vector Cox mixing absorbs \(U\)).**  
Assume \(f_j(u,\beta)>0\). Let a Cox model have random cell intensities

\[
\Lambda_j=A_j f_j(u,\beta),
\]

and, conditional on \(\Lambda\), let \(K_j\) follow any fixed observation law indexed by \(\Lambda\) (independent Poisson is the canonical example). If the joint law of the nonnegative vector \(A=(A_1,\ldots,A_d)\) is unrestricted and may vary with nuisance, then \(U\) is not identified.

**Proof.** Fix any target random vector \(Z\in(0,\infty)^d\) with law \(H\). At parameter value \(u\), define

\[
A_j^{(u)}=\frac{Z_j}{f_j(u,\beta)}.
\]

Then \(\Lambda^{(u)}=Z\) almost surely for every \(u\). The conditional law of \(K\) given \(Z\), the mixed law of \(K\), and the compressed law of \(C\) are therefore identical for all \(u\). \(\square\)

A **shared scalar** random effect \(A_i\) does not permit this cellwise construction. If it multiplies every cell in a sweep and the response-normalized shape \(f(u,\beta)/\sum_j f_j(u,\beta)\) is injective, angular shape can identify \(u\). But a conditional multinomial elimination requires the exact latent total \(\sum_jK_j\), not a sum of compressed lower endpoints and not the separately defined Total Counter. With only componentwise codes, one must integrate or sum over all latent totals consistent with the code rectangle.

The practical strengthening is therefore: specify a shared or low-rank mixing structure, tie its law across cells and repeated sweeps, and bound it away from degeneracies that erase all information.

## 6.3 Unknown calibration

**Theorem 6.3 (unknown monotone continuous calibration).**  
Let latent scalar observations \(Z_r\) under \(u_r\), \(r=1,2\), have continuous strictly increasing distribution functions \(F_r\). Suppose the observed value is \(Y=h_r(Z_r)\), where the nuisance class permits an arbitrary strictly increasing calibration function. For any increasing \(h_1\), define

\[
h_2=h_1\circ F_1^{-1}\circ F_2.
\tag{6.4}
\]

Then \(h_2\) is increasing and \(h_2(Z_2)\) has the same law as \(h_1(Z_1)\). Thus \(u_1\) and \(u_2\) are observationally equivalent.

**Proof.** The probability integral transform gives \(F_2(Z_2)\sim\operatorname{Unif}(0,1)\). Therefore \(F_1^{-1}\{F_2(Z_2)\}\) has distribution \(F_1\), the same as \(Z_1\). Applying \(h_1\) preserves equality in distribution. Monotonicity follows by composition. \(\square\)

For a finite discrete code, a common deterministic strictly increasing relabeling is one-to-one and preserves atom probabilities, so monotonicity alone does **not** necessarily erase code-law identification. The obstruction applies directly to continuously valued calibrated products and, more generally, to cell/state-specific maps or randomized calibration kernels. The Lunar Prospector calibrated product includes cell/state scaling, masking, and other processing, not merely a known common relabeling. The practical repair is raw \(C\), a fixed traceable calibration, or an uncertainty set narrow enough to keep profiled law sets separated.

## 6.4 Unknown outcome-dependent selection

Let \(P_u\) be a preselection observable law with density or mass \(p_u\), and let \(\pi_u(o)\in[0,1]\) be the probability that observation \(o\) is retained. The selected law is proportional to \(\pi_u(o)p_u(o)\).

**Theorem 6.4 (selection equivalence).**  
If the nuisance class permits arbitrary outcome-dependent selection, then:

1. whenever a target law \(Q\) is dominated by \(P_u\) and \(q/p_u\) is essentially bounded, selecting with
   \[
   \pi_u(o)=\frac{q(o)}{M_up_u(o)},\qquad
   M_u\ge\operatorname*{ess\,sup}\frac{q}{p_u},
   \tag{6.5}
   \]
   makes the retained law exactly \(Q\);
2. in any count/code model assigning positive probability to an all-zero array, the rule “retain iff \(C=0\)” gives the same degenerate retained law for every \(u\).

Therefore unrestricted selection destroys identification.

**Proof.** Under (6.5), the unnormalized retained density is \(q/M_u\), whose integral is \(1/M_u\); normalization gives \(q\). The second statement is immediate by conditioning on the common event \(C=0\). \(\square\)

The smallest practical repair is to make selection predetermined from external state, or to observe and model inclusion probabilities under a missing-at-random/bounded-selection assumption. When a gate is computed from the same flux used in the fit, every null simulation or resample must rerun that gate. Fixing the observed gate/mask changes the experiment and generally invalidates calibration.

## 6.5 Distributionally robust neighborhoods

**Proposition 6.5 (overlap boundary for total-variation neighborhoods).**  
Let \(B_{\rm TV}(P,\epsilon)=\{R:d_{\rm TV}(R,P)\le\epsilon\}\). Then

\[
B_{\rm TV}(P,\epsilon)\cap B_{\rm TV}(Q,\epsilon)\ne\varnothing
\quad\Longleftrightarrow\quad
d_{\rm TV}(P,Q)\le2\epsilon.
\tag{6.6}
\]

**Proof.** If \(R\) belongs to both balls, the triangle inequality gives \(d_{\rm TV}(P,Q)\le2\epsilon\). Conversely, let \(R=(P+Q)/2\). Directly from the \(L^1\) representation, \(d_{\rm TV}(R,P)=d_{\rm TV}(P,Q)/2\), and similarly for \(Q\). \(\square\)

Thus a robust radius is scientifically useful only relative to the minimum separation of profiled nominal law sets. Unbounded Huber contamination contains common laws and returns the model to class C.

## 6.6 Broadest-class triviality

**Theorem 6.6 (common-law lower bound for honest confidence sets).**  
Let \(\mathcal U\subset\mathbb R\) have finite Lebesgue measure \(|\mathcal U|>0\). Suppose a law \(Q\) belongs to \(\mathcal P_u^O\) for every \(u\in\mathcal U\). If \(\mathcal C_\alpha(O)\) is honest in the sense of (3.1), then

\[
E_Q\bigl|\mathcal C_\alpha(O)\bigr|
\ge(1-\alpha)|\mathcal U|.
\tag{6.7}
\]

For a finite grid of \(G\) values,

\[
E_Q\#\mathcal C_\alpha(O)\ge(1-\alpha)G.
\tag{6.8}
\]

If \(Q\) is a point mass and the confidence procedure is nonrandomized, honesty forces \(\mathcal C_\alpha(O)=\mathcal U\) at that observation for every \(\alpha<1\).

**Proof.** Honesty and common membership imply \(Q\{u\in\mathcal C_\alpha(O)\}\ge1-\alpha\) for every \(u\). Tonelli’s theorem gives

\[
E_Q|\mathcal C_\alpha(O)|
=\int_{\mathcal U}Q\{u\in\mathcal C_\alpha(O)\}\,du
\ge(1-\alpha)|\mathcal U|.
\]

The finite-grid version follows by summing indicators. If \(Q=\delta_{o_0}\) and the set is nonrandomized, \(Q\{u\in\mathcal C(o_0)\}\) is either zero or one; it must be one for every \(u\). \(\square\)

In the support-plus-compressor-only class, \(Q=\delta_{C\equiv0}\) is admissible for every \(u\), so the theorem applies. This is the requested impossibility result for the broadest nearly unrestricted class.

## 6.7 Point identification without uniform distinguishability

**Proposition 6.7 (negative-binomial overdispersion boundary).**  
Let \(K\) have negative-binomial mean \(\mu>0\) and shape \(r>0\), so

\[
P(K=0)=\left(\frac{r}{r+\mu}\right)^r.
\tag{6.9}
\]

For every fixed finite \(\mu\), \(P(K=0)\to1\) as \(r\downarrow0\). Hence \(\operatorname{Law}(K)\to\delta_0\) in total variation, uniformly over \(\mu\) in any compact subset of \((0,\infty)\). The compressed law also converges to the all-zero code law. Therefore, if the nuisance class allows \(r\downarrow0\), separated mean values are not uniformly distinguishable and no confidence sets can be both uniformly honest and uniformly shrinking over that class.

**Proof.** Taking logarithms,

\[
\log P(K=0)=r\log r-r\log(r+\mu)\to0,
\]

because \(r\log r\to0\) and \(r\log(r+\mu)\to0\), uniformly for \(\mu\) in a positive compact set. Thus \(P(K>0)\to0\). For any distribution on \(\mathbb N_0\), total variation from \(\delta_0\) equals \(P(K>0)\). Deterministic compression cannot increase total variation. For two separated means, the triangle inequality makes their distance tend to zero. If a uniformly shrinking honest confidence set existed, its inversion would yield tests with both uniformly controlled size and power tending to one for separated values, contradicting the elementary testing bound \(P(\text{reject})+Q(\text{accept})\ge1-d_{\rm TV}(P,Q)\). \(\square\)

The necessary strengthening is a scientifically justified lower bound \(r\ge r_{\min}>0\), or a different variance/tail restriction that supplies uniform separation.

## 6.8 Exchangeability is not an identification assumption

Finite exchangeability says only that a joint law is invariant to permutations of declared units. The all-zero common law is exchangeable, so Theorem 6.6 still applies. Infinite exchangeability gives a de Finetti mixture, but if the mixing distribution is unrestricted, Theorem 6.2’s absorption mechanism remains available. Exchangeability becomes useful only after specifying what is exchangeable, what state variables are conditioned upon, and an identifying restriction on the mixing law or conditional mean. It can justify permutation tests for a sharp invariant null; it cannot by itself produce a likelihood or identify \(U\).

---

# 7. Interval-code likelihood derivations

## 7.1 State-specific endpoints

For an observed scalar code \(c\), let \(a_s(c)=\inf I_s(c)\) and \(b_s(c)=\sup I_s(c)\), allowing \(b_s(c)=\infty\) only when the terminal rule is documented. For ordinary nonterminal unprescaled data,

\[
a(c)=L(c),\qquad b(c)=L(c+1)-1.
\tag{7.1}
\]

For a documented divide-by-\(d\) prescale followed by truncating compression,

\[
a_d(c)=dL(c),\qquad b_d(c)=dL(c+1)-1.
\tag{7.2}
\]

Every formula below uses \((a_s,b_s)\), not the decompressed lower endpoint alone.

## 7.2 Independent Poisson accepted counts

Let \(K\sim\operatorname{Pois}(\mu)\). Define \(F(k;\mu)=P_\mu(K\le k)\), with \(F(-1;\mu)=0\), and \(S(k;\mu)=P_\mu(K>k)\). For a finite interval \([a,b]\),

\[
\pi_{a,b}(\mu)
=P_\mu(a\le K\le b)
=F(b;\mu)-F(a-1;\mu)
=S(a-1;\mu)-S(b;\mu).
\tag{7.3}
\]

For a terminal upper tail,

\[
\pi_{a,\infty}(\mu)=S(a-1;\mu).
\tag{7.4}
\]

For ordinary code zero, \(a=b=0\) and \(\log\pi=-\mu\). For a prescaled zero code, \(\pi=F(d-1;\mu)\).

### Stable log differences

Let `log1mexp(x)` for \(x\le0\) be evaluated as

\[
\operatorname{log1mexp}(x)=
\begin{cases}
\log\{-\operatorname{expm1}(x)\},&-\log2<x\le0,\\
\log1p\{-e^x\},&x\le-\log2.
\end{cases}
\tag{7.5}
\]

If \(a=0\), use \(\log F(b;\mu)\). If \(b=\infty\), use \(\log S(a-1;\mu)\). Otherwise either exact identity is available:

\[
\begin{aligned}
\log\pi_{a,b}
&=\log F(b;\mu)
 +\operatorname{log1mexp}\{\log F(a-1;\mu)-\log F(b;\mu)\},\\
&=\log S(a-1;\mu)
 +\operatorname{log1mexp}\{\log S(b;\mu)-\log S(a-1;\mu)\}.
\end{aligned}
\tag{7.6}
\]

Use the lower-tail representation when the library returns accurate non-saturated log-CDFs and the upper-tail representation when the interval lies in the upper tail; if both are finite, their agreement is a numerical check. Never compute `cdf(b)-cdf(a-1)` in ordinary floating point when the terms are close.

A useful derivative is

\[
\frac{d}{d\mu}\pi_{a,b}(\mu)
=p_{\mu}(a-1)-p_{\mu}(b),
\tag{7.7}
\]

with \(p_\mu(-1)=0\); for \(b=\infty\), the derivative is \(p_\mu(a-1)\). Gradients of the log likelihood follow by division by \(\pi\), using log-domain ratios.

For independent cells and clusters,

\[
\ell(u,\nu;C,S)=
\sum_i\sum_j
\log\pi_{a_{S_i,j}(C_{ij}),b_{S_i,j}(C_{ij})}
\{\mu_{ij}(u,\nu,S_i)\}.
\tag{7.8}
\]

This factorization is invalid if accepted counts share a random effect, FIFO, normalization, or other dependence.

## 7.3 Negative-binomial interval likelihood

Use the mean/shape parameterization

\[
P(K=k)=
\frac{\Gamma(k+r)}{\Gamma(r)k!}
\left(\frac{r}{r+\mu}\right)^r
\left(\frac{\mu}{r+\mu}\right)^k,
\qquad r>0.
\tag{7.9}
\]

Then \(E K=\mu\) and \(\operatorname{Var}K=\mu+\mu^2/r\). The code probability is

\[
\pi^{\rm NB}_{a,b}(\mu,r)
=F_{\rm NB}(b;\mu,r)-F_{\rm NB}(a-1;\mu,r),
\tag{7.10}
\]

or the corresponding survival difference, evaluated by the same log-difference rule. A valid A-class model requires \(r\) known or profiled over a compact set \([r_{\min},r_{\max}]\) with \(r_{\min}>0\), plus response injectivity. Allowing \(r\downarrow0\) crosses the uniform-inference boundary proved in Proposition 6.7.

## 7.4 Shared-gamma Cox likelihood

A scientifically useful dependent alternative is

\[
A_i\sim\operatorname{Gamma}(r,r),\qquad
K_{ij}\mid A_i\ \overset{\rm ind}{\sim}\ 
\operatorname{Pois}\{A_i\mu_{ij}(u,\nu,S_i)\},
\tag{7.11}
\]

where the gamma uses shape \(r\), rate \(r\), so \(EA_i=1\). Marginally each cell is negative-binomial, while cells in a sweep are positively correlated through \(A_i\). The exact code-rectangle probability is

\[
P(C_i=c_i\mid u,\nu,S_i)
=\int_0^\infty
\prod_{j=1}^d
\pi_{a_{ij},b_{ij}}\{a\mu_{ij}(u,\nu,S_i)\}
\frac{r^r}{\Gamma(r)}a^{r-1}e^{-ra}\,da.
\tag{7.12}
\]

With \(x=ra\),

\[
P(C_i=c_i)=
\frac1{\Gamma(r)}
\int_0^\infty x^{r-1}e^{-x}
\prod_j\pi_{a_{ij},b_{ij}}
\left(\frac{x}{r}\mu_{ij}\right)dx.
\tag{7.13}
\]

This is a one-dimensional generalized Gauss–Laguerre integral. Adaptive quadrature in log scale is also feasible. The integrand should be accumulated as

\[
(r-1)\log x-x-\log\Gamma(r)+\sum_j\log\pi_{a_{ij},b_{ij}}(x\mu_{ij}/r),
\]

with log-sum-exp across nodes. Quadrature error must be bounded and propagated conservatively into the test statistic; Section 13 gives a certification rule.

A lognormal shared effect is similar but uses Gauss–Hermite or adaptive integration. An unrestricted vector effect is class C by Theorem 6.2.

## 7.5 Poisson-binomial and heterogeneous thinning

If \(K=\sum_{\ell=1}^N B_\ell\) with independent \(B_\ell\sim\operatorname{Bernoulli}(p_\ell)\), the probability generating polynomial is

\[
G(z)=\prod_{\ell=1}^N\{(1-p_\ell)+p_\ell z\},
\tag{7.14}
\]

and \(P(K=k)=[z^k]G(z)\). The interval-code probability is \(\sum_{k=a}^b[z^k]G(z)\). Coefficients can be computed by dynamic programming in \(O(N^2)\) for the full PMF, truncated DP in \(O(Nb)\), divide-and-conquer FFT, or a certified saddlepoint approximation with explicit error control. This model is appropriate only when the finite opportunities and efficiencies have physical meaning. Arbitrary unknown \(p_\ell\) or cellwise thinning recreates Theorem 6.1.

## 7.6 Nonparalyzable dead time and renewal models

For a Poisson attempted-arrival process of rate \(\lambda\) observed for exposure \(T\), with deterministic nonparalyzable dead time \(\tau\) after each accepted event, the time of the \(k\)th accepted event is

\[
T_k=G_k+(k-1)\tau,
\quad G_k\sim\operatorname{Gamma}(k,\lambda).
\tag{7.15}
\]

Therefore, for \(T-(k-1)\tau\ge0\),

\[
P(N_T\ge k)=P\{G_k\le T-(k-1)\tau\},
\tag{7.16}
\]

and it is zero otherwise. The accepted-count PMF is

\[
P(N_T=k)=P(N_T\ge k)-P(N_T\ge k+1).
\tag{7.17}
\]

Summing (7.17) over the code interval gives an exact scalar likelihood. The formula does not cover paralyzable dead time, event-dependent service, shared FIFO, or anode masks; those require their own state process.

For a finite-capacity FIFO with a finite-state continuous-time Markov chain, augment the generator with a counting reward. If \(Q(z)\) is the tilted generator whose accepted-transition rates are multiplied by \(z\), then the probability generating function over exposure \(T\) is

\[
G_T(z)=\pi_0^\top e^{Q(z)T}\mathbf 1.
\tag{7.18}
\]

Coefficients, and hence code intervals, can be obtained by FFT inversion on a circle with truncation/error certification. This becomes multivariate if accepted events are routed to cells. The model is A only when arrival, service, routing, mask, and initial-state rules are specified or bounded tightly enough for identification.

## 7.7 Conditional Poisson and the missing latent total

If exact counts in one row satisfy independent \(K_a\sim\operatorname{Pois}(\rho p_a(u,\nu))\), then conditional on \(N=\sum_aK_a\),

\[
(K_1,\ldots,K_A)\mid N
\sim\operatorname{Multinomial}\{N,p(u,\nu)\},
\tag{7.19}
\]

which eliminates \(\rho\). For interval codes, however, exact conditioning is

\[
P(C=c\mid N,u,\nu)
=\sum_{k\in I(c):\sum_a k_a=N}
\frac{N!}{\prod_a k_a!}\prod_a p_a(u,\nu)^{k_a}.
\tag{7.20}
\]

If \(N\) is unobserved, it must be summed with its model probability. Normalizing decompressed lower endpoints or calibrated flux by its row sum is not (7.19) or (7.20). The separately telemetered Total Counter is not the exact 3-D row total: it covers all energies including retrace and has its own masks, FIFO loss, and factor-16 prescale. `[Bundle: primary_sources/magerfmt.txt, lines 532–550]`

## 7.8 General dependent law

For any full multivariate accepted-count law,

\[
P(C_i=c_i)=
P\{K_i\in I_{S_i}(c_i)\}
=\sum_{k\in I_{S_i}(c_i)}p_{K,i}(k).
\tag{7.21}
\]

A composite likelihood formed by multiplying marginal interval probabilities may estimate marginal parameters under conditions, but it is not the full likelihood. A likelihood-ratio cutoff from the composite criterion is invalid unless its distribution is calibrated under the complete cluster law. Cluster sandwich covariance can support asymptotic estimating-equation inference; it cannot restore identification or make a misspecified composite likelihood exact.

---

# 8. Weaker stochastic models and their tractability

## 8.1 Conditional Poisson and shared Cox effects

A common sweep-level amplitude \(A_i\) is scientifically plausible when plasma density or exposure varies across a sweep while the response-normalized angular shape carries \(U\). It is less restrictive than independent Poisson and computationally tractable through (7.12). Identification requires that no nuisance combination makes the normalized mean shape equal for distinct \(u\), and that the mixing family is tied across cells. A free cell-specific random effect is not a robustness improvement; it is the nonidentifying class of Theorem 6.2.

If the exact latent row total were observed, conditioning could eliminate a common amplitude without specifying its distribution. With componentwise quantization it is not observed, so a valid conditional method must use (7.20) or augment the observation with a proved total. This is a central boundary between a genuine conditional likelihood and an analogy based on normalized flux.

## 8.2 Explicit overdispersion families

Negative-binomial, Poisson-lognormal, and finite mixtures can accommodate extra-Poisson variance. They are viable A classes when:

1. the mixing form is scientifically tied to a common physical effect;
2. mixing parameters are known or lie in a compact, uniformly identifiable set;
3. the full dependence induced by shared mixing is included;
4. response/state and selection are known or bounded.

A family label alone is insufficient. The dispersion boundary, zero inflation, and possible dependence of dispersion on energy, angle, or state must be specified. Unbounded mixtures are class C.

## 8.3 Heterogeneous efficiency and Poisson-binomial models

Poisson-binomial models are appropriate for a finite set of potential detected events with heterogeneous acceptance probabilities. They are not automatically weaker than Poisson: they impose finite support and conditional independence of Bernoulli trials. Known efficiencies can produce underdispersion relative to Poisson. Bounded efficiencies lead naturally to partial identification. If the event opportunities themselves are a Poisson process and each is independently thinned, the result returns to Poisson; if thinning shares detector state or dead time, it does not.

## 8.4 Renewal, refractory, and finite-capacity models

Dead time ordinarily induces underdispersion and rate-dependent distortion. A known nonparalyzable model is tractable via (7.15)–(7.17). FIFO losses couple cells through event order and service capacity; a finite-state queue is tractable by a Markov-additive likelihood when its transition rules are known. The Lunar Prospector source confirms the existence of approximately 300 ns processing dead time and a four-event FIFO in a related diagnostic path, but does not provide the state-resolved cell observation law required to instantiate these formulas. `[Bundle: primary_sources/magerfmt.txt, lines 532–550]`

An unrestricted loss fraction \(q_j(u)\) changes \(\mu_j\) to \(q_j(u)\lambda_j(u)\) and is algebraically identical to arbitrary response, so it is class C. A validated bound \(q_j\in[q_j^-,q_j^+]\) produces class B.

## 8.5 Clustered and finite-range dependent sweeps

Let \(O_i\) be a sweep/visit cluster. A defensible weaker model may allow any within-cluster law while assuming clusters are independent, \(m\)-dependent, or strongly mixing after conditioning on state and a common/regression parameter. Inference is then based on cluster features \(h(O_i)\), not 1320 independent cells. This can accommodate shared normalization, sequential energy rows, and random effects.

The assumption is meaningful only if \(U\) is common across the pooled clusters, follows a specified regression \(U_i=g(W_i;\theta)\), or is locally constant in a prespecified window. Pooling a day of changing plasma and surface conditions without such a model changes the estimand to an unspecified mixture and does not estimate a per-sweep \(U\).

## 8.6 Martingale/counting-process restrictions

Suppose an accepted counting path \(N_j(t)\) is observed and has compensator \(\Lambda_j(t;u,\nu)\), so \(M_j=N_j-\Lambda_j\) is a martingale with bounded jumps. Predictable weighted sums

\[
Z(u,\nu)=\sum_j\int w_j(t;u,\nu)\,dM_j(t)
\tag{8.1}
\]

can be tested by exact exponential-supermartingale or Freedman/Bernstein bounds when predictable variation is bounded. If only terminal interval codes are observed, one may form worst-case score bounds over \(K_j\in I(C_j)\); a candidate is rejected only if every latent count vector in the code rectangle violates the martingale concentration band. This is conservative and requires finite upper endpoints. The bundle lacks the accepted-event path and a certified compensator, so this is not currently operational.

## 8.7 Exchangeability

Finite exchangeability can justify treating a predeclared permutation as null-invariant, but it does not turn 88 angular cells into iid observations: their directions and response are not exchangeable without conditioning/transformation. Infinite exchangeability implies a mixture of iid laws only for an infinite sequence of comparable units; the de Finetti mixing law remains a nuisance and may be nonidentifying. For LP ER, time order, energy order, angular geometry, and state make a blanket exchangeability assumption scientifically implausible.

## 8.8 Estimating-equation and mean/variance models

Let \(m_i(u,\nu)\in\mathbb R^R\) be an observed cluster feature satisfying

\[
E_{u_0,\nu_0}m_i(u_0,\nu_0)=0.
\tag{8.2}
\]

No count distribution is required. Point identification needs the population zero set to be a singleton after nuisance profiling. Finite-sample coverage is available if components are bounded, using Theorem 11.4. With unbounded features, cluster-GMM or empirical likelihood is only uniformly asymptotically honest after an explicit uniform CLT, covariance consistency, and eigenvalue/rank conditions. A sandwich covariance estimates sampling variation of valid moments; it does not correct an incorrect mean or response map.

## 8.9 Moment inequalities

Code intervals and response envelopes naturally yield inequalities rather than equalities. These are scientifically preferable to reconstructing pseudo-counts. They can support a B-class outer region even under arbitrary within-sweep count law, provided repeated clusters estimate the moments and the upper endpoints/background/response bounds are finite. Section 9 gives the sharp componentwise construction.

---

# 9. Partial-identification construction

## 9.1 Bounded response under interval-coded Poisson marginals

Suppose repeated observations in a fixed state identify a vector of Poisson means \(\mu(Q)\) from the complete code law by Theorem 5.2. Let

\[
\mu_j=b_j+a_j f_j(u,\beta),
\tag{9.1}
\]

and let \(\mathcal H\) be a scientifically validated coupled set for \((\beta,a,b)\).

**Proposition 9.1 (identified set under bounded response).**  
The sharp identified set under model (9.1) is

\[
\Theta_I(Q)=
\left\{u:\exists(\beta,a,b)\in\mathcal H
\text{ such that }
\mu_j(Q)=b_j+a_jf_j(u,\beta),\ \forall j\right\}.
\tag{9.2}
\]

If nuisance bounds are uncoupled,

\[
a_j\in[a_j^-,a_j^+],\qquad
b_j\in[b_j^-,b_j^+],\qquad f_j(u,\beta)\ge0,
\]

then, for a fixed \((u,\beta)\), feasibility is equivalent to

\[
b_j^-+a_j^-f_j(u,\beta)
\le\mu_j(Q)\le
b_j^++a_j^+f_j(u,\beta)
\quad\forall j.
\tag{9.3}
\]

If physical coupling constraints are omitted when using (9.3), the resulting set is a computable outer approximation, not the sharp set.

**Proof.** By Theorem 5.2 the code law fixes \(\mu(Q)\). A value \(u\) is observationally compatible exactly when some admissible nuisance reproduces that mean, giving (9.2). Under independent interval bounds and nonnegative \(f_j\), the attainable scalar mean in cell \(j\) is the interval in (9.3), and cells can be chosen independently, proving equivalence. Dropping coupling enlarges the nuisance set and therefore can only enlarge the identified set. \(\square\)

The optimization is a feasibility problem, not a least-squares fit. With smooth finite-dimensional \(\mathcal H\), solve

\[
\rho(u)=
\inf_{(\beta,a,b)\in\mathcal H}
\max_j
\frac{|\mu_j-b_j-a_jf_j(u,\beta)|}{s_j},
\tag{9.4}
\]

and retain \(u\) iff a certified lower bound on \(\rho(u)\) is zero within declared arithmetic tolerance. For interval estimates of \(\mu_j\), replace equality by interval intersection.

## 9.2 Exact finite-sample mean intervals from zero codes

Suppose for a fixed cell/state there are \(n\) independent replicated clusters with common mean \(\mu\), and let

\[
Z_i=1\{C_i=0\},\qquad Z_i\sim\operatorname{Bernoulli}(p),
\quad p=F_{\operatorname{Pois}(\mu)}(d-1).
\]

An exact Clopper–Pearson interval \([p_L,p_U]\) with error probability \(\alpha_j\) transforms, because the map is decreasing, to

\[
\mu\in
[h_d^{-1}(p_U),h_d^{-1}(p_L)],
\quad h_d(\mu)=F_{\operatorname{Pois}(\mu)}(d-1).
\tag{9.5}
\]

Use \(h_d^{-1}(1)=0\) and \(h_d^{-1}(0)=\infty\). Allocating \(\sum_j\alpha_j\le\alpha\) gives simultaneous finite-sample coverage across cells without assuming cross-cell independence, because Bonferroni uses only each cell’s across-cluster binomial model. Intersect these mean intervals with (9.3) or the coupled feasibility set. The construction can be inefficient because it uses only code zero, but it is exact and demonstrates that quantization is not the obstacle.

If means or states vary across clusters, the \(Z_i\) are not identically distributed; use the full nonidentical likelihood, a regression model, or bounded-feature concentration instead of a binomial interval.

## 9.3 Distribution-free interval-moment identification

For finite inverse intervals define

\[
\ell_s(c)=\min I_s(c),\qquad r_s(c)=\max I_s(c).
\]

Then deterministically

\[
\ell_{S_i}(C_i)\le K_i\le r_{S_i}(C_i)
\tag{9.6}
\]

componentwise, so

\[
E\ell_{S}(C)\le EK\le Er_{S}(C).
\tag{9.7}
\]

Suppose physics and bounded response imply

\[
m_j^-(u,\nu)\le EK_j\le m_j^+(u,\nu).
\tag{9.8}
\]

A necessary compatibility condition is

\[
E\ell_j(C)\le m_j^+(u,\nu),
\qquad
Er_j(C)\ge m_j^-(u,\nu).
\tag{9.9}
\]

With no cross-cell restrictions on the latent conditional distribution inside each interval, (9.9) is also sharp for the scalar first moment: conditional randomization between interval endpoints realizes every mean between \(E\ell(C)\) and \(Er(C)\). Coupled physical/count constraints make (9.9) an outer approximation.

A terminal code without a finite documented upper endpoint makes \(r(C)\) unavailable. One may still use lower-moment inequalities, tail indicators \(1\{C\ge c\}\), or impose a separately justified counter maximum; one may not insert an invented upper count.

## 9.4 Targets of uncertainty statements

Three distinct objects must not be conflated:

1. A **confidence set for the true parameter** satisfies (3.1): \(P\{u_0\in\mathcal C\}\ge1-\alpha\).
2. A **confidence region covering the identified set** satisfies
   \[
   P\{\Theta_I(P_0)\subseteq\mathcal R\}\ge1-\alpha.
   \tag{9.10}
   \]
3. An **estimate of the identified set** may converge in Hausdorff distance but need not have finite-sample coverage.

The simultaneous moment-box construction in Theorem 11.4 covers the entire moment-identified set on one high-probability event and therefore satisfies both (1) and (2) under its maintained model. Pointwise inversion of separate tests guarantees (1) for the true \(u_0\) but does not automatically guarantee simultaneous inclusion of every observationally equivalent \(u\) unless the construction is based on a common simultaneous confidence region for the law/moments.

---

# 10. Derived optimization and testing objectives

## 10.1 Exact code-space likelihood route

For a specified count-process class, define

\[
\widehat\theta=(\widehat u,\widehat\nu)
\in\arg\max_{u\in\mathcal U,\nu\in\mathcal H_u}\ell(u,\nu;O),
\tag{10.1}
\]

where \(\ell\) uses interval probabilities and the full dependence law. This is an estimator, not a confidence set. For testing \(H_0:u=u_0\), use the generalized likelihood-ratio statistic

\[
T_{u_0}(O)=
2\left[
\sup_{u,\nu}\ell(u,\nu;O)
-
\sup_{\nu\in\mathcal H_{u_0}}\ell(u_0,\nu;O)
\right].
\tag{10.2}
\]

Nuisance is handled in the null distribution by least-favorable or Berger–Boos calibration, not merely by profiling the observed likelihood. Critical values are obtained from the exact null experiment as in Section 11. No Wilks or \(\chi^2_1\) cutoff is asserted at boundaries, under weak identification, with discrete codes, or with selected data.

A score or discrepancy statistic can replace (10.2) if its null law is easier to bound. Coverage comes from the test’s size, not from the choice of optimizer.

## 10.2 Cluster estimating equations and GMM

Let \(m_i(u,\nu)\in\mathbb R^R\) be cluster moments and \(\bar m_n=n^{-1}\sum_i m_i\). An estimator minimizes

\[
Q_n(u,\nu)=n\bar m_n(u,\nu)^\top W_n(u,\nu)\bar m_n(u,\nu).
\tag{10.3}
\]

A weak-identification-robust test of \(u=u_0\) is

\[
T_n(u_0)=
\inf_{\nu\in\mathcal H_{u_0}}
 n\bar m_n(u_0,\nu)^\top
\widehat\Sigma_n(u_0,\nu)^+
\bar m_n(u_0,\nu),
\tag{10.4}
\]

where \(+\) is the Moore–Penrose inverse on the estimable moment subspace. The covariance must be estimated across sweep/visit clusters, including all 15 rows and 88 cells in one cluster feature. Critical values require either the finite-sample concentration route of Theorem 11.4, a least-favorable Gaussian bound, subsampling, or a cluster bootstrap whose uniform validity conditions are stated and checked. Optimizer curvature in (10.3) is not a critical value.

## 10.3 Empirical likelihood

For independent clusters and a fixed \((u,\nu)\), empirical likelihood maximizes \(\prod_i(np_i)\) subject to

\[
p_i\ge0,\quad\sum_i p_i=1,\quad\sum_i p_i m_i(u,\nu)=0.
\tag{10.5}
\]

When feasible, the dual statistic is

\[
R_n(u,\nu)=2\sum_{i=1}^n\log\{1+\lambda^\top m_i(u,\nu)\},
\tag{10.6}
\]

where \(\lambda\) solves \(\sum_i m_i/(1+\lambda^\top m_i)=0\). Profile with \(\inf_\nu R_n(u,\nu)\). The familiar chi-square limit requires interiority of zero in the convex hull, fixed moment dimension, nonsingular covariance on the moment span, smoothness, and a uniform CLT. Quantization causes no special problem when moments are defined on observed codes, but sparse/degenerate code features can violate convex-hull and covariance conditions. Finite-sample concentration is safer near those boundaries.

## 10.4 Moment inequalities and partial identification

Let \(g_r(O;u,\nu)\) be oriented so that \(E g_r\le0\) under the null. A generic statistic is

\[
T_n(u)=
\inf_{\nu\in\mathcal H_u}
\max_{r\le R}
\left[
\frac{\sqrt n\,\bar g_r(u,\nu)}{\widehat\sigma_r(u,\nu)}
\right]_+.
\tag{10.7}
\]

Moment selection and studentization can improve power but complicate uniform calibration. The finite-sample alternative is to form simultaneous confidence intervals for \(Eg_r\) and retain \(u\) whenever some nuisance makes every inequality compatible. This directly yields an outer confidence region; it does not select an arbitrary optimizer within a flat identified set.

## 10.5 Distributionally robust tests

For a nominal law family \(P_{u,\nu}^0\) and a law neighborhood \(\mathcal N_{u,\nu}\), a level-\(\alpha\) robust test solves

\[
\sup_{\nu\in\mathcal H_u}
\sup_{P\in\mathcal N_{u,\nu}}E_P\varphi_u(O)
\le\alpha,
\tag{10.8}
\]

while maximizing worst-case power against separated alternatives. Equivalently, for a statistic \(T_u\), use the robust p-value

\[
p_u^{\rm rob}(o)=
\sup_{\nu}\sup_{P\in\mathcal N_{u,\nu}}
P\{T_u(O^*)\ge T_u(o)\}.
\tag{10.9}
\]

Inversion is honest over the neighborhood. If null and alternative neighborhoods overlap, Proposition 6.5 limits power and may force full sets. A robustness radius must come from calibration/validation evidence; choosing it after seeing interval width invalidates the guarantee.

## 10.6 Calibrated-flux quasi-likelihood

A calibrated-flux Gaussian criterion is a genuine likelihood only if, conditional on declared state and selection,

\[
Y_i\sim N\{m_i(u,\nu),\Sigma_i(u,\nu)\}
\tag{10.10}
\]

with the specified covariance and support treatment. If only \(E(Y_i)=m_i(u,\nu)\) is maintained, minimizing a weighted quadratic criterion is an M-estimator. Its asymptotic covariance is the cluster sandwich

\[
A^{-1}BA^{-\top},
\quad
A=E\left[-\frac{\partial\psi_i}{\partial\theta^\top}\right],
\quad B=E(\psi_i\psi_i^\top),
\tag{10.11}
\]

where \(\psi_i\) is the sweep/visit score. \(B\) must include shared row normalization, sequential rows, calibration correlations, and random effects. A single sweep cannot estimate a repeated-cluster sandwich without additional structure.

If rows are normalized to proportions, their covariance is singular and all free coordinates are correlated. A valid criterion uses a full-rank contrast and its full covariance, a generalized inverse on the simplex tangent, or a correctly transformed logistic-coordinate density. Dropping an arbitrary cell and assigning diagonal variances is generally order-dependent.

## 10.7 Invariance for compositional angular data

Let \(p,q\in\mathbb R^A\) be compositions with \(\mathbf1^\top p=\mathbf1^\top q=1\), let \(r=p-q\), and let \(\Sigma\) be positive definite on the tangent space

\[
\mathcal S=\{x:\mathbf1^\top x=0\},
\]

with null space spanned by \(\mathbf1\).

**Theorem 10.1 (contrast invariance).**  
Let \(A\) be any \((A-1)\times A\) matrix whose restriction to \(\mathcal S\) is one-to-one. Then

\[
(Ar)^\top(A\Sigma A^\top)^{-1}(Ar)
=r^\top\Sigma^+r.
\tag{10.12}
\]

Thus the full-covariance quadratic statistic is invariant to which cell is dropped and to any nonsingular coordinate basis on the simplex tangent.

**Proof.** Let \(B\) be an \(A\times(A-1)\) matrix with orthonormal columns spanning \(\mathcal S\). Write \(r=Bx\) and \(\Sigma=B\Omega B^\top\), where \(\Omega\) is positive definite. Since \(A\) is injective on \(\mathcal S\), \(AB\) is square and nonsingular. Then

\[
(Ar)^\top(A\Sigma A^\top)^{-1}(Ar)
=x^\top(AB)^\top\{AB\Omega(AB)^\top\}^{-1}ABx
=x^\top\Omega^{-1}x.
\]

Also \(\Sigma^+=B\Omega^{-1}B^\top\), so \(r^\top\Sigma^+r=x^\top\Omega^{-1}x\). \(\square\)

**Counterexample to diagonal dropped-cell weighting.** Let \(A=3\), residual \(r=(1,-1,0)\), and assign coordinate variances \((1,4,9)\). Dropping cell 3 gives \(1^2/1+(-1)^2/4=1.25\); dropping cell 2 gives \(1^2/1+0^2/9=1\). The criterion changes under relabeling. PR 14’s `cols[:-1]` plus diagonal model-dependent variances has this defect. `[Bundle: pr14/src/losscone/quasi_likelihood.py, lines 61–100]`

Log-ratio coordinates can be used, but invariance then requires the correct Jacobian/density or delta-method full covariance. Merely changing which denominator cell defines an additive log ratio does not preserve a diagonal working criterion.

## 10.8 When log-MSE is legitimate

**Proposition 10.2 (log-MSE likelihood and approximation regime).**  
Unweighted squared log error

\[
Q(\theta)=\sum_j\{\log Y_j-\log\mu_j(\theta)\}^2
\tag{10.13}
\]

is, up to a positive constant and parameter-independent terms, a negative log likelihood exactly when the maintained model is independent lognormal noise with common known log variance and median \(\mu_j(\theta)\):

\[
\log Y_j=\log\mu_j(\theta)+\varepsilon_j,
\qquad \varepsilon_j\overset{\rm iid}{\sim}N(0,\sigma^2).
\tag{10.14}
\]

If \(\mu_j\) denotes the arithmetic mean under a lognormal law, the log center is \(\log\mu_j-\sigma^2/2\), not \(\log\mu_j\). Unequal or correlated errors require inverse-covariance weighting.

For \(Y=\mu(1+x)\) with \(|x|\le\rho<1\),

\[
\{\log(Y/\mu)\}^2=x^2-x^3+R_4(x),
\tag{10.15}
\]

where

\[
|R_4(x)|\le C_\rho x^4,
\quad
C_\rho=\frac14+
\frac{2(1+\rho/2)}{3(1-\rho)^3}
+\frac{\rho^2}{9(1-\rho)^6}.
\tag{10.16}
\]

Thus log-MSE approximates relative-error least squares only for uniformly small relative errors, away from zero.

**Proof.** Under (10.14), the Gaussian log likelihood is

\[
-\ell=\frac1{2\sigma^2}Q(\theta)+\text{constant},
\]

proving the exact statement. The arithmetic-mean correction is the identity \(E e^{N(m,\sigma^2)}=e^{m+\sigma^2/2}\). For the approximation, Taylor’s theorem gives

\[
\log(1+x)=x-\frac{x^2}{2}+R_3,
\qquad |R_3|\le\frac{|x|^3}{3(1-\rho)^3}.
\]

Squaring,

\[
\log^2(1+x)=x^2-x^3+\frac{x^4}{4}
+2(x-x^2/2)R_3+R_3^2.
\]

Using \(|x-x^2/2|\le |x|(1+\rho/2)\) and \(|x|^6\le\rho^2x^4\) gives (10.16). \(\square\)

For large Poisson \(K\), the delta method gives approximately \(\operatorname{Var}(\log K)\approx1/\mu\) and bias approximately \(-1/(2\mu)\), so unweighted log-MSE is not the Poisson likelihood even before quantization. Code zeros make \(\log L(C)\) undefined; adding an arbitrary epsilon changes the objective. Wide code intervals, the positive missing sentinel, dead-time masking, and shared normalization further violate the approximation. The legacy Halekas log criterion is therefore a shape heuristic, not a likelihood or calibrated goodness-of-fit statistic. `[Bundle: current_branch/src/losscone/chi2.py, lines 35–55]`

---

# 11. Confidence-set algorithms and coverage proofs

## 11.1 Least-favorable test inversion

For each candidate \(u\), choose a scalar statistic \(T_u(O)\), larger values being less compatible with \(H_0:u=u_0\). For each nuisance \(\nu\in\mathcal H_u\), define the exact tail p-value

\[
p_{u,\nu}(o)=P_{u,\nu}\{T_u(O^*)\ge T_u(o)\}.
\tag{11.1}
\]

Define the least-favorable p-value

\[
p_u^{\rm LF}(o)=\sup_{\nu\in\mathcal H_u}p_{u,\nu}(o).
\tag{11.2}
\]

**Theorem 11.1 (finite-sample honest least-favorable inversion).**  
The test \(\varphi_u(o)=1\{p_u^{\rm LF}(o)\le\alpha\}\) has size at most \(\alpha\) for every \(\nu\in\mathcal H_u\). Consequently

\[
\mathcal C_\alpha^{\rm LF}(O)=
\{u\in\mathcal U:p_u^{\rm LF}(O)>\alpha\}
\tag{11.3}
\]

satisfies finite-sample honest coverage (3.1).

**Proof.** Fix \((u,\nu)\) and abbreviate \(T=T_u(O)\) and \(G(t)=P_{u,\nu}(T\ge t)\). The set \(A_\alpha=\{t:G(t)\le\alpha\}\) is an upper set because \(G\) is nonincreasing. If it is empty, \(P\{G(T)\le\alpha\}=0\). Otherwise write \(a=\inf A_\alpha\). If \(a\in A_\alpha\), then \(A_\alpha\subseteq[a,\infty)\) and

\[
P\{G(T)\le\alpha\}=P(T\in A_\alpha)
\le P(T\ge a)=G(a)\le\alpha.
\]

If \(a\notin A_\alpha\), upper-set structure gives \(A_\alpha\subseteq(a,\infty)\). Choose \(a_n\downarrow a\) with \(a_n\in A_\alpha\). Then continuity from below of probability gives

\[
P(T\in A_\alpha)\le P(T>a)
=\lim_{n\to\infty}P(T\ge a_n)
=\lim_{n\to\infty}G(a_n)\le\alpha.
\]

Thus the nonrandomized upper-tail p-value is super-uniform; atoms only make it conservative. Since \(p_u^{\rm LF}\ge p_{u,\nu}\),

\[
P_{u,\nu}(p_u^{\rm LF}\le\alpha)
\le P_{u,\nu}(p_{u,\nu}\le\alpha)\le\alpha.
\]

The event \(u\notin\mathcal C_\alpha^{\rm LF}\) is exactly rejection of the true composite null, proving coverage. \(\square\)

This theorem remains valid at parameter boundaries, with discrete quantization, disconnected sets, and nonstandard profile distributions. Its cost is conservatism and nuisance optimization.

## 11.2 Berger–Boos nuisance reduction

Let \(B_u(O)\subseteq\mathcal H_u\) be a \((1-\gamma)\) confidence set for the nuisance under every \((u,\nu)\):

\[
\inf_{\nu\in\mathcal H_u}P_{u,\nu}\{\nu\in B_u(O)\}\ge1-\gamma.
\]

Define

\[
p_u^{\rm BB}(O)=
\min\left\{1,\sup_{\nu\in B_u(O)}p_{u,\nu}(O)+\gamma\right\}.
\tag{11.4}
\]

**Theorem 11.2 (Berger–Boos validity).** The test rejecting when \(p_u^{\rm BB}\le\alpha\) has size at most \(\alpha\).

**Proof.** If the true nuisance \(\nu_0\in B_u(O)\) and \(p_u^{\rm BB}\le\alpha\), then \(p_{u,\nu_0}\le\alpha-\gamma\). Hence

\[
P(p_u^{\rm BB}\le\alpha)
\le P(p_{u,\nu_0}\le\alpha-\gamma)
+P(\nu_0\notin B_u)
\le(\alpha-\gamma)+\gamma=\alpha.
\]

\(\square\)

The nuisance confidence set must itself be honest under the null. A plug-in estimate without this protection is not Berger–Boos.

## 11.3 Exact Monte Carlo calibration

For a fully specified null \((u,\nu)\), simulate \(B\) complete datasets \(O^{*(1)},\ldots,O^{*(B)}\) independently from the same experiment as the observed \(O\), including state, latent count law, losses, quantization, ground processing, missingness, selection/gates, and nuisance refitting. Let \(T^{(0)}=T_u(O)\) and \(T^{(b)}=T_u(O^{*(b)})\). Define

\[
\widehat p_{u,\nu}=\frac{1+\sum_{b=1}^B1\{T^{(b)}\ge T^{(0)}\}}{B+1}.
\tag{11.5}
\]

**Theorem 11.3 (finite-sample Monte Carlo rank test).**  
Under the fully specified null and an identical measurable analysis algorithm for observed and simulated datasets, the test \(1\{\widehat p_{u,\nu}\le\alpha\}\) has size at most \(\alpha\).

**Proof.** The \(B+1\) complete datasets are exchangeable under the null, so the rank of \(T^{(0)}\) among them is uniform when there are no ties and conservative with the `greater than or equal` convention when ties occur. Formula (11.5) is the corresponding upper-tail rank divided by \(B+1\). Thus at most \(\lfloor\alpha(B+1)\rfloor\) ranks reject, giving probability at most \(\alpha\). \(\square\)

A parametric bootstrap at an estimated nuisance is not covered by this theorem because the observed dataset is not exchangeable with simulations from a random plug-in law conditional on the estimate. Combine Monte Carlo with Theorem 11.1 or 11.2, or prove a separate uniform bootstrap theorem.

## 11.4 Finite-sample bounded-feature moment set

Let \(O_1,\ldots,O_n\) be independent sweep/visit clusters having a common feature expectation under each maintained model (identical distribution is sufficient but not necessary). Choose predeclared features \(h_r(O_i)\in[a_r,b_r]\), \(r=1,\ldots,R\). Let the model-implied expectation set at \(u\) be

\[
\mathcal M_h(u)=
\{m(u,\nu)=E_{u,\nu}h(O):\nu\in\mathcal H_u\}.
\]

Define \(\bar h_r=n^{-1}\sum_i h_r(O_i)\) and

\[
\delta_r=(b_r-a_r)
\sqrt{\frac{\log(2R/\alpha)}{2n}}.
\tag{11.6}
\]

Let \(B_n=\prod_r[\bar h_r-\delta_r,\bar h_r+\delta_r]\), and set

\[
\mathcal C_\alpha^{\rm mom}
=\{u:\mathcal M_h(u)\cap B_n\ne\varnothing\}.
\tag{11.7}
\]

**Theorem 11.4 (finite-sample honest moment-box outer set).**  
Under cluster independence, the common-expectation condition, and the stated bounds,

\[
P_{u_0,\nu_0}\{u_0\in\mathcal C_\alpha^{\rm mom}\}\ge1-\alpha.
\tag{11.8}
\]

Moreover, on the same event, every \(u\) whose model expectation set contains the true expectation \(E h(O)\) is retained; thus (11.7) covers the entire moment-identified set with probability at least \(1-\alpha\).

**Proof.** Hoeffding’s inequality gives

\[
P\{ |\bar h_r-Eh_r|>\delta_r\}
\le2\exp\left[-\frac{2n\delta_r^2}{(b_r-a_r)^2}\right]
=\frac{\alpha}{R}.
\]

The union bound implies \(m_0=E h(O)\in B_n\) with probability at least \(1-\alpha\). Under the true model, \(m_0\in\mathcal M_h(u_0)\), so the intersection in (11.7) is nonempty. If another \(u\) is moment-observationally equivalent, then \(m_0\in\mathcal M_h(u)\), so it is retained on the same event. \(\square\)

This is the strongest fully finite-sample fallback available without a full count law. It can incorporate arbitrary dependence inside a sweep by defining each \(h(O_i)\) at cluster level. It is nontrivial only if response-derived expectation sets for separated \(u\) are separated by more than the concentration widths.

For \(m\)-dependent or mixing clusters, replace Hoeffding by a proved block/martingale concentration bound with its effective sample size. Calling every sweep independent without evidence is not permitted.

## 11.5 Uniformly asymptotic cluster-GMM route

A concise valid theorem can be stated by making the required stochastic result an explicit assumption rather than hiding it.

**Theorem 11.5 (uniformly calibrated GMM inversion under a uniform approximation).**  
Suppose for every null \(u\) and nuisance \(\nu\), a test statistic \(T_n(u)\) and critical value \(c_{n,\alpha}(u)\) satisfy

\[
\limsup_{n\to\infty}
\sup_{u,\nu}
P_{u,\nu}\{T_n(u)>c_{n,\alpha}(u)\}
\le\alpha.
\tag{11.9}
\]

Then \(\{u:T_n(u)\le c_{n,\alpha}(u)\}\) is uniformly asymptotically honest.

**Proof.** For the true \((u,\nu)\), noncoverage is the rejection event in (11.9). Taking the supremum and limsup gives the result directly. \(\square\)

For a multiplier/cluster bootstrap to establish (11.9), the project must verify: independent or uniformly mixing clusters; a uniform \((2+\delta)\)-moment bound; compact nuisance; stochastic equicontinuity of \(m_i(u,\nu)\); uniform covariance consistency; eigenvalues bounded away from zero on the tested moment span; and faithful rerunning of all preprocessing and selection. Near nonidentification, an Anderson–Rubin-style statistic such as (10.4) is preferable to Wald inversion. A theorem proved only for fixed nuisance values is pointwise, not honest over the class.

### 11.5.1 Complete route-to-inversion map for viable classes

Every A- or B-class recommendation is an inversion of a stated null test; changing the stochastic class changes the null law or valid restriction, not the coverage logic.

* **M1, M2a, M3a, M4a, M5a, and M6a (specified count laws).** For each \(u\), use the exact interval-code log likelihood appropriate to the class—Poisson (7.6), shared-gamma Cox (7.12), negative-binomial (7.10), Poisson-binomial (7.15), renewal (7.16–7.17), or finite-state FIFO (7.18). A recommended statistic is the generalized likelihood-ratio/deviance (10.2), with nuisance profiled in both null and alternative. Calibrate \(p_{u,\nu}\) by exact enumeration where feasible or complete-pipeline simulation; take the least-favorable supremum (11.2), or Berger–Boos (11.4) with an independently honest nuisance region; invert by (11.3). This is finite-sample honest and does not invoke Wilks.

* **M2b, M4b, M6b, M10b, M11b, and robust-overlap class M12b (bounded response/loss or moment partial identification).** For each \(u\), test whether the response/law envelope can satisfy the simultaneous code-probability or bounded-feature confidence region. The finite-sample statistic is the feasibility violation (10.7) or equivalently the indicator that \(\mathcal M_h(u)\cap B_n=\varnothing\); its level is controlled by the single simultaneous event in Theorem 11.4. Inversion gives an outer confidence region that covers the full moment-identified set, not merely one selected point.

* **M7a/M10a/M11a (replicated clusters with point-identifying moments).** The primary finite-sample test is the same moment-box feasibility test, now with a singleton population zero set. Cluster GMM (10.4) or empirical likelihood is a secondary, narrower route only after the uniform calibration condition (11.9) is established. Within-sweep dependence is unrestricted because the sweep/visit is the sampling cluster.

* **M8a/M8b (martingale/compensator classes).** For each \(u\), form the predeclared compensated score and reject only when a finite-sample exponential-supermartingale, bounded-jump, or predictable-variation inequality is violated uniformly over nuisance. Invert that upper bound exactly as in Theorem 11.1. If the compensator/response inequalities leave several \(u\)'s feasible, the result is B rather than A.

* **M12a (distributionally robust but separated law sets).** Use the minimax tail probability (10.9), with the inner law optimization certified over the fixed neighborhood; it is already a least-favorable p-value and is inverted by (11.3). If neighborhoods overlap, inference transitions to B or C according to Proposition 6.5 and Theorem 6.6.

No class is assigned A merely because an optimizer exists. A requires a point-identifying observable law/restriction plus the stated finite-sample or uniform-asymptotic calibration; B uses the same testing discipline but targets a compatible/identified set.

## 11.6 Bootstrap after quantization, refitting, and selection

**Theorem 11.6 (what exact simulation must reproduce).**  
Quantization and nuisance refitting do not invalidate Theorem 11.3 when they are deterministic/measurable components of the same algorithm applied to every observed and simulated complete dataset. Conversely, fixing a data-dependent mask/gate or replacing the full selected experiment by an unconditional simulator generally breaks exchangeability and has no exact-coverage guarantee.

**Proof.** In the valid case, each complete latent draw is passed through the same measurable map \(A\) consisting of compression, calibration, missingness, selection, fitting, and statistic calculation. Exchangeable inputs remain exchangeable after applying \(A\) componentwise, so Theorem 11.3 applies. If the observed statistic uses a random selection rule \(G(O)\) but simulations condition on the realized \(G(O_{\rm obs})\) rather than compute \(G(O^*)\), the observed and simulated statistics are images under different maps and need not be exchangeable. The counterexample below shows actual size distortion. \(\square\)

**Counterexample 11.6.1 (selection omitted from bootstrap).** Let \(Z\sim N(0,1)\), but analyze an observation only when \(Z>0\). The selected null law is half-normal. A bootstrap that ignores selection simulates \(N(0,1)\) and uses the nominal one-sided 5% cutoff 1.645. Under the actual selected law,

\[
P(Z>1.645\mid Z>0)=\frac{0.05}{0.5}=0.10.
\]

The true size is 10%, not 5%. Rerunning selection in unconditional simulations and conditioning on selected replicates, or simulating the exact conditional law, restores the experiment.

A plug-in nuisance bootstrap can be uniformly invalid at boundaries or weak identification even if selection is reproduced. Least-favorable or Berger–Boos calibration avoids relying on a nonuniform plug-in approximation.

## 11.7 Boundaries, weak identification, disconnected and empty sets

* A likelihood maximum on a boundary and a flat/plateau profile require no special asymptotic approximation under exact least-favorable simulation; they usually produce asymmetric, disconnected, or full-domain sets.
* Near observational equivalence, an honest procedure must lose power. A full-domain set is a valid result, not an algorithm failure.
* An **empty** inverted set means every candidate null was rejected by the procedure. It can occur with probability at most \(\alpha\) at a correctly specified true parameter, but it is evidence of global model conflict or numerical error—not a synonym for nonidentification. Nonidentification ordinarily produces a large/full set.
* A prefit regime failure should return “no inferential claim” or a full set under a declared conditional procedure, not an empty set labeled nonidentifiable.
* Grid inversion covers only grid-valued targets unless continuity plus a certified discretization bound is used. Linear interpolation of a noisy/profile grid can spuriously exclude off-grid values.

---

# 12. Nuisance, dependence, selection, and calibration treatment

## 12.1 Nuisance taxonomy

The nuisance vector must be divided because different components require different treatment:

\[
\nu=(\beta,\eta_{\rm resp},\eta_{\rm loss},\eta_{\rm dep},\eta_{\rm sel},S,\gamma).
\]

* \(\beta\): physical plasma/mirror/beam/background nuisance;
* \(\eta_{\rm resp}\): energy/angular response, exposure, anode efficiencies, geometric factors;
* \(\eta_{\rm loss}\): dead time, FIFO, rejection, masks, saturation, prescale;
* \(\eta_{\rm dep}\): Cox/cluster/renewal dependence parameters;
* \(\eta_{\rm sel}\): inclusion, gate, missingness mechanism;
* \(S\): discrete software/command/timing state;
* \(\gamma\): ground calibration and uncertainty.

Averaging over these quantities is a frequentist operation only if a repeated-sampling mixing law is part of the model. Integrating over a subjective prior is Bayesian sensitivity analysis and must be labeled separately. The frequentist default is profiling, conditioning, bounding, or least-favorable maximization.

## 12.2 Discrete state and association uncertainty

Unknown discrete state is not a continuous standard error. If a finite set \(\mathcal S_i\) of state decodings remains possible, the null nuisance set must include every state, and the p-value takes the supremum over them. If the state changes the inverse code interval or byte layout and the set is not enumerable from evidence, the observation law is undefined and the result is D.

A raw/calibrated association selected by maximizing shape correlation on the same data is another discrete nuisance/selection step. Valid options are:

1. establish association from independent timing/state information and treat it as fixed;
2. include all plausible associations in the nuisance set and use the least-favorable p-value;
3. split data: use one component to associate, another to infer, with the split and dependence justified;
4. simulate and rerun the association algorithm in every null replicate.

The 1998 \(-63.716\) s pattern supports a coarse ordering diagnostic, not option 1 by itself.

## 12.3 Response nuisance

A useful response parameterization should encode physical constraints rather than 1320 free scales. Examples include:

\[
\mu_i(u,\nu)=
D_i(\eta_{\rm eff})
R_i(\eta_E,\eta_\Omega)
\lambda_i(u,\beta)
+b_i,
\tag{12.1}
\]

where \(R_i\) is a nonnegative response matrix, \(D_i\) a diagonal exposure/efficiency map, and \(b_i\) background. Constraints should include nonnegativity, normalization where physically meaningful, shared anode factors across states, smooth energy response, and calibration anchors. Parameterize positive quantities on a log scale, probabilities by logits, and response-kernel rows by a simplex map. Bounds must come from calibration or validation, not be chosen to obtain a desired interval width.

When response dimension is large, exact least-favorable optimization may be replaced by an outer bound: enlarge the response set to a convex envelope, compute the robust p-value over that envelope, and accept the resulting conservatism. Shrinking the nuisance set because an optimizer failed is invalid.

## 12.4 Dependence and cluster definition

The natural minimum cluster is a full 15-row 3-D sweep. A visit-level cluster is required if adjacent sweeps share plasma state, instrument accumulation, or association/normalization effects. The cluster choice should be determined from timing/state and empirical autocorrelation before testing, not selected to minimize standard errors.

For cluster-GMM, write

\[
\widehat B=\frac1n\sum_i
(\psi_i-\bar\psi)(\psi_i-\bar\psi)^\top.
\]

It estimates arbitrary within-cluster covariance but assumes the chosen clusters satisfy the declared across-cluster condition. If clusters are serially dependent, use a predeclared block/HAC estimator or block bootstrap with a proved mixing rate and block sequence. Fifteen energy rows must never be entered as 15 independent clusters.

## 12.5 Missingness and sentinels

Missing constants must be removed before positivity masks, logarithms, row sums, gates, and normalization. For the bundled 3-D files, the exact predicate includes `flux == 9.999E-99` as missing, in addition to nonfinite values and any source-backed mask. A 1970 parsed timestamp associated with an all-sentinel spectrum is an invalid-time condition, not a physical observation.

Let \(R\) denote observation/retention. If

\[
P(R=1\mid Y,W,U,\nu)=P(R=1\mid W)=\pi(W)
\]

with observed \(W\) and known \(\pi(W)>0\), inverse-probability moments can be used. If \(\pi\) is bounded, partial identification is possible. Dead-time masking based on the unobserved high rate, flux-dependent gates, and sunlight contamination are generally not ignorable without a model. Hiding them in `mask` changes the target law and invokes Theorem 6.4.

## 12.6 Regime gates and conditional claims

A gate based only on externally observed state can define the experiment. A gate based on the same \(Y\) used to fit \(U\) is data-dependent selection. A valid confidence statement must be one of:

* unconditional, with the gate rerun in simulation and “no result” treated as an outcome;
* conditional on passing the gate, with the exact conditional null law simulated or derived;
* selective, with a selection-adjusted test.

A physics validity gate may be scientifically necessary, but it does not by itself supply a statistical distribution. If a gate declares the forward model invalid, the output should be “no model-based claim,” not a narrow interval and not an empty set called nonidentifiable.

## 12.7 Calibrated-flux covariance and quasi-likelihood

The calibrated product shares several covariance sources: logarithmic digitization, response/geometric-factor calibration, dead-time correction, row totals, background estimation, sequential energy timing, and any normalization. A diagonal per-cell variance of the form \((\sigma_{\rm rel}p)^2+\sigma_{\rm abs}^2\) is a proposed working choice, not a source fact. If used for point estimation, uncertainty must be based on cluster scores and a full covariance or on a simulator validated against replicated calibrated data.

A calibrated-flux quasi-likelihood can be a sensitivity analysis under a declared observation law. It becomes an A-class method only after the law or uniform moment conditions, covariance, response, selection, and nuisance calibration are supplied. The archive statement that reported errors include Poisson and digitization components does not validate independent Gaussian errors or the PR 14 covariance.

## 12.8 Contamination

For an \(\epsilon\)-contamination model

\[
P=(1-\epsilon)P_{u,\nu}^0+\epsilon H,
\]

with arbitrary \(H\), robust testing must take the supremum over \(H\). For a fixed event \(A\), the worst-case probability is

\[
\sup_H P(A)=(1-\epsilon)P_{u,\nu}^0(A)+\epsilon.
\tag{12.2}
\]

Thus a nominal p-value can be conservatively inflated to \((1-\epsilon)p^0+\epsilon\) for an event-tail test, before nuisance maximization. This is useful only for small, externally justified \(\epsilon\); if \(\epsilon\ge\alpha\), many rejection events cannot have robust size \(\alpha\), and if neighborhoods overlap across \(u\), identification becomes partial or absent.

## 12.9 Common random numbers

Common random numbers are not required for validity, but they are important for stable comparisons of nuisance values, response bounds, or model variants. Use a counter-based generator whose key is a tuple such as

\[
(\text{master seed},\text{null-}u\text{ box},\text{replicate},\text{latent stream}),
\]

and transform the same latent uniforms/normals under compared nuisance values. This reduces Monte Carlo noise in differences and makes branch-and-bound envelopes reproducible. PR 14’s floor/no-floor diagnostic uses `seed` and `seed+10000`, not common random numbers. `[Bundle: pr14/scripts/diagnostics/r_bound_artifact_check.py, lines 302–317]`

---

# 13. Numerical implementation and certification

## 13.1 Primary method: exact code-space composite-null inversion

This method is recommended only after the Section 17 inputs define a validated state-specific law.

### Pseudocode

```text
INPUT:
    raw merged records; detached labels/state files;
    physical forward family f(U,beta);
    response/loss model and compact nuisance sets H_u;
    confidence error alpha;
    numerical/Monte Carlo error budget delta_num;
    master RNG seed.

A. Decode observation
    1. Verify file hashes, record length, merge/status flags.
    2. Decode Digital Subcom/software/command state and event-time windows.
    3. Assemble each complete 15 x 88 3-D sweep.
    4. For every cell construct exact inverse set I_s(c), including only
       documented prescale and terminal rules.
    5. Mark unsupported or ambiguous state. If its nuisance set cannot be
       enumerated, output NO MODEL-BASED CLAIM; do not fit lower endpoints.

B. Define law
    6. Fold the continuous physical model through exposure, energy/angular
       response, efficiencies, background, masks, dead time/FIFO, and selection.
    7. Evaluate exact code-rectangle probabilities under the selected stochastic
       class (Poisson, shared-gamma Cox, renewal, etc.).
    8. Validate probability normalization and limiting cases by state.

C. Observed statistic for H0: U = u
    9. Compute a feasible lower bound L_alt on sup_{v,nu} log L(v,nu).
   10. Compute a certified upper bound U_null on sup_{nu in H_u} log L(u,nu).
   11. Set T_lower(u) = 2 max{0, L_alt - U_null}.
       If either optimization is unresolved, set T_lower(u)=0 and retain u.

D. Nuisance-calibrated p-value
   12. Construct an honest nuisance set B_u with error gamma, or use all H_u.
   13. Cover B_u by nuisance boxes. For each box choose a representative and
       derive a certified TV bound epsilon_box from every law in the box to it.
   14. Simulate the complete null experiment at each representative, including
       quantization, missingness, association/selection/gates, and refitting.
   15. Estimate the tail probability of T_lower by a one-sided binomial upper
       bound UCB_box, allocating total simulation failure probability delta_MC.
   16. p_cert(u) = min{1, gamma + delta_MC
                          + max_box(UCB_box + epsilon_box)}.

E. Continuum inversion
   17. Partition the admissible U domain into intervals.
   18. Reject a U interval only if a certified upper bound on p_cert(u) is below
       alpha for every u in the interval. Otherwise retain it.
   19. Refine boundary and unresolved intervals until the declared U tolerance.
   20. Output the union of retained closed intervals, full-domain/no-claim flags,
       nuisance and numerical certificates, hashes, seed, and diagnostics.
```

### Why the statistic is conservative

Let \(A=\sup_{v,\nu}\ell(v,\nu)\) and \(N_u=\sup_{\nu\in\mathcal H_u}\ell(u,\nu)\). If \(L_{\rm alt}\le A\) and \(U_{\rm null}\ge N_u\), then

\[
T_u^{\rm lower}=2[L_{\rm alt}-U_{\rm null}]_+
\le2[A-N_u]_+=T_u.
\tag{13.1}
\]

Using the same conservative algorithm in every simulation defines a legitimate statistic in its own right. Setting it to zero on failure makes rejection harder and cannot spuriously narrow the set.

### Certified nuisance approximation

For a fixed observed statistic threshold, tail probability is an event probability. If every law in nuisance box \(B\) is within \(\epsilon_B\) in total variation of representative law \(P_B\), then

\[
\sup_{\nu\in B}P_{u,\nu}(T\ge t)
\le P_B(T\ge t)+\epsilon_B.
\tag{13.2}
\]

A one-sided Clopper–Pearson upper bound \(U_B\) for the simulated binomial tail satisfies \(P\{p_B\le U_B\}\ge1-\delta_B\). Allocate \(\sum_B\delta_B\le\delta_{\rm MC}\). On the simultaneous good event, `max(U_B+epsilon_B)` upper-bounds the nuisance supremum. Adding \(\gamma+\delta_{\rm MC}\) to the reported p-value, or equivalently rejecting only below \(\alpha-\gamma-\delta_{\rm MC}\), keeps total size at most \(\alpha\) by the proofs of Theorems 11.1–11.2 and a union bound.

A TV bound can be obtained from interval-probability derivatives, Hellinger/KL bounds, or a validated coupling. For independent cells, product-law TV is at most the sum of marginal TVs; this can be loose but safe. For shared-gamma models, bound the conditional product distance and integrate over the common effect.

### Optimization and nuisance parameterization

* Enumerate discrete format/state alternatives.
* Use log parameters for positive scales and backgrounds; logits for probabilities; softmax rows for response kernels; Cholesky factors for covariance.
* Keep every nuisance domain compact. Infinite/unbounded domains either require an analytic tail reduction or place the model in a nonuniform/C class.
* Exploit conditional concavity in simple scale/background parameters when available. Profile those with bracketed one-dimensional solvers.
* Use interval branch-and-bound, deterministic Lipschitz global optimization, or a verified convex relaxation for the remaining nuisance. Multistart local optimization is useful for feasible lower bounds but is not a global certificate.
* A failure to prove exclusion retains the candidate \(u\).

### Stable probability calculations

Use (7.5)–(7.6), library log-CDF/log-SF implementations, and log-sum-exp. For shared-gamma quadrature, increase order until an interval/error estimator bounds the difference below a declared likelihood tolerance. If a cell probability underflows, retain a rigorous log upper/lower bound rather than replacing it by zero. Verify each scalar code probability sums to one over codes for a grid of \(\mu\) and state values.

### Continuum and set representation

A point grid followed by linear interpolation is not a coverage certificate. Represent the result as retained \(U\)-boxes. A box is excluded only when the entire composite null over that box is rejected after adding statistic/law Lipschitz bounds. All unresolved boxes remain. This automatically allows disconnected and one-sided sets. Store all components dynamically; never truncate to a fixed component count without an explicit overflow representation.

### Complexity

Let \(n\) be clusters, \(d=1320\), \(G\) candidate \(U\)-boxes, \(J\) nuisance boxes, \(B\) simulations, and \(Q\) quadrature nodes. A direct independent-count likelihood costs \(O(nd)\) per parameter evaluation and roughly \(O(GJnd)\) before simulation. Shared-gamma evaluation costs \(O(Qnd)\). Full calibration is approximately \(O(GJB\times\text{fit cost})\), so nuisance reduction, reusable latent simulations, response precomputation, and sequential Monte Carlo stopping are essential. Computational cost does not justify replacing a least-favorable calibration by an unqualified plug-in.

## 13.2 Best fallback: bounded-feature moment outer set

This route requires repeated independent/mixing clusters and a response-derived expectation set, but no count distribution.

```text
INPUT:
    observed C or correctly masked Y clusters O_i;
    predeclared bounded features h_r(O_i);
    model expectation set M_h(u) from response/calibration bounds;
    alpha and U-domain tolerance.

1. Define clusters before inspecting fit residuals.
2. Apply the same deterministic missingness/state rules to every cluster.
3. Compute h_r(O_i) and verify hard bounds [a_r,b_r].
4. Form the simultaneous Hoeffding box B_n using equation (11.6), or a
   proved block/martingale bound for dependent clusters.
5. For each U-box, solve whether some u in the box and nuisance nu satisfy
   m(u,nu) in B_n. Use interval/convex outer bounds.
6. Exclude a U-box only when infeasibility is certified. Retain unresolved boxes.
7. Return the union as an outer confidence region, with the exact moment target,
   cluster assumption, response bounds, and numerical certificate.
```

Candidate features include code-threshold indicators \(1\{C_j\le c\}\), bounded angular contrasts, missingness indicators, and interval lower/upper endpoints when finite. Threshold indicators are especially robust to terminal uncertainty. The feature set must be chosen before outcome-driven tuning or that selection must be included in calibration.

The feasibility problem is often convex if mean envelopes are affine in bounded response parameters; hard loss-cone masks can make it mixed-integer/nonconvex. An outer convex relaxation preserves coverage at the cost of width.

## 13.3 Monte Carlo reproducibility and stopping

Use master seed `20260722` for the proposed validation campaign, with a counter-based RNG and stored substream keys. For each tail estimate, use sequential batches and stop only when a one-sided confidence bound proves either:

\[
U_B+\epsilon_B+\text{adjustments}<\alpha
\quad\text{or}\quad
L_B-\epsilon_B>\alpha,
\]

or when a declared simulation cap is reached; unresolved candidates are retained. Record the number of draws and exact binomial bounds. At level 0.05, a few dozen draws give only a coarse rank test; precise least-favorable decisions generally require far more. No fixed simulation grid proves coverage over unsearched nuisance values.

## 13.4 Numerical audit trail

Every confidence-set record should include:

* observable type and source hashes;
* state decoder/version and inverse-interval rule per cell;
* physical/response/loss model version and nuisance bounds;
* all preprocessing, missingness, gate, and association rules;
* statistic definition and optimization lower/upper bounds;
* retained \(U\) boxes before display rounding;
* Monte Carlo seed/substreams, draws, tail counts, and confidence bounds;
* quadrature and arithmetic tolerances;
* unresolved/failure flags that force retention;
* a declaration of finite-sample, uniform asymptotic, pointwise, or sensitivity status.

---

# 14. Simulation and coverage-validation plan

Simulation is an implementation audit, not a substitute for Theorems 11.1–11.4. It should attack the exact places where finite-sample size or numerical certification can fail.

## 14.1 Reproducible design

Use master seed `20260722` with counter-based substreams. Run both a reduced geometry for unit tests and the full \(15\times88\) geometry. Store every scenario, parameter, state, and seed key. A proposed operational stress domain, inherited only as an engineering range from PR 14 rather than a source fact, is \(U\in[-300,20]\) V.

For each scenario, simulate until either at least 10,000 independent repeated experiments have been completed or the two-sided 95% Clopper–Pearson interval for empirical noncoverage has width at most 0.01, with a larger cap for near-boundary cases. Report the exact binomial interval. A passing finite panel supports code correctness only; it does not establish uniform coverage between panel points.

## 14.2 Required scenario families

| Family | Concrete settings and purpose |
|---|---|
| Ordinary interior | Known-response Poisson and shared-gamma Cox; \(U\) at several interior values; moderate means spanning codes 0 through wide bins; nuisance away from bounds |
| Parameter boundaries | \(U\) at both domain limits; mirror/beam/background/dispersion at lower and upper bounds; zero beam; response coefficients at bound faces |
| Weak identification | Very low efficiencies; most probability in code zero; hard-grid plateau; nearly collinear \(\partial_U\mu\) and nuisance tangent; shared gamma variance large; NB \(r\) near \(r_{\min}\) |
| Quantization/tails | Means immediately below/above code thresholds; broad bins; terminal-code mass; prescaled-state tests where documented; all-zero and near-saturation sweeps |
| Dependence | Shared random effects, finite-range serial dependence, visit-level effects, row-correlated calibration; compare correct clustering to false cell/row independence |
| Loss processes | Nonparalyzable dead time at low/high rates; finite FIFO near capacity; bounded versus misspecified loss; total/rate diagnostic strata |
| Response uncertainty | Least-favorable efficiency/energy-kernel distortions, anchor-cell perturbation, correlated flat-field error, state mismatch, energy-center shift |
| Missingness/selection | Full and sparse `9.999E-99`; invalid timestamps; sunlight masks; flux-dependent regime gates; association selected from multiple candidates; rerun versus fixed selection |
| Contamination | Huber contamination at 0, half the declared radius, and the radius boundary; adversarial all-zero/high-tail contamination |
| Model misspecification | Poisson analysis of Cox/NB/dead-time data; PR14 Gaussian analysis of quantized/missing calibrated data; response kernel outside bounds |
| Numerical adversaries | Multimodal nuisance profile, disconnected true set, optimizer failure, quadrature under-resolution, Monte Carlo tail near \(\alpha\), many components |

## 14.3 Metrics

For each scenario report:

* true-parameter coverage and exact binomial uncertainty;
* coverage of the entire identified set for the moment/partial-ID method;
* probability of full-domain, disconnected, one-sided, and empty sets;
* set Lebesgue measure and component count;
* type-I error by nuisance value and the empirical least-favorable nuisance;
* power against fixed separations \(|u-u_0|\);
* numerical failure/retention frequency;
* calibration error attributable to finite simulation and nuisance covering;
* sensitivity to clustering and block length;
* runtime by \(n,d,G,J,B,Q\).

For the exact test, empirically verify that rejection never exceeds the theorem’s nominal bound plus simulation uncertainty. For the fallback moment set, verify the simultaneous moment-box event and identified-set inclusion separately.

## 14.4 Adversarial equivalence tests

Every claimed A model should include a numerical search for distinct \(u_1,u_2\) and nuisance values minimizing a law distance such as Hellinger or TV. The search should include:

\[
\inf_{\nu_1,\nu_2} d\{P_{u_1,\nu_1}^O,P_{u_2,\nu_2}^O\}.
\]

If the certified lower bound is zero for separated values, narrow the claim to B/C. For bounded response, compare the computed identified set against the explicit response-absorption construction as bounds widen. For NB, verify convergence to all-zero as \(r\downarrow0\). For selection, reproduce Counterexample 11.6.1 and confirm that rerunning the gate restores nominal size.

## 14.5 PR 14 validation replacement

Replace the current twelve-trial threshold smoke with:

1. exact self-model tests of the complete simulator and criterion;
2. nuisance-grid and least-favorable coverage panels;
3. selection/missingness rerun tests;
4. permutation/order-invariance tests;
5. tests that `bootstrap_n=0`, optimizer failure, and unsupported state cannot create a narrow set;
6. a held-out archive diagnostic after single-sweep loading is wired;
7. an explicit label: self-model Monte Carlo validation, not a proof of archive coverage.

---

# 15. Empirical diagnostics using the bundle

## 15.1 Reproducibility files

The read-only audit scripts and outputs are supplied in `/mnt/data/lp_er_audit/`:

```text
audit_reference.py            reference_audit.json / reference_audit_stdout.txt
raw_frame_audit.py            raw_frame_audit.txt
sentinel_patterns.py          sentinel_patterns.txt
current_code_smoke.py         current_code_smoke.txt
pr14_audit_debug.py           pr14_audit_debug_stdout.txt
code_snippets_full.txt        line-numbered audit excerpts
```

The main audit uses no random sampling. The PR 14 simulator/covariance diagnostic uses seed `20260722`; its small profile smoke reproduces the repository seed 42. The commands are:

```bash
cd /mnt/data/lp-er-honest-ci-research-bundle
python /mnt/data/lp_er_audit/audit_reference.py
python /mnt/data/lp_er_audit/raw_frame_audit.py
python /mnt/data/lp_er_audit/sentinel_patterns.py
python /mnt/data/lp_er_audit/current_code_smoke.py
python /mnt/data/lp_er_audit/pr14_audit_debug.py
```

## 15.2 Raw and calibrated record results

| Check | 1998 anchor | 1999 PR day | Interpretation |
|---|---:|---:|---|
| Raw merged records | 7,047 | 7,285 | Matches fixed-record parsing |
| Accepted complete sweeps under current parser | 165 | 0 | 1998 version 1 supported; 1999 version 5 unsupported |
| Raw code range | 0–153 | not decoded | No terminal-code evidence in 1998 sample |
| Raw zero-code fraction | 0.458636 | not decoded | Many low-count/zero bins; interval likelihood must handle atoms |
| Calibrated 15-row spectra | 1,032 | 1,055 | These are time-varying spectra, not automatic replicates of one \(U\) |
| Missing sentinel cells | 1,320 | 2,658 | One all-missing 1998 spectrum; two all-missing plus 18 sparse cells in 1999 |
| Zero calibrated flux cells | 602,203 | 495,061 | Log criteria require explicit zero law/treatment |
| PR 14 mask retains sentinels | all 1,320 | all 2,658 | Definite preprocessing defect |

The single largest 1998 code, 153, has lower endpoint 6,400 and ordinary upper endpoint 6,655. Treating it as exactly 6,400 loses 256 possible accepted-count values. No code at or above 240 was observed in the anchor, so terminal behavior cannot be inferred empirically from that file.

The version audit found 441 version-1 observations in the 1998 relevant frames and 455 version-5 observations in 1999. Since the bundle documents version 3 but not version 5, no byte-layout extrapolation was made.

## 15.3 Association result

The 1998 association search generated 2,084 time-near raw/calibrated pairs. The top correlation was 0.841996; the dominant offset was ERT minus calibrated sweep end \(-63.716\) s. This supports common ordering and a stable latency pattern. It does not identify event time or response, and it uses shape similarity selected from candidates. The 1999 association has no candidates because the raw parser does not support the state.

## 15.4 Current-code smoke audit

Ten manual tests passed: lower endpoints; complete assembly; non-real-time interruption; duplicate and out-of-order rejection; hybrid, format-3, mixed-version, and missing-version rejection; and ERT offsets. Full repository `pytest` could not be completed in the execution environment because the test import path requires `spiceypy`, which is not installed and network package installation was unavailable. The manual result therefore verifies only the isolated decoder behaviors listed, not the full suite.

## 15.5 PR 14 reproducible diagnostics

1. **Sentinel.** `build_calibration_mask([[1, 9.999e-99, 0, nan]])` returned `[True, True, False, False]`.

2. **Order dependence.** For the same three-cell data/model pair under different cell permutations, the quasi-NLL was `-3.1971`, `-4.5485`, `-4.2481`, and `-4.5485`. A coordinate-invariant simplex statistic cannot change this way.

3. **Simulator/criterion mismatch.** With 200,000 draws at seed 20260722, simulated normalized proportions had covariance

\[
\begin{pmatrix}
0.002204&-0.001457&-0.000747\\
-0.001457&0.001619&-0.000161\\
-0.000747&-0.000161&0.000908
\end{pmatrix},
\]

with correlations \(-0.772\), \(-0.528\), and \(-0.133\), while the working diagonal variances were `(0.005626, 0.002026, 0.000901)`. Clipping and renormalization necessarily induce the negative correlations omitted by the criterion.

4. **Threshold smoke.** Under the self-simulator, twelve trials with cutoff 25 covered 12/12 and had median span 23.01 V. The same twelve trials with cutoff 1 covered 3/12 and had median span 1.26 V. This illustrates cutoff sensitivity and reproduces why curvature alone is not coverage.

5. **Fallback and serialization.** `bootstrap_n=0` returned `c_alpha=1.0`. A synthetic ten-component set reported ten components but serialized only eight non-NaN lower endpoints because the storage schema has eight slots.

These empirical observations support the PR disposition below. None is a coverage theorem for Lunar Prospector data.

---

# 16. PR 14 audit and disposition

## 16.1 Mathematical claim

The document correctly labels the target as a calibrated-flux quasi-likelihood rather than a Poisson or multinomial count likelihood and recognizes two-sided, one-sided, disconnected, and full-domain sets. `[Bundle: pr14/docs/physics/profile_likelihood_ci.md, lines 1–16 and 38–43]` However, “parametric bootstrap through the same observation model” is not enough for honest coverage unless that model is maintained, nuisance is handled uniformly, and the complete selected pipeline is reproduced. Those conditions are not currently met.

## 16.2 Component disposition

| Component | Audit finding | Disposition |
|---|---|---|
| Measurement-contract distinctions | Correctly distinguishes calibrated flux, raw codes, pseudo-counts, and missing response; correctly says raw codes are currently unusable for inference | **Retain** |
| Quasi-likelihood naming | Avoids false Poisson/multinomial label | **Retain**, but call present output a working-criterion sensitivity profile, not an honest confidence set |
| Default noise and response knobs | `sigma_rel=.15`, `sigma_abs=.001`, `de_over_e=.5`, blur/background are not supplied by the archive law | **Relabel as heuristic** until calibrated/validated; replace defaults by sourced bounds/models |
| Row normalization | May remove row amplitude as a working transformation, but is not conditional multinomial likelihood from coded counts | **Relabel as heuristic**; derive the actual transformed mean/covariance or use code likelihood |
| Dropped-cell diagonal NLL | Drops the last valid cell and uses diagonal, model-dependent variances; violates Theorem 10.1 and empirical permutation invariance | **Replace** with full tangent-space covariance/generalized inverse or valid logistic-coordinate law |
| Bootstrap simulator | Adds independent Gaussian noise, clips, renormalizes, and rescales; this creates correlated proportions not represented by the NLL | **Replace** with a coherent probability law or treat criterion as GMM with cluster calibration |
| Calibration mask | `finite & >0` includes `9.999E-99` and all-missing spectra | **Repair immediately** before any fitting/gating/normalization |
| Energy quadrature | Uniform nodes over an assumed width; product source states 25% resolution but no kernel; PR default is 50% | **Relabel as heuristic**; replace with state-specific response kernel/uncertainty set |
| Pitch blur | Assumes uniform pitch sampling from one row and applies ordinary convolution; 88 3-D cells are geometry-dependent | **Replace** with the actual angular response matrix or bounded response operator |
| Prefit energy gate | Weighted variance of \(1/E\) is a diagnostic proxy, not a proved Fisher-information condition; row sums include positive sentinels | **Repair and relabel diagnostic**; define external/selection law and rerun in calibration |
| Postfit beam gate | Scientifically useful discrepancy flag, but postfit data-dependent selection | **Retain as diagnostic**, not automatic confidence-set calibration unless simulated/conditioned |
| Nuisance profiling | Coarse \((r,A)\) grids and one-dimensional local polish in \(r\) at the best amplitude have no global certificate `[profile_ci.py, lines 70–140]` | **Replace/repair** with certified optimization; failure must retain \(u\) |
| Plug-in bootstrap | Simulates at the fitted \((\hat U,\hat r,\hat A)\), keeps the mask fixed, uses a coarser free-\(U\) grid, and does not take a nuisance supremum `[profile_ci.py, lines 221–291]` | **Reject as honest calibration**; use LF/BB or prove uniform plug-in validity |
| Preprocessing/selection in bootstrap | Mask and prefit gate are built once; missingness, gate, association, and state uncertainty are not regenerated | **Replace** with full-pipeline simulation |
| `bootstrap_n=0` / `skip_bootstrap` | Returns or uses cutoff 1 with an “asymptotic chi-square” comment `[profile_ci.py, lines 287–291 and 376–381]` | **Reject for inferential output**; no interval unless a valid cutoff is supplied |
| Grid refinement | Bootstrap cutoff is calibrated before profile-grid refinement; refined fit can change \(\hat U,\hat r,\hat A\) without recalibration `[profile_ci.py, lines 383–421]` | **Repair**; define one algorithm and apply it identically in all replicates |
| Level-set interpolation | Linearly interpolates grid crossings and treats nonfinite profile values as rejected `[confidence_set.py, lines 261–338]` | **Replace** with outer box certification; nonfinite/failure values retain, not reject |
| Empty-set semantics | Empty components are labeled `NONIDENTIFIABLE` `[profile_ci.py, lines 423–435]` | **Repair**: empty means global rejection/numerical issue; nonidentification usually yields full/large set |
| Component schema | Stores only eight components but records the untruncated count | **Repair** with dynamic/ragged representation or explicit overflow/full envelope |
| Coverage tests | Eight MLE trials; twelve “coverage” trials; cutoff 25; pass condition coverage at least 0.5 `[tests/test_profile_ci_coverage.py, lines 33–140]` | **Replace** with Section 14 validation; current test is a geometry smoke only |
| Archive diagnostic | Script explicitly says archive single-sweep loading is not wired and only demo is available `[scripts/diagnostics/losscone_profile_ci.py, lines 129–133]` | **Reject archive-readiness claim** |
| CLI/batch routing | Main CLI does not expose/select profile inference; top-level run calls the ordinary pipeline; batch choices are only Halekas/Lillis `[cli.py; pipeline.py, lines 823–846; batch.py, lines 325–330]` | **Repair** before any user-facing claim; record method in outputs |
| Common random numbers | Floor/no-floor comparison changes seed by 10,000 | **Repair** for stable paired comparisons |
| Disconnected/full/one-sided schema | Conceptually correct and worth preserving | **Retain**, after numerical and serialization repair |

## 16.3 What survives

The forward mean and profile code can remain as a **diagnostic sensitivity surface** after sentinel correction and explicit parameter labeling. It can help expose plateaus, nuisance boundaries, and response sensitivity. It cannot be promoted to an honest confidence set by adding more bootstrap repetitions while leaving the observation law, nuisance, selection, and response unidentified.

The major PR claim should therefore be changed from “coverage-calibrated profile-likelihood confidence set” to “calibrated-flux working-model profile/sensitivity set” unless and until one of the A/B constructions in this memorandum is implemented and its assumptions are documented.

---

# 17. Exact additional information needed from the TBS/calibration package

## 17.1 Necessary operational inputs

A code-space observation law requires, by state and epoch:

1. **Format/state decoder:** the authoritative mapping of Digital Subcom/software versions—including version 5—to 3-D byte layout, energy table, command changes, and header semantics.
2. **Compressor terminal rule:** counter width, maximum, overflow/saturation behavior, and exact inverse set for `FF`; explicit confirmation of whether any 3-D cells are prescaled and how that state is telemetered.
3. **Event-time/exposure:** reconstruction of each 3-D cell’s integration window from major-frame time, spin phase/period, header state, merge/delayed selection, and latency; exposure for polar sums and sequential energies.
4. **Energy response:** passband/response matrix for each commanded energy state, not merely center energies or \(\Delta E/E\).
5. **Angular/anode response:** the missing 256-effective-anode efficiency table \(F\), anode-to-3-D-cell mapping, ADC differential-nonlinearity correction, MCP gain/flat field, and uncertainty/covariance.
6. **Geometric normalization:** exact relationship among the full-instrument \(0.07E\) approximation, cell factors, and the product-level \(0.02\) conversion.
7. **Accepted-event chain:** masks, pulse-height discriminators, background, dead-time model, FIFO routing/capacity/service, rejection, and their state dependence.
8. **Onboard arithmetic:** the `Appendix TBS` phase-space accumulation sequence, integer precision, rounding/truncation, overflow, summing order, and retrace handling.
9. **Ground calibration:** the software/algorithm mapping codes and state to archived \(Y\), including dead-time mask, background treatment, rotations, rounding, missing sentinels, and validation examples.
10. **Uncertainty envelope:** calibration covariance or defensible bounds tied across cells/states, not independent arbitrary cell scales.
11. **Selection/quality:** sunlight, saturation/dead-time validity, outage, merge flags, and any product-generation exclusion rules.
12. **Cross-product validation:** reference cases reproducing 3-D calibrated flux and the low/high-resolution aggregate products from raw codes and state.

The bundle specifically names the anode efficiency table, onboard arithmetic appendix, detailed operation description, MAG/ER Commanding document, and spacecraft-to-instrument ICD as absent or referenced-but-unavailable. `[Bundle: KNOWN_FACTS_AND_OPEN_QUESTIONS.md, lines 80–106; primary_sources/magerfmt.txt, lines 4–8, 503–514, and 698–721]`

## 17.2 Sufficiency proposition

**Proposition 17.1 (information sufficient to make code-space inference operational).**  
Suppose the recovered package supplies either:

* an evaluable probability mass function \(p_{u,\nu}(C,S)\) for every candidate \(u\), all nuisance \(\nu\) in a compact documented set, and every observed state; or
* a validated exact simulator for that law plus certified TV/likelihood bounds over nuisance boxes;

and suppose the profiled law sets for separated \(u\) are disjoint. Then the least-favorable construction of Theorem 11.1, with the numerical certification of Section 13, yields an operational finite-sample honest confidence set. Exact recovery of latent \(K\) is unnecessary.

**Proof.** The evaluator/simulator defines every composite null distribution of the exact observable. Compact nuisance and certified bounds make the supremum in (11.2) computably upper-bounded. Disjoint profiled law sets provide identification; Theorem 11.1 gives coverage; Section 13 ensures numerical approximations only enlarge p-values/retained sets. Lemma 5.1 supplies the code push-forward, so latent count recovery is not required. \(\square\)

The conditions are close to necessary in this experiment: without inverse intervals the observable law is undefined; without a response restriction Theorem 6.1 applies; without selection control Theorem 6.4 applies; without a dependence/count evaluator only moment-based class B is available.

## 17.3 Acceptable substitute for historical TBS

A first-principles historical package is not the only possible route. A state-aware empirical archive emulator can substitute if it:

1. predicts the joint code-to-calibrated observation law on held-out periods;
2. is tied across cells/states strongly enough to defeat response absorption;
3. supplies a validated uncertainty set or robust neighborhood covering emulator error;
4. reproduces 3-D and independent low/high-resolution products;
5. preserves quantization, missingness, state, and selection;
6. is used as an empirical observation model, not mislabeled as detector physics.

Without a coverage-calibrated error envelope, an emulator is a point predictor and remains a sensitivity tool.

---

# 18. Recommended staged research and implementation plan

## Stage 0 — Freeze claims

Do not report PR 14 outputs as confidence intervals. Correct the sentinel immediately, preserve raw codes as codes, and label all current \(U\) profiles as heuristic/sensitivity results. Unsupported version/state must produce no model-based claim.

## Stage 1 — State and timing provenance

Implement an authoritative mission-wide decoder from source documents, beginning with version 5. Decode 3-D layout, header, spin period/phase, merge flags, and cell integration windows. Validate the 1998 \(-64\) s association on held-out raw/calibrated segments and then the 1999 day. Keep association and event-time claims separate.

**Exit criterion:** every analyzed cell has a documented state, code interval, and physical exposure window, or is excluded with an explicit reason known before the fit.

## Stage 2 — Observation operator

Recover the TBS/calibration package or build a state-aware emulator with uncertainty bounds. Reproduce the archived 3-D flux from raw codes on training cases and both low/high-resolution aggregate products on held-out cases. Quantify response and calibration error jointly across cells.

**Exit criterion:** an evaluable/simulable \(P(C\mid U,\nu,S)\) or a bounded expectation/law set; no free cellwise response capable of Theorem 6.1.

## Stage 3 — Identification audit

For the continuous response-folded physical model, verify global or local rank and search numerically/certifiably for observational equivalence across \(U\). Test hard-grid plateaus, nuisance boundaries, near-zero efficiency, and shared mixing. If separation fails, define and compute the identified set rather than forcing a point estimate.

**Exit criterion:** an explicit A/B/C classification for each state/regime and a scientifically approved nuisance domain.

## Stage 4 — Inferential engine

Implement exact interval probabilities and the shared-gamma Cox alternative first. Add least-favorable/BB inversion, conservative optimization, continuum box retention, and full-pipeline simulation. In parallel implement the finite-sample bounded-feature moment outer set as the weaker-law fallback.

**Exit criterion:** theorem assumptions encoded in configuration/output, optimizer failures retain candidates, and no uncalibrated cutoff path exists.

## Stage 5 — Validation

Run Section 14’s simulation panel, equivalence attacks, numerical certificates, and held-out 1998/1999 diagnostics. Use common random numbers for paired model comparisons. Publish empirical coverage with exact binomial intervals and distinguish it from the theorem.

**Exit criterion:** self-model size at or below nominal within Monte Carlo uncertainty, least-favorable nuisance coverage, and documented degradation under misspecification.

## Stage 6 — Scientific deployment

Report per-sweep \(U\) only in states/regimes classified A. Report identified/outer sets in B. Report no nontrivial frequentist claim in C, and the exact missing fact in D. Archive all state, response, selection, nuisance, numerical, and random-seed provenance with each result.

---

# 19. Theorem, proposition, and lemma index

| Number | Result | Required deliverable addressed |
|---|---|---|
| Lemma 5.1 | Exact push-forward from accepted integer accumulators to code intervals, with prescale and terminal qualifications | Quantized-observation lemma |
| Corollary 5.1.1 | Lower endpoints/code support do not identify latent \(K\) or its mean | Nonconflation of code and count |
| Theorem 5.2 | Identification of Poisson means from code-zero probabilities; point ID iff profiled mean sets are disjoint; compact uniform separation | Known-response identification theorem |
| Corollary 5.2.1 | Local Jacobian/rank sufficient condition | Local identification boundary |
| Proposition 5.3 | Finite hard-threshold forward grids create observationally equivalent \(U\)-plateaus | Weak/partial identification in current model |
| Proposition 5.4 | Known bijective calibration preserves ID; coarsening contracts TV | Code-space versus flux-space analysis |
| Theorem 6.1 | Arbitrary cellwise response/background exactly absorbs any loss-cone change | Explicit response nonidentification theorem |
| Theorem 6.2 | Arbitrary vector Cox mixing produces the same random intensity law for every \(U\) | Explicit stochastic-law nonidentification theorem |
| Theorem 6.3 | Quantile-map equivalence under arbitrary monotone continuous calibration | Unknown-calibration obstruction |
| Theorem 6.4 | Arbitrary outcome-dependent selection can generate a common retained law | Selection obstruction |
| Proposition 6.5 | TV robust neighborhoods overlap iff center distance is at most twice radius | Distributionally robust boundary |
| Theorem 6.6 | Common observable law forces expected confidence-set measure at least \((1-\alpha)|\mathcal U|\) | Broadest-class impossibility/triviality theorem |
| Proposition 6.7 | NB shape \(r\downarrow0\) converges to all-zero and destroys uniform distinguishability | Point ID versus honest uniform inference boundary |
| Proposition 9.1 | Sharp bounded-response identified set and coordinatewise feasibility formula | Partial-identification proposition |
| Theorem 10.1 | Full-covariance simplex statistic is invariant to dropped cell/coordinate basis | Compositional invariance result |
| Proposition 10.2 | Exact lognormal conditions and finite-remainder regime for log-MSE | Log-MSE theorem/counterexample |
| Theorem 11.1 | Least-favorable p-value inversion has finite-sample uniform coverage | Primary confidence-set coverage theorem |
| Theorem 11.2 | Berger–Boos nuisance adjustment is finite-sample valid | Nuisance calibration theorem |
| Theorem 11.3 | Full-pipeline Monte Carlo rank test is exact/conservative | Exact simulation calibration |
| Theorem 11.4 | Bounded cluster features give a finite-sample honest moment outer set | Best fallback coverage theorem |
| Theorem 11.5 | Uniform size approximation implies uniformly asymptotic GMM coverage | Explicit asymptotic status |
| Theorem 11.6 and Counterexample 11.6.1 | Quantization/refitting are harmless if reproduced; fixed data-dependent selection can double size | Bootstrap/selection theorem and counterexample |
| Proposition 17.1 | A complete state-specific law evaluator/simulator plus separation is sufficient for operational code-space inference | TBS/calibration sufficiency proposition |

No result uses the missing `TBS` material as if it existed. Positive A/B statements are conditional on the assumptions stated in their theorem or decision-matrix row.

---

# 20. Assumption ledger

## 20.1 Source facts used as facts

| ID | Fact | Evidence |
|---|---|---|
| F1 | The 3-D product is 88 angular cells by 15 sequential energies, accumulated over a half-spin and read every 16 spins | `primary_sources/magerfmt.txt`, lines 23–62, 551–560, 691–697 |
| F2 | Compressor lower endpoints and truncation are documented for nonterminal codes | `primary_sources/magerfmt.txt`, lines 964–991; PDF p. 27 |
| F3 | Factor-eight prescale is documented for specified pitch-angle accumulations, with header state; it is not explicitly documented for 3-D cells | `primary_sources/magerfmt.txt`, lines 342–351, 503–531 |
| F4 | Total/rate counters have different prescales, masks, dead-time/FIFO/rejection content and are not exact 3-D totals | `primary_sources/magerfmt.txt`, lines 532–550 |
| F5 | Anode efficiency table and exact onboard arithmetic are `TBS` | `primary_sources/magerfmt.txt`, lines 698–721 |
| F6 | ERT is station receive time in a merged prior/delayed stream, not event time | `primary_sources/lvl0sis.txt`, lines 492–626 |
| F7 | Calibrated flux processing includes decompression, exposure/rate conversion, dead-time correction/mask, retained background, and \(0.02\) geometric conversion | `primary_sources/data-3deleflux-desc.txt`, lines 80–114 |
| F8 | Product states 25% energy resolution, sunlight/potential cautions, and no potential correction | `primary_sources/data-3deleflux-desc.txt`, lines 19–26 and 256–281 |
| F9 | Reported product errors omit absolute geometric-factor uncertainty | `primary_sources/data-electron-desc.txt`, lines 104–112 |
| F10 | `9.999E-99` is a missing constant | `reference_data/1999_pr_day/3D990429.LBL`, lines 156–165 |
| F11 | Current code extracts lower endpoints and supported version-1/2 structural sweeps but disclaims calibration/timing/response | `current_branch/src/losscone/level0.py` and tests |
| F12 | The 1998 anchor has 165 accepted sweeps; 1999 relevant state is version 5 and unsupported | Reproducible scripts in `/mnt/data/lp_er_audit/` |

## 20.2 Proposed assumptions required for positive inference

| ID | Proposed assumption | Consequence if adopted and verified |
|---|---|---|
| A1 | Every analyzed cell has a correct state-specific inverse code interval, including terminal behavior | Makes code probability well-defined |
| A2 | A continuous, state-specific response/loss mean or full law is known up to compact nuisance; response is tied across cells/states | Prevents Theorem 6.1 absorption and enables A/B inference |
| A3 | The profiled observation-law sets are separated for distinct \(U\); local rank holds where point inference is claimed | Point identification/uniform distinguishability |
| A4 | Accepted-count law is one specified family: independent Poisson, shared-gamma Cox, compact NB, Poisson-binomial, renewal, or finite-state loss | Defines exact interval likelihood and simulator |
| A5 | Nuisance domains, including dispersion and efficiency, exclude degeneracies such as \(r\downarrow0\) or zero response unless full sets are accepted | Uniform honest power/width possible |
| A6 | Selection, missingness, gates, and association are predetermined, modeled, bounded, conditioned upon exactly, or rerun in simulation | Prevents Theorem 6.4 and bootstrap failure |
| A7 | For repeated-cluster methods, clusters are independent/mixing and share a common \(U\), specified regression, or predeclared local-homogeneity target | Makes cluster moments estimate a defined parameter |
| A8 | Bounded-feature ranges or uniform moment/CLT/covariance conditions hold | Enables finite-sample moment or uniform asymptotic GMM coverage |
| A9 | Response/calibration bounds and robust-neighborhood radii are source/validation based and fixed before fitting | Gives interpretable B/robust A sets |
| A10 | Global/numerical and Monte Carlo errors are certified conservatively; failure retains \(U\) | Prevents numerical undercoverage |

## 20.3 Unverified conjectures or unresolved facts—not used as established premises

| ID | Item | Status and resolution |
|---|---|---|
| U1 | Digital Subcom version 5 uses a particular format-3-compatible 3-D layout | Unresolved; requires authoritative commanding/format/ICD source or independently validated decoding |
| U2 | `FF` is a saturation bin with a particular finite or infinite upper inverse | Unresolved; requires counter width/overflow rule |
| U3 | Any 3-D cells use factor-eight prescaling | Not established by cited source; require explicit 3-D state documentation |
| U4 | PR 14’s \(\Delta E/E=0.5\) uniform quadrature represents the instrument response | Unsupported; require response kernel/calibration |
| U5 | PR 14’s Gaussian relative/absolute noise floors describe calibrated-flux sampling | Unsupported; require repeated calibration data or a declared sensitivity model |
| U6 | A shared-gamma sweep random effect is physically adequate | Proposed viable family, not a source fact; validate covariance/tails against repeated stable-state sweeps |
| U7 | A day’s 1,032/1,055 spectra share one \(U\) or an iid cluster law | Not assumed; requires a scientific regression/local-stationarity design |
| U8 | The current hard-grid loss-cone forward model is an accurate continuous instrument-folded mean | Unverified and structurally plateaued; require continuous response integration and validation |
| U9 | The \(-63.716\) s association offset is a fixed physical latency | Only a coarse empirical pattern; decode event time and validate held out |
| U10 | A raw-to-calibrated empirical emulator can achieve a sufficiently small, stable uncertainty set | Research hypothesis; validate on held-out 3-D and aggregate products |

---

# 21. Limitations, adversarial self-check, and claims the project must not make

## 21.1 Adversarial self-check of positive claims

The following attacks were applied before final classification.

1. **Attack:** choose \(u_1\ne u_2\) and rescale every response cell to match the mean.  
   **Result:** succeeds unless response is known, finite-dimensional with rank, or bounded tightly. All A claims explicitly impose that restriction; otherwise the class is B/C.

2. **Attack:** choose a cell-specific Cox multiplier to hold the random intensity vector fixed.  
   **Result:** succeeds for arbitrary vector mixing. The positive Cox result is restricted to a shared/low-rank specified mixing law.

3. **Attack:** exploit the current finite hard mask so no grid indicator changes with \(U\).  
   **Result:** succeeds on plateaus. Point-ID claims require a continuous response-folded map/rank or are narrowed to partial-ID plateaus.

4. **Attack:** let NB dispersion approach zero so all observations are code zero.  
   **Result:** succeeds. A-class NB imposes \(r\ge r_{\min}>0\); otherwise C for useful uniform inference.

5. **Attack:** select only an outcome common to all \(U\), or hold the observed flux-derived gate fixed in bootstrap.  
   **Result:** succeeds. Positive coverage requires known/bounded selection or full-pipeline rerunning.

6. **Attack:** treat a lower endpoint or normalized calibrated row as an exact count/multinomial composition.  
   **Result:** invalidated by Lemma 5.1 and equation (7.20). Positive code likelihoods sum over intervals; quasi routes are labeled working moment models.

7. **Attack:** reorder angular cells or change the dropped simplex coordinate.  
   **Result:** PR 14 changes numerically. The recommended statistic uses a full tangent-space covariance and Theorem 10.1.

8. **Attack:** let nuisance optimization or profile-grid failure inflate a test statistic.  
   **Result:** Section 13 uses lower statistic bounds, upper null-likelihood bounds, and retention on failure; it cannot narrow from unresolved optimization.

9. **Attack:** choose the true nuisance between evaluated nuisance-grid points.  
   **Result:** a plain grid is invalid. The recommended method adds certified TV/Lipschitz box bounds or retains the box.

10. **Attack:** vary \(U\) across pooled sweeps.  
    **Result:** cluster inference no longer targets a per-sweep \(U\). The positive repeated-sweep result explicitly requires common/regression/local-homogeneity structure.

11. **Attack:** use an unknown monotone calibration to map one continuous flux law to another.  
    **Result:** succeeds. Calibrated-flux A/B claims require a fixed or bounded calibration law; the current broad Y class is C/D.

12. **Attack:** use a terminal code with no upper endpoint in a bounded-moment method.  
    **Result:** the upper moment is unavailable. The method uses lower/tail inequalities only or remains D until a terminal rule is supplied.

No positive coverage proof contains an unstated independent-cell Poisson premise: Theorem 11.4 permits arbitrary within-cluster dependence, and the exact likelihood route states the chosen dependence law explicitly.

## 21.2 Claims prohibited by the case record and proofs

The project must not claim:

* that decompressed lower endpoints are detector counts;
* that rounded flux-derived pseudo-counts are \(K\);
* that a counter plus a mean rate implies independent Poisson sampling;
* that the pitch-angle prescale applies to 3-D cells without state evidence;
* that the `FF` upper endpoint is known;
* that ERT or the observed \(-64\) s association is event time;
* that 15 energy rows or 88 normalized cells are independent;
* that the full-instrument and product-level geometric factors are interchangeable;
* that a response kernel can be inferred silently from a reported resolution or rounded flux;
* that robust/sandwich standard errors cure nonidentification;
* that an optimizer profile width or \(\chi^2_1\) cutoff is a confidence interval;
* that a plug-in parametric bootstrap is exact merely because it simulates from the fitted model;
* that a fixed mask/gate bootstrap reproduces data-dependent selection;
* that twelve simulations, or any finite nuisance grid, prove honest coverage;
* that an empty inverted set means nonidentification;
* that a full-domain set is a failure rather than an honest weak-ID outcome;
* that current PR 14 is wired to archive single-sweep inference;
* that missing `TBS` response coefficients exist or can be reconstructed uniquely from calibrated flux;
* that absolute lunar and spacecraft potentials are separately identified when only their difference enters the model;
* that averaging a frequentist nuisance over an unstated distribution is profiling or robustness;
* that current bundle-only raw or calibrated observations yield an operational nontrivial honest per-sweep confidence set.

## 21.3 Remaining limitations of this memorandum

The positive results are conditional mathematical constructions. The memorandum does not assert that any proposed count-process family is the true ER accepted-count law. The empirical audits cover two reference periods, not the mission. The version-5 mapping, terminal code, and TBS response are unresolved. No full repository test suite was run because of the unavailable `spiceypy` dependency. The exact physical sign/trajectory convention for \(U\) must be fixed consistently in the eventual response model; the statistical results apply to the declared relative-shift parameter. Practical confidence-set width cannot be predicted until response, exposure, count scale, and repeated-state structure are recovered.

---

# 22. Final decision matrix

Each row defines a sufficiently specific class to receive exactly one category. The status refers to that class, not to a family name with unbounded variants. Where a mathematical class is A/B but the current Lunar Prospector record lacks inputs, a separate D row is included.

| Assumption class | Status | Identification and recommended objective | Coverage guarantee or obstruction | Smallest practical strengthening / exact resolver |
|---|---|---|---|---|
| 1a. Independent Poisson accepted counts, exact state/response, continuous injective mean | **A** | Point ID by Theorem 5.2; exact interval-code log likelihood (7.8); LF/BB LR inversion (10.2) | Finite-sample uniform coverage by Theorems 11.1–11.3, with Section 13 certification | Already sufficient; verify global separation/rank and selection |
| 1b. Current LP instantiation of class 1 | **D** | Objective cannot yet be evaluated for archive sweeps | Missing state/response means no defined \(P(C\mid U,\nu,S)\) | Resolve every item in Section 17, especially version 5, terminal rule, anode table, exposure/arithmetic, loss/selection |
| 2a. Independent Poisson with compact finite-dimensional response nuisance and separated profiled laws | **A** | Profile/minimax interval likelihood; statistic (10.2) | Finite-sample LF or Berger–Boos coverage | Supply compact response parameterization and honest nuisance set; prove rank/separation |
| 2b. Poisson with validated bounded response/calibration whose feasible sets overlap | **B** | Identified set (9.2); feasibility/robust likelihood or exact zero-code mean intervals | Finite-sample outer confidence region via simultaneous mean/moment bounds; moment-box can cover the entire ID set | Narrow bounds/anchors until profiled sets become disjoint to move to A |
| 2c. Arbitrary cellwise response/background | **C** | Not identified | Theorem 6.1 gives exact observational equivalence | Known response ratios, anchor cells, low-dimensional tied response, or bounded distortion |
| 3a. Specified shared-scalar Cox/gamma mixing with exact response and separated law sets | **A** | Integral likelihood (7.12) with full sweep dependence | Finite-sample LF/BB inversion | Validate shared/low-rank mixing and compact dispersion |
| 3b. Arbitrary vector Cox/mixed-Poisson law | **C** | Not identified | Theorem 6.2 holds the random intensity law fixed for all \(U\) | Restrict mixing to a shared/low-rank known family or identifying conditional moments |
| 4a. Poisson-binomial with known trial structure/efficiencies and rank | **A** | DP/FFT interval likelihood | Finite-sample LF inversion | Validate finite opportunities, efficiencies, and dependence |
| 4b. Poisson-binomial with validated bounded efficiencies | **B** | Partial ID through interval-probability/mean envelopes | Finite-sample robust/moment outer set | Calibration anchors or tighter coupled bounds |
| 4c. Arbitrary heterogeneous thinning | **C** | Not identified | Equivalent to response absorption | Tie efficiencies across cells/states or bound them |
| 5a. Negative-binomial with \(r\in[r_{\min},r_{\max}]\), \(r_{\min}>0\), exact response, separated laws | **A** | Interval NB likelihood (7.10), profile over \(r\) | Finite-sample LF/BB coverage | Validate dispersion family and lower bound |
| 5b. NB/overdispersion permitting \(r\downarrow0\) | **C** | Fixed-\(r\) point ID may hold, but useful uniform inference fails over the class | Proposition 6.7; laws approach all-zero | Impose \(r\ge r_{\min}>0\) or a uniform variance/tail bound |
| 6a. Known renewal/nonparalyzable-dead-time or finite-state FIFO model with injective accepted law | **A** | Renewal tails (7.16–7.17) or Markov-additive likelihood (7.18) | Finite-sample LF inversion | Exact exposure, dead time, queue, routing, mask, and arrival model |
| 6b. Validated bounded loss/dead-time/FIFO uncertainty | **B** | Accepted-mean/code-probability inequalities | Finite-sample moment/robust outer set | Tighter loss envelopes and terminal bounds can move toward A |
| 6c. Current LP cell-level loss model | **D** | Diagnostic counters establish losses but not a cell law | Missing state-specific accepted-event law | Recover PPA/FIFO/mask/dead-time command/calibration package and cross-check |
| 6d. Unrestricted cell-specific loss | **C** | Not identified | Same construction as Theorem 6.1 | Bound or mechanistically model losses |
| 7a. Independent/mixing sweep clusters, arbitrary within-cluster dependence, bounded features, common/regression \(U\), singleton moment zero set | **A** | Moment-box objective (11.7); cluster GMM (10.4) as an asymptotic alternative | Finite-sample coverage by Theorem 11.4; GMM only uniformly asymptotic under (11.9) conditions | Establish cluster/replication design and response-derived moments |
| 7b. Same repeated-cluster class, but nonsingleton moment-compatible set | **B** | Compute the moment-identified set by feasibility | Theorem 11.4 covers the entire moment-identified set on one simultaneous event | Add moments/anchors or tighten response bounds |
| 7c. Single arbitrary sweep with only support or an unverified mean/covariance label | **C** | No distribution-free identifying experiment | Common-law/support construction | Specify a full law or obtain genuine replicated clusters with valid bounded moments |
| 8a. Counting-process martingale with observed path/compensator, bounded jumps/variation, and point-identifying rank | **A** | Martingale score and finite-sample concentration/test inversion | Finite-sample honest coverage from the specified supermartingale/concentration bound | Observe accepted path or certified terminal score and compensator |
| 8b. Same martingale class with only inequality/rank bounds yielding a set | **B** | Worst-case score/moment feasibility | Finite-sample outer set | Stronger compensator/response information |
| 8c. Current LP martingale route | **D** | Required path, compensator, and terminal bounds are unavailable | Observation is only a compressed terminal array | Recover event/accumulator dynamics and state, or use another class |
| 9. Finite or infinite exchangeability alone | **C** | Symmetry does not identify \(U\) | All-zero exchangeable law; unrestricted de Finetti mixing | Add identifying conditional means, shared parametric mixing, state conditioning, and replication |
| 10a. Valid estimating equations with bounded cluster features and point-identifying zero set | **A** | Finite-sample moment-box feasibility; GMM/EL as secondary objectives | Theorem 11.4 finite-sample coverage | Verify mean restrictions, rank, cluster design, and hard bounds |
| 10b. Valid estimating equations with a nonsingleton zero set | **B** | Compute the moment-identified set | Honest outer region via simultaneous moments | Add valid moments/anchors or tighter response bounds |
| 10c. Broad working estimating equations allowing an arbitrary misspecified mean | **C** | The optimizer need not target \(U\) | Sandwich covariance does not repair an incorrect mean or nonidentification | Restrict and validate the mean/response model |
| 10d. A specific intended LP estimating-equation law not yet validated | **D** | Could become A/B after validation | Required mean/covariance/cluster facts are missing | Validate against repeated state-resolved data and provide a uniform covariance model |
| 11a. Moment equalities from code/response bounds with a separated singleton solution | **A** | Equality-feasibility or bounded-feature test inversion | Finite-sample coverage by Theorem 11.4 | Preserve simultaneous bounds and cluster assumptions |
| 11b. Moment inequalities from code and bounded response with a nonsingleton solution | **B** | Equations (9.7–9.9), statistic (10.7), feasibility | Finite-sample outer coverage by Theorem 11.4 | Finite terminal bounds, more moments, or tighter response envelope |
| 12a. Fixed robust neighborhood whose profiled law sets remain disjoint | **A** | Minimax robust p-value (10.9) | Finite-sample robust LF inversion | Calibrate metric/radius externally and certify law optimization |
| 12b. Fixed robust neighborhood with some overlap but a proper identified set | **B** | Robust identified-set feasibility/minimax inversion | Honest outer region for the robust identified set | Reduce radius or add response anchors to obtain disjoint sets |
| 12c. Overlapping neighborhoods containing a common law for all \(U\), or unbounded contamination | **C** | Not uniformly distinguishable | Proposition 6.5 and Theorem 6.6 | Reduce radius below the separation boundary; restrict response/selection |
| 13. Code support plus deterministic compressor only | **C** | Not identified | Theorem 6.6; common all-zero law forces an essentially full set | Add at minimum an identifying response/moment relation and selection restriction |
| 14a. Calibrated flux with a fixed ground map, valid full conditional law/covariance, and separated law sets | **A** | Exact push-forward likelihood or valid cluster estimating equation | LF finite-sample or uniformly asymptotic cluster coverage, according to the maintained law | Supply ground algorithm, covariance, response, state, selection, and validation |
| 14b. Calibrated flux with validated bounded calibration/response and a proper compatible set | **B** | Robust/moment feasibility over the calibration set | Honest outer region | Tighter source-backed bounds and repeated-cluster design |
| 14c. Broad calibrated-flux law allowing arbitrary cellwise calibration or outcome-dependent selection | **C** | Not identified | Theorems 6.1, 6.3, and 6.4 | Fixed/bounded calibration, explicit selection, covariance, and response law |
| 14d. The narrower intended calibrated-flux law for current archive data | **D** | Could become A/B, but no validated law or bound is supplied | Current PR working model is not established by sources | Recover ground calibration/uncertainty or validate an emulator and full-pipeline simulator |
| 15. Current PR 14 implementation, interpreted as an honest confidence-set method | **C** | It supplies a working profile objective, not a validated observable likelihood | No coverage theorem; empirical and structural defects in Sections 15–16 | Replace with an A/B method; repair sentinel, covariance/invariance, selection, nuisance, optimization, and tests |

## Final decision

**No nontrivial honest frequentist confidence set for per-sweep \(U\) is currently justified by the bundle alone.** The raw-code route is scientifically promising but operationally **D** because named state/response information is missing. The broad calibrated-flux route is **C** under a defensible class broad enough to include arbitrary cellwise response/calibration or selection, and the narrower intended route is **D** until its observation law is validated.

**A nontrivial honest set is possible in principle under assumptions materially weaker than independent Poisson.** The strongest recommended A route is exact code-space least-favorable test inversion under a specified shared-effect, renewal, or other count-process law and fixed response. The strongest weak-law fallback is the finite-sample bounded-feature moment outer set across genuinely replicated clusters. If only response bounds are defensible, the scientifically correct result is a B-class identified/outer set. If the bounds are so wide that Theorem 6.1’s equivalence remains feasible, the result must be full or C.
