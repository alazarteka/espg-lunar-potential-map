# κ‑Fit Electron Temperature: **Count‑Weighted vs Unweighted** Fits — Design & Analysis Summary

**Date:** 2025‑08‑22
**Module/Script:** `scripts/profiling/temperature_weighting_study.py`
**Scope:** Effect of count‑derived weighting on κ‑fit electron temperature $T_e$ (eV) for all 199x Lunar Prospector ER spectra using the **original** fit (no illumination correction or $U_{SC}$ energy shift).

---

## 1) Purpose & Context

We want to know how much the inferred electron temperature $T_e$ from a κ distribution fit depends on whether we **weight residuals by their count‑derived uncertainties** or not. The fear is that **unweighted** least squares in **log‑flux space** gives undue leverage to **low‑count, high‑energy** bins, biasing $T_e$ upward. This document quantifies that effect, interprets the mechanisms, and recommends next steps.

---

## 2) Dataset & Acceptance Criteria

- Files: `data/199*/*/*.TAB` (all available 199x years).

- For each file/spectrum `spec_no`, we run two independent fits with identical configuration except `use_weights`:
    - **Weighted fit:** `use_weights=True`
    - **Unweighted fit:** `use_weights=False`

- A pair is **accepted** if both fits succeed, are marked `is_good_fit=True`, and produce finite $T_e>0$.

**Accepted pairs:** **437,800**

---

## 3) What “weighting” means here (objective & uncertainties)

- We fit in **log‑flux space** with a squared‑error objective.
- For each energy bin $i$, we have an uncertainty in log‑flux, $\sigma_{\log F,i}$, propagated from reconstructed **counts** (with background handling and the log transform).
- The implementation multiplies each residual by $1/(\sigma_{\log F,i}+\epsilon)$ **before squaring**, which is equivalent to minimizing

    $$
    \sum_i \frac{r_i^2}{\sigma_{\log F,i}^2}\quad\text{(i.e., WLS with weights }w_i = 1/\sigma_{\log F,i}^2\text{)}.
    $$

- **Unweighted** fits set $w_i\equiv 1$.

**Intuition:** in log‑space, low counts $\Rightarrow$ large $\sigma_{\log F}$ $\Rightarrow$ **down‑weight** those bins. This is the statistically correct treatment for heteroscedastic (count‑driven) noise.

---

## 4) Fitting Configuration (identical for both branches)

- Bounds: $\kappa \in (2.5, 6.0)$, $\log_{10}\theta\,[\text{m/s}]\in(6,8)$.
- Starts: LHS $n\_starts=10$ (seed 42); optimizer: L‑BFGS‑B.
- **Energy‑response convolution**: log‑Gaussian kernel $W$ with `ENERGY_WINDOW_WIDTH_RELATIVE=0.5`.
- Flux handling: directional $\to$ integrated over incident hemisphere and scaled to approximate $4\pi$.
- Negative raw fluxes are clamped to 0 upstream of counts reconstruction.
- Conversion $(\theta,\kappa)\to T_e$ via the project function `theta_to_temperature_ev`.

> **Important:** This analysis uses the **original** κ fit only—no day/night correction and **no** spacecraft‑potential energy shift.

---

## 5) Results (dataset‑wide)

**Pairs:** 437,800

- **Te (weighted)**: mean **33.4 eV**, median **19.5 eV**, P5 **3.79**, P95 **90.6**
- **Te (unweighted)**: mean **44.2 eV**, median **24.1 eV**, P5 **12.6**, P95 **116**

**Pairwise differences** $\Delta T_e = T_{e,\mathrm{w}} - T_{e,\mathrm{uw}}$

- **Absolute ΔTe**: mean **−10.8 eV**, median **+0.132 eV**, P5 **−64.1**, P95 **+0.643**
- **Relative ΔTe/Te_uw**: mean **−20.8%**, median **+0.574%**, P5 **−93.8%**, P95 **+2.77%**

**Key patterns:**

- Most spectra change little (pairwise **median ΔTe ≈ +0.13 eV**; **median relative ≈ +0.57%**).
- Means are strongly **negative** because a **small tail** of spectra sees unweighted fits run **much hotter**; weighting **pulls them back down**.
- Distribution medians differ substantially (**19.5 eV vs 24.1 eV**). This is not inconsistent with the tiny **median of pairwise differences**; see §7 (median‑of‑differences ≠ difference‑of‑medians).

---

## 6) Figures (placeholders)

> Attach the four PNGs here. Captions supplied so they drop straight into a paper or report.

**[Figure 1]** Weighted vs Unweighted κ‑Fit Temperatures (original fits).
![Figure 1](images/te_weighted_unweighted_hist.png)

**[Figure 2]** Te comparison: weighted vs unweighted (original fits).
![Figure 2](images/te_weighted_vs_unweighted_scatter.png)

**[Figure 3]** Difference in temperatures: $\Delta T_e = T_{e,\mathrm{w}}-T_{e,\mathrm{uw}}$ (eV).

![Figure 3](images/te_diff_hist.png)

**[Figure 4]** Relative difference: $100\%\times \Delta T_e/T_{e,\mathrm{uw}}$.

![Figure 4](images/te_rel_diff_hist.png)

---

## 7) Interpretation of the figures and numbers

### 7.1 What the histograms say

- **Figure 1 (overlaid Te):** The **unweighted** distribution has a **fatter high‑$T_e$** tail. The **weighted** distribution shifts mass toward **3–30 eV** and trims extremes. Medians move from **24.1 eV (uw)** to **19.5 eV (w)** (\~**−19%**).
- **Figure 3 (ΔTe)** and **Figure 4 (relative ΔTe)**: Tall spike near **0** with a **long negative shoulder**. At least **5%** of spectra (≈ **21.9 k**) have **ΔTe ≤ −64 eV** or **≤ −93.8%** relative—precisely the failure mode we worried about: **noisy high‑E bins** push the **unweighted** fit **too hot**.

### 7.2 What the scatter shows

- **Figure 2 (scatter, log–log)** splits into two regimes:
    1. A **tight 1:1 band**: high‑SNR spectra where weighting barely matters.
    2. A **“pancake” at $T_{e,\mathrm{w}}\sim4–12$ eV** vs **$T_{e,\mathrm{uw}}$ spanning \~20–500 eV**: cases where a few **low‑count high‑energy** points tilt the unweighted fit hot; weighting anchors the solution to the well‑measured core.

### 7.3 Why the effect occurs (mechanism)

- In **log‑flux space**, bins with tiny counts have **large** $\sigma_{\log F}$.
    - **Unweighted** least squares treats them like core bins ⇒ a few noisy tail points can **tilt** the spectrum and inflate θ, hence **$T_e$**.
    - **Weighted** least squares down‑weights them ⇒ solution is controlled by the **count‑rich region**, yielding **cooler** and more stable $T_e$.

- **Median-of-differences vs difference-of-medians:**
  The **pairwise median ΔTe ≈ +0.13 eV** can coexist with **median(Te_w) < median(Te_uw)** because medians are **order statistics** of different marginal distributions. A small subset with **large negative ΔTe** reorders the combined sample enough to reduce the **marginal median** of $T_{e,\mathrm{w}}$ by \~4.6 eV, even though the _typical_ pairwise change is near zero.

---

## 8) Implications for science products

- **Bias correction:** Unweighted κ‑fits **overestimate $T_e$** in a non‑negligible minority of spectra. Switching to **weighted** fits removes a methodological bias and reduces spurious hot outliers.
- **Aggregate metrics:** Expect **cooler medians** and **truncated high‑$T_e$ tails** in campaign summaries, maps, and trend analyses. Any downstream quantity that **nonlinearly** depends on $T_e$ (e.g., heat fluxes, mean free paths) may shift appreciably in affected intervals.
- **Quality control:** The tail cases are identifiable (see §10); they can be flagged or down‑weighted in higher‑level products.

---

## 9) Recommended default & documentation note

- **Adopt count‑weighted fits as the default** for $T_e$ estimates.
- **Document the change**: at the dataset level this yielded $\sim$ **−19%** shift in the median and $\sim$ **−24%** in the mean (33.4 vs 44.2 eV), primarily by suppressing hot outliers.

---

## 10) Diagnostics to add immediately (low effort, high value)

1. **ΔTe vs tail counts:** For each spectrum, compute counts in the top‑N energy bins and plot ΔTe (or ΔTe/Te_uw) vs that SNR proxy. Expect monotonic trends isolating the failure regime.
2. **Edge‑of‑range influence:** ΔTe vs **max energy with nonzero counts** and vs **fraction of bins with near‑zero counts**.
3. **Parameter‑bound hits:** Count how often **θ** or **κ** hits bounds; cross‑tab with large negative ΔTe and with the “pancake” cloud in Fig. 2.
4. **Goodness‑of‑fit sanity:** Check **weighted reduced $\chi^2$** distribution; if it clusters far from 1, revisit the $\sigma_{\log F}$ model.
5. **Rare ΔTe>0 cases:** Inspect a few; these often indicate core underestimation (background subtraction too aggressive) where weighting legitimately nudges $T_e$ up by a few percent.

> These diagnostics will give a concise causal story suitable for a methods section or review response.

---

## 11) Method upgrades (short‑to‑medium term)

- **Poisson (count‑space) likelihood:** Forward‑model **counts** through the instrument response and maximize the **Poisson log‑likelihood**. This removes log‑space heteroscedasticity entirely and handles zeros naturally.
- **Robust loss in log‑space:** If staying in log‑space, replace squared loss on standardized residuals with **Huber** or **Tukey** to further curb extreme‑bin influence.
- **Censoring / detection thresholds:** Treat sub‑threshold bins as **left‑censored** (Tobit‑style term), or jointly fit a small **additive background** parameter.
- **Response‑coupling awareness:** The convolution matrix $W$ couples bins; consider an approximate covariance or inflate $\sigma$ where leakage is strong.
- **Audit the weight formula:** Confirm the implementation indeed yields $w_i=1/\sigma^2$ after squaring (it should, given current residual scaling).

---

## 12) Caveats & assumptions

- Only **original** fits: no day/night correction and **no** energy shift from $U_{SC}$.
- The **“good fit”** threshold is permissive; some marginal spectra likely pass.
- Counts/uncertainty modeling inherits assumptions from `ERData._add_count_columns` (gains, backgrounds, EPS in the log transform).
- The hemispherical $\to 4\pi$ approximation may indirectly affect inferred densities; $T_e$ is less sensitive but still coupled through the fit.

---

## 13) Reproducibility

From repo root (Python 3.12 with `uv`):

```bash
uv run python scripts/profiling/temperature_weighting_study.py
```

**Outputs:** written to `temp/`

- `te_weighted_unweighted_hist.png` (Fig. 1)
- `te_weighted_vs_unweighted_scatter.png` (Fig. 2)
- `te_diff_hist.png` (Fig. 3)
- `te_rel_diff_hist.png` (Fig. 4)

**Arrays and summary stats** are printed/logged by the script; seed fixed for reproducibility of multi‑start initializations.

---

## 14) Practical recommendations (checklist)

- [ ] **Switch default** production $T_e$ to **weighted** fits.
- [ ] **Publish the delta**: include Fig. 1–4 and the table in §5 when introducing the change.
- [ ] **Add QC flag** for spectra with `hot_tail_suspect = (T_e_w / T_e_uw < 0.5) or (ΔTe/Te_uw < −50%)` to mark tail‑driven corrections.
- [ ] **Run the diagnostics in §10** and keep 6–8 exemplar spectra (before/after) for the method note.
- [ ] **Plan the Poisson‑likelihood prototype** and compare on a representative subset.

---

## 15) Glossary & notes

- $T_e$ (eV): κ‑temperature derived from $(\kappa,\theta)$ via `theta_to_temperature_ev`.
- **Weighted** = WLS in log‑flux with weights $1/\sigma_{\log F}^2$ from **count** uncertainties.
- **Unweighted** = unit weights in log‑flux space.
- **P5 / P95** = 5th/95th percentiles across accepted spectra.
- **ΔTe** = $T_{e,\mathrm{w}} - T_{e,\mathrm{uw}}$ per spectrum.

---

### TL;DR

Weighting by count‑derived uncertainties **stabilizes** the κ temperature: most spectra barely change, but a **small, influential tail** where unweighted fits run **too hot** is **pulled down**, reducing the dataset **median by \~19%** and **mean by \~24%**. Use **weighted** fits as the default, document the shift, and add simple diagnostics/QC to isolate the tail regime. A Poisson (count‑space) likelihood is the natural next upgrade.
