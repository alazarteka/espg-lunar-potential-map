# Spacecraft Potential Calculator: Design & Analysis Summary

**Date:** 2025‑08‑16
**Module:** `src/spacecraft_potential.py`
**Scope:** Day–night (sunlit vs. shaded) spacecraft charging for Lunar Prospector ER spectra

---

## 1) Purpose & Context

This module estimates the **spacecraft floating potential** $U_{SC}$ (in volts) for each measured electron spectrum. It unifies:

* **Sunlit (dayside) charging:** governed by photoelectron emission balanced by ambient electron collection (ions negligible).
* **Shaded (nightside) charging:** governed by the **current balance** among ambient electron collection, **secondary‑electron emission (SEE)** from primaries that do impact, and attractive ion collection.

It integrates tightly with the **Kappa fitter** (`src/kappa.py`), using the fitted electron velocity distribution (κ model) as the plasma input.

---

## 2) High‑Level Workflow

Given an `ERData` spectrum (`spec_no`):

1. **Geometry test** (`is_day`): cast a line from spacecraft to Sun; if it intersects the lunar sphere → **shadow**, else **daylight**.
2. **Initial κ fit:** run `Kappa.fit()` on the measured, uncorrected spectrum to obtain $(n_e,\kappa,\theta_{\text{uc}})$.
3. **Branch:**

   * **Sunlight:** estimate $U_{SC}>0$ using a **J↔U** photoyield mapping; **pre‑shift** energies by $-U$; refit; recompute $U$.
   * **Shade:** do **no pre‑shift**; set up a current‑balance equation $F(U)=0$ and solve for **$U<0$** with Brent’s method. Convert the uncorrected κ temperature to the **ambient** temperature at the solved $U$ and return the corresponding ambient $(n_e,\kappa,\theta_c)$.

Outputs are:

* **Sunlight:** corrected κ parameters + $U_{SC}>0$.
* **Shade:** ambient κ parameters + $U_{SC}<0$.

---

## 3) Physics Model (by branch)

### 3.1 Sunlight (photoemission‑dominated)

**Key idea:** A sunlit conductive spacecraft charges positive until the **photoelectron emission current** $J_{\text{ph}}(U)$ equals the **ambient electron collection** $J_e$ (ions are negligible at the relevant $U$). The module:

1. Computes a **target collection current density** $J_e$ by integrating the fitted κ omni‑flux (details in §5).
2. Uses `U_from_J` to invert the **JU curve** (instrument‑/material‑specific parameterization of $J_{\text{ph}}(U)$) and obtain $U_{SC}$ with $0\le U \le 150$ V.
3. **Energy pre‑shift**: spectra are in eV; a +$U$ potential lowers measured energies by $U$, so correct by

   $$
   E_{\text{corrected}} = E_{\text{meas}} - U\quad(\text{**subtract volts**, not }eU).
   $$
4. **Refit κ** on corrected energies and recompute $J_e$ and $U$ for the final output.

> **Why pre‑shift only in sunlight?** The photoelectron “knee” enables a one‑step $U$ estimate; once $U$ is known, the entire spectrum is rigidly shifted by $U$. On the nightside there is **no knee** and SEE/ions matter, so you must solve $U$ **inside** the current balance.

---

### 3.2 Shade (SEE + ions + electron collection)

**Key idea:** With $U<0$ and no photoemission, the spacecraft floats where:

$$
\underbrace{J_e(U)}_{\text{ambient e⁻ collected}} \;+\; \underbrace{J_i(U)}_{\text{ions collected}}
\;-\; \underbrace{J_{\text{SEE}}(U)}_{\text{secondaries emitted}} \;=\; 0.
$$

* **Electron collection $J_e(U)$** is suppressed by the negative barrier; only primaries with $E\ge |U|$ can reach the surface.
* **Secondary emission $J_{\text{SEE}}(U)$** depends on impact energy $E_{\text{imp}}=E-|U|$ through a **Sternglass‑like yield** $\delta(E)$.
* **Ion collection $J_i(U)$** is **enhanced** by attraction; we adopt a thick‑sheath/OML‑like scaling (no bulk flow).

A crucial κ identity relates the **measured uncorrected** κ temperature $T_{\text{uc}}$ to the **ambient** $T_c$ outside the sheath at the trial potential $U$:

$$
k_B T_c \;=\; k_B T_{\text{uc}} \;+\; \frac{e\,U}{\kappa-3/2}
\quad\Longleftrightarrow\quad
T_c[\text{eV}] \;=\; T_{\text{uc}}[\text{eV}] \;+\; \frac{U[\text{V}]}{\kappa-3/2}.
$$

Because $U<0$, $T_c<T_{\text{uc}}$. This mapping is applied **inside each evaluation** of $F(U)$.

---

## 4) Detailed Algorithmic Structure

### 4.1 Inputs

* `ERData` with columns:

  * time (for SPICE geometry),
  * energy centers (eV),
  * directional fluxes and counts (converted upstream to omnidirectional flux).
* **Constants**: electron/ion masses, electron charge, lunar radius.
* **SEE parameters**: `sey_E_m` (eV at peak yield) and `sey_delta_m` (peak yield).
* **Energy grid** for current integrals: $[E_{\min}, E_{\max}]$, log‑spaced.
* **Bracket for nightside U**: typically $[-1500, -1]$ V, expanded if needed.

### 4.2 Day branch (function steps)

1. `initial_fit = Kappa.fit()` → $(n_e,\kappa,\theta_{\text{uc}})$.
2. Evaluate $J_e$ via `electron_current_density_magnitude` (integral of omni‐flux → number flux → current).
3. `U_from_J(J_e)` on $[0, 150]$ V → provisional $U$.
4. **Energy correction** $E \leftarrow E - U$ (in eV), re‑prepare data (+density estimate), refit κ.
5. Recompute $J_e$, then $U$.
6. Output: corrected κ params and $U$ in volts.

### 4.3 Shade branch (function steps)

1. `initial_fit = Kappa.fit()` → $(n_e,\kappa,\theta_{\text{uc}})$; compute $T_{\text{uc}}$.

2. Build **energy grid** `E` (eV), typically `geomspace(max(E_min,0.5), E_max, n_steps)`.

3. Define **current calculator** at a trial $U<0$:

   **(a) Map to ambient κ:** $T_c = T_{\text{uc}} + U/(\kappa - 1.5)$. If $T_c\le 0$, declare the trial unphysical (see robustness in §10).

   **(b) Ambient electron omni‑flux** $F_{4\pi}(E; n_e,\kappa,\theta_c)$ from `omnidirectional_flux_magnitude`.

   **(c) Flux to a plane** (hemisphere, cosine‑weighted):

   $$
   \Phi(E) = \alpha \, F_{4\pi}(E)\times 10^{4}
   \quad\text{with}\quad \alpha=
   \begin{cases}
     \tfrac{1}{4}, & \text{if }F_{4\pi}\text{ is 4π-integrated}\\[2pt]
     \tfrac{1}{2}, & \text{if }F\text{ is 2π-integrated}
   \end{cases}
   $$

   (cm⁻²→m⁻² via $10^4$).

   **(d) Electron collection current:**

   $$
   J_e(U) = e \int_{E\ge |U|} \Phi(E)\, dE.
   $$

   **(e) SEE emission current:**

   $$
   J_{\text{SEE}}(U) = e \int_{E\ge |U|} \underbrace{\delta\!\big(E - |U|\big)}_{\text{Sternglass}}\,
   \Phi(E)\, dE.
   $$

   With $\delta(E) = 7.4\,\delta_m (E/E_m)\exp\{-2\sqrt{E/E_m}\}$.

   **(f) Ion collection current (OML‑like):**

   $$
   v_{\mathrm{th},i}=\sqrt{\frac{e\,T_i}{2\pi m_i}},\quad
   J_{i0}=e\,n_i\,v_{\mathrm{th},i},\quad
   J_i(U)=J_{i0}\sqrt{1-\frac{U}{T_i}}
   $$

   where by default $T_i=T_c$, $n_i=n_e$, $m_i=m_p$, and $U<0\Rightarrow 1-U/T_i>1$.

4. **Balance function:** $F(U)=J_e(U)+J_i(U)-J_{\text{SEE}}(U)$.

5. **Root‑finding (Brent):**

   * Check signs at `spacecraft_potential_low` and `spacecraft_potential_high`.
   * If needed, expand the low bound more negative until signs differ (or limit reached).
   * Solve with `brentq(F, low, high, tol=1e-3, maxiter=200)` → $U_{SC}<0$.

6. Compute **ambient** $T_c(U_{SC})$ and $\theta_c$; package ambient κ params + $U_{SC}$.

---

## 5) From κ parameters to currents (unit conventions)

* **Temperature/energy:** kept in **eV**; potentials in **V** (numerically equal for single‑charge electrons/ions).
* **Omnidirectional differential flux:** `omnidirectional_flux_magnitude(...)` returns **# cm⁻² s⁻¹ eV⁻¹** as a function of energy **outside the sheath** (ambient), given $(n_e,\kappa,\theta)$.
* **To plane flux:** multiply by **$10^4$** (cm⁻²→m⁻²) and geometry factor **$\alpha\in\{1/4,1/2\}$** to represent the hemispherical cosine‑weighted incidence onto a small surface.
* **Current density:** multiply **number flux integrals** by $e$ (C) to get **A m⁻²**.

> Only multiply by $e$ when converting a **number flux** to **current**; do **not** multiply potentials/energies by $e$ in the eV/V arithmetic.

---

## 6) Secondary‑Electron Emission (SEE)

**Yield model (Sternglass‑like):**

$$
\delta(E) \;=\; 7.4\,\delta_m \left(\frac{E}{E_m}\right)\exp\!\big[-2\sqrt{E/E_m}\big],\quad
\delta(E\le0)=0.
$$

* $E$ is the **impact** energy at the surface, i.e., $E_{\text{imp}}=E-|U|$ for a negative surface.
* $\delta_m$ is the peak yield at $E_m$.
* The form reproduces the universal rise–peak–fall shape of SEE curves compactly.

**Parameterization:** `sey_E_m` and `sey_delta_m` are exposed as function arguments for calibration to spacecraft surface properties (contamination/aging can shift both).

**Physical effect in $F(U)$:** As $U$ becomes more negative, fewer primaries cross the barrier (smaller domain $E\ge|U|$), and those that do arrive with smaller $E_{\text{imp}}$, pushing $\delta$ down the curve → **SEE weakens**, favoring more negative $U$.

---

## 7) Ion Current Model

Default is a **no‑flow**, thick‑sheath **OML‑like** enhancement:

* **Thermal flux to a plane at zero bias:**

  $$
  \Gamma_{i0} = n_i\sqrt{\frac{k_B T_i}{2\pi m_i}} \;\Rightarrow\;
  J_{i0} = e\,\Gamma_{i0} = e\,n_i\sqrt{\frac{eT_i}{2\pi m_i}}.
  $$
* **Bias dependence:**
  $J_i(U)=J_{i0}\sqrt{1-U/T_i}$ for $U\le0$.
  This increases with $|U|$ (ions are attracted and focused).

**Notes:**

* If hot/heavy ions or flow are expected, consider upgrading to a **κ‑integral ion model** mirroring the electron integral (no barrier, but energy gain and focusing) or adding bulk drift.

---

## 8) Numerical Methods & Robustness

### 8.1 Energy grid

* Log‑spaced grid `E` from `E_min_ev` to `E_max_ev`.
* Choose `E_max_ev` high enough that $E_{\max} \gtrsim |U_{SC}|$ and SEE tails converge (rule‑of‑thumb: $\max(20\,T_c,\,1.2|U|)$).

### 8.2 Barrier mask

* For $U<0$, only energies $E\ge |U|$ contribute to $J_e$ and $J_{\text{SEE}}$.
* **Edge case:** If no energies pass the barrier (e.g., $|U|>E_{\max}$), set $J_e=J_{\text{SEE}}=0$ explicitly.

### 8.3 Unphysical ambient temperature

* If the κ mapping yields $T_c\le 0$ at a trial $U$, treat that $U$ as **invalid** and force the balance to a large **positive** value so bracketing drifts **toward** physical $U$ (i.e., toward 0 from below). Returning mixed infinities can produce `NaN` in the bracketing.

### 8.4 Geometry factor α

* **Critical**: ensure the ¼ (4π) vs. ½ (2π) factor matches the definition of `omnidirectional_flux_magnitude`. Mismatch biases all three currents and can shift $U$ by tens of volts.

### 8.5 Root finding & bracketing

* Initial bracket $[-1500,-1]$ V is expanded on the negative end until a sign change is found (cap expansions to avoid runaway).
* Brent (`brentq`) is robust with a well‑bracketed sign change; `xtol=1e-3` (1 mV accuracy) is typically sufficient.

---

## 9) Interfaces & Return Types

### `calculate_potential(...) -> Optional[tuple[KappaParams, VoltageType]]`

* **Inputs:** ER spectrum (`er_data`, `spec_no`), SEE params, energy grid parameters, bracketing.
* **Output (on success):**
  `KappaParams` (ambient κ params for shade; corrected κ params for sunlight) and $U_{SC}$ (units: `ureg.volt`).
* **Output (on failure):** `None`.

### `current_balance(U, fit, E, sey_E_m, sey_delta_m) -> float`

* Returns $F(U)$ (A m⁻²) to be zeroed by the root finder.

### `calculate_shaded_currents(U, fit, E, sey_E_m, sey_delta_m) -> (Je,Jsee,Ji)`

* Returns each current component (A m⁻²) at a given $U$.
* **Sign convention:** all **magnitudes**; the balance uses `Je + Ji − Jsee`.

---

## 10) Validation, Diagnostics, and Unit Tests

**Quick invariants:**

* **Units:** Only multiply by $e$ when converting number flux → current. Energies/potentials remain in eV/V elsewhere.
* **Limits:**

  * As $U\to 0^-$: $J_e$ large; $J_{\text{SEE}}$ can be comparable if $\delta>1$; $J_i$ small.
  * As $U\to -\infty$: $J_e\to0$, $J_{\text{SEE}}\to0$, $J_i$ increases ⇒ $F(U)>0$.
* **Geometry factor test:** In a Maxwellian limit at $U=0$, $\int\Phi(E)\,dE$ should equal $\tfrac{1}{4}n\langle v\rangle$ (4π) or $\tfrac{1}{2}n\langle v\rangle$ (2π).

**Suggested unit tests:**

1. **Barrier empty:** With $|U|>E_{\max}$, verify `Je=Jsee=0` and `F(U)=Ji>0`.
2. **Unphysical $T_c$:** For a very negative $U$ that yields $T_c\le0$, ensure `current_balance` returns a large **positive** value and no `NaN`.
3. **SEE parameter sensitivity:** Increasing `sey_delta_m` should make $U_{SC}$ **less negative** (stronger emission balances sooner).
4. **α sensitivity:** Switch α between ¼ and ½ and verify expected proportional shift in all currents.

**Diagnostics to log at the solution $U_{SC}$:**

* $U_{SC}$, $T_{\text{uc}}$, $T_c$, $\kappa$, $(E_m,\delta_m)$.
* $J_e$, $J_{\text{SEE}}$, $J_i$ (A m⁻²) and their fractions of the total.
* Number of Brent iterations; bracketing endpoints and signs.

---

## 11) Sensitivity & Calibration

* **SEE parameters ($E_m,\delta_m$):** dominate the shade solution, particularly when the electron spectrum is hot. Treat them as **effective surface parameters** and calibrate per instrument configuration or by fitting intervals where you have independent $U$ constraints.
* **Geometry factor α:** a systematic factor‑of‑2 error here directly rescales all currents and biases $U$.
* **Ion model:** $T_i$ and mass $m_i$ matter when electrons are cold (deep lobes). Expose `ion_mass_kg` and `Ti_eV` if you expect deviations from $T_i=T_e$.
* **Energy grid ceiling:** too low `E_max_ev` suppresses Je/Jsee and can yield overly negative $U$.

---

## 12) Known Limitations & Future Extensions

* **Ions as Maxwellian OML:** No bulk flow, no κ tail; upgradeable to a symmetric κ‑integral with optional drift.
* **SEE spectral detail:** We use a scalar yield $\delta(E)$. Detailed models split true/elastic/rediffused components and include angular/energy distributions of secondaries—overkill for $U$, but relevant for detailed spectra.
* **Surface heterogeneity:** A single $(E_m,\delta_m)$ approximates mixed materials. A weighted mixture (areas or view factors) could be introduced if needed.
* **Anisotropy / pitch‑angle effects:** The omni‑flux assumes isotropy. If the environment is anisotropic, a directional model would improve fidelity.
* **Sunlight branch dependence on `U_from_J`:** Accuracy hinges on how the photoyield $J_{\text{ph}}(U)$ is parameterized. Document and validate that curve separately.

---

## 13) Practical Recommendations

* **Choose α correctly** (¼ vs ½) and make it a module constant. Add a one‑time self‑check against a Maxwellian reference.
* **Guard edge cases:** `T_c<=0`, empty barrier mask, and `NaN` from mixed infinities.
* **Record component currents** at the solution for QA.
* **Tune SEE params** per campaign; keep defaults `(E_m≈500 eV, δ_m≈1.5)` only as starting points for spacecraft‑like surfaces.
* **Use generous `E_max_ev`** (e.g., `max(20*T_uc, 2*|U_high|)`).

---

## 14) Example Usage

```python
from src.flux import ERData
from src.spacecraft_potential import calculate_potential

er = ERData.from_file("lp_er_spectrum.parquet")  # includes time + flux columns
spec_no = 123456

# Calibrate SEE params if known; otherwise start with defaults:
out = calculate_potential(
    er_data=er,
    spec_no=spec_no,
    E_min_ev=1.0,
    E_max_ev=2.0e4,
    n_steps=600,
    spacecraft_potential_low=-2000.0,
    spacecraft_potential_high=-0.5,
    sey_E_m=500.0,
    sey_delta_m=1.5,
)

if out is None:
    print("Potential could not be determined for this spectrum.")
else:
    params, U = out
    print(f"U_SC = {U:~P}")  # pint formatting, prints "xx V"
    print(f"kappa={params.kappa:.2f}, theta={params.theta:~P}, n={params.density:~P}")
```

---

## 15) Function‑by‑Function Notes

* **`theta_to_temperature_ev` / `temperature_ev_to_theta`**
  κ identity; ensure $\kappa>1.5$. Your fitter bounds already enforce this.

* **`sternglass_secondary_yield(E_imp, E_m, delta_m)`**
  Clamp $E_{\text{imp}}\le0\to0$. Consider exposing alternative yield families if you later need material‑specific curves.

* **`calculate_shaded_currents(U, fit, E, sey_E_m, sey_delta_m)`**
  Performs one evaluation of $(J_e,J_{\text{SEE}},J_i)$ at a trial $U<0$.
  Robustness tips: handle `T_c<=0`, empty barrier masks, and make the geometry factor explicit.

* **`current_balance(U, fit, E, sey_E_m, sey_delta_m)`**
  Returns $F(U)$. Keep it simple and deterministic; log when it returns non‑finite values (should not happen after guards).

* **`calculate_potential(...)`**
  Orchestrates the branch, bracketing, solving, and packaging of outputs. Return `None` cleanly for failure cases (e.g., poor κ fit or no bracketed root).

---

## 16) Change Log & Open Questions

* **2025‑08‑16:** SEE parameters surfaced; nightside current balance integrated; explicit κ temperature mapping $T_{\text{uc}}\to T_c(U)$.
* **Open:**

  * Document `U_from_J` and the photoyield model used in the day branch; add tests comparing predicted $U$ to knee‑based inferences on clear sunlit intervals.
  * Decide and lock **α** after verifying the `omnidirectional_flux_magnitude` convention (4π vs 2π).
  * Add optional κ‑ion + drift capability for plasma sheet cases.

---

### TL;DR

* **Sunlight:** compute $J_e$ from κ → invert photoyield $J_{\text{ph}}(U)$ to get $U$ → **pre‑shift** energies by $-U$ → refit → return corrected κ + $U$.
* **Shade:** **no pre‑shift**; for each trial $U<0$, map $T_{\text{uc}}\to T_c(U)$, compute $J_e$, $J_{\text{SEE}}$ (Sternglass), and $J_i$ (OML), solve $J_e+J_i-J_{\text{SEE}}=0$ by Brent, then report **ambient** κ + $U$.

This document should equip you (and future readers) to maintain, validate, and extend the potential calculator with confidence.
