# Clean-Room Reconstruction of the Halekas et al. (2008) Lunar Surface Potential Method

**Target paper:** Halekas, J. S., G. T. Delory, R. P. Lin, T. J. Stubbs, and W. M. Farrell (2008),
*Lunar Prospector observations of the electrostatic potential of the lunar surface and its response
to incident currents*, J. Geophys. Res., 113, A09102, doi:10.1029/2008JA013194 (16 pages).

**Provenance of this reconstruction:** the paper PDF only (page/paragraph citations below refer to
its pages 1–16 and AGU paragraph numbers [n]), plus public-web sources: NASA PDS / WUSTL ODE /
NSSDCA documentation of the LP ER data products, Lin et al. (1998, Science 281, 1480), and the
Halekas lineage papers (2002 GRL, 2003, 2005a GRL, 2005b JGR, 2007; Mitchell et al. 2008 Icarus;
Lillis et al. 2008 Icarus; Halekas et al. 2010 JGR-Planets). Where lineage-paper content could only
be recovered from search-engine indexing of paywalled text, it is marked *[search-index]* and given
lower weight. Supporting detail lives in the companion files in this directory:
`methods_digest.md` (verbatim-level digest of the PDF), `lineage_research.md`,
`instrument_pds.md`, `notes_director.md`.

---

## 0. One-paragraph summary of the method

For each 80-s LP ER 3-D electron distribution on a magnetic field line connected to the lunar
surface, Halekas et al. (i) estimate the LP **spacecraft potential** U_SC from the electron
spectrum alone (photoelectron-break / current–voltage curve in sunlight; a tuned current-balance
charging model in shadow); (ii) form the **normalized energy–pitch-angle distribution** by dividing
the whole distribution by its incident (surface-going) half, so the fit target is the ratio of
reflected to incident flux; (iii) build a **synthetic normalized distribution** consisting of a
sharp, energy-dependent adiabatic loss cone,
sin²α_c = (B_S/B_M)·[1 + eU_M/(E − eU_SC)], **plus** an upward-going secondary-electron beam
centered at measured energy e|U_M − U_SC|; and (iv) **grid-search** the lunar surface potential U_M
and the magnetic-field ratio B_S/B_M (U_SC held fixed) to minimize a least-squares mismatch between
the **logarithms** of the synthetic and measured normalized fluxes over the entire distribution.
The method is automated but every step was hand-checked; it returns valid results only for
U_M ≲ −20 V (negative potentials), referenced to the **local ambient plasma**.

---

## 1. Data product and preprocessing

### 1.1 Instrument and product (paper §3, p.4, [18]; PDS)

- Instrument: LP Electron Reflectometer, a symmetric hemispherical ("top-hat"-class) electrostatic
  analyzer with a 360°×14° disk FOV, imaging MCP detector (~1.4° intrinsic angular resolution),
  sharing a 2.5-m boom with the fluxgate magnetometer (Lin et al. 1998, Science 281:1480; NSSDCA
  1998-001A-06).
- **Data product used:** the full-energy 3-D distribution product, archived as PDS
  **LP-L-ER-3-RDR-3DELEFLUX-80SEC** (`3Dyymmdd.TAB`, directory `DATA/ER_3DIST/` of the MAG/ER
  Level-1B volumes; PDS4 `urn:nasa:pds:lp-er-calibrated:data-3deleflux`). Each record: time tag +
  **88 solid-angle sectors × 15 log-spaced energy channels** + the **average B vector over the
  2.5-s accumulation** (supplied expressly for pitch-angle computation). Units: differential
  number flux, cm⁻² s⁻¹ sr⁻¹ eV⁻¹ (calibrated flux, not counts, not phase-space density).
- Energies: 15 channels spanning ~10 eV–20 keV with intrinsic ΔE/E ≈ 0.25; onboard summing of
  adjacent bins gives an **effective ΔE/E ≈ 0.5** ([18]). The energy-bin centers are
  **time-varying** (PDS `E_BINS.TAB`): the lowest bin is ~38 eV for most of the mission and drops
  to ~7 eV during the 1999 terrestrial-magnetosphere passages ([19], p.4). A reimplementation must
  read the time-tagged energy table, not assume a fixed grid.
- Cadence: one 3-D distribution per **80 s** (16 spins; ~120 km along-track), accumulated over
  **2.5 s** (half a ~5-s spin), so each measurement has few-km intrinsic spatial resolution ([18]).
- They explicitly do **not** use the higher-cadence pitch-angle-sorted time-series product
  (ER_HI_TS, ~5 s) because it carries only 2 selected energy channels — "we require full energy
  coverage not available at higher cadence" ([18]). Confirmed by the archive (5-s product cycles
  among 5 possible energies used two at a time).
- Orbit/altitude: ~100 km (1998), ~15–50 km (1999); data Jan 1998–Jul 1999, covering solar wind,
  magnetosheath, tail lobes, plasma sheet, and lunar wake ([18]).

### 1.2 Preprocessing / selection

- **Pitch angles** are computed per sector from the record's co-averaged MAG B vector; the 88
  sectors are rebinned into an energy–pitch-angle distribution (bin count is a processing choice;
  the paper never states it — see Ambiguity A5).
- **Magnetic connection to the surface** is determined (Figure 4 shows a "magnetic connection to
  the surface" color bar with red/black polarity, p.7 caption): the loss-cone/beam analysis is only
  meaningful when the local field line intersects the surface; the "foot point" (surface point
  magnetically connected to LP) and its solar zenith angle are used throughout §6–7 ([46], Figs 7,
  11). Which hemisphere of the distribution is "incident" (surface-going) vs "reflected"
  (surface-leaving) follows from the field polarity.
- **Spacecraft-potential correction of spectra:** measured spectra are shifted in energy by the
  calculated U_SC before deriving plasma moments (density, temperature, current of the
  downward-going electrons) ([34]–[35], §4.4). For the loss-cone fit itself, U_SC enters the model
  equation instead (see §2 below).
- **Environment-dependent spectral-fit choices** ([33], §4.3): sudden low-energy flux enhancements
  are treated as spacecraft-generated secondaries in the plasma sheet (fit the higher-energy
  component, neglect the low one), but as the solar-wind core in the solar wind (fit the lowest
  resolvable component after removing photoelectrons).
- The study analyzes **selected time periods** ("reanalysis of selected LP ER data", [59]):
  29 Apr 1999 (terrestrial magnetosphere: lobes + plasma sheet), 23 Mar 1998 (solar wind + wake),
  6 May 1998 (SEP event). This is not a whole-mission batch pipeline; "to ensure the highest
  quality measurements, we hand check every step of this process" ([45], p.9).

---

## 2. The physical forward model

### 2.1 Loss cone at the spacecraft (outside the LP sheath) — §5.1, p.7, [37]

Adiabatic motion (conservation of energy and magnetic moment) in the combined crustal/ambient
magnetic mirror and surface electrostatic field gives the critical (loss-cone) pitch angle α_c at
the spacecraft:

$$\sin^2\alpha_c \;=\; \frac{B_S}{B_M}\left(1 + \frac{e\,U_M}{E}\right)$$

with (paper's own definitions, [37]):
- **U_M** = lunar surface potential relative to the ambient plasma (volts; **negative** for the
  measurable regime);
- **B_S/B_M = magnetic field at the spacecraft divided by field at the lunar surface** — note
  **S = Spacecraft, M = Moon**, the *opposite* of the crustal-field-mapping literature where B_S
  usually denotes the *surface* field (see Pitfall P1). Typically B_S/B_M ≲ 1 (Fig. 5 example:
  0.975);
- **E** = electron kinetic energy; **e** = elementary charge (so eU_M/E is the ratio of potential
  energy to kinetic energy; with E in eV, this is numerically U_M/E).

Electrons at the spacecraft with pitch angle inside the cone (sin²α < sin²α_c on the surface-going
side) reach the surface and are absorbed; outside it they mirror and return. Physics check of the
sign convention: for U_M < 0, at E ≤ e|U_M| the bracket goes ≤ 0 — the cone closes and *all*
electrons are reflected electrostatically; as E → ∞ the cone opens to the purely magnetic value
sin²α_c → B_S/B_M. The **energy dependence** of the loss-cone boundary is what encodes U_M; the
high-energy asymptote encodes B_S/B_M.

The model is **fully adiabatic by assumption**; the paper flags this as quasi-adiabatic at best
(gyroradii comparable to lunar field scales) and cites the non-adiabatic "tip-angle effect" of
Halekas et al. (2003) for obliquely intersecting field lines ([37], p.7). No non-adiabatic
correction is applied in the 2008 fits.

### 2.2 Transit through the spacecraft sheath — [38], p.8, Eq. (2)

The distribution is measured *after* passing through LP's own sheath. Two limiting treatments are
considered ([38]):
1. fully adiabatic through the sheath → replace U_M by ΔU = U_M − U_SC;
2. pitch angle conserved, energy shifted only → replace E by E − eU_SC.

Because the electron gyroradii greatly exceed the sheath dimensions, **they adopt option 2**
(supported, they say, by simple particle-tracing simulations), giving the fitted equation
(their Eq. (2)):

$$\boxed{\;\sin^2\alpha_c \;=\; \frac{B_S}{B_M}\left(1 + \frac{e\,U_M}{E - e\,U_{SC}}\right)\;}$$

where **E is the energy measured at the spacecraft** and U_SC is the spacecraft floating potential
from §3 below (the paper writes "E − U_SC"; dimensional consistency requires eU_SC, i.e. E in eV
and U_SC in volts). They note both options share the same asymptotes and yield approximately the
same U_M ([38]).

### 2.3 The loss-cone boundary is SHARP in the model

The synthetic distribution uses a **sharp step** at α_c(E). The paper explicitly observes that the
*measured* loss cone (and beam) "do not appear quite as sharp as the synthetic case" — a
constant-energy cut shows "a modest slope to the loss cone rather than the sharp step function of
the synthetic distribution" — attributing the broadening to non-gyrotropic sheath effects or waves,
and argues the fit still recovers the correct average boundary and its energy dependence
([41], p.8). **No broadening/smearing kernel is part of the 2008 model.** (Contrast: the Mars ER
work of Lillis et al. models finite width / backscatter explicitly.)

### 2.4 The upward-going secondary electron beam — [37]–[40]

- Physics: secondary electrons produced at the surface at ~eV energies are accelerated upward
  through the (negative-surface) sheath potential drop, forming a field-aligned upward beam whose
  energy marks the potential (heritage: Halekas et al. 2002, GRL 29, 1435, which detected the beam
  and, after subtracting a theoretical loss-cone distribution, fit the residual to a **Gaussian**
  — e.g. center ≈ 51 eV, FWHM ≈ 30 eV in that event *[search-index]*).
- In the 2008 model, "the final electron distribution consists of a superposition of the loss cone
  distribution and this secondary beam" ([37]); the synthetic distribution **adds an upward-going
  beam centered at an electron energy of (U_M − U_SC)** — i.e. at measured energy
  e|U_M − U_SC| = e(U_SC − U_M) for U_M < U_SC ([40]). The beam is a **co-fit model component**,
  not masked out, and not used as a separate independent estimator in the automated fit — though
  the paper calls the beam "perhaps the best marker of the lunar potential" because its energy
  shift through the sheath is trivial ([38]).
- Sheath transit of the beam: it "simply shifts in energy according to the spacecraft potential";
  if U_SC is sufficiently negative (spacecraft below surface potential), the beam cannot reach the
  spacecraft — "we take this into account in our analysis" ([38]) (i.e., omit the beam component
  when e(U_SC − U_M) ≤ 0 or below the lowest energy bin).
- **The paper specifies neither the beam's energy width, pitch-angle extent, nor amplitude** in the
  synthetic distribution (Ambiguity A3). Best-supported inference: width comparable to the
  instrument's effective ΔE/E ≈ 0.5 at the beam energy (the 2002 Gaussian fit had FWHM ≈ 0.6× its
  center energy, consistent with instrument-resolution-dominated width), confined to the upward
  (loss-cone) pitch-angle region, with amplitude either fixed to match the observed upward flux at
  the beam energy or included as a nuisance parameter — the text is silent.

### 2.5 Regime of validity ([42]–[44], §5.2)

- Works **only for negative U_M**: magnetic mirror + repulsive potential combine monotonically for
  electrons. Positive surface potentials make the effective potential non-monotonic; because the
  dayside photoelectron sheath is only meters thick (≪ crustal-field scale heights), electron
  reflectometry cannot in general measure positive potentials from orbit.
- Occasional small positive best-fit values are **considered invalid** — a measure of the intrinsic
  fit error ([44]).
- **Detection threshold: |U_M| ≲ 20 V is statistically indistinguishable from zero** given the
  ER's energy resolution/coverage and pitch-angle resolution ([44]; dashed "measurement threshold"
  lines in Figs. 7 and 11).

---

## 3. Spacecraft potential U_SC (paper §4)

The 2008 paper's headline "first advance" ([36]) — needed both to shift measured energies (moments)
and to plug into Eq. (2) and the beam energy.

### 3.1 Sunlight ([20]–[24], pp.4–5; Figs 1–2)

1. **Calibration phase** (1999 magnetosphere passes, sweep down to 7 eV): fit the total measured
   distribution to
   - a **Maxwellian** for spacecraft photoelectrons recaptured by the positive spacecraft:
     f(v) = n·(m/2πkT)^{3/2}·exp(−mv²/2kT), plus
   - a **kappa distribution** for ambient plasma electrons at the spacecraft (their Eq. (1)):
     $$f(v) = \frac{\Gamma(\kappa+1)}{(\pi\kappa)^{3/2}\,\Gamma(\kappa-1/2)}\;\frac{n}{\Theta^3}\left[1+\frac{v^2}{\kappa\Theta^2}\right]^{-\kappa-1},\qquad T_\kappa=\frac{\kappa}{\kappa-3/2}\,\frac{m\Theta^2}{2k}.$$
   Distributions are modeled *at the spacecraft* (no potential-shift term), and — critically — the
   fitted model is **convolved with the instrument energy resolution** before comparison;
   neglecting this shifts the apparent spectral break and overestimates U_SC ([22], Fig. 1).
2. The **break energy** between photoelectron and plasma components equals e·U_SC (positive);
   the first velocity moment of the kappa component gives the plasma electron current, which in
   equilibrium equals the escaping photocurrent. Many such fits build the spacecraft's
   **current–voltage curve** (Fig. 2), fit by a double exponential
   $$J_{ph,esc}(U_{SC}) = A\,e^{-U_{SC}/B} + C\,e^{-U_{SC}/D},\quad A=1.07~\mu A/m^2,\ B=5~V,\ C=0.016~\mu A/m^2,\ D=60~V$$
   ([23]; the paper mislabels the second pair "A, B" and prints "mA/m²" — magnitudes and the
   Fig. 8 photocurrent argument only close in µA/m²; the functional form matches Polar/ISEE/GEOS/
   Voyager heritage, Scudder et al. 2000 etc.).
3. **Routine use** (any sunlit time, even with a 38-eV sweep floor): fit only a kappa to the
   plasma part (excluding low-energy photo/secondary contamination), integrate it for the electron
   current, invert the J–V curve for U_SC ([24]). Estimated error **~<10 V** ([22]). Assumes the
   photoelectron yield of the spacecraft is constant over the mission.

### 3.2 Shadow ([25]–[32], pp.5–6; Fig. 3)

No measurable returning photoelectrons/secondaries → current-balance **charging model**:
- Balance incident electron current against secondary-electron emission from the spacecraft plus
  incident ion current, all as functions of U_SC (< 0).
- Secondary yield: **Sternglass formula** δ(E) = 7.4·δ_m·(E/E_m)·exp(−2√(E/E_m)) (Sternglass 1954),
  integrated over the incident electron spectrum.
- Electron spectrum: kappa fit to the measurement. Ions: assumed kappa with same κ, n_i = n_e,
  T_i = T_e, non-flowing; **thick-sheath** (orbit-limited, Horányi 1996) ion collection for the
  small spacecraft.
- Liouville/sheath correction: the measured (at-spacecraft) kappa temperature T_uc relates to the
  ambient one by **kT_c = kT_uc + eU_SC/(κ − 3/2)** (Halekas et al. 2005b, JGR 110, A07222 — the
  shifted-kappa mapping: same κ, transformed effective n and T; a shifted Maxwellian keeps T and
  changes effective n). Combining this with the charging model gives a transcendental equation
  solved numerically for U_SC ([28]).
- The spacecraft's unknown secondary-emission parameters were **tuned empirically**: search
  δ_m ∈ [0.75, 2.0], E_m ∈ [200, 800] eV for the values that best eliminate *artificial
  discontinuities in Earthward-going electron flux at light/shadow boundaries in the tail lobes*
  (where no real discontinuity should exist) → **δ_m = 1.5, E_m = 500 eV** ([29]–[31]).
- Model behavior (Fig. 3): U_SC ≈ 0 for kT_e/e ≈ 60–100 eV up to a few hundred eV (secondary
  emission balances), negative outside that window; κ ≈ 1.6 stays negative everywhere. Overall
  U_SC generally within ±50 V ([34]).
- SEP-event exception (§8, [57]): with unmeasured high-energy particles, U_SC cannot be
  self-consistently determined; they apply the surface-potential method **without** the U_SC
  correction (arguing secondary emission keeps U_SC near zero then).

---

## 4. Normalization of the measured distribution — [39]

Verbatim-level statement: *"We begin by normalizing the distribution by dividing both reflected and
incident halves of the distribution by the incident half, so that we fit the ratio of reflected to
incident flux."*

Implementable reading (heritage-supported): at each energy channel E, and for each pitch-angle bin,
divide the flux by the incident-hemisphere flux **at the mirror-conjugate pitch angle** (or,
operationally near-equivalently, by the incident-half flux at the same |cos α| or the mean incident
flux at that energy):

$$r(E,\alpha) \;=\; \frac{J(E,\alpha)}{J_{inc}(E,\alpha')}\,,\qquad \alpha' = 180°-\alpha \ \ (\text{conjugate incident bin}),$$

so that the incident half maps to ≈1 identically, and the reflected half carries the loss-cone
signature: r ≈ 1 outside the loss cone (adiabatic mirror returns the incident flux), r ≪ 1 inside
the loss cone (electrons absorbed by the surface; only backscatter/secondaries partially refill
it), plus the beam spike. This per-energy, per-pitch ratio construction is the standard ER
normalization inherited from the crustal-field mapping technique (Lin et al. 1998; Halekas et al.
2001; Mitchell et al. 2008: "ratios of upward to downward flux") *[search-index]*. The 2008 paper
does not spell out conjugate-bin vs hemisphere-average division (Ambiguity A2), but Figure 5
("Normalized energy pitch angle distribution") shows a 2-D (E, α) map whose incident half is flat,
consistent with either.

Notes:
- Normalization removes the absolute calibration, the spectral shape of the ambient plasma, and
  (to first order) any gyrotropic anisotropy — the fit quantity is dimensionless.
- Whether "flux" is differential number flux (as archived) or energy flux is immaterial after the
  per-energy ratio; what matters is that numerator and denominator are the same quantity at the
  same energy.
- Whether the crustal-mapping practice of **backscatter subtraction** (removing the diffuse
  secondary/backscatter refilling of the loss cone before fitting; cf. Lillis et al. 2008 for Mars,
  Mitchell et al. 2008 for the Moon) was applied in the 2008 potential fit is **not stated** in the
  paper; the described synthetic distribution (sharp empty loss cone + discrete beam) suggests it
  was not explicitly subtracted, with the log-space objective tolerating the refilled floor
  (Ambiguity A6).

---

## 5. The fit itself — [39]–[40]

- **Synthetic distribution:** a normalized (ratio) model on the same (E, α) grid: value 1 outside
  the loss cone; a depleted (nominally empty) loss cone interior bounded by Eq. (2)'s α_c(E); plus
  the upward beam component centered at measured energy e(U_SC − U_M) (=|U_M − U_SC| in eV).
- **Objective:** *"we then calculate a least squares fit parameter for the entire distribution
  using the logarithms of the synthetic and measured fluxes"* — i.e. minimize
  $$\chi^2(U_M, B_S/B_M) \;=\; \sum_{E,\alpha}\Big[\log J^{syn}_{norm}(E,\alpha) - \log J^{meas}_{norm}(E,\alpha)\Big]^2$$
  summed over the **entire** normalized distribution (both hemispheres, all energies/pitch bins),
  with no stated weighting, no stated exclusion windows, and no stated count-statistics floor
  (Ambiguities A4, A7). The log-space residual makes the fit sensitive to the order-of-magnitude
  depletion inside the loss cone rather than dominated by the near-unity region.
- **Search:** *"We repeat this procedure for a full range of magnetic ratios and lunar surface
  potentials to find the best fit"* — an exhaustive **2-D grid search over (B_S/B_M, U_M)**, with
  **U_SC held fixed** at the §4 value. Grid ranges and steps are not given (Ambiguity A8);
  physically B_S/B_M spans ~(0.5–1.05) for LP altitudes over weak-to-moderate crustal fields, and
  U_M spans ~(−4500, 0) V across the studied environments. Free parameters: **U_M and B_S/B_M
  only** (plus, implicitly, whatever sets the beam amplitude); fixed: U_SC, the sharpness of the
  boundary, the beam center (slaved to U_M − U_SC).
- **Per-measurement fitting:** one fit per 80-s 3-D distribution (Fig. 5 fits a single 14:45 UT
  measurement; Figs. 6/10/13 show time series at the measurement cadence). No pre-averaging of
  sweeps is mentioned.
- **Automation + manual QC:** Fig. 5 caption calls it the "automated surface potential
  determination"; §6 [45] says every step was hand-checked for the selected intervals.
- Worked example (Fig. 5, p.8): 29 Apr 1999 14:45 UT, plasma sheet; best fit U_SC = +11 V,
  **U_M = −160 V**, B_S/B_M = 0.975; synthetic reproduces both the beam and the energy-dependent
  loss-cone boundary.

---

## 6. Uncertainty and quality control

Stated in the paper:
- U_SC error **~<10 V** in sunlight ([22]); no formal error given for shadow U_SC.
- Surface-potential floor: **|U_M| < ~20 V indistinguishable from zero**; positive fitted values
  treated as invalid/no-detection ([44]; threshold lines on Figs. 7, 11).
- "Hand check every step" for the selected periods; same "level of quality control" applied to the
  solar-wind/wake interval ([45], [53]).
- No χ²-based confidence intervals, no per-fit error bars, no automated rejection thresholds are
  described anywhere in the paper (Ambiguity A7). Scatter in Figs. 7–9, 11–12, 14–15 is the de
  facto uncertainty display.
- Implicit physical cross-checks (heritage): consistency between the beam energy and the loss-cone
  energy dependence ([41]: the fit boundary is "consistent with the energy of the secondary
  beam"); dependence of results on foot-point SZA and electron temperature behaving as
  current-balance physics predicts (§6–7).
- Interpretation caveats the authors themselves attach: possible **non-monotonic near-surface
  potentials** mean the technique senses the potential *minimum* above the surface rather than the
  surface itself ([49]); in the wake, part of the inferred drop may be the **ambipolar wake
  potential**, and U_M is referenced to the **local wake plasma**, not the upstream solar wind
  ([53], [56]); non-adiabatic (tip-angle) effects are neglected ([37]; cf. Halekas et al. 2010
  reexamination of ER *[search-index]*).

---

## 7. Lineage: inherited vs changed

**Inherited from ER crustal-field mapping** (Anderson et al. 1975 heritage; Lin et al. 1998;
Halekas et al. 2001 JGR; Mitchell et al. 2008 Icarus; generic formulation also in Lillis et al.
2008 for Mars):
- Adiabatic mirror relation sin²α_c = B_ratio (energy-independent when purely magnetic) and the
  practice of reading the loss-cone boundary directly from the ER's fine pitch-angle resolution
  (rather than the isotropic-flux reflection coefficient R = |cos α_c| of the 1970s technique).
- The **up/down flux-ratio normalization** per energy.
- Least-squares fitting of a synthetic loss-cone distribution over a parameter grid.
- The 88-sector × 15-energy 3-D data handling and MAG-based pitch angles.

**Added/changed for the potential measurement** (Halekas et al. 2002 → 2005a → 2008):
- The electrostatic term making the loss cone **energy-dependent**: sin²α_c = (B_S/B_M)(1 + eU/E)
  (introduced observationally in Halekas et al. 2002, which measured ~−35 V surface-relative-to-LP
  potentials in shadow and detected the upward secondary beam, Gaussian-fit after loss-cone
  subtraction *[search-index]*).
- The **beam as a co-fit component** of the synthetic distribution (2002 fit it separately/post
  hoc; 2005a used uncorrected energies; 2008 integrates it and slaves its center to U_M − U_SC).
- The **spacecraft-potential correction** (all of §4) — the specific 2008 innovation. 2002/2005a
  potentials were surface-relative-to-spacecraft with unknown offsets; 2008 produces
  surface-relative-to-ambient-plasma values "with no arbitrary offsets" ([35], [36]).
- The choice E → E − eU_SC (pitch-conserving sheath transit) rather than U_M → U_M − U_SC in the
  loss-cone equation ([38]).
- Joint (U_M, B_S/B_M) grid fit over the whole 2-D normalized distribution in log space.
- The shifted-Maxwellian/shifted-kappa Liouville mappings used for the shadow U_SC come from
  Halekas et al. 2005b (JGR 110, A07222).
- **Not** inherited from the Mars work: Monte Carlo backscatter subtraction (Lillis et al. 2008)
  and explicit loss-cone-broadening modeling — the 2008 lunar-potential fit uses a sharp boundary
  and no stated backscatter model.

---

## 8. Ambiguities and assumptions (numbered)

Where the paper underdetermines the method, with best-supported inference:

- **A1 — Units in Eq. (2).** Printed as "E − U_SC"; must be E − eU_SC (E in eV, U_SC in volts →
  numerically subtract). Similarly "beam centered at an electron energy of U_M − U_SC" means
  measured energy e|U_M − U_SC| (a positive number for U_M < U_SC). Evidence: dimensional
  consistency; Fig. 5 numbers (U_SC = +11 V, U_M = −160 V → beam at ≈171 eV).
- **A2 — Exact normalization operator.** "Divide both halves by the incident half" could mean
  (i) conjugate-pitch-angle division J(E,α)/J(E,180°−α), (ii) division by the incident-hemisphere
  mean at each energy, or (iii) division by an incident-half model. Heritage (up/down ratio at
  conjugate angles) favors (i); for a gyrotropic, nearly isotropic incident half, (i) and (ii)
  nearly coincide. Not resolvable from the paper.
- **A3 — Beam model shape/amplitude.** Center energy is specified; width (energy and pitch),
  shape, and amplitude are not. Best inference: width ~ instrument effective ΔE/E ≈ 0.5 (2002's
  Gaussian FWHM/center ≈ 0.6 supports resolution-dominated width); pitch extent = upward
  field-aligned (plausibly filling the upward loss cone, through which the beam travels);
  amplitude either matched to data or a free/nuisance parameter — silent.
- **A4 — Objective-function details.** No weighting, no error model (no one-count floor
  mentioned), no stated energy or pitch-angle exclusion windows, no handling prescription for
  log(0) inside the empty synthetic loss cone (a floor value must exist in practice; perhaps the
  synthetic cone interior is set to a small nonzero ratio, or interior bins with no counts are
  skipped). "Least squares fit parameter" suggests an unweighted sum of squared log-residuals.
- **A5 — Pitch-angle binning.** Number of pitch bins in the normalized (E, α) map is unstated (the
  88 sectors at one energy limit useful bins to a few dozen at most; Fig. 5's rendering suggests
  tens of bins over 0–180°).
- **A6 — Backscatter/secondary refilling of the loss cone.** Not addressed for the potential fit;
  crustal-mapping heritage subtracts or models it. Likely left in the data with the sharp-empty
  synthetic cone absorbing the mismatch (contributes to the "not as sharp as synthetic" remark,
  [41]).
- **A7 — QC thresholds.** No numeric fit-acceptance criterion; QC was manual for the selected
  intervals. The only quantitative gates stated: reject U_M > 0 results; treat |U_M| < 20 V as
  zero/no-detection.
- **A8 — Grid ranges/steps** for U_M and B_S/B_M: unstated.
- **A9 — Energy channels included.** Presumably all 15 (subject to sweep configuration); whether
  the lowest (7–38 eV) channels — contaminated by photoelectrons in sunlight — are excluded from
  the loss-cone fit is unstated.
- **A10 — B_S/B_M as a free parameter vs from crustal-field maps.** They fit it per measurement
  (grid), rather than fixing it from LP magnetometer or crustal-field maps; nothing constrains it
  across neighboring measurements.
- **A11 — What "fluxes" are fit.** Differential energy flux vs number flux is immaterial after
  normalization, but figure labels ("differential energy flux") vs archive units (number flux)
  differ; the fit quantity is the dimensionless ratio.
- **A12 — Shadow U_SC self-consistency during environment transitions** (e.g., plasma sheet with
  flows, SEP events): treated approximately or skipped ([57]); assumptions T_i = T_e, n_i = n_e,
  non-flowing are acknowledged first approximations.

---

## 9. Likely reimplementation pitfalls

- **P1 — B_S/B_M naming trap.** In this paper **B_S = field at the Spacecraft, B_M = field at the
  Moon's surface** ([37], p.7), so B_S/B_M ≤ ~1. The crustal-field ER literature uses B_S for the
  *surface* field; blindly copying that convention inverts the ratio. Check against Fig. 5's
  0.975.
- **P2 — Sign of U_M and the closing loss cone.** U_M < 0 in the measurable regime; for
  E − eU_SC ≤ e|U_M| the bracket in Eq. (2) is ≤ 0 → **no loss cone** (total reflection). An
  implementation must clamp sin²α_c to [0, 1]: bracket ≤ 0 → cone closed; (B_S/B_M)(…) ≥ 1 → cone
  fully open (no reflection). Getting the sign backwards (potential term widening instead of
  narrowing the cone at low E for negative U_M) inverts the energy dependence and biases U_M.
- **P3 — Where U_SC enters.** Three distinct places, easy to conflate: (i) Eq. (2) replaces
  **E → E − eU_SC** (NOT U_M → U_M − U_SC — the paper explicitly rejects that for the loss cone);
  (ii) the beam center is at **e|U_M − U_SC|** in measured energy; (iii) plasma moments come from
  spectra **shifted by U_SC**. Using ΔU = U_M − U_SC inside the loss-cone bracket is a documented
  alternative the authors considered and discarded (though they note results are similar).
- **P4 — Fit the whole normalized 2-D distribution in log space**, not just the boundary. The
  objective is squared differences of **logarithms** of normalized fluxes over the *entire* (E, α)
  map (incident half included, ≈ trivially matched). Fitting linear-space ratios, or extracting
  α_c(E) per energy and fitting the curve, are different estimators with different noise behavior
  and will not reproduce the paper's behavior off-benchmark.
- **P5 — Sharp synthetic boundary vs broadened data.** The model is a step; the data are smeared.
  A least-squares log fit against a step tolerates this ([41]), but adding ad-hoc smearing, or
  fitting only the steepest-gradient bin, changes the estimator. Also decide the synthetic value
  *inside* the cone (empty ⇒ log floor needed) — an arbitrary floor choice shifts χ² weighting.
- **P6 — Beam inclusion.** The upward beam must be **in the model** (its energy slaved to
  U_M − U_SC); masking it out discards the strongest constraint and leaves the beam flux to
  corrupt the loss-cone region of the fit (the beam sits *inside* the upward loss cone in pitch).
  Conversely, when U_SC < U_M (spacecraft more negative), drop the beam — it can't reach the
  detector ([38]).
- **P7 — Potential reference frame.** Output U_M is relative to the **local ambient plasma at the
  spacecraft's plasma environment** (in the wake: relative to wake plasma, which itself sits below
  the solar wind by the ambipolar drop, [53]). Comparing to models referenced to the upstream
  solar wind or to the spacecraft without the U_SC chain gives systematic offsets.
- **P8 — Time-varying energy sweep.** Lowest bin 38 eV for most of the mission, 7 eV only in the
  1999 magnetosphere passes; energy centers come from the time-tagged PDS table. A fixed energy
  grid silently misplaces the beam and the low-E loss-cone closure. The effective ΔE/E is 0.5
  (onboard summed), not the intrinsic 0.25 — this matters for beam width and for the U_SC
  spectral-break convolution ([18], [22]).
- **P9 — Use the 80-s 3-D product**, not the 5-s pitch-angle product (2 energies only); one fit
  per 2.5-s-accumulated 3-D snapshot; no multi-sweep averaging is described.
- **P10 — Sunlight photoelectron contamination.** Below ~50 eV in sunlight the spectrum is
  photoelectron-dominated ([34]); those channels cannot be used naively in the normalized ratio
  (incident half contaminated). The paper is silent on the exclusion (A9) — but not handling it at
  all will poison low-energy ratios exactly where the potential signature lives.
- **P11 — µA vs mA typo** in the photocurrent J–V coefficients ([23]): the printed "mA/m²" must be
  µA/m² (cross-check: Fig. 8's surface photocurrent argument, 0.1 vs 4.5 µA/m²). Copying "mA"
  makes the sunlight U_SC inversion nonsense.
- **P12 — Instrument-resolution convolution in the U_SC fit** ([22], Fig. 1): fitting the
  photoelectron/plasma break without convolving the model with ΔE/E ≈ 0.5 systematically
  overestimates U_SC (which then propagates into Eq. (2) and the beam center).
- **P13 — Positive/near-zero results are non-detections**, not measurements: clamp/flag U_M > −20 V
  rather than reporting small positive or small negative values as physical ([44]).
- **P14 — Hand-checked, selected data.** The published numbers come from a few hand-vetted
  intervals, not an unattended mission-wide batch. An automated reimplementation run over
  arbitrary times will include fits (no surface connection, sheath crossings, non-adiabatic
  geometries, SEP contamination) that the original authors would have discarded by eye; poor
  behavior "elsewhere" than a benchmark is expected without an equivalent vetting layer.

---

## 10. Minimal implementable spec (pseudocode)

```
for each 80-s ER 3-D record (88 sectors × 15 energies, avg B vector):
    if field line does not intersect surface: skip
    compute pitch angles from B; rebin to (E, α) grid; identify incident half by B polarity
    U_SC = spacecraft_potential(record)          # §3: J-V inversion (sunlit) or charging model (shadow)
    r(E, α) = J(E, α) / J_incident(E, conjugate α)   # §4 normalization; r≈1 on incident half
    best = argmin over grid (U_M ∈ [−few kV, 0), R ∈ (0, ~1.05]) of
        Σ_{E,α} [ log r_syn(E, α; U_M, R, U_SC) − log r(E, α) ]²
      where r_syn = 1 outside cone; ~empty inside cone with sin²α_c = R·(1 + U_M/(E − U_SC)) (eV/V),
            clamped to [0,1];
            + upward beam at energy |U_M − U_SC| (width ~ ΔE/E≈0.5, upward pitch range)
              included only if U_SC > U_M
    report U_M (valid only if U_M ≤ −20 V), R = B_S/B_M, U_SC; hand-check
```

---

## 11. Key sources

Paper (all page/[¶] cites above): Halekas et al. 2008, JGR 113, A09102, doi:10.1029/2008JA013194.
Full digest: `methods_digest.md`. Lineage evidence and URLs: `lineage_research.md`. Instrument/PDS:
`instrument_pds.md` (LP-L-ER-3-RDR-3DELEFLUX-80SEC at PDS-PPI/WUSTL; Lin et al. 1998 Science
281:1480; NSSDCA 1998-001A-06). Lineage papers: Halekas et al. 2002 GRL 29(10):1435
(doi:10.1029/2001GL014428); Halekas et al. 2003 (tip-angle); Halekas et al. 2005a GRL 32:L09102
(doi:10.1029/2005GL022627); Halekas et al. 2005b JGR 110:A07222 (doi:10.1029/2004JA010991);
Halekas et al. 2007 (SEP); Mitchell et al. 2008 Icarus 194:401; Lillis et al. 2008 Icarus 194:544;
Halekas et al. 2010 JGR-Planets 115:E03006 (doi:10.1029/2009JE003516). No derived surface-potential
product exists in PDS/SPDF — the paper is the sole algorithm documentation.

---

# Round 2 (appended 2026-07-11): Ambiguity attack, selection function, and error-filter spec

Second investigation wave: five parallel research agents (later restatements/thesis hunt; quantitative
figure forensics on the PDF; shared ER-toolchain internals; selection/filtering practice; consumers of
the dataset) plus direct PDF verification. New evidence files: `figure_forensics.md`,
`toolchain_research.md`, `restatements_research.md`, `selection_research.md`, `consumers_research.md`
(plus `pdf_extraction_findings.md` from a page-verified PDF-retrieval wave, round 3 below).
New authoritative input: private correspondence from Prof. Halekas to the project's researcher:
*"You will absolutely have to filter out points with large errors to get a physically meaningful
result. Much of this can be done automatically, but I would recommend some 'by-hand' quality control
to check the results as well."* Treated as confirming that (a) unfiltered fits are NOT expected to be
physically meaningful, (b) filtering on fit error is essential and largely automatable, (c) by-hand QC
is part of the intended workflow.

## R2.1 Newly page-verified lineage fact (changes emphasis of the reconstruction)

**¶[16], p.3 of the 2008 paper (verified in the PDF):** the pre-2008 crustal-field ER processing used
energy channels only down to ~200 eV, which capped loss-cone-only potential estimates at
|ΔU| ≈ 100 V, "since larger potential drops produce a secondary beam which begins to contaminate the
loss cone at ~200 eV". Halekas 2002 used loss cones only (on crustal-mapping-processed data);
Halekas 2005a/2007 used the beam only. The 2008 method is explicitly the FIRST to treat the entire
distribution self-consistently. Consequences for a reimplementation:
- The 2008 fit **must include energy channels below 200 eV** (unlike the crustal-mapping heritage).
- The **beam is not a nuisance — it is the constraint that breaks the ≥100-V degeneracy**: at large
  |U_M| the loss-cone shape saturates as an estimator and the beam energy carries the information.
  A fit that mishandles the beam will do fine at small |U_M| and fail at plasma-sheet/SEP magnitudes.

## R2.2 Ambiguities resolved or narrowed (numbering from §8)

- **A1 (Eq. 2 units) — resolved before; unchanged.** eU_SC in energy units; beam at e|U_M − U_SC|.
- **A2 (normalization operator) — narrowed, not closed.** Toolchain heritage evidence (Anderson 1975;
  Mitchell 2007; Lillis 2008/2018) strongly supports **conjugate-pitch-angle ratio**
  R(E,α)=J_up(E,α)/J_down(E,α_mirror) [toolchain_research.md, high confidence]; a search-engine
  paraphrase of the 2008 text supports "normalized by the incident portion ... as a function of energy
  and pitch angle" (per-energy incident-half division) [restatements_research.md]. The two coincide
  for gyrotropic, pitch-symmetric incident halves. **Spec decision: use conjugate-bin division**
  (heritage continuity); flag hemisphere-average division as an admissible variant with near-identical
  results on quiet distributions but different noise weighting.
- **A3 (beam model) — largely RESOLVED by figure forensics** (Figure 5 synthetic panel, 300-dpi
  render; `figure_forensics.md`): the synthetic beam is **~1 energy bin wide** (at the effective
  ΔE/E≈0.5 binning) at the bin containing e|U_M−U_SC| (171 eV case sits in the ~100–200 eV bin), and
  is **confined to the ~2 most field-aligned upward pitch bins (~155–180°, ~25° extent)** — it does
  NOT fill the whole upward loss cone. Its **amplitude ≈ 3.6× the incident level** in the synthetic
  panel vs ~4.5× peak in the measured panel — i.e. amplitude is of the observed order, either matched
  to the data or a coarse free/preset scale; not resolvable further at print resolution. The measured
  beam is broader (~45° in pitch). Model shape at this resolution is effectively a **top-hat/delta in
  (E-bin, α-bin) space**, consistent with the 2002 precursor characterizing the *measured* beam as a
  Gaussian (center 51 eV, FWHM 30 eV in that event).
- **A4 (objective details / log(0) floor) — substantially narrowed.** Figure 5's synthetic loss-cone
  interior renders as a **uniform hard floor at ratio ≈ 0.1** (exactly the color-scale minimum) with a
  hard one-bin step boundary. Since log(0) is undefined, the model must carry a finite floor; the
  display is consistent with a floor at or below 0.1 (display clamping cannot be excluded, but a
  floor ~0.05–0.1 reproduces the panel). The color scale itself (log, 0.1–10, green=1) tells us the
  fit quantity spans ±1 decade around unity — consistent with unweighted least squares on
  log10(ratio). No evidence anywhere of Poisson weighting or a one-count floor in the 2008 fit
  (the toolchain's χ² practice is standard but its weighting is unrecoverable publicly).
- **A5 (pitch binning) — RESOLVED.** Two independent lines agree: the group's LP ER pitch-angle
  distributions use the **native 11.25° bins (16 bins over 0–180°)** [Halekas et al. 2010, read in
  full], and Figure 5's panels show **~15 discrete pitch bins ≈ 12° each** [figure_forensics.md].
- **A6 (backscatter handling) — clarified.** The crustal-mapping and Mars toolchains model/subtract
  backscatter (MarMCET etc.); the 2008 potential paper's synthetic distribution instead uses the
  sharp cone + floor, and the measured cone interior refills to ~0.2–0.36 near the boundary
  [figure_forensics.md]. Conclusion stands: **no explicit backscatter subtraction in the 2008
  potential fit**; the floor value absorbs it.
- **A7 (QC thresholds) — see selection function, §R2.3–R2.5.** The ~20 V floor is page-verified
  ([44] and the ≈20 V dashed lines measured in Figs. 7/11).
- **A8 (grid ranges/steps) — narrowed by figure forensics.** The U_M time traces in Figs. 6/10/13 are
  **visibly grid-quantized with steps ~10–20 V** (print-limited upper bound; possibly multiplicative
  spacing at kV scales), while U_SC traces are continuous — direct evidence that U_M comes from a
  discrete grid search and U_SC from a continuous (iterative) solve. Grid extent must reach at least
  −4000 V (Fig. 13) and slightly positive values (small positive fits occur, [44]).
- **A9 (energy channels included) — partially resolved by ¶[16]:** channels below 200 eV are included
  by design (that's the 2008 point). Fig. 5 shows the fit domain spanning ~10 eV–10 keV (~13
  rendered energy columns of the 15). Sunlight photoelectron channels: still no explicit statement
  (the Fig. 5 case is in shadow... actually 14:45 UT 29 Apr 1999 with U_SC=+11 V is sunlit —
  suggesting low-E channels stay in the fit with the photoelectron population removed as in §4.3).
- **A10 (B_S/B_M free per fit) — unchanged**, reinforced: Halekas 2010 notes cutoffs near 90°
  (B_S/B_M→1) are "rather unreliable" — the known degeneracy motivates it being fit, then filtered.
- **A11, A12 — unchanged.**

## R2.3 The selection function (reconstructed)

**What was analyzed.** Three hand-vetted intervals: 29 Apr 1999 ≈12:57–22:02 UT (~405 80-s
measurements; magnetosphere), 23 Mar 1998 ≈00:00–12:00 UT (~540; solar wind/wake), 6 May 1998
≈00:12–23:43 UT (~1080; SEP) [UT spans measured from the rendered figure axes]. "Selected time
periods" ([59], abstract) — the method was never run mission-wide in the 2008 paper.

**What survived.** On these vetted days, the U_M panel plots values at nearly every time step
(near-continuous traces in Figs. 6/10/13 — kept fraction near 1 after hand-vetting *on days chosen
for good conditions*). The scatter plots keep only above-threshold negative potentials: ~180–220
points in Fig. 7 vs ~405 measurements → **~45–55% of measurements on the best day yield
|U_M| > 20 V negative fits**; the rest are below threshold (dayside ≈ 0 V), not necessarily failed
fits. For scale, the group's best-documented sibling pipeline (Lillis et al. 2018, Mars, read in
full) keeps ~36–40% of ~10M distributions via automated gates.

**Explicit selection screens in the group's own companion work** (PSS 57:78–82): orbits chosen with
**no strong crustal fields** and **a clear secondary beam** — i.e. B_S/B_M near 1 (weak-mirror,
loss-cone-degenerate) geometries were *preferred* for beam clarity in that companion study, while
strong-anomaly regions complicate the adiabatic model; and beam visibility itself was a gate.

**Why manual QC was necessary — failure-mode inventory** (each acknowledged in the paper or lineage;
symptom an automated fitter would hit):
1. Field line not intersecting the surface / polarity confusion → no loss cone at all; fit returns
   garbage or B_S/B_M rail ([Fig. 4 connection bar]; Lillis 2018 elevation cut ~30°).
2. B_S/B_M → 1 with small |U_M| → cutoff at 90°, flat χ² valley, unstable fit (Halekas 2010:
   "rather unreliable"); also positive-U_M results, invalid by construction ([43]–[44]).
3. Beam below the lowest energy channel or beam blocked (U_SC < U_M) → the kV-range constraint
   vanishes; loss-cone-only information saturates ([16], [38]).
4. Photoelectron/secondary contamination of low-energy channels in sunlight ([33]–[34]) → corrupted
   normalization ratios exactly where eU/E is large.
5. Non-adiabatic geometry — oblique ("tip-angle") field lines, small gyroradius-to-field-scale
   ratios → broadened/filled cones, biased-shallow boundaries (Halekas 2003; Halekas 2010).
6. Non-gyrotropy through the spacecraft sheath and wave smoothing → smeared boundary ([41]),
   biasing a sharp-step fit if severe.
7. Low counts at high energies / tenuous plasma (lobes, central wake) → noisy log-ratios (no
   Poisson weighting to absorb them).
8. Time variability across the 2.5-s sweep or 80-s cadence (plasma-sheet boundary crossings,
   SEP onsets) → mixed distributions violating the static model ([34] flux discontinuities).
9. Wake ambipolar potentials entangled with surface potential ([53], [56]) → interpretation, not
   fit, failure — must be flagged by region, not by χ².
10. SEP penetrating background and unmeasured high-energy populations → U_SC undetermined ([57]),
   background counts flatten the measured ratio contrast.

## R2.4 "Points with large errors": what the error metric plausibly is

Direct evidence for the group's error practice, in order of relevance:
- **Halekas et al. 2002 (GRL, the direct precursor) quoted per-fit Monte Carlo error bars** on the
  loss-cone potential (ΔU = −48 ± 15 V) [restatements_research.md] — i.e. propagate count/statistical
  noise through the same fit, or equivalently sample the χ² valley.
- The wider toolchain quotes **Δχ² = 1 (1-σ) parameter intervals / error ellipsoids** in the
  (field-ratio, potential) plane (Mitchell 2007 ~30% per-PAD errors; Lillis 2018 1-σ ellipsoids from
  the χ² surface) [toolchain_research.md, selection_research.md].
- The 2008 grid search makes the Δχ²-based interval free to compute: the width of the χ² valley in
  U_M at fixed best B_S/B_M (and vice versa) IS the natural per-fit error, and a "flat valley"
  (degenerate B_S/B_M→1, weak contrast, low counts) directly produces a "large error".
So **"error" most plausibly = the 1-σ half-width δU_M of the χ² valley from the grid search
(Δχ² ≤ 1 contour, or Monte Carlo resampling giving the same thing)**, with "large" meaning δU_M
comparable to |U_M| itself or to the 20-V floor.

## R2.5 Best-supported automatic filter spec (faithful to intent)

Ordered gate stack; a fit is kept only if it passes all. Sources per gate in brackets.

- **G0 (a priori):** field line connects to the surface with adequate elevation (≥~30°) and stable
  footpoint; instrument in full-sweep mode; no data-outage flags. [Fig. 4 connection bar; Lillis
  2018; PDS outages.tab]
- **G1 (counts):** adequate counts/flux in the channels spanning the expected boundary and beam
  (≥2 channels with usable loss-cone contrast). [Lillis 2018 gate (a)]
- **G2 (coverage):** pitch-angle coverage samples both hemispheres and the cutoff region (α_c not
  outside the sampled range; 16×11.25° bins). [Lillis 2018 gate (b); Halekas 2010]
- **G3 (contrast/degeneracy):** reject fits with best-fit cutoff within ~1 pitch bin of 90°
  (B_S/B_M·(1+…) ≈ 1) — flat χ² valley, unmeasurable small/positive potentials. [Halekas 2010;
  §5.2 of the paper]
- **G4 (validity):** reject U_M > 0; clamp/flag |U_M| ≤ 20 V as "no detection" rather than a value.
  [¶(44); measured Fig. 7/11 threshold lines]
- **G5 (rails):** reject best fits on the edge of the U_M or B_S/B_M grid. [standard grid-search
  hygiene; implied by grid quantization seen in Figs. 6/10/13]
- **G6 (error):** compute δU_M from the Δχ²=1 contour (or Monte Carlo resampling à la Halekas 2002);
  reject if δU_M > max(20 V, f·|U_M|) with f ≈ 0.3 (Mitchell 2007's ~30% per-PAD precedent; tune
  by hand-checking a benchmark day). [2002 GRL; Mitchell 2007; Lillis 2018; Halekas correspondence]
- **G7 (consistency):** where the beam is detected independently (peak in the upward field-aligned
  ratio), require the beam energy and loss-cone U_M to agree within the energy-bin width; where
  U_SC < U_M or beam energy < lowest channel, accept loss-cone-only fits only for |U_M| ≲ 100 V.
  [¶(16), ¶(38), ¶(41); Halekas 2005a heritage]
- **G8 (environment flags):** flag (not necessarily reject) sunlight low-E photoelectron
  contamination, wake ambipolar entanglement, SEP background, and plasma-sheet boundary crossings
  within the averaging window — these are the by-hand QC categories. [¶(33), (53), (56), (57)]
- **G9 (by-hand pass):** visual review of the normalized (E,α) map + fit overlay for a random/flagged
  subsample — the step the group never automated away. [¶(45); Halekas correspondence]

## R2.6 What changed in the overall reconstruction

1. The beam's role upgraded from "co-fit component" to **the essential large-|U_M| constraint**
   (loss-cone-only information saturates at ~100 V; ¶[16] page-verified).
2. Beam model concretized: single energy bin at e|U_M−U_SC|, ~2 field-aligned upward pitch bins,
   amplitude ~3–5× incident (Fig. 5 forensics).
3. Loss-cone interior floor concretized: uniform, ≈0.05–0.1 in ratio units (Fig. 5 forensics).
4. Pitch binning fixed at 16 × 11.25° native bins (Halekas 2010 + Fig. 5 forensics agree).
5. Grid search directly evidenced (quantized U_M traces, step ≲10–20 V; continuous U_SC).
6. Selection function reconstructed (§R2.3): selected days, near-complete kept fraction after
   hand-vetting on those days, ~45–55% above-threshold negative detections on the best day;
   companion-paper screens (no strong crustal fields; clear secondary beam).
7. Error filter spec (§R2.5) added, anchored on the 2002 Monte Carlo precedent, Δχ² practice, and
   the author's own (private) confirmation that error filtering + by-hand QC are required.
8. Normalization pinned to conjugate-bin division as the spec choice (heritage), with the
   hemisphere-average variant documented as admissible (A2 narrowed, not closed).

---

# Round 3 (appended 2026-07-11): Page-verified closures from retrieved lineage PDFs

A PDF-retrieval wave (curl + local text extraction, bypassing the web-fetcher's PDF blindness)
obtained page-verified full text of: **Halekas et al. 2002 GRL** (via Wayback snapshot of the
Arizona repository bitstream; `h2002_grl_p*.txt`), **Halekas et al. 2010 JGR** (`h2010_jgr_p*.txt`),
**LPSC 2008 #1365 and 2009 #1357**, **Lillis et al. 2018 JGR-Planets** (`lillis2018_p*.txt`), both
**Lillis 2008 Icarus** methods papers (author copies; `lillis2008_*_p*.txt`), and **Farrell et al.
2010** (NTRS). Consolidated: `pdf_extraction_findings.md`, `lillis_findings.md`. Unobtainable after
exhaustive attempts: Halekas et al. 2009 PSS 57:78–82 (closed everywhere; its reported "no strong
crustal fields and a clear secondary beam" selection screen remains snippet-level, unverified),
Halekas 2005a GRL and Halekas 2007 GRL (10.1029/2006GL028517) full texts (lapsed bronze OA,
no mirrors/snapshots) — their content is adequately restated in the 2008 paper's ¶[13]–[16].

## R3.1 Closures (page-verified)

- **Beam width/shape (closes most of A3):** Halekas 2002 (the technique's origin): after
  subtracting a theoretical loss-cone distribution, the residual upward beam is **Gaussian with
  center 51 eV and FWHM 30 eV**, and the paper attributes this width to the instrument's
  **effective ΔE/E = 0.48** (24% analyzer resolution, adjacent sweep steps summed). So the beam is
  **resolution-limited**: a synthetic beam one effective energy bin wide (ΔE/E ≈ 0.5) at
  e|U_M − U_SC| is the faithful model — exactly what Figure 5's synthetic panel shows (~1 bin).
- **Per-fit error metric (anchors §R2.4):** Halekas 2002, Figure 2 caption: loss-cone measurements
  at **110/200/340/590 eV** carry "**error bars estimated from Monte Carlo simulations**", and the
  potential fit result is "**ΔU = −48 ± 15 V**, with the error again estimated from Monte Carlo
  simulations." The group's per-fit potential error is a **Monte Carlo propagation through the
  fit**; a Δχ²=1 interval is the cheap modern equivalent (used in the Lillis branch: 1-σ error
  ellipsoid where "χ² is less than its minimum plus unity", Lillis 2008 crustal-map App. C p.588).
- **2002-era selection/statistics context:** H2002 analyzed nightside geotail/wake data at
  20–40 km altitude, ~7000 near-side points, mean surface-minus-spacecraft potentials −32 V (wake)
  and −38 V (geotail), headline ~−35 V. (Correction to an earlier note: that "35 V" figure is the
  H2002 abstract itself, not Farrell 2010, which cites H2002/2009a only for secondary-emission
  yield 0.2–0.8 and the beam technique.)
- **Pitch handling in the shared LP ER code (confirms A5):** Halekas 2010 uses the ER's native
  **11.25° pitch bins**, oversampled and fit with a **step-function** cutoff model; its 0.55 nT
  floor is a crustal-B reliability cutoff for near-90° loss-cone angles (a B_S/B_M→1 degeneracy
  gate), not a potential-pipeline threshold.
- **The two toolchain branches now cleanly separated (major divergence-point evidence):**

  | Aspect | Lillis/Mars branch (page-verified) | Halekas 2008 lunar potential (paper text) |
  |---|---|---|
  | Fit space | **linear** relative flux (survival prob.) | **log** of normalized flux |
  | Domain | **masked to 0.07 ≤ ratio ≤ 0.79** (error-window rationale, Lillis 2008a §7.1 p.556) | **entire distribution** (both halves, all bins) |
  | Energies | 3 channels, 90–400 eV | all 15 channels, ~10 eV–20 keV (incl. <200 eV; ¶[16]) |
  | Pitch bins | 50 model bins → 8 upward instrumental bins | 16 × 11.25° bins, both hemispheres |
  | Beam | not modeled (atmosphere instead); backscatter subtracted via MC lookup | **co-fit** upward beam at e\|U_M−U_SC\|; no backscatter subtraction |
  | Optimizer | amoeba/Nelder–Mead (~1 s/fit) | brute-force 2-D grid (U_M quantization visible in Figs. 6/10/13) |
  | Errors | Δχ²=1 ellipsoid | Monte Carlo (2002 precedent); grid-χ² valley equivalent |
  | QC gate | **reject χ² > 2.0** (set by visual inspection); SZA > 120°, \|B\| > 10 nT, elevation > 40°; **~58% kept** (2.3M/4M); 2018: ~40% | hand-checking + 20-V floor + U_M>0 invalid; no published numeric χ² gate |

  A reimplementation that mixes branches (e.g., masked linear χ² à la Lillis applied to the lunar
  potential problem, or Lillis-style low-ratio masking that would **remove the beam and the cone
  interior** — the two most potential-sensitive regions) can reproduce a single benchmark yet
  diverge everywhere else. Conversely the Lillis mask rationale explains why an *unmasked linear*
  fit fails: ratio errors blow up as the ratio → 0 or 1, which is precisely what the 2008 choice of
  **log space** mitigates without discarding the cone interior and beam.

## R3.2 Final disposition of the ambiguity list

Resolved: A1 (units), A3 (beam: resolution-width Gaussian/top-hat, ~2 field-aligned upward pitch
bins, amplitude ~3–5× incident level — only the amplitude's exact parameterization stays open),
A5 (16 × 11.25° pitch bins), A9 (channels <200 eV included by design).
Narrowed: A2 (conjugate-bin division preferred; hemisphere-average admissible), A4 (log-space
unweighted least squares; finite cone floor ~0.05–0.1; no Poisson weighting evidenced), A6 (no
backscatter subtraction; floor absorbs it), A7 (gate stack §R2.5; Monte Carlo/Δχ² error filter;
20-V floor page-verified), A8 (grid step ≲10–20 V in U_M, extent to −4 kV; B_S/B_M ∈ ~(0.5, 1.05]).
Still open (publicly undocumented; live only in the original IDL code or author correspondence):
exact χ² weighting/count floor, beam amplitude parameterization, exact grid bounds/steps, the
hemisphere-vs-conjugate normalization choice, and any numeric auto-reject threshold used in 2008.

## R3.3 Leads chased to exhaustion
- Halekas PhD thesis (Berkeley 2003): no public copy; algorithm postdates it. DEAD.
- Halekas 2005a GRL, Halekas 2007 GRL full texts: lapsed bronze OA, Cloudflare-gated, no
  mirrors/Wayback. DEAD (content restated in 2008 ¶[13]–[16], which we have verbatim).
- Halekas 2009 PSS 57:78–82: closed access everywhere (Unpaywall/OpenAlex/S2 zero OA). DEAD.
- PDS derived surface-potential product: confirmed nonexistent. DEAD.
- Halekas 2009 JGR (SEP statistical study) full text: bot-walled; LPSC 2009 #1357 abstract
  obtained instead. PARTIALLY OPEN (an institutional fetch would close it).
- Everything else (2002, 2008, 2010, Lillis 2008 ×2, 2018, Farrell 2010, LPSC abstracts,
  PDS/instrument docs): obtained and page-verified. EXHAUSTED.
