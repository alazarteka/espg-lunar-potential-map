# ER Measurement Contract

**Status:** Phase 1–4 bridge — 2026-07-20. Defines what the project observes and
what must be established for the D2 spacecraft-relative potential (`ΔU`)
error estimator.

## Scope

The production pipeline reads the PDS-calibrated Lunar Prospector ER 3-D
electron flux collection (`LP-L-ER-3-RDR-3DELEFLUX-80SEC-V1.1`), not Level-0
telemetry. Calibrated flux is downstream of:

1. Onboard logarithmic compression (truncated lower endpoint).
2. Ground decompression and division by integration time.
3. Non-extendable dead-time correction (values with factor > 1.25 masked).
4. Unsubtracted ~10 count/s background.
5. Division by geometric factor and energy → differential flux.

**Consequence:** reconstructed “pseudo-counts” from calibrated flux are
**not** detector counts and are not valid inputs to a Poisson or multinomial
count likelihood.

## Observation-model decision (Phase 4)

Level-0 telemetry codes are publicly available, but the MAG/ER SIS leaves the
per-anode efficiency table and onboard phase-space arithmetic as `TBS`. Until
those are recovered and a decoded-to-calibrated cross-check succeeds, the
publishable path is:

> **Calibrated-flux quasi-likelihood** with a **parametric bootstrap that
> simulates the same calibrated-data process**.

This must **not** be called a Poisson or multinomial count likelihood. The
fresh-look multinomial construction remains the *analogue* for
conditional-on-row-total structure; naming follows this contract.

Implemented in:

- [`src/losscone/quasi_likelihood.py`](../../src/losscone/quasi_likelihood.py)
- [`src/losscone/profile_ci.py`](../../src/losscone/profile_ci.py)
- [`src/losscone/confidence_set.py`](../../src/losscone/confidence_set.py)
- [`src/losscone/response_folded.py`](../../src/losscone/response_folded.py)
- [`src/losscone/regime_gate.py`](../../src/losscone/regime_gate.py)

Observation level string (stable NPZ metadata):
`calibrated_flux_quasi_likelihood`.

## What is *not* a confidence interval

| Legacy product | Role | CI status |
|---|---|---|
| Halekas log-distance | Point-estimate loss | Not a χ²; no CI |
| Lillis masked linear loss + 0.07–0.79 band | Point-estimate loss | Not a χ²; band selects on target |
| LHS `u_width_dchi2red_*` / `u_is_identifiable_*` | Optimizer-geometry diagnostic | **Not** a CI |

The honest uncertainty product is a **profile-likelihood confidence set**
`{U : Λ(U) ≤ c_α}` with `c_α` from the parametric bootstrap, reported with
endpoints, components, and one-sided / full-domain flags — never collapsed to
a lone scalar width.

## Regime gate (outside the likelihood)

Magnetic polarity / footpoint, energy leverage (`Var_w(1/K)`), and beam–edge
consistency are model-validity conditions. Failures emit limits /
nonidentifications (`GateReason`), never finite potentials with tight sets.

## Selection and independence

- Documented calibration masks only (finite flux, `E ≥ U_sc`, sunlight when
  available). **No 0.07–0.79 band** on the D2 path.
- Statistical unit is the **sweep** (15 energy rows), not 15 independent
  observations.

## Related docs

- Methods note: [profile_likelihood_ci.md](../physics/profile_likelihood_ci.md)
- Priority-0 r-bound check: [r_bound_artifact_check.md](../archive/analysis/r_bound_artifact_check.md)
- Diagnostics CLI: [diagnostics.md](../cli/diagnostics.md)
