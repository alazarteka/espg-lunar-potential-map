# Profile-likelihood confidence sets for ΔU

## Claim

The strongest routine uncertainty statement attachable to a single-sweep
spacecraft-relative potential is a **coverage-calibrated profile-likelihood
confidence set** under an explicitly conditional **calibrated-flux
quasi-likelihood**. It is **not** a Poisson or multinomial count CI.

\[
\{ U :\; \Lambda(U)=2[\ell_{\max}-\ell_p(U)] \le c_\alpha \}
\]

with \(c_\alpha\) from a parametric bootstrap through the same observation
model. Sets may be two-sided, one-sided, disconnected, or the full admissible
domain.

## Why not the legacy “chi-squares”

Halekas (unweighted log-distance), Lillis (unweighted linear / \((n-3)\)), and
any Cash-on-pseudo-counts construction lack a valid observational variance for
the *archived* calibrated flux. Their numeric values have no goodness-of-fit
distribution, so they cannot host a CI. The LHS `u_width` diagnostic measures
near-optimal sampled optimizer geometry, not a profile confidence interval.

See [er_measurement_contract.md](../architecture/er_measurement_contract.md).

## Implementation map

| Module | Role |
|---|---|
| `src/losscone/response_folded.py` | Phase-3 mean: ΔE/E≈0.5 energy quadrature + optional edge blur |
| `src/losscone/quasi_likelihood.py` | Row-conditional calibrated-flux quasi-NLL + bootstrap draws |
| `src/losscone/regime_gate.py` | Pre/post-fit physics gates (outside the likelihood) |
| `src/losscone/profile_ci.py` | Adaptive profile + bootstrap \(c_\alpha\) → set |
| `src/losscone/confidence_set.py` | Schema / NPZ serialization |

## Naming discipline

- Say **calibrated-flux quasi-likelihood confidence set**.
- Do **not** say Poisson CI, multinomial CI, or “U-width = 1σ error”.
- Per-sweep intervals are **conditional on adiabatic / magnetic-connection
  assumptions**; statistics alone cannot police energy-dependent effective \(r\).

## Priority 0

The legacy `LOSS_CONE_BS_OVER_BM_MIN = 0.3` floor is an identifying constraint.
Run `scripts/diagnostics/r_bound_artifact_check.py` before staking narrative
claims about a ~−100 V population median. The D2 path defaults to
`PROFILE_CI_R_MIN = 0.02`.
