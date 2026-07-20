"""Profile-likelihood confidence sets for DeltaU / U_surface (Phase 4 core).

Builds ``{U : 2[ℓ_max - ℓ_p(U)] <= c_alpha}`` with ``c_alpha`` from a parametric
bootstrap under the calibrated-flux quasi-likelihood. Sets are reported
verbatim (endpoints, components, one-sided / full-domain flags) - never
collapsed to a scalar U-width that discards scientifically useful limits.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize_scalar

from src import config
from src.losscone.confidence_set import (
    GateReason,
    SweepConfidenceSet,
    level_set_from_profile,
)
from src.losscone.quasi_likelihood import (
    QuasiLikelihoodConfig,
    mean_model_for_params,
    nll_at_params,
    quasi_negloglik,
    simulate_calibrated_flux,
)
from src.losscone.regime_gate import (
    RegimeGateResult,
    beam_edge_consistency,
    prefit_regime_gate,
)
from src.losscone.response_folded import build_calibration_mask


@dataclass(frozen=True)
class ProfileCIConfig:
    """Profiler / bootstrap controls."""

    alpha: float = 0.32  # ~1sigma one-sided analogue; two-sided ~68%
    u_min: float = config.LOSS_CONE_U_SURFACE_MIN
    u_max: float = config.LOSS_CONE_U_SURFACE_MAX
    # Allow r below the legacy 0.3 floor (artifact hypothesis).
    r_min: float = 0.02
    r_max: float = 1.0
    beam_amp_min: float = config.LOSS_CONE_BEAM_AMP_MIN
    beam_amp_max: float = config.LOSS_CONE_BEAM_AMP_MAX
    n_r_grid: int = 17
    n_beam_grid: int = 9
    coarse_u_count: int = 41
    refine_tol_v: float = 5.0
    refine_factor: int = 4
    bootstrap_n: int = 40
    bootstrap_seed: int = 0
    min_usable_channels: int = 2
    ql: QuasiLikelihoodConfig | None = None


@dataclass(frozen=True)
class ProfileResult:
    confidence_set: SweepConfidenceSet
    gate: RegimeGateResult
    u_grid: np.ndarray
    lambda_profile: np.ndarray
    r_profile: np.ndarray
    beam_profile: np.ndarray


def _nuisance_grid(cfg: ProfileCIConfig) -> tuple[np.ndarray, np.ndarray]:
    r = np.linspace(cfg.r_min, cfg.r_max, cfg.n_r_grid)
    a = np.linspace(cfg.beam_amp_min, cfg.beam_amp_max, cfg.n_beam_grid)
    return r, a


def profile_nll_at_u(
    data_flux: np.ndarray,
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    u_surface: float,
    *,
    u_spacecraft: float | np.ndarray = 0.0,
    mask: np.ndarray | None = None,
    cfg: ProfileCIConfig | None = None,
) -> tuple[float, float, float]:
    """Minimize quasi-NLL over (r, A_beam) at fixed U. Returns (nll, r, A)."""
    cfg = cfg or ProfileCIConfig()
    ql = cfg.ql or QuasiLikelihoodConfig()
    if mask is None:
        mask = build_calibration_mask(data_flux)

    r_grid, a_grid = _nuisance_grid(cfg)
    best_nll = 1e30
    best_r = float("nan")
    best_a = float("nan")
    for r in r_grid:
        for a in a_grid:
            nll = nll_at_params(
                data_flux,
                energy_centers,
                pitch_grid,
                u_surface=float(u_surface),
                bs_over_bm=float(r),
                beam_amp=float(a),
                u_spacecraft=u_spacecraft,
                mask=mask,
                cfg=ql,
            )
            if nll < best_nll:
                best_nll = nll
                best_r = float(r)
                best_a = float(a)

    # Local 1-D polish in r at best A (cheap).
    if np.isfinite(best_a):

        def _obj(r: float) -> float:
            return nll_at_params(
                data_flux,
                energy_centers,
                pitch_grid,
                u_surface=float(u_surface),
                bs_over_bm=float(r),
                beam_amp=float(best_a),
                u_spacecraft=u_spacecraft,
                mask=mask,
                cfg=ql,
            )

        polished = minimize_scalar(
            _obj,
            bounds=(cfg.r_min, cfg.r_max),
            method="bounded",
            options={"xatol": 1e-3},
        )
        if polished.success and polished.fun < best_nll:
            best_nll = float(polished.fun)
            best_r = float(polished.x)

    return best_nll, best_r, best_a


def build_u_grid(cfg: ProfileCIConfig) -> np.ndarray:
    """Coarse U grid spanning the admissible domain."""
    return np.linspace(cfg.u_min, cfg.u_max, cfg.coarse_u_count)


def refine_u_grid(
    u_grid: np.ndarray,
    lambda_profile: np.ndarray,
    c_alpha: float,
    *,
    cfg: ProfileCIConfig,
) -> np.ndarray:
    """Insert refined nodes near Lambda ~ c_alpha threshold crossings."""
    u = np.asarray(u_grid, dtype=np.float64)
    lam = np.asarray(lambda_profile, dtype=np.float64)
    extras: list[float] = []
    for i in range(len(u) - 1):
        a, b = lam[i] - c_alpha, lam[i + 1] - c_alpha
        if not (np.isfinite(a) and np.isfinite(b)):
            continue
        if a == 0.0 or b == 0.0 or a * b < 0.0:
            # Crossing: densify between u[i] and u[i+1].
            seg = np.linspace(u[i], u[i + 1], cfg.refine_factor + 2)[1:-1]
            extras.extend(float(x) for x in seg)
        elif min(abs(a), abs(b)) < 0.5 and abs(u[i + 1] - u[i]) > cfg.refine_tol_v:
            seg = np.linspace(u[i], u[i + 1], cfg.refine_factor + 2)[1:-1]
            extras.extend(float(x) for x in seg)
    if not extras:
        return u
    merged = np.unique(np.concatenate([u, np.asarray(extras, dtype=np.float64)]))
    return merged


def compute_lambda_profile(
    data_flux: np.ndarray,
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    u_grid: np.ndarray,
    *,
    u_spacecraft: float | np.ndarray = 0.0,
    mask: np.ndarray | None = None,
    cfg: ProfileCIConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    """Profile Lambda(U) = 2 (nll(U) - nll_min). Returns grids + MLE triple."""
    cfg = cfg or ProfileCIConfig()
    if mask is None:
        mask = build_calibration_mask(data_flux)

    nlls = np.empty(len(u_grid), dtype=np.float64)
    rs = np.empty(len(u_grid), dtype=np.float64)
    amps = np.empty(len(u_grid), dtype=np.float64)
    for i, u in enumerate(u_grid):
        nll, r, a = profile_nll_at_u(
            data_flux,
            energy_centers,
            pitch_grid,
            float(u),
            u_spacecraft=u_spacecraft,
            mask=mask,
            cfg=cfg,
        )
        nlls[i] = nll
        rs[i] = r
        amps[i] = a

    imin = int(np.nanargmin(nlls))
    nll_min = float(nlls[imin])
    lam = 2.0 * (nlls - nll_min)
    return (
        lam,
        rs,
        amps,
        float(u_grid[imin]),
        float(rs[imin]),
        float(amps[imin]),
    )


def bootstrap_c_alpha(
    data_flux: np.ndarray,
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    *,
    u_hat: float,
    r_hat: float,
    beam_amp_hat: float,
    u_spacecraft: float | np.ndarray = 0.0,
    mask: np.ndarray | None = None,
    cfg: ProfileCIConfig | None = None,
) -> float:
    """Empirical (1-alpha) quantile of Lambda(U_true) under parametric bootstrap at MLE.

    At the true U used to generate the replicate, Lambda should be near zero under a
    correct model; the null distribution of Lambda at the constrained true value
    (vs free MLE on the replicate) calibrates c_alpha, including active bounds.
    """
    cfg = cfg or ProfileCIConfig()
    ql = cfg.ql or QuasiLikelihoodConfig()
    if mask is None:
        mask = build_calibration_mask(data_flux)

    mean = mean_model_for_params(
        energy_centers,
        pitch_grid,
        u_surface=u_hat,
        bs_over_bm=r_hat,
        beam_amp=beam_amp_hat,
        u_spacecraft=u_spacecraft,
        cfg=ql,
    )
    rng = np.random.default_rng(cfg.bootstrap_seed)
    stats: list[float] = []

    # Coarse U grid for bootstrap free fits (keep cheap).
    boot_u = np.linspace(cfg.u_min, cfg.u_max, max(11, cfg.coarse_u_count // 2))

    for _ in range(int(cfg.bootstrap_n)):
        y = simulate_calibrated_flux(mean, mask, cfg=ql, rng=rng)
        # Free MLE on replicate.
        free_nlls = []
        for u in boot_u:
            nll, _, _ = profile_nll_at_u(
                y,
                energy_centers,
                pitch_grid,
                float(u),
                u_spacecraft=u_spacecraft,
                mask=mask,
                cfg=cfg,
            )
            free_nlls.append(nll)
        nll_free = float(np.min(free_nlls))
        # Constrained at true u_hat.
        nll_con, _, _ = profile_nll_at_u(
            y,
            energy_centers,
            pitch_grid,
            float(u_hat),
            u_spacecraft=u_spacecraft,
            mask=mask,
            cfg=cfg,
        )
        stats.append(2.0 * (nll_con - nll_free))

    if not stats:
        return 1.0  # asymptotic χ²_1 68% fallback
    q = float(np.quantile(stats, 1.0 - cfg.alpha))
    # Guard against numerical negatives / zeros.
    return max(q, 1e-6)


def fit_profile_confidence_set(
    data_flux: np.ndarray,
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    *,
    u_spacecraft: float | np.ndarray = 0.0,
    polarity: float | int | None = 1,
    beam_centroid_eV: float | None = None,
    cfg: ProfileCIConfig | None = None,
    skip_bootstrap: bool = False,
    c_alpha_override: float | None = None,
) -> ProfileResult:
    """End-to-end: regime gate → profile → bootstrap c_alpha → confidence set."""
    cfg = cfg or ProfileCIConfig()
    ql = cfg.ql or QuasiLikelihoodConfig()
    mask = build_calibration_mask(data_flux)

    gate = prefit_regime_gate(
        energy_centers,
        polarity=polarity,
        flux=data_flux,
        min_usable_channels=cfg.min_usable_channels,
    )
    empty_grid = np.asarray([], dtype=np.float64)
    if not gate.ok:
        cs = SweepConfidenceSet(
            gate_reason=gate.reason,
            domain_lo=cfg.u_min,
            domain_hi=cfg.u_max,
            notes=gate.message,
            alpha=cfg.alpha,
            bootstrap_n=0,
        )
        return ProfileResult(cs, gate, empty_grid, empty_grid, empty_grid, empty_grid)

    u_grid = build_u_grid(cfg)
    lam, rs, amps, u_hat, r_hat, a_hat = compute_lambda_profile(
        data_flux,
        energy_centers,
        pitch_grid,
        u_grid,
        u_spacecraft=u_spacecraft,
        mask=mask,
        cfg=cfg,
    )
    nll_min = float(
        nll_at_params(
            data_flux,
            energy_centers,
            pitch_grid,
            u_surface=u_hat,
            bs_over_bm=r_hat,
            beam_amp=a_hat,
            u_spacecraft=u_spacecraft,
            mask=mask,
            cfg=ql,
        )
    )

    post = beam_edge_consistency(
        u_edge=u_hat,
        beam_centroid_eV=beam_centroid_eV,
        u_spacecraft=float(np.asarray(u_spacecraft).reshape(-1)[0])
        if np.size(u_spacecraft)
        else 0.0,
    )
    if not post.ok:
        cs = SweepConfidenceSet(
            u_hat=u_hat,
            r_hat=r_hat,
            beam_amp_hat=a_hat,
            nll_min=nll_min,
            alpha=cfg.alpha,
            domain_lo=cfg.u_min,
            domain_hi=cfg.u_max,
            gate_reason=post.reason,
            notes=post.message,
            profile_u=tuple(float(x) for x in u_grid),
            profile_lambda=tuple(float(x) for x in lam),
        )
        return ProfileResult(cs, post, u_grid, lam, rs, amps)

    if c_alpha_override is not None:
        c_alpha = float(c_alpha_override)
        boot_n = 0
    elif skip_bootstrap:
        c_alpha = 1.0  # asymptotic χ²_1 ~68%
        boot_n = 0
    else:
        c_alpha = bootstrap_c_alpha(
            data_flux,
            energy_centers,
            pitch_grid,
            u_hat=u_hat,
            r_hat=r_hat,
            beam_amp_hat=a_hat,
            u_spacecraft=u_spacecraft,
            mask=mask,
            cfg=cfg,
        )
        boot_n = int(cfg.bootstrap_n)

    # Refine near threshold crossings and rebuild Lambda on the denser grid.
    u_ref = refine_u_grid(u_grid, lam, c_alpha, cfg=cfg)
    if u_ref.size != u_grid.size or not np.allclose(u_ref, u_grid):
        lam, rs, amps, u_hat, r_hat, a_hat = compute_lambda_profile(
            data_flux,
            energy_centers,
            pitch_grid,
            u_ref,
            u_spacecraft=u_spacecraft,
            mask=mask,
            cfg=cfg,
        )
        u_grid = u_ref
        nll_min = float(
            nll_at_params(
                data_flux,
                energy_centers,
                pitch_grid,
                u_surface=u_hat,
                bs_over_bm=r_hat,
                beam_amp=a_hat,
                u_spacecraft=u_spacecraft,
                mask=mask,
                cfg=ql,
            )
        )

    comps, t_lo, t_hi, is_full, is_one = level_set_from_profile(
        u_grid,
        lam,
        c_alpha,
        domain_lo=cfg.u_min,
        domain_hi=cfg.u_max,
    )
    if is_full:
        gate_reason = GateReason.FULL_DOMAIN
    elif len(comps) == 0:
        gate_reason = GateReason.NONIDENTIFIABLE
    else:
        gate_reason = GateReason.OK

    cs = SweepConfidenceSet(
        u_hat=u_hat,
        r_hat=r_hat,
        beam_amp_hat=a_hat,
        nll_min=nll_min,
        c_alpha=c_alpha,
        alpha=cfg.alpha,
        bootstrap_n=boot_n,
        components=comps,
        domain_lo=cfg.u_min,
        domain_hi=cfg.u_max,
        touches_bound_lo=t_lo,
        touches_bound_hi=t_hi,
        is_full_domain=is_full,
        is_one_sided=is_one,
        gate_reason=gate_reason,
        profile_u=tuple(float(x) for x in u_grid),
        profile_lambda=tuple(float(x) for x in lam),
    )
    return ProfileResult(cs, gate, u_grid, lam, rs, amps)


# Re-export for callers that need the deviance evaluator.
__all__ = [
    "ProfileCIConfig",
    "ProfileResult",
    "bootstrap_c_alpha",
    "compute_lambda_profile",
    "fit_profile_confidence_set",
    "profile_nll_at_u",
    "quasi_negloglik",
]
