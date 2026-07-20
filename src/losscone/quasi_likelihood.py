"""Calibrated-flux quasi-likelihood for spacecraft-relative DeltaU (Phase 4).

This is deliberately *not* named Poisson or multinomial count likelihood.
Calibrated LP-ER flux is downstream of compression, dead time, background, and
G*E conversion; the ER measurement contract requires an explicitly conditional
calibrated-flux quasi-likelihood whose parametric bootstrap simulates the same
calibrated-data process.

The row-conditional structure mirrors the fresh-look multinomial analogue:
per-energy rows are compared after shared-normalizer conditioning so incident
amplitudes cancel without treating ratios as independent.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.losscone.response_folded import (
    ResponseFoldedParams,
    build_calibration_mask,
    response_folded_mean,
    response_folded_mean_batch,
)

EPS = 1e-12


@dataclass(frozen=True)
class QuasiLikelihoodConfig:
    """Observation-model knobs for the calibrated-flux quasi-likelihood."""

    # Relative flux noise floor after calibration (digitization / residual).
    sigma_rel: float = 0.15
    # Absolute floor so empty cells do not dominate.
    sigma_abs: float = 1e-3
    # Soften zeros in log terms.
    eps: float = 1e-8
    use_response_folding: bool = True
    de_over_e: float = 0.5
    n_energy_quad: int = 5
    beam_width_eV: float = 40.0
    beam_pitch_sigma_deg: float = 15.0
    edge_broadening_deg: float = 0.0
    background: float = 0.05


def _row_normalize(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Normalize each energy row by its masked sum (conditional-on-total)."""
    x = np.asarray(x, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    out = np.zeros_like(x, dtype=np.float64)
    masked = np.where(m, np.maximum(x, 0.0), 0.0)
    row_sum = masked.sum(axis=-1, keepdims=True)
    good = row_sum > 0
    out = np.divide(masked, row_sum, out=out, where=good)
    return out


def quasi_negloglik(
    data_flux: np.ndarray,
    model_flux: np.ndarray,
    mask: np.ndarray,
    *,
    cfg: QuasiLikelihoodConfig | None = None,
) -> float:
    """Row-conditional Gaussian quasi-NLL on calibrated flux proportions.

    For each energy row, both data and model are normalized by their masked
    cell sum (the calibrated-flux analogue of conditioning on the row total).
    Residuals use a diagonal weight sigma² = (sigma_rel * p)² + sigma_abs² on the free
    simplex coordinates (one masked cell dropped per row) so the statistic
    is not inflated by the singular sum-to-one constraint.
    """
    cfg = cfg or QuasiLikelihoodConfig()
    data = np.asarray(data_flux, dtype=np.float64)
    model = np.asarray(model_flux, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    if data.shape != model.shape or data.shape != m.shape:
        raise ValueError("data, model, and mask must share shape")

    p_data = _row_normalize(data, m)
    p_model = _row_normalize(model, m)

    nll = 0.0
    nE, _nP = m.shape
    for i in range(nE):
        cols = np.flatnonzero(m[i])
        if cols.size < 2:
            continue
        free = cols[:-1]  # drop one cell for the simplex constraint
        resid = p_data[i, free] - p_model[i, free]
        sigma2 = (cfg.sigma_rel * np.maximum(p_model[i, free], cfg.eps)) ** 2 + (
            cfg.sigma_abs**2
        )
        nll += 0.5 * float(np.sum(resid * resid / sigma2 + np.log(sigma2)))
    if not np.isfinite(nll):
        return 1e30
    return float(nll)


def mean_model_for_params(
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    *,
    u_surface: float,
    bs_over_bm: float,
    beam_amp: float,
    u_spacecraft: float | np.ndarray = 0.0,
    cfg: QuasiLikelihoodConfig | None = None,
) -> np.ndarray:
    """Forward mean under the configured response-folded (or point) model."""
    cfg = cfg or QuasiLikelihoodConfig()
    if cfg.use_response_folding:
        params = ResponseFoldedParams(
            u_surface=float(u_surface),
            bs_over_bm=float(bs_over_bm),
            beam_amp=float(beam_amp),
            beam_width_eV=cfg.beam_width_eV,
            beam_pitch_sigma_deg=cfg.beam_pitch_sigma_deg,
            edge_broadening_deg=cfg.edge_broadening_deg,
            background=cfg.background,
            de_over_e=cfg.de_over_e,
            n_energy_quad=cfg.n_energy_quad,
        )
        return response_folded_mean(
            energy_centers, pitch_grid, params, u_spacecraft=u_spacecraft
        )
    from src.losscone.model import synth_losscone

    return synth_losscone(
        energy_grid=energy_centers,
        pitch_grid=pitch_grid,
        U_surface=float(u_surface),
        U_spacecraft=u_spacecraft,
        bs_over_bm=float(bs_over_bm),
        beam_width_eV=cfg.beam_width_eV,
        beam_amp=float(beam_amp),
        beam_pitch_sigma_deg=cfg.beam_pitch_sigma_deg,
        background=cfg.background,
    )


def nll_at_params(
    data_flux: np.ndarray,
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    *,
    u_surface: float,
    bs_over_bm: float,
    beam_amp: float,
    u_spacecraft: float | np.ndarray = 0.0,
    mask: np.ndarray | None = None,
    cfg: QuasiLikelihoodConfig | None = None,
) -> float:
    """Evaluate quasi-NLL at a single parameter point."""
    cfg = cfg or QuasiLikelihoodConfig()
    model = mean_model_for_params(
        energy_centers,
        pitch_grid,
        u_surface=u_surface,
        bs_over_bm=bs_over_bm,
        beam_amp=beam_amp,
        u_spacecraft=u_spacecraft,
        cfg=cfg,
    )
    if mask is None:
        mask = build_calibration_mask(data_flux)
    return quasi_negloglik(data_flux, model, mask, cfg=cfg)


def simulate_calibrated_flux(
    model_flux: np.ndarray,
    mask: np.ndarray,
    *,
    cfg: QuasiLikelihoodConfig | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Parametric bootstrap draw under the stated calibrated-flux noise model.

    Draws row-conditional noisy proportions and rescales to the observed row
    totals so the bootstrap matches the conditional quasi-likelihood.
    """
    cfg = cfg or QuasiLikelihoodConfig()
    rng = rng or np.random.default_rng()
    model = np.asarray(model_flux, dtype=np.float64)
    m = np.asarray(mask, dtype=bool)
    p = _row_normalize(model, m)
    sigma = np.sqrt((cfg.sigma_rel * np.maximum(p, cfg.eps)) ** 2 + cfg.sigma_abs**2)
    noise = rng.normal(0.0, 1.0, size=p.shape) * sigma
    p_draw = np.clip(p + noise, 0.0, None)
    p_draw = _row_normalize(p_draw, m)

    # Preserve observed row scale when available via model row sums as proxy.
    row_scale = np.where(m, np.maximum(model, 0.0), 0.0).sum(axis=-1, keepdims=True)
    row_scale = np.maximum(row_scale, cfg.eps)
    out = p_draw * row_scale
    out = np.where(m, out, np.nan)
    return out


def mean_model_batch(
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    u_surface: np.ndarray,
    bs_over_bm: np.ndarray,
    beam_amp: np.ndarray,
    *,
    u_spacecraft: float | np.ndarray = 0.0,
    cfg: QuasiLikelihoodConfig | None = None,
) -> np.ndarray:
    """Batch forward means for profiling / LHS."""
    cfg = cfg or QuasiLikelihoodConfig()
    if cfg.use_response_folding:
        return response_folded_mean_batch(
            energy_centers,
            pitch_grid,
            u_surface,
            bs_over_bm,
            beam_amp,
            u_spacecraft=u_spacecraft,
            beam_width_eV=cfg.beam_width_eV,
            beam_pitch_sigma_deg=cfg.beam_pitch_sigma_deg,
            edge_broadening_deg=cfg.edge_broadening_deg,
            background=cfg.background,
            de_over_e=cfg.de_over_e,
            n_energy_quad=cfg.n_energy_quad,
        )
    from src.losscone.model import synth_losscone_batch

    n = int(np.asarray(u_surface).size)
    return synth_losscone_batch(
        energy_grid=energy_centers,
        pitch_grid=pitch_grid,
        U_surface=np.asarray(u_surface, dtype=np.float64),
        U_spacecraft=u_spacecraft,
        bs_over_bm=np.asarray(bs_over_bm, dtype=np.float64),
        beam_width_eV=np.full(n, cfg.beam_width_eV),
        beam_amp=np.asarray(beam_amp, dtype=np.float64),
        beam_pitch_sigma_deg=cfg.beam_pitch_sigma_deg,
        background=np.full(n, cfg.background),
    )
