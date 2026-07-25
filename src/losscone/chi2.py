"""Chi-squared objective functions for loss-cone fitting: Halekas log-χ² and
Lillis masked linear χ²."""

from __future__ import annotations

import numpy as np

from src import config
from src.losscone.masks import build_lillis_mask


def _get_eps(eps: float | None) -> float:
    if eps is not None:
        return float(eps)
    return float(getattr(config, "EPS", 1e-6))


def _as_model_mask_3d(
    model_mask: np.ndarray | None, n_models: int
) -> np.ndarray | None:
    if model_mask is None:
        return None
    mask = np.asarray(model_mask)
    if mask.ndim == 2:
        return np.broadcast_to(mask[None, :, :], (n_models, *mask.shape))
    if mask.ndim == 3:
        if mask.shape[0] != n_models:
            raise ValueError(
                f"model_mask has {mask.shape[0]} models, expected {n_models}"
            )
        return mask
    raise ValueError(f"model_mask must be 2D or 3D, got shape {mask.shape}")


def compute_halekas_chi2(
    norm2d: np.ndarray,
    model: np.ndarray,
    model_mask: np.ndarray | None = None,
    eps: float | None = None,
) -> float:
    eps = _get_eps(eps)
    data_mask = np.isfinite(norm2d) & (norm2d > 0)
    combined_mask = data_mask & model_mask if model_mask is not None else data_mask
    if not combined_mask.any():
        return float("nan")

    log_data = np.zeros_like(norm2d, dtype=float)
    np.log(norm2d + eps, out=log_data, where=combined_mask)

    log_model = np.zeros_like(model, dtype=float)
    np.log(model + eps, out=log_model, where=combined_mask)

    diff = log_data - log_model
    chi2 = np.sum(diff[combined_mask] ** 2)
    return float(chi2)


def compute_halekas_chi2_batch(
    norm2d: np.ndarray,
    models: np.ndarray,
    model_mask: np.ndarray | None = None,
    eps: float | None = None,
) -> np.ndarray:
    """
    Vectorized Halekas-style chi2 over many models.

    Args:
        norm2d: Measured normalized flux, shape (nE, nPitch).
        models: Model normalized flux, shape (n_models, nE, nPitch).
        model_mask: Optional validity mask, shape (nE, nPitch) or
            (n_models, nE, nPitch).
        eps: Epsilon for log stability.

    Returns:
        chi2 values, shape (n_models,). Entries are NaN if no valid bins.
    """
    eps = _get_eps(eps)
    models = np.asarray(models)
    if models.ndim != 3:
        raise ValueError(
            f"models must be 3D (n_models, nE, nPitch), got {models.shape}"
        )
    n_models = models.shape[0]

    data_mask = np.isfinite(norm2d) & (norm2d > 0)
    if not data_mask.any():
        return np.full(n_models, np.nan, dtype=float)

    log_data = np.zeros_like(norm2d, dtype=float)
    np.log(norm2d + eps, out=log_data, where=data_mask)

    combined_mask = np.broadcast_to(data_mask[None, :, :], models.shape)
    model_mask_3d = _as_model_mask_3d(model_mask, n_models)
    if model_mask_3d is not None:
        combined_mask = combined_mask & model_mask_3d

    n_valid = np.count_nonzero(combined_mask, axis=(1, 2)).astype(int)

    log_models = np.zeros_like(models, dtype=float)
    np.log(models + eps, out=log_models, where=combined_mask)
    diff = (log_data[None, :, :] - log_models) * combined_mask
    chi2 = np.sum(diff * diff, axis=(1, 2))
    chi2 = chi2.astype(float, copy=False)
    chi2[n_valid == 0] = np.nan
    return chi2


def compute_cash_deviance_batch(
    models: np.ndarray,
    counts: np.ndarray,
    count_scale: np.ndarray,
    background: np.ndarray,
    model_mask: np.ndarray | None = None,
    *,
    reduced: bool = False,
) -> np.ndarray:
    """Poisson deviance (Cash statistic) for many normalized loss-cone models.

    ``D = 2 Σ_masked [μ − C + C·ln(C/μ)]`` with model expected counts
    ``μ = model·κ + b`` and observed counts ``C``. Uses the ``C·ln(C/μ) = 0``
    convention at ``C = 0`` (Poisson deviance is well-defined there). ``μ > 0``
    always because the model floor (background) and the count scale ``κ`` are
    positive and ``b ≥ 0``.

    The profile interval ``ΔD ≤ 1`` is the 1-parameter 68% CI (Wilks). This is a
    genuine variance-weighted objective (per-bin Poisson statistics), unlike the
    unweighted :func:`compute_lillis_chi2`.

    Args:
        models: (n_models, nE, nPitch) normalized model flux.
        counts: (nE, nPitch) observed counts ``C = λ`` (float, unrounded).
        count_scale: (nE, nPitch) per-bin ``κ`` mapping model → expected counts.
        background: (nE, nPitch) or (nPitch,) per-bin background counts ``b``.
        model_mask: optional (nE, nPitch) or (n_models, nE, nPitch) validity mask.
        reduced: if True, divide by ``max(n_valid − 3, 1)`` (per-dof).

    Returns:
        (n_models,) deviance; NaN where fewer than ``LILLIS_MIN_VALID_BINS`` valid.
    """
    models = np.asarray(models, dtype=np.float64)
    if models.ndim != 3:
        raise ValueError(
            f"models must be 3D (n_models, nE, nPitch), got {models.shape}"
        )
    n_models = models.shape[0]
    kappa = np.asarray(count_scale, dtype=np.float64)
    c_obs = np.asarray(counts, dtype=np.float64)
    b = np.asarray(background, dtype=np.float64)
    if b.ndim == 1:
        b = np.broadcast_to(b[None, :], c_obs.shape)

    mu = models * kappa[None, :, :] + b[None, :, :]  # (n_models, nE, nPitch), > 0
    c3 = c_obs[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        c_ln = np.where(c3 > 0.0, c3 * np.log(c3 / mu), 0.0)
    dev = 2.0 * (mu - c3 + c_ln)  # per bin, >= 0

    mask3d = _as_model_mask_3d(model_mask, n_models)
    if mask3d is not None:
        dev = np.where(mask3d, dev, 0.0)
        n_valid = np.count_nonzero(mask3d, axis=(1, 2)).astype(int)
    else:
        n_valid = np.full(n_models, c_obs.size, dtype=int)

    total = dev.sum(axis=(1, 2)).astype(float)
    if reduced:
        dof = np.maximum(n_valid - 3, 1)
        total = total / dof.astype(float)
    total[n_valid < config.LILLIS_MIN_VALID_BINS] = np.nan
    return total


def compute_lillis_chi2(
    norm2d: np.ndarray,
    model: np.ndarray,
    raw_flux: np.ndarray,
    pitches: np.ndarray,
    model_mask: np.ndarray | None = None,
) -> float:
    data_mask = build_lillis_mask(raw_flux, pitches)
    combined_mask = data_mask & model_mask if model_mask is not None else data_mask
    n_valid = int(np.count_nonzero(combined_mask))
    if n_valid < config.LILLIS_MIN_VALID_BINS:
        return float("nan")
    diff = np.zeros_like(norm2d, dtype=float)
    np.subtract(norm2d, model, out=diff, where=combined_mask)
    chi2 = np.sum(diff[combined_mask] ** 2)
    dof = max(n_valid - 3, 1)
    return float(chi2) / float(dof)


def compute_lillis_chi2_batch(
    norm2d: np.ndarray,
    models: np.ndarray,
    raw_flux: np.ndarray,
    pitches: np.ndarray,
    model_mask: np.ndarray | None = None,
) -> np.ndarray:
    """
    Vectorized Lillis-style reduced chi2 over many models.

    Args:
        norm2d: Measured normalized flux, shape (nE, nPitch).
        models: Model normalized flux, shape (n_models, nE, nPitch).
        raw_flux: Raw (unnormalized) flux for mask construction, shape (nE, nPitch).
        pitches: Pitch angles in degrees, shape (nE, nPitch).
        model_mask: Optional validity mask, shape (nE, nPitch) or
            (n_models, nE, nPitch).

    Returns:
        Reduced chi2 values, shape (n_models,). Entries are NaN if too few valid bins.
    """
    models = np.asarray(models)
    if models.ndim != 3:
        raise ValueError(
            f"models must be 3D (n_models, nE, nPitch), got {models.shape}"
        )
    n_models = models.shape[0]

    data_mask = build_lillis_mask(raw_flux, pitches)
    model_mask_3d = _as_model_mask_3d(model_mask, n_models)
    combined_mask = np.broadcast_to(data_mask[None, :, :], models.shape)
    if model_mask_3d is not None:
        combined_mask = combined_mask & model_mask_3d

    n_valid = np.count_nonzero(combined_mask, axis=(1, 2)).astype(int)
    too_few = n_valid < config.LILLIS_MIN_VALID_BINS

    diff = np.zeros_like(models, dtype=float)
    np.subtract(norm2d[None, :, :], models, out=diff, where=combined_mask)
    chi2 = np.sum(diff * diff, axis=(1, 2))
    dof = np.maximum(n_valid - 3, 1)
    reduced = chi2 / dof.astype(float, copy=False)
    reduced = reduced.astype(float, copy=False)
    reduced[too_few] = np.nan
    return reduced
