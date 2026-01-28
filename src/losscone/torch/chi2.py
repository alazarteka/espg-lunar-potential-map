"""Torch chi² utilities for loss-cone fitting."""

from __future__ import annotations

import torch
from torch import Tensor

from src import config


def precompute_log_data_torch(
    data: Tensor,
    data_mask: Tensor,
    eps: float = 1e-6,
) -> tuple[Tensor, Tensor]:
    """
    Precompute log(data) and expanded mask for chi2 computation.

    Call this once before LHS/DE phases to avoid redundant log() calls.

    Args:
        data: (N_chunks, nE, nPitch) observed normalized flux per chunk
        data_mask: (N_chunks, nE, nPitch) boolean mask for valid data points
        eps: small value to avoid log(0)

    Returns:
        log_data_exp: (N_chunks, 1, nE, nPitch) precomputed log(data), broadcast-ready
        data_mask_exp: (N_chunks, 1, nE, nPitch) expanded mask
    """
    data_safe = torch.where(data_mask, data, torch.ones_like(data))
    log_data = torch.log(data_safe + eps)
    log_data_exp = log_data.unsqueeze(1)
    data_mask_exp = data_mask.unsqueeze(1)
    return log_data_exp, data_mask_exp


def compute_chi2_batch_torch(
    model: Tensor,
    data: Tensor,
    data_mask: Tensor,
    eps: float = 1e-6,
    model_mask: Tensor | None = None,
) -> Tensor:
    """
    Compute chi-squared for batch of models against data.

    Args:
        model: (n_params, nE, nPitch) model predictions
        data: (nE, nPitch) observed normalized flux
        data_mask: (nE, nPitch) boolean mask for valid data points
        eps: small value to avoid log(0)
        model_mask: Optional (n_params, nE, nPitch) mask indicating where
            model is physically valid (E >= U_spacecraft). Combined with data_mask.

    Returns:
        (n_params,) chi-squared values
    """
    # Replace invalid data with 1.0 before log to avoid NaN propagation
    # (masked out anyway, but prevents NaN * 0 = NaN issue)
    data_safe = torch.where(data_mask, data, torch.ones_like(data))

    log_model = torch.log(model + eps)
    log_data = torch.log(data_safe + eps)

    # Broadcast data to match model shape
    log_data_exp = log_data.unsqueeze(0)  # (1, nE, nPitch)
    data_mask_exp = data_mask.unsqueeze(0)  # (1, nE, nPitch)

    # Combine data_mask with model_mask if provided
    if model_mask is not None:
        combined_mask = data_mask_exp & model_mask
    else:
        combined_mask = data_mask_exp

    combined_mask = combined_mask.expand_as(log_model)

    # Compute diff only where mask is True
    diff = torch.where(
        combined_mask, log_data_exp - log_model, torch.zeros_like(log_model)
    )
    chi2 = (diff**2).sum(dim=(1, 2))
    n_valid = combined_mask.sum(dim=(1, 2))
    chi2 = torch.where(n_valid > 0, chi2, torch.full_like(chi2, float("nan")))

    return chi2


def compute_lillis_chi2_batch_torch(
    model: Tensor,
    data: Tensor,
    data_mask: Tensor,
    model_mask: Tensor | None = None,
) -> Tensor:
    """
    Compute Lillis-style reduced chi-squared for batch of models (linear space).

    Args:
        model: (n_params, nE, nPitch) model predictions
        data: (nE, nPitch) observed normalized flux
        data_mask: (nE, nPitch) boolean mask for valid bins (Lillis mask)
        model_mask: Optional (n_params, nE, nPitch) model validity mask

    Returns:
        (n_params,) reduced chi-squared values
    """
    data_exp = data.unsqueeze(0)  # (1, nE, nPitch)
    data_mask_exp = data_mask.unsqueeze(0)  # (1, nE, nPitch)

    if model_mask is not None:
        combined_mask = data_mask_exp & model_mask
    else:
        combined_mask = data_mask_exp

    combined_mask = combined_mask.expand_as(model)
    diff = torch.where(combined_mask, data_exp - model, torch.zeros_like(model))
    chi2 = (diff**2).sum(dim=(1, 2))
    n_valid = combined_mask.sum(dim=(1, 2))
    dof = torch.clamp(n_valid.to(dtype=chi2.dtype) - 3.0, min=1.0)
    reduced = chi2 / dof
    reduced = torch.where(
        n_valid >= config.LILLIS_MIN_VALID_BINS,
        reduced,
        torch.full_like(reduced, float("nan")),
    )
    return reduced


def compute_chi2_multi_chunk_torch(
    models: Tensor,
    data: Tensor,
    data_mask: Tensor,
    eps: float = 1e-6,
    log_data_precomputed: tuple[Tensor, Tensor] | None = None,
    model_mask: Tensor | None = None,
) -> Tensor:
    """
    Compute chi-squared for multiple chunks × multiple candidates.

    Args:
        models: (N_chunks, n_pop, nE, nPitch) model predictions
        data: (N_chunks, nE, nPitch) observed normalized flux per chunk
        data_mask: (N_chunks, nE, nPitch) boolean mask for valid data points
        eps: small value to avoid log(0)
        log_data_precomputed: Optional (log_data_exp, data_mask_exp) from
            precompute_log_data_torch(). If provided, skips redundant log(data).
        model_mask: Optional (N_chunks, n_pop, nE, nPitch) mask indicating where
            model is physically valid (E >= U_spacecraft). Combined with data_mask.

    Returns:
        (N_chunks, n_pop) chi-squared values
    """
    log_model = torch.log(models + eps)

    if log_data_precomputed is not None:
        log_data_exp, data_mask_exp = log_data_precomputed
    else:
        # Fallback: compute log_data inline (for backwards compatibility)
        data_safe = torch.where(data_mask, data, torch.ones_like(data))
        log_data = torch.log(data_safe + eps)
        log_data_exp = log_data.unsqueeze(1)
        data_mask_exp = data_mask.unsqueeze(1)

    # Combine data_mask with model_mask if provided
    if model_mask is not None:
        combined_mask = data_mask_exp & model_mask
    else:
        combined_mask = data_mask_exp

    combined_mask = combined_mask.expand_as(log_model)

    # Compute diff only where mask is True
    diff = torch.where(
        combined_mask, log_data_exp - log_model, torch.zeros_like(log_model)
    )
    chi2 = (diff**2).sum(dim=(2, 3))  # Sum over (E, A) -> (N, P)
    n_valid = combined_mask.sum(dim=(2, 3))
    chi2 = torch.where(n_valid > 0, chi2, torch.full_like(chi2, float("nan")))

    return chi2


def compute_lillis_chi2_multi_chunk_torch(
    models: Tensor,
    data: Tensor,
    data_mask: Tensor,
    model_mask: Tensor | None = None,
) -> Tensor:
    """
    Compute Lillis-style reduced chi-squared for multiple chunks × candidates.

    Args:
        models: (N_chunks, n_pop, nE, nPitch) model predictions
        data: (N_chunks, nE, nPitch) observed normalized flux per chunk
        data_mask: (N_chunks, nE, nPitch) boolean Lillis mask per chunk
        model_mask: Optional (N_chunks, n_pop, nE, nPitch) model validity mask

    Returns:
        (N_chunks, n_pop) reduced chi-squared values
    """
    data_exp = data.unsqueeze(1)  # (N, 1, nE, nPitch)
    data_mask_exp = data_mask.unsqueeze(1)  # (N, 1, nE, nPitch)

    if model_mask is not None:
        combined_mask = data_mask_exp & model_mask
    else:
        combined_mask = data_mask_exp

    combined_mask = combined_mask.expand_as(models)
    diff = torch.where(combined_mask, data_exp - models, torch.zeros_like(models))
    chi2 = (diff**2).sum(dim=(2, 3))
    n_valid = combined_mask.sum(dim=(2, 3))
    dof = torch.clamp(n_valid.to(dtype=chi2.dtype) - 3.0, min=1.0)
    reduced = chi2 / dof
    reduced = torch.where(
        n_valid >= config.LILLIS_MIN_VALID_BINS,
        reduced,
        torch.full_like(reduced, float("nan")),
    )
    return reduced
