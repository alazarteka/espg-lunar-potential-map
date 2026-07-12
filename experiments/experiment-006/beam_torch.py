"""Batched PyTorch forward model for Experiment 006 beam inference.

All floating-point inputs are expected to share a dtype and device. Rows and
samples may be padded; masks keep padding out of both the model and objective.
This module intentionally contains no optimizer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence

N_CONTINUUM_QUANTITIES = 4
N_CONTINUUM_COEFFICIENTS = 12
N_PARAMETERS = 15


def _as_batch_column(value: float | Tensor, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.ndim == 0:
        return result
    if result.shape != reference.shape[:-1]:
        raise ValueError("Batched scalar shape must match the leading dimensions")
    return result.unsqueeze(-1)


def _bounds_tensor(value: Sequence[float] | Tensor, reference: Tensor) -> Tensor:
    result = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    if result.shape == (N_CONTINUUM_QUANTITIES,):
        return result
    expected = (*reference.shape[:-1], N_CONTINUUM_QUANTITIES)
    if result.shape != expected:
        raise ValueError("Continuum bounds must contain four values per batch")
    return result.unsqueeze(-2)


def integrated_energy_response(
    channel_low_ev: Tensor,
    channel_high_ev: Tensor,
    center_ev: float | Tensor,
    fwhm_fraction: float | Tensor,
) -> Tensor:
    """Average a linear-energy Gaussian response over padded channel rows."""
    if channel_low_ev.shape != channel_high_ev.shape:
        raise ValueError("Channel lower and upper edges must have matching shapes")
    if (
        not channel_low_ev.is_floating_point()
        or not channel_high_ev.is_floating_point()
    ):
        raise TypeError("Channel edges must be floating-point tensors")
    if channel_low_ev.dtype != channel_high_ev.dtype:
        raise TypeError("Channel edges must have the same dtype")
    if channel_low_ev.device != channel_high_ev.device:
        raise ValueError("Channel edges must be on the same device")

    center = _as_batch_column(center_ev, channel_low_ev)
    fraction = _as_batch_column(fwhm_fraction, channel_low_ev)
    width = channel_high_ev - channel_low_ev
    sigma = fraction * center / 2.355
    sqrt_two = channel_low_ev.new_tensor(2.0).sqrt()
    upper = (channel_high_ev - center) / (sqrt_two * sigma)
    lower = (channel_low_ev - center) / (sqrt_two * sigma)
    coefficient = sigma * (channel_low_ev.new_tensor(torch.pi / 2.0)).sqrt()
    integral = coefficient * (torch.erf(upper) - torch.erf(lower))
    return (integral / width).clamp_min(0.0)


def energy_basis(
    energies_ev: Tensor,
    energy_domain_ev: tuple[float, float] | Tensor,
) -> Tensor:
    """Return the quadratic basis on a fixed normalized log-energy domain."""
    if not energies_ev.is_floating_point():
        raise TypeError("Energies must be a floating-point tensor")
    domain = torch.as_tensor(
        energy_domain_ev, dtype=energies_ev.dtype, device=energies_ev.device
    )
    if domain.shape != (2,):
        raise ValueError("Energy domain must contain lower and upper bounds")
    scaled = (
        2.0
        * (
            (torch.log(energies_ev) - torch.log(domain[0]))
            / (torch.log(domain[1]) - torch.log(domain[0]))
        )
        - 1.0
    )
    return torch.stack((torch.ones_like(scaled), scaled, scaled.square()), dim=-1)


def continuum_row_parameters(
    energies_ev: Tensor,
    coefficients: Tensor,
    energy_domain_ev: tuple[float, float] | Tensor,
    physical_lower: Sequence[float] | Tensor,
    physical_upper: Sequence[float] | Tensor,
) -> Tensor:
    """Map batched latent coefficients to four bounded parameters per row."""
    expected = (*energies_ev.shape[:-1], N_CONTINUUM_COEFFICIENTS)
    if coefficients.shape != expected:
        raise ValueError("Expected 12 continuum coefficients per batch")
    if (
        coefficients.dtype != energies_ev.dtype
        or coefficients.device != energies_ev.device
    ):
        raise ValueError("Energies and coefficients must share dtype and device")

    basis = energy_basis(energies_ev, energy_domain_ev)
    coefficient_matrix = coefficients.reshape(
        *coefficients.shape[:-1], N_CONTINUUM_QUANTITIES, 3
    )
    latent = torch.einsum("...ri,...qi->...rq", basis, coefficient_matrix)
    lower = _bounds_tensor(physical_lower, energies_ev)
    upper = _bounds_tensor(physical_upper, energies_ev)
    return lower + (upper - lower) * torch.sigmoid(latent)


def model_values(
    energies_ev: Tensor,
    channel_low_ev: Tensor,
    channel_high_ev: Tensor,
    row_ids: Tensor,
    pitches_deg: Tensor,
    parameters: Tensor,
    *,
    energy_domain_ev: tuple[float, float] | Tensor,
    fwhm_fraction: float | Tensor,
    physical_lower: Sequence[float] | Tensor,
    physical_upper: Sequence[float] | Tensor,
    row_valid_mask: Tensor | None = None,
    sample_valid_mask: Tensor | None = None,
) -> Tensor:
    """Evaluate the batched channel-integrated continuum-plus-beam model.

    ``energies_ev`` and channel edges have shape ``(..., rows)``; ``row_ids``
    and pitches have shape ``(..., samples)``. Invalid output samples are zero.
    """
    if (
        energies_ev.shape != channel_low_ev.shape
        or energies_ev.shape != channel_high_ev.shape
    ):
        raise ValueError("Energies and channel edges must have matching shapes")
    if row_ids.shape != pitches_deg.shape:
        raise ValueError("Row identifiers and pitch samples must match")
    if row_ids.shape[:-1] != energies_ev.shape[:-1]:
        raise ValueError("Row and sample tensors must have matching batches")
    if parameters.shape != (*energies_ev.shape[:-1], N_PARAMETERS):
        raise ValueError("Expected 15 beam-model parameters per batch")
    if row_ids.dtype != torch.long:
        raise TypeError("Row identifiers must have dtype torch.long")
    for value in (channel_low_ev, channel_high_ev, pitches_deg, parameters):
        if value.dtype != energies_ev.dtype or value.device != energies_ev.device:
            raise ValueError("All floating-point inputs must share dtype and device")

    if row_valid_mask is None:
        row_valid = torch.ones_like(energies_ev, dtype=torch.bool)
    else:
        if row_valid_mask.shape != energies_ev.shape:
            raise ValueError("Row mask must match the padded energy rows")
        row_valid = row_valid_mask.to(device=energies_ev.device, dtype=torch.bool)
    if sample_valid_mask is None:
        sample_valid = torch.ones_like(row_ids, dtype=torch.bool)
    else:
        if sample_valid_mask.shape != row_ids.shape:
            raise ValueError("Sample mask must match the padded samples")
        sample_valid = sample_valid_mask.to(device=energies_ev.device, dtype=torch.bool)

    row_count = energies_ev.shape[-1]
    safe_row_ids = row_ids.clamp(0, max(row_count - 1, 0))
    in_range = (row_ids >= 0) & (row_ids < row_count)
    sample_valid = sample_valid & in_range
    selected_row_valid = torch.gather(row_valid, -1, safe_row_ids)
    sample_valid = sample_valid & selected_row_valid
    safe_pitches = torch.where(sample_valid, pitches_deg, pitches_deg.new_tensor(180.0))

    domain = torch.as_tensor(
        energy_domain_ev, dtype=energies_ev.dtype, device=energies_ev.device
    )
    safe_energies = torch.where(row_valid, energies_ev, domain[0])
    safe_low = torch.where(row_valid, channel_low_ev, domain[0])
    safe_high = torch.where(row_valid, channel_high_ev, domain[0] * 1.01)

    row_parameters = continuum_row_parameters(
        safe_energies,
        parameters[..., 3:],
        energy_domain_ev,
        physical_lower,
        physical_upper,
    )
    gather_index = safe_row_ids.unsqueeze(-1).expand(*safe_row_ids.shape, 4)
    selected = torch.gather(row_parameters, -2, gather_index)
    transition = torch.sigmoid((selected[..., 2] - safe_pitches) / selected[..., 3])
    continuum = selected[..., 0] + selected[..., 1] * transition

    row_response = integrated_energy_response(
        safe_low, safe_high, parameters[..., 1], fwhm_fraction
    )
    selected_response = torch.gather(row_response, -1, safe_row_ids)
    pitch_response = torch.exp(
        -0.5 * ((safe_pitches - 180.0) / parameters[..., 2].unsqueeze(-1)).square()
    )
    beam = parameters[..., 0].unsqueeze(-1) * selected_response * pitch_response
    return torch.where(sample_valid, continuum + beam, torch.zeros_like(continuum))


def log_residuals(
    observed: Tensor, modeled: Tensor, valid_mask: Tensor | None = None
) -> Tensor:
    """Return log residuals, with optionally masked positions set to zero."""
    if observed.shape != modeled.shape:
        raise ValueError("Observed and modeled tensors must match")
    if observed.dtype != modeled.dtype or observed.device != modeled.device:
        raise ValueError("Observed and modeled tensors must share dtype and device")
    if valid_mask is None:
        valid = torch.ones_like(observed, dtype=torch.bool)
    else:
        if valid_mask.shape != observed.shape:
            raise ValueError("Validity mask must match observed values")
        valid = valid_mask.to(device=observed.device, dtype=torch.bool)
    one = torch.ones((), dtype=observed.dtype, device=observed.device)
    safe_observed = torch.where(valid, observed, one)
    safe_modeled = torch.where(valid, modeled, one)
    residuals = torch.log(safe_modeled) - torch.log(safe_observed)
    return torch.where(valid, residuals, torch.zeros_like(residuals))


def masked_log_residuals(
    observed: Tensor, modeled: Tensor, valid_mask: Tensor
) -> Tensor:
    """Return padded log residuals, with invalid positions set to zero."""
    return log_residuals(observed, modeled, valid_mask)


def masked_log_objective(
    observed: Tensor, modeled: Tensor, valid_mask: Tensor
) -> Tensor:
    """Return the unweighted sum of squared log residuals for each batch."""
    return masked_log_residuals(observed, modeled, valid_mask).square().sum(dim=-1)
