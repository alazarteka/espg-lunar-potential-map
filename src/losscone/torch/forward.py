"""
Torch loss-cone forward model.

This module contains the PyTorch equivalents of the NumPy loss-cone model in
`src.losscone.model`, implemented with vectorized tensor operations.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Small epsilon to avoid division by zero
EPS = 1e-12

# Default flux value outside loss cone
DEFAULT_BACKGROUND = 0.05


def synth_losscone_batch_torch(
    energy_grid: Tensor,
    pitch_grid: Tensor,
    U_surface: Tensor,
    U_spacecraft: float | Tensor = 0.0,
    bs_over_bm: Tensor | None = None,
    beam_width_eV: Tensor | None = None,
    beam_amp: Tensor | None = None,
    beam_pitch_sigma_deg: float = 0.0,
    background: Tensor | None = None,
    return_mask: bool = False,
) -> Tensor | tuple[Tensor, Tensor]:
    """
    PyTorch batch loss-cone model for GPU acceleration.

    Computes loss-cone models for multiple parameter combinations simultaneously.
    All tensor parameters must have the same length (n_params).

    This implementation uses hard masks (non-differentiable) to match
    the CPU implementation exactly.

    Args:
        energy_grid: (nE,) electron energies in eV
        pitch_grid: (nE, nPitch) pitch angles in degrees
        U_surface: (n_params,) lunar surface potentials in volts
        U_spacecraft: spacecraft potential in volts (scalar or (nE,))
        bs_over_bm: (n_params,) B_spacecraft/B_surface ratios (default: all 1.0)
        beam_width_eV: (n_params,) beam widths in eV (default: all 0.0)
        beam_amp: (n_params,) beam amplitudes (default: all 0.0)
        beam_pitch_sigma_deg: beam angular spread in degrees (scalar)
        background: (n_params,) flux outside loss cone (default: all 0.05)
        return_mask: if True, return (model, valid_mask) tuple

    Returns:
        If return_mask=False: Model flux tensor of shape (n_params, nE, nPitch)
        If return_mask=True: Tuple of (model, valid_mask) where valid_mask is
            shape (n_params, nE, nPitch) indicating where E >= U_spacecraft
    """
    device = energy_grid.device
    dtype = energy_grid.dtype

    U_surface = U_surface.to(device=device, dtype=dtype)
    n_params = U_surface.size(0)

    # Set defaults for optional tensors
    if bs_over_bm is None:
        bs_over_bm = torch.ones(n_params, device=device, dtype=dtype)
    else:
        bs_over_bm = bs_over_bm.to(device=device, dtype=dtype)

    if beam_width_eV is None:
        beam_width_eV = torch.zeros(n_params, device=device, dtype=dtype)
    else:
        beam_width_eV = beam_width_eV.to(device=device, dtype=dtype)

    if beam_amp is None:
        beam_amp = torch.zeros(n_params, device=device, dtype=dtype)
    else:
        beam_amp = beam_amp.to(device=device, dtype=dtype)

    if background is None:
        background = torch.full(
            (n_params,), DEFAULT_BACKGROUND, device=device, dtype=dtype
        )
    else:
        background = background.to(device=device, dtype=dtype)

    # Reshape for broadcasting: params -> (nParams, 1, 1)
    U_surface = U_surface.view(-1, 1, 1)
    bs_over_bm = bs_over_bm.view(-1, 1, 1)
    beam_amp = beam_amp.view(-1, 1, 1)
    beam_width_eV = beam_width_eV.view(-1, 1, 1)
    background = background.view(-1, 1, 1)

    # Reshape grids for broadcasting
    nE, nPitch = pitch_grid.shape
    pitch_exp = pitch_grid.unsqueeze(0)  # (1, nE, nPitch)
    E_exp = energy_grid.unsqueeze(0).unsqueeze(-1)  # (1, nE, 1)

    # Handle U_spacecraft: scalar -> (1,1,1), array(nE,) -> (1,nE,1)
    if isinstance(U_spacecraft, (int, float)):
        U_spacecraft_t = torch.tensor(U_spacecraft, device=device, dtype=dtype).view(
            1, 1, 1
        )
    else:
        U_spacecraft_t = (
            U_spacecraft.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        )

    # Compute validity mask: E >= U_spacecraft
    valid_energy = E_exp >= U_spacecraft_t

    # Compute loss cone angle using Halekas 2008 formula
    # sin²(αc) = (BS/BM) × (1 + UM / (E - U_spacecraft))
    E_corrected = torch.clamp(E_exp - U_spacecraft_t, min=EPS)
    x = bs_over_bm * (1.0 + U_surface / E_corrected)
    x_clipped = torch.clamp(x, 0.0, 1.0)
    ac_deg = torch.rad2deg(torch.arcsin(torch.sqrt(x_clipped)))

    # Build model: background everywhere, 1.0 inside loss cone
    model = background.expand(n_params, nE, nPitch).clone()

    # Inside loss cone: pitch <= 180 - αc (hard mask, matches CPU)
    inside_cone = pitch_exp <= (180.0 - ac_deg)
    model = torch.where(inside_cone, torch.ones_like(model), model)

    # Add secondary electron beam if enabled
    has_beam = (beam_width_eV > 0).any() and (beam_amp > 0).any()
    has_accel = (U_spacecraft_t > U_surface).any()
    if has_beam and has_accel:
        delta_u = U_spacecraft_t - U_surface
        accel_mask = (delta_u > 0).to(dtype=dtype)
        beam_center = torch.maximum(delta_u, beam_width_eV)
        beam_width_safe = torch.clamp(beam_width_eV, min=EPS)
        energy_profile = (
            beam_amp
            * accel_mask
            * torch.exp(-0.5 * ((E_exp - beam_center) / beam_width_safe) ** 2)
        )

        if beam_pitch_sigma_deg > 0:
            pitch_profile = torch.exp(
                -0.5 * ((pitch_exp - 180.0) / beam_pitch_sigma_deg) ** 2
            )
        else:
            pitch_profile = 1.0

        beam = energy_profile * pitch_profile
        model = model + beam

    if return_mask:
        # Broadcast valid_energy to full shape (n_params, nE, nPitch)
        valid_mask = valid_energy.expand(n_params, nE, nPitch)
        return model, valid_mask
    else:
        return model


def synth_losscone_multi_chunk_torch(
    energy_grids: Tensor,
    pitch_grids: Tensor,
    U_surface: Tensor,
    U_spacecraft: Tensor | None = None,
    bs_over_bm: Tensor | None = None,
    beam_width_eV: Tensor | None = None,
    beam_amp: Tensor | None = None,
    beam_pitch_sigma_deg: float = 0.0,
    background: Tensor | None = None,
) -> Tensor:
    """
    PyTorch multi-chunk loss-cone model for batched GPU acceleration.

    Computes loss-cone models for multiple chunks × multiple parameter candidates.
    Each chunk has its own energy/pitch grids.

    Args:
        energy_grids: (N_chunks, nE) electron energies in eV per chunk
        pitch_grids: (N_chunks, nE, nPitch) pitch angles in degrees per chunk
        U_surface: (N_chunks, n_pop) lunar surface potentials in volts
        U_spacecraft: (N_chunks, nE) or (N_chunks,) spacecraft potential per chunk
        bs_over_bm: (N_chunks, n_pop) B_spacecraft/B_surface ratios
        beam_width_eV: (N_chunks, n_pop) beam widths in eV
        beam_amp: (N_chunks, n_pop) beam amplitudes
        beam_pitch_sigma_deg: beam angular spread in degrees (scalar)
        background: (N_chunks, n_pop) flux outside loss cone

    Returns:
        Model flux tensor of shape (N_chunks, n_pop, nE, nPitch)
    """
    device = energy_grids.device
    dtype = energy_grids.dtype

    N_chunks, nE = energy_grids.shape
    n_pop = U_surface.size(1)
    nPitch = pitch_grids.size(2)

    # Set defaults for optional tensors
    if bs_over_bm is None:
        bs_over_bm = torch.ones(N_chunks, n_pop, device=device, dtype=dtype)

    if beam_width_eV is None:
        beam_width_eV = torch.zeros(N_chunks, n_pop, device=device, dtype=dtype)

    if beam_amp is None:
        beam_amp = torch.zeros(N_chunks, n_pop, device=device, dtype=dtype)

    if background is None:
        background = torch.full(
            (N_chunks, n_pop), DEFAULT_BACKGROUND, device=device, dtype=dtype
        )

    # Reshape for broadcasting:
    # U_surface: (N, P) -> (N, P, 1, 1)
    # energy_grids: (N, E) -> (N, 1, E, 1)
    # pitch_grids: (N, E, A) -> (N, 1, E, A)
    U_surface_exp = U_surface.view(N_chunks, n_pop, 1, 1)
    bs_over_bm_exp = bs_over_bm.view(N_chunks, n_pop, 1, 1)
    beam_amp_exp = beam_amp.view(N_chunks, n_pop, 1, 1)
    beam_width_exp = beam_width_eV.view(N_chunks, n_pop, 1, 1)
    background_exp = background.view(N_chunks, n_pop, 1, 1)

    # Energy: (N, E) -> (N, 1, E, 1)
    valid_E = energy_grids > 0
    E_safe = torch.where(valid_E, energy_grids, torch.ones_like(energy_grids))
    E_exp = E_safe.view(N_chunks, 1, nE, 1)
    valid_E_exp = valid_E.view(N_chunks, 1, nE, 1)

    # Pitch: (N, E, A) -> (N, 1, E, A)
    pitch_exp = pitch_grids.unsqueeze(1)

    # Handle U_spacecraft: (N, E) -> (N, 1, E, 1) or (N,) -> (N, 1, 1, 1)
    if U_spacecraft is None:
        U_spacecraft_exp = torch.zeros(N_chunks, 1, 1, 1, device=device, dtype=dtype)
    elif U_spacecraft.dim() == 1:
        # Per-chunk scalar: (N,) -> (N, 1, 1, 1)
        U_spacecraft_exp = U_spacecraft.view(N_chunks, 1, 1, 1)
    else:
        # Per-chunk per-energy: (N, E) -> (N, 1, E, 1)
        U_spacecraft_exp = U_spacecraft.view(N_chunks, 1, nE, 1)

    # Compute loss cone angle using Halekas 2008 formula
    # sin²(αc) = (BS/BM) × (1 + UM / (E - U_spacecraft))
    E_corrected = torch.clamp(E_exp - U_spacecraft_exp, min=EPS)
    x = bs_over_bm_exp * (1.0 + U_surface_exp / E_corrected)
    x_clipped = torch.clamp(x, 0.0, 1.0)
    ac_deg = torch.rad2deg(torch.arcsin(torch.sqrt(x_clipped)))

    # Build model: background everywhere, 1.0 inside loss cone
    model = background_exp.expand(N_chunks, n_pop, nE, nPitch).clone()

    # Inside loss cone: pitch <= 180 - αc (hard mask)
    inside_cone = (pitch_exp <= (180.0 - ac_deg)) & valid_E_exp
    model = torch.where(inside_cone, torch.ones_like(model), model)

    # Add secondary electron beam if enabled
    has_beam = (beam_width_exp > 0).any() and (beam_amp_exp > 0).any()
    has_accel = (U_spacecraft_exp > U_surface_exp).any()
    if has_beam and has_accel:
        delta_u = U_spacecraft_exp - U_surface_exp
        accel_mask = (delta_u > 0).to(dtype=dtype)
        beam_center = torch.maximum(delta_u, beam_width_exp)
        beam_width_safe = torch.clamp(beam_width_exp, min=EPS)
        energy_profile = (
            beam_amp_exp
            * accel_mask
            * torch.exp(-0.5 * ((E_exp - beam_center) / beam_width_safe) ** 2)
        )

        if beam_pitch_sigma_deg > 0:
            pitch_profile = torch.exp(
                -0.5 * ((pitch_exp - 180.0) / beam_pitch_sigma_deg) ** 2
            )
        else:
            pitch_profile = 1.0

        beam = energy_profile * pitch_profile
        model = model + beam

    return model
