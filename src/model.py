import math

import numpy as np

###########
# Constants
###########


e_0 = 1.602e-19  # Elementary charge in Coulombs


def synth_losscone(
    energy_grid: np.ndarray,
    pitch_grid: np.ndarray,
    U_surface: float,
    bs_over_bm: float,
    beam_width_eV: float = 0.0,
    beam_amp: float = 0.0,
    beam_pitch_sigma_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a loss-cone model that never returns NaN/Inf.

    Supports broadcasting for vectorized LHS:
    - energy_grid: (nE,)
    - pitch_grid: (nE, nPitch)
    - U_surface: scalar or (nParams,) - lunar surface potential in volts
    - bs_over_bm: scalar or (nParams,)

    Returns:
    - If params are scalar: (nE, nPitch)
    - If params are arrays: (nParams, nE, nPitch)
    """
    # Ensure inputs are arrays
    energy_grid = np.asarray(energy_grid)
    pitch_grid = np.asarray(pitch_grid)

    # Handle parameter broadcasting
    U_surface = np.asarray(U_surface)
    bs_over_bm = np.asarray(bs_over_bm)
    beam_amp = np.asarray(beam_amp)

    # Check if we are doing a batch calculation
    is_batch = U_surface.ndim > 0 or bs_over_bm.ndim > 0 or beam_amp.ndim > 0

    # Guard against E <= 0 (mask invalid energies)
    valid_E = energy_grid > 0
    E_safe = np.where(valid_E, energy_grid, 1.0)  # Avoid div by zero

    # Reshape inputs for broadcasting
    # Target shape: (nParams, nE, nPitch) if batch, else (nE, nPitch)

    if is_batch:
        # Ensure params are at least 1D
        if U_surface.ndim == 0:
            U_surface = U_surface[None]
        if bs_over_bm.ndim == 0:
            bs_over_bm = bs_over_bm[None]
        if beam_amp.ndim == 0:
            beam_amp = beam_amp[None]

        n_params = max(U_surface.size, bs_over_bm.size, beam_amp.size)

        # Reshape params to (nParams, 1, 1)
        U_surface = U_surface.reshape(-1, 1, 1)
        bs_over_bm = bs_over_bm.reshape(-1, 1, 1)
        beam_amp = beam_amp.reshape(-1, 1, 1)

        # Reshape grids to (1, nE, nPitch)
        # pitch_grid is (nE, nPitch)
        pitch_grid_exp = pitch_grid[None, :, :]
        # E_safe is (nE,) -> (1, nE, 1)
        E_safe_exp = E_safe[None, :, None]
        valid_E_exp = valid_E[None, :, None]

        # Calculate x = B_s/B_m * (1 + U_surface / E)
        # (nParams, 1, 1) * (1 + (nParams, 1, 1) / (1, nE, 1)) -> (nParams, nE, 1)
        x = bs_over_bm * (1.0 + U_surface / E_safe_exp)

        # Initialize model
        # (nParams, nE, nPitch)
        model = np.zeros((n_params, pitch_grid.shape[0], pitch_grid.shape[1]))

        # Calculate critical angle
        x_clipped = np.clip(x, 0.0, 1.0)
        ac_rad = np.arcsin(np.sqrt(x_clipped))
        ac_deg = np.degrees(ac_rad)  # (nParams, nE, 1)

        # Mask: pitch <= 180 - ac
        # (1, nE, nPitch) <= (nParams, nE, 1) -> (nParams, nE, nPitch)
        mask = pitch_grid_exp <= (180.0 - ac_deg)

        # Apply mask
        final_mask = mask & valid_E_exp
        model[final_mask] = 1.0

        # Beam
        if np.any(beam_width_eV > 0):
             # beam_center = max(abs(U_surface), beam_width_eV)
             # U_surface is (nParams, 1, 1)
             # beam_width_eV might be scalar or (nParams,), reshape to (nParams, 1, 1)
             beam_width_eV_arr = np.asarray(beam_width_eV)
             if beam_width_eV_arr.ndim == 0:
                 beam_width_eV_exp = beam_width_eV_arr
             else:
                 beam_width_eV_exp = beam_width_eV_arr.reshape(-1, 1, 1)

             beam_center = np.maximum(np.abs(U_surface), beam_width_eV_exp)

             # beam calculation
             # E is (1, nE, 1)
             # (nParams, nE, 1)
             beam = beam_amp * np.exp(-0.5 * ((E_safe_exp - beam_center) / beam_width_eV_exp) ** 2)
             
             if beam_pitch_sigma_deg > 0:
                 pitch_weight = np.exp(
                     -0.5 * ((pitch_grid_exp - 180.0) / beam_pitch_sigma_deg) ** 2
                 )
             else:
                 pitch_weight = np.ones_like(pitch_grid_exp)
                 
             model += beam * pitch_weight
             
        return model

    else:
        # Scalar path (optimized for single call)
        # Initialize model with zeros
        model = np.zeros_like(pitch_grid)

        if not np.any(valid_E):
            return model

        # Vectorized calculation for valid energies
        # Reshape E for broadcasting against pitch_grid (nE, nPitch)
        # Assuming energy_grid is 1D (nE,) and pitch_grid is 2D (nE, nPitch) or compatible

        # Calculate x = B_s/B_m * (1 + U_surface / E)
        # We need to handle the shape carefully.
        # If energy_grid is 1D, we reshape to (nE, 1) to broadcast against (nE, nPitch)
        x = bs_over_bm * (1.0 + U_surface / E_safe)

        # If x is scalar or 1D, broadcast to match pitch_grid
        if x.ndim == 1 and pitch_grid.ndim == 2:
            x = x[:, None]

        # Calculate critical angle (ac)
        x_clipped = np.clip(x, 0.0, 1.0)
        ac_rad = np.arcsin(np.sqrt(x_clipped))
        ac_deg = np.degrees(ac_rad)

        # Mask: pitch <= 180 - ac
        mask = pitch_grid <= (180.0 - ac_deg)

        # Apply mask only where E was valid and x was valid logic
        if valid_E.ndim == 1 and mask.ndim == 2:
            final_mask = mask & valid_E[:, None]
        else:
            final_mask = mask & valid_E

        model[final_mask] = 1.0

        # Optional narrow beam
        if beam_width_eV > 0 and beam_amp > 0:
            beam_center = max(abs(U_surface), beam_width_eV)
            # Vectorized beam calculation
            # E is (nE,), beam is (nE,)
            beam = beam_amp * np.exp(-0.5 * ((energy_grid - beam_center) / beam_width_eV) ** 2)
            
            if beam_pitch_sigma_deg > 0:
                # pitch is (nE, nPitch)
                pitch_weight = np.exp(
                    -0.5 * ((pitch_grid - 180.0) / beam_pitch_sigma_deg) ** 2
                )
            else:
                pitch_weight = np.ones_like(pitch_grid)

            # Add beam: beam is (nE,), pitch_weight is (nE, nPitch)
            # Broadcast beam to (nE, 1)
            if beam.ndim == 1:
                model += beam[:, None] * pitch_weight
            else:
                model += beam * pitch_weight

        return model


def _chi2(params, energies, pitches, data, eps):
    U_surface, bs_over_bm = params
    model = synth_losscone(energies, pitches, U_surface, bs_over_bm)

    # Bail-out if the model went pathological
    if not np.all(np.isfinite(model)) or (model <= 0).all():
        return 1e30  # huge penalty

    diff = np.log(data + eps) - np.log(model + eps)
    return np.sum(diff * diff)
