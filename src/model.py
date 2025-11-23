import math

import numpy as np

###########
# Constants
###########


e_0 = 1.602e-19  # Elementary charge in Coulombs


def synth_losscone(
    energy_grid: np.ndarray,
    pitch_grid: np.ndarray,
    delta_U: float,
    bs_over_bm: float,
    beam_width_eV: float = 0.0,
    beam_amp: float = 0.0,
    beam_pitch_sigma_deg: float = 0.0,
) -> np.ndarray:
    """
    Build a loss-cone model that never returns NaN/Inf.
    """
    # Ensure inputs are arrays
    energy_grid = np.asarray(energy_grid)
    pitch_grid = np.asarray(pitch_grid)
    
    # Guard against E <= 0 (mask invalid energies)
    valid_E = energy_grid > 0
    
    # Initialize model with zeros
    model = np.zeros_like(pitch_grid)
    
    if not np.any(valid_E):
        return model

    # Vectorized calculation for valid energies
    # Reshape E for broadcasting against pitch_grid (nE, nPitch)
    # Assuming energy_grid is 1D (nE,) and pitch_grid is 2D (nE, nPitch) or compatible
    
    # Calculate x = B_s/B_m * (1 + delta_U / E)
    # We need to handle the shape carefully. 
    # If energy_grid is 1D, we reshape to (nE, 1) to broadcast against (nE, nPitch)
    E_safe = np.where(valid_E, energy_grid, 1.0) # Avoid div by zero
    x = bs_over_bm * (1.0 + delta_U / E_safe)
    
    # If x is scalar or 1D, broadcast to match pitch_grid
    if x.ndim == 1 and pitch_grid.ndim == 2:
        x = x[:, None]
    
    # Calculate critical angle (ac)
    # x <= 0 -> ac = 0
    # x >= 1 -> ac = 90
    # else -> ac = asin(sqrt(x))
    
    # We can use numpy operations directly
    # sqrt(x) might be invalid if x < 0, so clip first
    x_clipped = np.clip(x, 0.0, 1.0)
    ac_rad = np.arcsin(np.sqrt(x_clipped))
    ac_deg = np.degrees(ac_rad)
    
    # Handle x <= 0 explicitly if needed, but clip(0,1) handles the math.
    # However, the original logic had:
    # if x <= 0: ac = 0 (handled by clip -> 0 -> asin(0)=0)
    # if x >= 1: ac = 90 (handled by clip -> 1 -> asin(1)=pi/2=90)
    
    # Mask: pitch <= 180 - ac
    mask = pitch_grid <= (180.0 - ac_deg)
    
    # Apply mask only where E was valid and x was valid logic
    # (The original loop skipped E<=0, leaving model=0 there)
    # We need to ensure we don't set 1s where E<=0
    if valid_E.ndim == 1 and mask.ndim == 2:
        final_mask = mask & valid_E[:, None]
    else:
        final_mask = mask & valid_E
        
    model[final_mask] = 1.0

    # Optional narrow beam
    if beam_width_eV > 0 and beam_amp > 0:
        beam_center = max(abs(delta_U), beam_width_eV)
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
    delta_U, bs_over_bm = params
    model = synth_losscone(energies, pitches, delta_U, bs_over_bm)

    # Bail-out if the model went pathological
    if not np.all(np.isfinite(model)) or (model <= 0).all():
        return 1e30  # huge penalty

    diff = np.log(data + eps) - np.log(model + eps)
    return np.sum(diff * diff)
