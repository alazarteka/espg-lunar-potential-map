import numpy as np
import pandas as pd
import spiceypy as spice
import math

###########
# Constants
###########



e_0 = 1.602e-19  # Elementary charge in Coulombs

def synth_losscone(energy_grid: np.ndarray,
                   pitch_grid: np.ndarray,
                   delta_U: float,
                   bs_over_bm: float,
                   beam_width_eV: float = 0.0,
                   beam_amp: float = 0.0) -> np.ndarray:
    """
    Build a loss-cone model that never returns NaN/Inf.
    """
    nE, nPitch = pitch_grid.shape
    model = np.zeros((nE, nPitch))

    for i, E in enumerate(energy_grid):
        # Guard against E ≤ 0
        if E <= 0:
            continue

        x = bs_over_bm * (1.0 + delta_U / E)   # dimensionless

        # Map illegal values onto physically plausible limits
        if x <= 0.0:
            ac = 0.0           # full loss cone (no mirroring)
        elif x >= 1.0:
            ac = 90.0          # mirror point at 90°, loss cone closed
        else:
            ac = math.degrees(math.asin(math.sqrt(x)))

        mask = pitch_grid[i] <= 180 - ac
        model[i, mask] = 1.0

    # Optional narrow beam
    if beam_width_eV > 0 and beam_amp > 0:
        beam = beam_amp * np.exp(-0.5 * ((energy_grid - delta_U) / beam_width_eV) ** 2)
        model += beam[:, None]

    return model

def _chi2(params, energies, pitches, data, eps):
    delta_U, bs_over_bm = params
    model = synth_losscone(energies, pitches, delta_U, bs_over_bm)

    # Bail-out if the model went pathological
    if not np.all(np.isfinite(model)) or (model <= 0).all():
        return 1e30          # huge penalty

    diff = np.log(data + eps) - np.log(model + eps)
    return np.sum(diff * diff)
