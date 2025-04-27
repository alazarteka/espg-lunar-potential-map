import numpy as np
import pandas as pd
import spiceypy as spice

###########
# Constants
###########



e_0 = 1.602e-19  # Elementary charge in Coulombs

def synth_losscone(energy_grid: np.ndarray,
                   pitch_grid: np.ndarray,
                   delta_U: float,
                   bs_over_bm: float,
                   beam_width_eV: float = 0.0,
                   beam_amp: float = 0.0,
                   ):

    
    """
    Synthesizes a loss cone distribution in energy and pitch angle space.
    """
    # Initialize the distribution
    model = np.zeros((len(energy_grid), len(pitch_grid)))

    for i, energy in enumerate(energy_grid):
        ac = np.degrees(np.arcsin(np.sqrt(bs_over_bm * (1 + e_0 * delta_U / (energy * e_0)))))
        model[i, pitch_grid>=ac] =1.0

    if beam_width_eV>0 and beam_amp>0:
        beam_profile = beam_amp * np.exp(-0.5*((energy_grid - delta_U)/beam_width_eV)**2)
        # add it across all pitches
        model += beam_profile[:,None]

    return model

