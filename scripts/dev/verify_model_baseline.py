
import numpy as np
from src.model import synth_losscone as old_synth_losscone

def test_vectorization():
    # Generate random inputs
    np.random.seed(42)
    n_energy = 15
    n_pitch = 88
    energy_grid = np.geomspace(10, 20000, n_energy)
    pitch_grid = np.linspace(0, 180, n_pitch)
    # Broadcast pitch grid to (n_energy, n_pitch) as expected by the function
    pitch_grid_2d = np.tile(pitch_grid, (n_energy, 1))
    
    delta_U = -50.0
    bs_over_bm = 0.5
    beam_width = 10.0
    beam_amp = 5.0
    beam_pitch_sigma = 10.0

    print("Running old implementation...")
    old_result = old_synth_losscone(
        energy_grid, 
        pitch_grid_2d, 
        delta_U, 
        bs_over_bm, 
        beam_width, 
        beam_amp, 
        beam_pitch_sigma
    )
    
    # Save result for comparison
    np.save("old_model_result.npy", old_result)
    print("Saved old_model_result.npy")

if __name__ == "__main__":
    test_vectorization()
