
import numpy as np
from src.model import synth_losscone as new_synth_losscone

def test_vectorization_compare():
    # Load old result
    try:
        old_result = np.load("old_model_result.npy")
    except FileNotFoundError:
        print("Baseline file not found!")
        exit(1)

    # Generate same inputs
    np.random.seed(42)
    n_energy = 15
    n_pitch = 88
    energy_grid = np.geomspace(10, 20000, n_energy)
    pitch_grid = np.linspace(0, 180, n_pitch)
    pitch_grid_2d = np.tile(pitch_grid, (n_energy, 1))
    
    delta_U = -50.0
    bs_over_bm = 0.5
    beam_width = 10.0
    beam_amp = 5.0
    beam_pitch_sigma = 10.0

    print("Running new implementation...")
    new_result = new_synth_losscone(
        energy_grid, 
        pitch_grid_2d, 
        delta_U, 
        bs_over_bm, 
        beam_width, 
        beam_amp, 
        beam_pitch_sigma
    )
    
    # Compare
    if np.allclose(old_result, new_result, rtol=1e-10, atol=1e-10):
        print("SUCCESS: Results match!")
    else:
        print("FAILURE: Results do not match!")
        diff = np.abs(old_result - new_result)
        print(f"Max difference: {np.max(diff)}")
        exit(1)

if __name__ == "__main__":
    test_vectorization_compare()
