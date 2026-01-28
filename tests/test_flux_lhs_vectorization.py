"""Integration tests for flux fitting LHS vectorization."""

import numpy as np
import pytest

from src import config
from src.losscone.chi2 import compute_halekas_chi2, compute_halekas_chi2_batch
from src.model import synth_losscone, synth_losscone_batch


def test_lhs_chi2_batch_matches_sequential():
    """
    Test that vectorized LHS chi2 calculation matches sequential evaluation.

    This verifies the core optimization from commit c288412.
    """
    np.random.seed(42)

    # Setup realistic test data
    n_energy = config.SWEEP_ROWS  # 15
    n_pitch = 88
    n_samples = 50  # Typical LHS sample count

    # Generate energy and pitch grids
    energies = np.geomspace(10, 20000, n_energy)
    pitch_grid_1d = np.linspace(0, 180, n_pitch)
    pitches = np.tile(pitch_grid_1d, (n_energy, 1))

    # Generate synthetic "measured" data
    true_U_surface = -50.0
    true_bs_over_bm = 0.5
    true_beam_amp = 2.0
    beam_width = abs(true_U_surface) * 0.2
    beam_pitch_sigma = 10.0
    eps = 1e-6

    norm2d = synth_losscone(
        energies,
        pitches,
        true_U_surface,
        U_spacecraft=0.0,
        bs_over_bm=true_bs_over_bm,
        beam_width_eV=beam_width,
        beam_amp=true_beam_amp,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )
    # Add noise
    norm2d = norm2d * np.random.uniform(0.8, 1.2, norm2d.shape)
    norm2d = np.clip(norm2d, eps, None)

    # Generate LHS samples (simplified - just random)
    lhs_U_surface = np.random.uniform(-100, -10, n_samples)
    lhs_bs_over_bm = np.random.uniform(0.1, 0.9, n_samples)
    lhs_beam_amp = np.random.uniform(0, 5, n_samples)
    lhs_beam_width = np.maximum(np.abs(lhs_U_surface) * 0.2, config.EPS)

    # VECTORIZED PATH (batch implementation)
    models_batch = synth_losscone_batch(
        energies,
        pitches,
        lhs_U_surface,
        U_spacecraft=0.0,
        bs_over_bm=lhs_bs_over_bm,
        beam_width_eV=lhs_beam_width,
        beam_amp=lhs_beam_amp,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )

    chi2_batch = compute_halekas_chi2_batch(norm2d=norm2d, models=models_batch, eps=eps)

    # SEQUENTIAL PATH (original implementation for comparison)
    chi2_sequential = np.zeros(n_samples)
    for i in range(n_samples):
        model_i = synth_losscone(
            energies,
            pitches,
            lhs_U_surface[i],
            U_spacecraft=0.0,
            bs_over_bm=lhs_bs_over_bm[i],
            beam_width_eV=lhs_beam_width[i],
            beam_amp=lhs_beam_amp[i],
            beam_pitch_sigma_deg=beam_pitch_sigma,
        )

        chi2_sequential[i] = compute_halekas_chi2(norm2d=norm2d, model=model_i, eps=eps)

    # Should match exactly
    np.testing.assert_allclose(
        chi2_batch,
        chi2_sequential,
        rtol=1e-10,
        atol=1e-10,
        err_msg="Vectorized chi2 does not match sequential evaluation",
    )

    # Best fit should be the same
    assert np.argmin(chi2_batch) == np.argmin(chi2_sequential)


def test_lhs_vectorization_with_edge_cases():
    """Test LHS vectorization handles edge cases correctly."""

    np.random.seed(123)

    n_energy = 10
    n_pitch = 20
    n_samples = 10

    energies = np.geomspace(10, 1000, n_energy)
    pitches = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))
    norm2d = np.random.uniform(0.1, 10, (n_energy, n_pitch))
    eps = 1e-6

    # Edge case parameters
    lhs_U_surface = np.array(
        [
            -100.0,  # Normal
            -1.0,  # Very small (almost zero)
            -500.0,  # Very large
            0.0,  # Zero
            -50.0,  # Normal
            -200.0,  # Large
            -10.0,  # Small
            -75.0,  # Normal
            -150.0,  # Moderate
            -25.0,  # Small
        ]
    )
    lhs_bs_over_bm = np.array([0.5, 0.1, 0.9, 0.5, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    lhs_beam_amp = np.array([0.0, 5.0, 2.0, 1.0, 0.0, 3.0, 4.0, 0.5, 2.5, 1.5])
    lhs_beam_width = np.maximum(np.abs(lhs_U_surface) * 0.2, config.EPS)

    # Batch
    models_batch = synth_losscone_batch(
        energies,
        pitches,
        lhs_U_surface,
        U_spacecraft=0.0,
        bs_over_bm=lhs_bs_over_bm,
        beam_width_eV=lhs_beam_width,
        beam_amp=lhs_beam_amp,
        beam_pitch_sigma_deg=10.0,
    )

    chi2_batch = compute_halekas_chi2_batch(norm2d=norm2d, models=models_batch, eps=eps)

    # Sequential
    chi2_seq = []
    for i in range(n_samples):
        model = synth_losscone(
            energies,
            pitches,
            lhs_U_surface[i],
            U_spacecraft=0.0,
            bs_over_bm=lhs_bs_over_bm[i],
            beam_width_eV=lhs_beam_width[i],
            beam_amp=lhs_beam_amp[i],
            beam_pitch_sigma_deg=10.0,
        )
        chi2_seq.append(compute_halekas_chi2(norm2d=norm2d, model=model, eps=eps))

    chi2_seq = np.array(chi2_seq)

    np.testing.assert_allclose(chi2_batch, chi2_seq, rtol=1e-10)


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_lhs_vectorization_randomized(seed):
    """Randomized testing of LHS vectorization equivalence."""

    np.random.seed(seed)

    # Random dimensions
    n_energy = np.random.randint(10, 20)
    n_pitch = np.random.randint(50, 100)
    n_samples = np.random.randint(10, 50)

    energies = np.geomspace(10, 20000, n_energy)
    pitches = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    # Random data
    norm2d = np.random.uniform(0.1, 10, (n_energy, n_pitch))
    eps = 1e-6

    # Random LHS samples
    lhs_U_surface = np.random.uniform(-200, -1, n_samples)
    lhs_bs_over_bm = np.random.uniform(0.1, 0.9, n_samples)
    lhs_beam_amp = np.random.uniform(0, 5, n_samples)
    lhs_beam_width = np.maximum(np.abs(lhs_U_surface) * 0.2, config.EPS)
    beam_pitch_sigma = np.random.uniform(5, 20)

    # Batch
    models_batch = synth_losscone_batch(
        energies,
        pitches,
        lhs_U_surface,
        U_spacecraft=0.0,
        bs_over_bm=lhs_bs_over_bm,
        beam_width_eV=lhs_beam_width,
        beam_amp=lhs_beam_amp,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )

    chi2_batch = compute_halekas_chi2_batch(norm2d=norm2d, models=models_batch, eps=eps)

    # Sequential
    chi2_seq = []
    for i in range(n_samples):
        model = synth_losscone(
            energies,
            pitches,
            lhs_U_surface[i],
            U_spacecraft=0.0,
            bs_over_bm=lhs_bs_over_bm[i],
            beam_width_eV=lhs_beam_width[i],
            beam_amp=lhs_beam_amp[i],
            beam_pitch_sigma_deg=beam_pitch_sigma,
        )
        chi2_seq.append(compute_halekas_chi2(norm2d=norm2d, model=model, eps=eps))

    chi2_seq = np.array(chi2_seq)

    np.testing.assert_allclose(
        chi2_batch,
        chi2_seq,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Mismatch for seed {seed}",
    )


def test_beam_width_broadcasting():
    """
    Test that beam_width parameter broadcasts correctly in batch mode.

    This is a critical aspect of the LHS vectorization.
    """
    np.random.seed(42)

    n_energy = 10
    n_pitch = 20
    n_samples = 5

    energies = np.linspace(100, 1000, n_energy)
    pitches = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    U_surfaces = np.array([-100, -50, -25, -75, -150])
    bs_over_bms = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    beam_amps = np.array([2.0, 2.0, 2.0, 2.0, 2.0])
    beam_widths = np.abs(U_surfaces) * 0.2  # Different widths for each sample

    # Batch call
    models_batch = synth_losscone_batch(
        energies,
        pitches,
        U_surfaces,
        U_spacecraft=0.0,
        bs_over_bm=bs_over_bms,
        beam_width_eV=beam_widths,
        beam_amp=beam_amps,
        beam_pitch_sigma_deg=10.0,
    )

    # Individual calls
    for i in range(n_samples):
        model_individual = synth_losscone(
            energies,
            pitches,
            U_surfaces[i],
            U_spacecraft=0.0,
            bs_over_bm=bs_over_bms[i],
            beam_width_eV=beam_widths[i],
            beam_amp=beam_amps[i],
            beam_pitch_sigma_deg=10.0,
        )

        np.testing.assert_allclose(
            models_batch[i],
            model_individual,
            rtol=1e-10,
            err_msg=f"Beam width broadcasting failed for sample {i}",
        )
