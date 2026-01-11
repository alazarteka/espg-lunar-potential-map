import math

import numpy as np
import pytest

from src import config
from src.model import synth_losscone, synth_losscone_batch


def reference_synth_losscone(
    energy_grid,
    pitch_grid,
    U_surface,
    bs_over_bm,
    beam_width_eV=0,
    beam_amp=0,
    beam_pitch_sigma_deg=0,
    background=config.LOSS_CONE_BACKGROUND,
):
    """
    Original loop-based implementation of synth_losscone for equivalence testing.
    """
    nE, nPitch = pitch_grid.shape
    model = np.full((nE, nPitch), background, dtype=float)

    for i, E in enumerate(energy_grid):
        # Guard against E <= 0
        if E <= 0:
            continue

        x = bs_over_bm * (1.0 + U_surface / E)  # dimensionless

        # Map illegal values onto physically plausible limits
        if x <= 0.0:
            ac = 0.0  # full loss cone (no mirroring)
        elif x >= 1.0:
            ac = 90.0  # mirror point at 90°, loss cone closed
        else:
            ac = math.degrees(math.asin(math.sqrt(x)))

        mask = pitch_grid[i] <= 180 - ac
        model[i, mask] = 1.0

    # Optional narrow beam
    if beam_width_eV > 0 and beam_amp > 0:
        # Reference uses U_spacecraft = 0.0, so delta_u = -U_surface.
        delta_u = -U_surface
        if delta_u > 0:
            beam_center = max(delta_u, beam_width_eV)
            beam = beam_amp * np.exp(
                -0.5 * ((energy_grid - beam_center) / beam_width_eV) ** 2
            )
            if beam_pitch_sigma_deg > 0:
                pitch_weight = np.exp(
                    -0.5 * ((pitch_grid - 180.0) / beam_pitch_sigma_deg) ** 2
                )
            else:
                pitch_weight = np.ones_like(pitch_grid)

            # Original implementation broadcasting logic
            # beam is (nE,), pitch_weight is (nE, nPitch)
            model += beam[:, None] * pitch_weight

    return model


@pytest.mark.parametrize("seed", [42, 123, 999])
def test_single_vs_reference_randomized(seed):
    """
    Compare single-input synth_losscone against reference implementation
    with randomized inputs.
    """
    np.random.seed(seed)

    # Generate random dimensions
    n_energy = np.random.randint(10, 50)
    n_pitch = np.random.randint(20, 100)

    # Generate random inputs
    energy_grid = np.geomspace(1.0, 20000.0, n_energy)
    # Add some negative/zero energies to test masking
    if np.random.rand() < 0.5:
        energy_grid[0] = -5.0
        energy_grid[1] = 0.0

    pitch_grid_1d = np.linspace(0, 180, n_pitch)
    pitch_grid = np.tile(pitch_grid_1d, (n_energy, 1))

    U_surface = np.random.uniform(-100, 0)
    bs_over_bm = np.random.uniform(0.1, 0.9)

    # Randomize beam parameters
    beam_width = np.random.uniform(0, 20)
    beam_amp = np.random.uniform(0, 10)
    beam_pitch_sigma = np.random.uniform(0, 20)

    # Run both implementations
    ref_result = reference_synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        bs_over_bm,
        beam_width,
        beam_amp,
        beam_pitch_sigma,
    )

    vec_result = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        U_spacecraft=0.0,
        bs_over_bm=bs_over_bm,
        beam_width_eV=beam_width,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )

    # Assert equivalence
    # We use a small tolerance for floating point differences
    np.testing.assert_allclose(
        vec_result,
        ref_result,
        rtol=1e-10,
        atol=1e-10,
        err_msg=f"Mismatch for seed {seed}",
    )


def test_single_vs_reference_edge_cases():
    """Test single-input synth_losscone against reference for edge cases."""
    n_energy = 5
    n_pitch = 10
    energy_grid = np.array([100.0] * n_energy)
    pitch_grid = np.zeros((n_energy, n_pitch))

    # Case 1: U_surface makes x > 1 (closed loss cone)
    # x = bs/bm * (1 + dU/E)
    # Try bs/bm=1.0, dU=0 -> x=1 -> ac=90 -> mask: pitch <= 90.
    # If pitch is 0, model should be 1.

    ref = reference_synth_losscone(energy_grid, pitch_grid, 0.0, 1.0)
    vec = synth_losscone(energy_grid, pitch_grid, 0.0, U_spacecraft=0.0, bs_over_bm=1.0)
    np.testing.assert_allclose(ref, vec)

    # Case 2: U_surface makes x < 0 (full loss cone)
    # x < 0 -> ac=0 -> mask: pitch <= 180.
    # All 1s.
    vec = synth_losscone(
        energy_grid, pitch_grid, -200.0, U_spacecraft=0.0, bs_over_bm=1.0
    )
    np.testing.assert_allclose(ref, vec)


def test_beam_suppressed_when_surface_above_spacecraft():
    """Beam should be suppressed when U_spacecraft <= U_surface."""
    n_energy = 20
    n_pitch = 30
    energy_grid = np.geomspace(10.0, 1000.0, n_energy)
    pitch_grid = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    U_surface = 10.0
    bs_over_bm = 0.5
    beam_width = 10.0
    beam_amp = 5.0
    beam_pitch_sigma = 10.0

    with_beam = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        U_spacecraft=0.0,
        bs_over_bm=bs_over_bm,
        beam_width_eV=beam_width,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )
    no_beam = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        U_spacecraft=0.0,
        bs_over_bm=bs_over_bm,
        beam_width_eV=beam_width,
        beam_amp=0.0,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )

    np.testing.assert_allclose(with_beam, no_beam, rtol=1e-12, atol=1e-12)


def test_batch_vs_single():
    """
    Test that synth_losscone_batch produces the same results as
    looped synth_losscone calls.
    """
    np.random.seed(42)
    n_energy = 15
    n_pitch = 20
    n_batch = 10

    energy_grid = np.geomspace(10, 20000, n_energy)
    pitch_grid = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    U_surfaces = np.random.uniform(-100, 0, n_batch)
    bs_over_bms = np.random.uniform(0.1, 0.9, n_batch)
    beam_amps = np.random.uniform(0, 10, n_batch)
    beam_widths = np.full(n_batch, 10.0)
    beam_pitch_sigma = 10.0

    # Run batch version
    batch_result = synth_losscone_batch(
        energy_grid,
        pitch_grid,
        U_surfaces,
        U_spacecraft=0.0,
        bs_over_bm=bs_over_bms,
        beam_width_eV=beam_widths,
        beam_amp=beam_amps,
        beam_pitch_sigma_deg=beam_pitch_sigma,
    )

    # Compare against looped single calls
    for i in range(n_batch):
        single_result = synth_losscone(
            energy_grid,
            pitch_grid,
            U_surfaces[i],
            U_spacecraft=0.0,
            bs_over_bm=bs_over_bms[i],
            beam_width_eV=beam_widths[i],
            beam_amp=beam_amps[i],
            beam_pitch_sigma_deg=beam_pitch_sigma,
        )

        np.testing.assert_allclose(
            batch_result[i],
            single_result,
            rtol=1e-10,
            atol=1e-10,
            err_msg=f"Batch mismatch at index {i}",
        )
