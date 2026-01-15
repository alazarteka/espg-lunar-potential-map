"""
Test validity masking for E < U_spacecraft cases.
"""

import numpy as np

from src.model import synth_losscone, synth_losscone_batch


def test_mask_returns_invalid_energies():
    """Test that return_mask=True identifies E < U_spacecraft."""
    n_energy = 10
    n_pitch = 20

    # Energy grid from 5 to 100 eV
    energy_grid = np.linspace(5.0, 100.0, n_energy)
    pitch_grid = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    # Set U_spacecraft = 50V, so first half of energies are invalid
    U_spacecraft = 50.0
    U_surface = -10.0

    model, mask = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        U_spacecraft=U_spacecraft,
        return_mask=True,
    )

    # Check mask shape
    assert mask.shape == (n_energy, n_pitch)

    # Check that energies < U_spacecraft are masked out
    for i, E in enumerate(energy_grid):
        if U_spacecraft > E:
            assert not mask[i].any(), f"Energy {E} < {U_spacecraft} should be invalid"
        else:
            assert mask[i].all(), f"Energy {E} >= {U_spacecraft} should be valid"


def test_batch_mask_returns_per_parameter():
    """Test that batch masking works with multiple U_spacecraft values."""
    n_energy = 10
    n_pitch = 20
    n_params = 5

    energy_grid = np.linspace(10.0, 100.0, n_energy)
    pitch_grid = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    # Different U_spacecraft for each parameter set
    # Use scalar U_spacecraft but vary U_surface
    U_surfaces = np.linspace(-50, -10, n_params)
    U_spacecraft = 0.0  # All energies should be valid

    models, masks = synth_losscone_batch(
        energy_grid,
        pitch_grid,
        U_surfaces,
        U_spacecraft=U_spacecraft,
        return_mask=True,
    )

    # Check shapes
    assert models.shape == (n_params, n_energy, n_pitch)
    assert masks.shape == (n_params, n_energy, n_pitch)

    # With U_spacecraft=0 and all positive energies, all should be valid
    assert masks.all(), "All energies should be valid when E > U_spacecraft=0"


def test_mask_combines_with_data_mask_in_chi2():
    """Test that model validity mask combines with data mask in chi2 computation."""
    # This is an integration test concept - the actual combination happens
    # in flux.py and model_torch.py chi2 functions
    n_energy = 10
    n_pitch = 20

    energy_grid = np.linspace(5.0, 100.0, n_energy)
    pitch_grid = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    U_spacecraft = 50.0  # Half the energies are invalid
    U_surface = -10.0

    model, model_mask = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        U_spacecraft=U_spacecraft,
        return_mask=True,
    )

    # Simulate data mask (e.g., some data points are bad)
    data_mask = np.ones((n_energy, n_pitch), dtype=bool)
    data_mask[5:7, :] = False  # Mark some data as invalid

    # Combined mask should exclude both invalid energies AND bad data
    combined_mask = data_mask & model_mask

    # Check that low energies are excluded
    for i, E in enumerate(energy_grid):
        if U_spacecraft > E:
            assert not combined_mask[i].any()

    # Check that bad data rows are excluded
    assert not combined_mask[5:7].any()

    # Check that valid regions remain
    valid_energies = energy_grid >= U_spacecraft
    valid_rows = np.where(valid_energies)[0]
    # Exclude the bad data rows
    valid_rows = valid_rows[~np.isin(valid_rows, [5, 6])]
    for i in valid_rows:
        assert combined_mask[i].all()


def test_backward_compatibility_no_mask():
    """Test that return_mask=False maintains backward compatibility."""
    n_energy = 10
    n_pitch = 20

    energy_grid = np.linspace(10.0, 100.0, n_energy)
    pitch_grid = np.tile(np.linspace(0, 180, n_pitch), (n_energy, 1))

    U_surface = -10.0

    # Default behavior (no mask)
    model_no_mask = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        return_mask=False,
    )

    # Explicit return_mask=True
    model_with_mask, mask = synth_losscone(
        energy_grid,
        pitch_grid,
        U_surface,
        return_mask=True,
    )

    # Models should be identical
    np.testing.assert_allclose(model_no_mask, model_with_mask)

    # Check that result is just array, not tuple
    assert isinstance(model_no_mask, np.ndarray)
    assert not isinstance(model_no_mask, tuple)
