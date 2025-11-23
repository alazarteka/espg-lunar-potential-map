"""Tests for flux fitting warm starting feature."""

import numpy as np
import pandas as pd
import pytest

from src import config
from src.flux import LossConeFitter
from src.model import synth_losscone


class MockERData:
    """Mock ER data for testing."""

    def __init__(self, n_rows=config.SWEEP_ROWS):
        energies = np.geomspace(10, 20000, n_rows)
        self.data = pd.DataFrame({
            config.ENERGY_COLUMN: energies,
            config.UTC_COLUMN: ["2000-01-01 00:00:00"] * n_rows,
        })


class MockPitchAngle:
    """Mock pitch angle calculator."""

    def __init__(self, n_rows=config.SWEEP_ROWS):
        n_pitch = 88
        pitch_grid_1d = np.linspace(0, 180, n_pitch)
        self.pitch_angles = np.tile(pitch_grid_1d, (n_rows, 1))


def test_warm_starting_accepts_good_solution(monkeypatch):
    """Test that a good previous solution is accepted for warm starting."""

    np.random.seed(42)

    # True parameters for synthetic data
    true_delta_U = -75.0
    true_bs_over_bm = 0.6
    true_beam_amp = 2.5

    # Create mock data
    er_data = MockERData()
    pitch_angle = MockPitchAngle()

    # Generate synthetic flux data
    energies = er_data.data[config.ENERGY_COLUMN].to_numpy()
    pitches = pitch_angle.pitch_angles

    true_model = synth_losscone(
        energies, pitches, true_delta_U, true_bs_over_bm,
        beam_width_eV=abs(true_delta_U) * 0.2,
        beam_amp=true_beam_amp,
        beam_pitch_sigma_deg=10.0
    )

    # Add noise to make it realistic
    noisy_model = true_model * np.random.uniform(0.9, 1.1, true_model.shape)

    # Mock _get_normalized_flux to return our synthetic data
    def mock_get_normalized_flux(self, energy_bin, measurement_chunk):
        return noisy_model[energy_bin, :]

    # Mock loadtxt to return dummy thetas
    def mock_loadtxt(fname, dtype=None):
        return np.zeros(config.SWEEP_ROWS)

    monkeypatch.setattr(LossConeFitter, "_get_normalized_flux", mock_get_normalized_flux)
    monkeypatch.setattr("numpy.loadtxt", mock_loadtxt)

    # Create fitter (pass dummy path and pre-constructed pitch_angle)
    fitter = LossConeFitter(er_data, "dummy.txt", pitch_angle=pitch_angle)

    # Fit without warm start
    result_no_warm = fitter._fit_surface_potential(0, previous_solution=None)

    # Fit with good warm start (close to true values)
    good_previous = np.array([true_delta_U + 5, true_bs_over_bm + 0.05, true_beam_amp + 0.2])
    result_with_warm = fitter._fit_surface_potential(0, previous_solution=good_previous)

    # Both should find reasonable solutions
    assert np.isfinite(result_no_warm[0])
    assert np.isfinite(result_with_warm[0])

    # Results should be similar (both converge to truth)
    assert abs(result_no_warm[0] - result_with_warm[0]) < 20  # Within 20V
    assert abs(result_no_warm[1] - result_with_warm[1]) < 0.2  # Within 0.2


def test_warm_starting_rejects_bad_solution(monkeypatch):
    """Test that a bad previous solution is rejected in favor of LHS."""

    np.random.seed(123)

    # True parameters
    true_delta_U = -50.0
    true_bs_over_bm = 0.5
    true_beam_amp = 2.0

    # Create mock data
    er_data = MockERData()
    pitch_angle = MockPitchAngle()

    energies = er_data.data[config.ENERGY_COLUMN].to_numpy()
    pitches = pitch_angle.pitch_angles

    true_model = synth_losscone(
        energies, pitches, true_delta_U, true_bs_over_bm,
        beam_width_eV=abs(true_delta_U) * 0.2,
        beam_amp=true_beam_amp,
        beam_pitch_sigma_deg=10.0
    )

    noisy_model = true_model * np.random.uniform(0.9, 1.1, true_model.shape)

    def mock_get_normalized_flux(self, energy_bin, measurement_chunk):
        return noisy_model[energy_bin, :]

    def mock_loadtxt(fname, dtype=None):
        return np.zeros(config.SWEEP_ROWS)

    monkeypatch.setattr(LossConeFitter, "_get_normalized_flux", mock_get_normalized_flux)
    monkeypatch.setattr("numpy.loadtxt", mock_loadtxt)

    fitter = LossConeFitter(er_data, "dummy.txt", pitch_angle=pitch_angle)

    # Fit with terrible warm start (far from truth)
    bad_previous = np.array([-500.0, 0.05, 0.1])  # Way off
    result_with_bad_warm = fitter._fit_surface_potential(0, previous_solution=bad_previous)

    # Should still find reasonable solution by falling back to LHS
    assert np.isfinite(result_with_bad_warm[0])
    # Should be closer to true value than the bad warm start
    assert abs(result_with_bad_warm[0] - true_delta_U) < abs(bad_previous[0] - true_delta_U)


def test_warm_starting_doesnt_change_final_result(monkeypatch):
    """
    Test that warm starting doesn't significantly change final result quality.

    Warm starting should improve convergence speed, not final accuracy.
    """

    np.random.seed(999)

    true_delta_U = -60.0
    true_bs_over_bm = 0.55
    true_beam_amp = 1.8

    er_data = MockERData()
    pitch_angle = MockPitchAngle()

    energies = er_data.data[config.ENERGY_COLUMN].to_numpy()
    pitches = pitch_angle.pitch_angles

    true_model = synth_losscone(
        energies, pitches, true_delta_U, true_bs_over_bm,
        beam_width_eV=abs(true_delta_U) * 0.2,
        beam_amp=true_beam_amp,
        beam_pitch_sigma_deg=10.0
    )

    noisy_model = true_model * np.random.uniform(0.95, 1.05, true_model.shape)

    def mock_get_normalized_flux(self, energy_bin, measurement_chunk):
        return noisy_model[energy_bin, :]

    def mock_loadtxt(fname, dtype=None):
        return np.zeros(config.SWEEP_ROWS)

    monkeypatch.setattr(LossConeFitter, "_get_normalized_flux", mock_get_normalized_flux)
    monkeypatch.setattr("numpy.loadtxt", mock_loadtxt)

    # Test that warm starting doesn't break correctness
    # Two fits: one without warm start, one with a good warm start
    fitter = LossConeFitter(er_data, "dummy.txt", pitch_angle=pitch_angle)

    result_no_warm = fitter._fit_surface_potential(0, previous_solution=None)

    # Use a reasonable warm start
    good_warm = np.array([true_delta_U + 10, true_bs_over_bm + 0.1, true_beam_amp + 0.5])
    result_with_warm = fitter._fit_surface_potential(0, previous_solution=good_warm)

    # Both should find reasonable solutions (within 30V of truth)
    assert abs(result_no_warm[0] - true_delta_U) < 30
    assert abs(result_with_warm[0] - true_delta_U) < 30

    # Warm starting shouldn't significantly degrade accuracy
    # (allow 2x worse as a conservative bound)
    error_no_warm = abs(result_no_warm[0] - true_delta_U)
    error_with_warm = abs(result_with_warm[0] - true_delta_U)
    assert error_with_warm < error_no_warm * 2.0 + 5.0  # +5V tolerance


def test_warm_starting_with_none_falls_back_to_lhs(monkeypatch):
    """Test that passing None for previous_solution works correctly."""

    np.random.seed(42)

    er_data = MockERData()
    pitch_angle = MockPitchAngle()

    energies = er_data.data[config.ENERGY_COLUMN].to_numpy()
    pitches = pitch_angle.pitch_angles

    # Simple synthetic data
    model = synth_losscone(energies, pitches, -50.0, 0.5, beam_width_eV=10.0, beam_amp=2.0)

    def mock_get_normalized_flux(self, energy_bin, measurement_chunk):
        return model[energy_bin, :]

    def mock_loadtxt(fname, dtype=None):
        return np.zeros(config.SWEEP_ROWS)

    monkeypatch.setattr(LossConeFitter, "_get_normalized_flux", mock_get_normalized_flux)
    monkeypatch.setattr("numpy.loadtxt", mock_loadtxt)

    fitter = LossConeFitter(er_data, "dummy.txt", pitch_angle=pitch_angle)

    # Should not crash and should return valid result
    result = fitter._fit_surface_potential(0, previous_solution=None)

    assert len(result) == 4  # (delta_U, bs_over_bm, beam_amp, chi2)
    assert np.isfinite(result[0])
    assert np.isfinite(result[1])
    assert np.isfinite(result[2])
    assert np.isfinite(result[3])


def test_warm_starting_handles_invalid_previous_solution(monkeypatch):
    """Test that warm starting handles invalid (NaN/Inf) previous solutions gracefully."""

    np.random.seed(42)

    er_data = MockERData()
    pitch_angle = MockPitchAngle()

    energies = er_data.data[config.ENERGY_COLUMN].to_numpy()
    pitches = pitch_angle.pitch_angles

    model = synth_losscone(energies, pitches, -50.0, 0.5, beam_width_eV=10.0, beam_amp=2.0)

    def mock_get_normalized_flux(self, energy_bin, measurement_chunk):
        return model[energy_bin, :]

    def mock_loadtxt(fname, dtype=None):
        return np.zeros(config.SWEEP_ROWS)

    monkeypatch.setattr(LossConeFitter, "_get_normalized_flux", mock_get_normalized_flux)
    monkeypatch.setattr("numpy.loadtxt", mock_loadtxt)

    fitter = LossConeFitter(er_data, "dummy.txt", pitch_angle=pitch_angle)

    # Test with NaN
    invalid_previous = np.array([np.nan, 0.5, 2.0])
    result_nan = fitter._fit_surface_potential(0, previous_solution=invalid_previous)

    # Should still work (fall back to LHS)
    assert np.isfinite(result_nan[0])

    # Test with Inf
    invalid_previous = np.array([-np.inf, 0.5, 2.0])
    result_inf = fitter._fit_surface_potential(0, previous_solution=invalid_previous)

    # Should still work
    assert np.isfinite(result_inf[0])
