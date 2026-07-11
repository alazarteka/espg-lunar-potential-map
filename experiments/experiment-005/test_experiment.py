"""Tests for the Experiment 005 response-aware beam estimator."""

from __future__ import annotations

import importlib.util
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np


def _load_beam_module():
    path = Path(__file__).with_name("beam.py")
    spec = importlib.util.spec_from_file_location("experiment_005_beam", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


beam = _load_beam_module()
sys.modules["beam"] = beam


def _load_calibrate_module():
    path = Path(__file__).with_name("calibrate.py")
    spec = importlib.util.spec_from_file_location("experiment_005_calibrate", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


calibrate = _load_calibrate_module()


def synthetic_observation(
    *,
    center_ev: float = 150.0,
    amplitude: float = 8.0,
    pitch_sigma_deg: float = 25.0,
    noise_sigma: float = 0.0,
    seed: int = 7,
):
    energies = np.array([30.0, 50.0, 80.0, 130.0, 210.0, 340.0, 550.0, 900.0])
    pitches = np.linspace(92.0, 178.0, 32)
    row_ids = np.repeat(np.arange(len(energies)), len(pitches))
    sample_energies = np.repeat(energies, len(pitches))
    sample_pitches = np.tile(pitches, len(energies))
    shape = beam.beam_shape(
        sample_energies,
        sample_pitches,
        center_ev,
        pitch_sigma_deg,
        0.48,
    )
    values = 0.8 + amplitude * shape
    if noise_sigma:
        rng = np.random.default_rng(seed)
        values *= np.exp(rng.normal(0.0, noise_sigma, size=len(values)))
    return beam.BeamObservation(
        normalization="incident_hemisphere_mean",
        energies_ev=energies,
        energy_indices=np.arange(len(energies)),
        row_ids=row_ids,
        sample_energies_ev=sample_energies,
        pitches_deg=sample_pitches,
        values=values,
        incident_counts=np.full(len(energies), 40),
        reflected_counts=np.full(len(energies), len(pitches)),
    )


def fast_configuration():
    configuration = deepcopy(beam.load_configuration())
    configuration["beam"]["multistart_count"] = 12
    configuration["beam"]["bootstrap_replicates"] = 30
    return configuration


def test_beam_energy_response_has_configured_fwhm() -> None:
    center = 200.0
    values = beam.beam_shape(
        np.array([center, center + 0.24 * center]),
        np.array([180.0, 180.0]),
        center,
        20.0,
        0.48,
    )
    assert np.isclose(values[0], 1.0)
    assert np.isclose(values[1], 0.5, rtol=2e-3)


def test_fit_recovers_synthetic_beam() -> None:
    configuration = fast_configuration()
    observation = synthetic_observation()
    fitted = beam.fit_beam(observation, configuration["beam"])
    assert fitted.success
    assert abs(fitted.center_ev - 150.0) < 1.0
    assert abs(fitted.amplitude - 8.0) < 0.1
    assert abs(fitted.pitch_sigma_deg - 25.0) < 0.2


def test_bootstrap_is_deterministic_and_contains_truth() -> None:
    configuration = fast_configuration()
    observation = synthetic_observation(noise_sigma=0.08)
    fitted = beam.fit_beam(observation, configuration["beam"])
    first = beam.bootstrap_beam(
        observation,
        fitted,
        configuration["beam"],
        np.random.default_rng(42),
    )
    second = beam.bootstrap_beam(
        observation,
        fitted,
        configuration["beam"],
        np.random.default_rng(42),
    )
    assert first == second
    assert first.successes == first.requested
    assert first.center_q025_ev < 150.0 < first.center_q975_ev


def test_assessment_accepts_resolved_beam() -> None:
    configuration = fast_configuration()
    observation = synthetic_observation(amplitude=12.0, noise_sigma=0.04)
    assessment = beam.assess_beam(
        observation,
        observation,
        configuration["beam"],
        configuration["beam_qc"],
        np.random.default_rng(44),
    )
    assert assessment.beam_valid, assessment.reason_codes
    assert assessment.positive_excess_rows >= 2


def test_assessment_rejects_beam_absent_control() -> None:
    configuration = fast_configuration()
    observation = synthetic_observation(amplitude=0.0, noise_sigma=0.02)
    assessment = beam.assess_beam(
        observation,
        observation,
        configuration["beam"],
        configuration["beam_qc"],
        np.random.default_rng(45),
    )
    assert not assessment.beam_valid
    assert "amplitude_not_resolved" in assessment.reason_codes


def test_relative_boundary_closes_below_voltage_scale() -> None:
    boundaries = calibrate.relative_boundary_pitch(
        np.array([50.0, 150.0, 1000.0]), -150.0, 0.975
    )
    assert np.allclose(boundaries[:2], 180.0)
    assert boundaries[2] < 120.0
    assert np.all(np.diff(boundaries) <= 0.0)


def test_simulated_beam_is_additive_on_same_physical_background() -> None:
    support = synthetic_observation(amplitude=0.0)
    arguments = {
        "center_ev": 150.0,
        "pitch_sigma_deg": 25.0,
        "log_noise_sigma": 0.0,
        "field_ratio": 0.975,
        "cone_floor": 0.2,
        "boundary_width_deg": 8.0,
    }
    absent = calibrate.simulate_observation(
        support,
        amplitude=0.0,
        rng=np.random.default_rng(2),
        **arguments,
    )
    present = calibrate.simulate_observation(
        support,
        amplitude=4.0,
        rng=np.random.default_rng(2),
        **arguments,
    )
    expected = 4.0 * beam.beam_shape(
        support.sample_energies_ev,
        support.pitches_deg,
        150.0,
        25.0,
        0.48,
    )
    assert np.allclose(present.values - absent.values, expected)


def test_json_safe_replaces_nonfinite_values() -> None:
    assert calibrate.json_safe({"a": [1.0, float("nan")]}) == {"a": [1.0, None]}
