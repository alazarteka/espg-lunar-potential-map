"""Tests for the independent Experiment 006 sector-level simulator."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("beam_model.py", "beam_model")
support_data = _load("support_data.py", "support_data")
simulator = _load("simulator.py", "experiment_006_simulator")


def _support():
    energies = np.geomspace(30.0, 1500.0, 15)
    low = energies / 1.15
    high = energies * 1.15
    pitches = np.tile(np.linspace(5.0, 175.0, 32), (15, 1))
    return support_data.GeometrySupport(1, energies, low, high, pitches, np.ones(15))


def _row() -> dict:
    row = {
        "true_center_ev": 220.0,
        "amplitude": 8.0,
        "pitch_sigma_deg": 24.0,
        "background_family": "asymmetric_shoulder",
        "response_family": "nominal_linear_gaussian",
        "response_direction": 0,
        "temporal_family": "ramp",
        "log_noise_sigma": 0.2,
    }
    config = json.loads(Path(__file__).with_name("configuration-c1.json").read_text())
    row.update(
        {
            f"seed_{name}": index + 100
            for index, name in enumerate(config["seeds"]["namespaces"])
        }
    )
    return row


def test_simulation_is_positive_deterministic_and_branches_after_noise() -> None:
    config = json.loads(Path(__file__).with_name("configuration-c1.json").read_text())
    first = simulator.simulate_case(_support(), _row(), config)
    second = simulator.simulate_case(_support(), _row(), config)
    np.testing.assert_array_equal(first.native_flux, second.native_flux)
    np.testing.assert_array_equal(first.incident_mean, second.incident_mean)
    assert np.all(first.native_flux > 0.0)
    finite = np.isfinite(first.incident_mean) & np.isfinite(first.conjugate)
    assert np.any(finite)
    assert not np.allclose(first.incident_mean[finite], first.conjugate[finite])


def test_normalizations_match_hand_calculation_and_use_reflected_only() -> None:
    pitches = np.tile(
        np.array([10, 20, 30, 40, 50, 60, 120, 130, 140, 150, 160, 170]), (1, 1)
    )
    flux = np.arange(1.0, 13.0)[None, :]
    mean, conjugate = simulator.normalize_native(
        flux, pitches, np.ones_like(flux, dtype=bool)
    )
    assert np.isnan(mean[0, :6]).all()
    np.testing.assert_allclose(mean[0, 6:], flux[0, 6:] / 3.5)
    np.testing.assert_allclose(conjugate[0, 6:], flux[0, 6:] / flux[0, 5::-1])


def test_simulator_does_not_import_estimator_model() -> None:
    source = Path(simulator.__file__).read_text(encoding="ascii")
    assert "beam_model" not in source
    assert "model_values" not in source
