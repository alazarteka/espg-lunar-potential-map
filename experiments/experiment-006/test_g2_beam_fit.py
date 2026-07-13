"""Tests for the G2 DE beam-objective adapter."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load(filename: str, name: str):
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


beam_model = _load("beam_model.py", "beam_model")
_load("beam_torch.py", "beam_torch")
reference = _load("fit_reference.py", "fit_reference")
_load("start_parameters.py", "start_parameters")
_load("g2_de_torch.py", "g2_de_torch")
g2 = _load("g2_beam_fit.py", "experiment_006_g2_beam_fit")

TRUE = np.array([8.0, 220.0, 24.0, *([0.0] * 12)])


def _observation():
    energies = np.geomspace(35.0, 1000.0, 10)
    low, high = beam_model.channel_edges_ev(energies)
    pitch_grid = np.array([100.0, 125.0, 145.0, 160.0, 171.0, 178.0])
    rows = np.repeat(np.arange(energies.size), pitch_grid.size)
    pitches = np.tile(pitch_grid, energies.size)
    values = beam_model.model_values(
        energies,
        low,
        high,
        rows,
        pitches,
        TRUE,
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=np.array([0.01, 0.01, 90.0, 0.5]),
        physical_upper=np.array([2.0, 3.0, 180.0, 45.0]),
    )
    return reference.Observation(energies, low, high, rows, pitches, values)


def test_packed_torch_objective_matches_numpy() -> None:
    observation = _observation()
    packed = g2.pack_observations([observation], device="cpu")
    parameters = torch.as_tensor(TRUE).reshape(1, 1, 1, 15)
    actual = g2.beam_objective(packed)(parameters).item()
    residual = reference._residual(observation, TRUE)
    expected = float(np.dot(residual, residual))
    np.testing.assert_allclose(actual, expected, atol=1e-24, rtol=1e-12)


def test_small_de_run_recovers_matched_center_after_cpu_polish() -> None:
    config = g2.load_configuration()
    config["estimator"].update(
        islands=2,
        population_per_island=32,
        maximum_generations=80,
        migration_interval_generations=20,
        migration_elites_per_island=1,
    )
    result = g2.fit_observations(
        [_observation()],
        [17],
        device="cpu",
        configuration=config,
        cpu_max_nfev=500,
    )
    assert result.island_parameters.shape == (1, 2, 15)
    assert result.results[0] is not None
    assert abs(result.results[0].parameters[1] - TRUE[1]) < 1e-5
