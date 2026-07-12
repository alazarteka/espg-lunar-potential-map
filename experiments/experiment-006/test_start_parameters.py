"""Tests for Experiment 006's deterministic observation-only starts."""

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


model = _load("beam_model.py", "beam_model")
reference = _load("fit_reference.py", "fit_reference")
starts_module = _load("start_parameters.py", "experiment_006_start_parameters")


def _observation(center: float = 350.0, amplitude: float = 8.0):
    energies = np.geomspace(40.0, 1000.0, 12)
    low, high = model.channel_edges_ev(energies)
    pitch_grid = np.array([100.0, 120.0, 135.0, 145.0, 158.0, 168.0, 176.0, 180.0])
    rows = np.repeat(np.arange(energies.size), pitch_grid.size)
    pitches = np.tile(pitch_grid, energies.size)
    parameters = np.array([amplitude, center, 18.0, *([0.0] * 12)])
    values = model.model_values(
        energies,
        low,
        high,
        rows,
        pitches,
        parameters,
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=np.array([0.01, 0.01, 90.0, 0.5]),
        physical_upper=np.array([2.0, 3.0, 180.0, 45.0]),
    )
    return reference.Observation(energies, low, high, rows, pitches, values)


def _starts(observation):
    return starts_module.construct_starts(observation, *reference.parameter_bounds())


def test_deterministic_exactly_twelve_and_in_bounds() -> None:
    observation = _observation()
    first = _starts(observation)
    second = _starts(observation)
    lower, upper = reference.parameter_bounds()
    assert first.shape == (12, 15)
    assert np.isfinite(first).all()
    assert np.all((first >= lower) & (first <= upper))
    np.testing.assert_array_equal(first, second)


def test_invariant_to_row_and_sample_permutation() -> None:
    observation = _observation()
    row_order = np.array([8, 2, 10, 0, 6, 4, 11, 1, 9, 3, 7, 5])
    inverse = np.empty_like(row_order)
    inverse[row_order] = np.arange(row_order.size)
    sample_order = np.arange(observation.observed.size)[::-1]
    permuted = reference.Observation(
        observation.energies_ev[row_order],
        observation.channel_low_ev[row_order],
        observation.channel_high_ev[row_order],
        inverse[observation.row_ids[sample_order]],
        observation.pitches_deg[sample_order],
        observation.observed[sample_order],
    )
    np.testing.assert_allclose(_starts(observation), _starts(permuted), atol=1e-12)


def test_null_safe_and_beam_center_coverage() -> None:
    observation = _observation(center=150.0, amplitude=0.0)
    null = reference.Observation(
        observation.energies_ev,
        observation.channel_low_ev,
        observation.channel_high_ev,
        np.array([], dtype=np.int64),
        np.array([]),
        np.array([]),
    )
    assert np.isfinite(_starts(null)).all()

    for center in (150.0, 350.0):
        centers = _starts(_observation(center=center))[:, 1]
        assert np.min(np.abs(centers - center)) < 0.3 * center


def test_start_grid_matches_frozen_configuration() -> None:
    configuration = json.loads(
        Path(__file__)
        .with_name("configuration-d1-starts.json")
        .read_text(encoding="ascii")
    )
    assert configuration["start_count"] == starts_module.N_STARTS
    assert configuration["grid"]["center_factors"] == list(starts_module.CENTER_FACTORS)
