"""Tests for the canonical Experiment 006 beam forward model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).with_name("beam_model.py")
    spec = importlib.util.spec_from_file_location("experiment_006_beam_model", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


model = _load_module()


def physical_bounds() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([0.01, 0.01, 90.0, 0.5]),
        np.array([2.0, 3.0, 180.0, 45.0]),
    )


def test_channel_edges_preserve_input_order_and_bracket_centers() -> None:
    energies = np.array([340.0, 113.0, 196.0, 65.0])
    low, high = model.channel_edges_ev(energies)
    assert np.all(low < energies)
    assert np.all(energies < high)
    ordered = np.argsort(energies)
    np.testing.assert_allclose(high[ordered][:-1], low[ordered][1:])


def test_integrated_response_converges_to_point_response_for_narrow_bin() -> None:
    center = 200.0
    low = np.array([center - 1e-4])
    high = np.array([center + 1e-4])
    response = model.integrated_energy_response(low, high, center, 0.48)
    np.testing.assert_allclose(response, 1.0, rtol=1e-7)


def test_continuum_midpoint_coefficients_map_to_physical_midpoints() -> None:
    energies = np.array([65.0, 113.0, 196.0, 340.0])
    lower, upper = physical_bounds()
    rows = model.continuum_row_parameters(
        energies,
        np.zeros(12),
        (20.0, 2000.0),
        lower,
        upper,
    )
    expected = np.broadcast_to(0.5 * (lower + upper), rows.shape)
    np.testing.assert_allclose(rows, expected)


def test_model_is_positive_and_beam_is_surface_leaving_localized() -> None:
    energies = np.array([65.0, 113.0, 196.0, 340.0, 588.0])
    low, high = model.channel_edges_ev(energies)
    pitches = np.array([100.0, 140.0, 170.0, 179.0])
    row_ids = np.repeat(np.arange(len(energies)), len(pitches))
    sample_pitches = np.tile(pitches, len(energies))
    lower, upper = physical_bounds()
    no_beam = np.concatenate(([0.0, 180.0, 25.0], np.zeros(12)))
    with_beam = no_beam.copy()
    with_beam[0] = 8.0
    baseline = model.model_values(
        energies,
        low,
        high,
        row_ids,
        sample_pitches,
        no_beam,
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=lower,
        physical_upper=upper,
    )
    represented = model.model_values(
        energies,
        low,
        high,
        row_ids,
        sample_pitches,
        with_beam,
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=lower,
        physical_upper=upper,
    )
    excess = (represented - baseline).reshape(len(energies), len(pitches))
    assert np.all(represented > 0.0)
    assert np.argmax(excess[:, -1]) in {1, 2}
    assert np.all(excess[:, -1] > excess[:, 0])


def test_log_residuals_are_zero_for_exact_model() -> None:
    values = np.array([0.2, 1.0, 4.0])
    np.testing.assert_allclose(model.log_residuals(values, values), 0.0)
