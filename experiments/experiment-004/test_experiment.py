"""Focused tests for the Experiment 004 historical reconstruction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).with_name("run.py")
    spec = importlib.util.spec_from_file_location("experiment_004_run", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


run = _load_module()


def test_rebin_pitch_uses_fixed_native_bins() -> None:
    pitches = np.tile(run.PITCH_CENTERS_DEG, (2, 1))
    flux = np.tile(np.arange(1.0, 17.0), (2, 1))
    result = run.rebin_pitch(flux, pitches, np.array([100.0, 200.0]))
    np.testing.assert_allclose(result.flux, flux)
    np.testing.assert_array_equal(result.support, np.ones_like(result.support))


def test_conjugate_normalization_maps_incident_half_to_one() -> None:
    incident = np.arange(1.0, 9.0)
    reflected = 0.5 * incident[::-1]
    normalized = run.normalize_flux(
        np.concatenate([incident, reflected])[None, :], "conjugate"
    )
    np.testing.assert_allclose(normalized[0, :8], 1.0)
    np.testing.assert_allclose(normalized[0, 8:], 0.5)


def test_hard_boundary_closes_at_low_energy_for_negative_potential() -> None:
    models, u_values, b_values = run.losscone_base_models(
        np.array([50.0, 500.0]), u_spacecraft_v=0.0, floor=0.05
    )
    match = np.flatnonzero((u_values == -100.0) & np.isclose(b_values, 1.0))
    assert match.size == 1
    model = models[int(match[0])]
    np.testing.assert_allclose(model[0], 1.0)
    assert np.any(model[1, 8:] == 0.05)


def test_beam_occupies_one_energy_and_two_pitch_bins() -> None:
    energies = np.array([300.0, 200.0, 100.0])
    beam, nearest = run.beam_profiles(energies, np.array([-180.0]), u_spacecraft_v=11.0)
    assert nearest[0] == 1
    assert np.count_nonzero(beam) == 2
    np.testing.assert_allclose(beam[0, 1, -2:], 1.0)
