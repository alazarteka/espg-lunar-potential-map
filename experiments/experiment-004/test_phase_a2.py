"""Focused tests for Experiment 004 Phase A2."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).with_name("phase_a2.py")
    spec = importlib.util.spec_from_file_location("experiment_004_phase_a2", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a2 = _load_module()


def test_sector_ratios_pair_reflected_with_conjugate_incident() -> None:
    pitches = np.array([[10.0, 40.0, 140.0, 170.0]])
    flux = np.array([[2.0, 4.0, 2.0, 1.0]])
    ratios = phase_a2.sector_ratios(flux, pitches)
    np.testing.assert_allclose(ratios[0, :2], 1.0)
    np.testing.assert_allclose(ratios[0, 2:], 0.5)


def test_support_integration_produces_fractional_boundary_cell() -> None:
    spectrum = phase_a2.SectorSpectrum(
        energies_ev=np.array([100.0]),
        flux=np.ones((1, 2)),
        pitches_deg=np.array([[126.0, 132.0]]),
    )
    models, u_values, b_values = phase_a2.support_integrated_base_models(
        spectrum, u_spacecraft_v=0.0, floor=0.10
    )
    match = np.flatnonzero((u_values == -40.0) & np.isclose(b_values, 1.0))
    assert match.size == 1
    represented = models[int(match[0])]
    finite = represented[np.isfinite(represented)]
    assert np.any((finite > 0.10) & (finite < 1.0))


def test_fractional_beam_response_has_expected_half_maximum() -> None:
    center = 200.0
    half_offset = 0.5 * 0.48 * center
    energies = np.array([center - half_offset, center, center + half_offset])
    beam = phase_a2.beam_profiles(
        energies,
        u_surface_v=np.array([-189.0]),
        u_spacecraft_v=11.0,
        mode="fractional_gaussian",
    )
    np.testing.assert_allclose(beam[0, 1, -1], 1.0)
    np.testing.assert_allclose(beam[0, [0, 2], -1], 0.5, atol=1e-3)


def test_beam_centroid_recovers_response_limited_synthetic_peak() -> None:
    energies = np.array([600.0, 340.0, 196.0, 113.0, 65.0, 38.0, 22.0])
    center = 160.0
    sigma = 0.48 * center / 2.355
    response = np.exp(-0.5 * ((energies - center) / sigma) ** 2)
    observed = np.ones((len(energies), 16))
    observed[:, -2:] += 5.0 * response[:, None]
    fitted, _, _, _ = phase_a2.fit_beam_centroid(observed, energies)
    assert abs(fitted - center) < 1.0
