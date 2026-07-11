"""Focused tests for Experiment 004 Phase A3."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).with_name("phase_a3.py")
    spec = importlib.util.spec_from_file_location("experiment_004_phase_a3", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a3 = _load_module()


def test_broad_beam_pitch_response_has_expected_half_maximum() -> None:
    center = 180.0
    half_offset = 0.5 * phase_a3.BEAM_BROAD_FWHM_DEG
    response = np.exp(
        -0.5
        * (np.array([center - half_offset, center]) - center) ** 2
        / phase_a3.BEAM_BROAD_SIGMA_DEG**2
    )
    np.testing.assert_allclose(response, np.array([0.5, 1.0]), atol=1e-3)


def test_tau_boundary_is_fractional_for_single_supported_sector() -> None:
    spectrum = phase_a3.phase_a2.SectorSpectrum(
        energies_ev=np.array([100.0]),
        flux=np.ones((1, 1)),
        pitches_deg=np.array([[130.0]]),
    )
    models, u_values, b_values = phase_a3.integrated_boundary_models(
        spectrum,
        support_values=spectrum.flux,
        u_spacecraft_v=0.0,
        floor=0.10,
        tau_deg=5.0,
    )
    match = np.flatnonzero((u_values == -40.0) & np.isclose(b_values, 1.0))
    assert match.size == 1
    finite = models[int(match[0])][np.isfinite(models[int(match[0])])]
    assert np.any((finite > 0.10) & (finite < 1.0))


def test_2d_beam_fit_recovers_synthetic_center_and_width() -> None:
    energies = np.array([600.0, 340.0, 196.0, 113.0, 65.0, 38.0, 22.0])
    pitches = phase_a3.phase_a.PITCH_CENTERS_DEG
    center = 160.0
    pitch_sigma = 18.0
    energy_sigma = 0.48 * center / 2.355
    energy_response = np.exp(-0.5 * ((energies - center) / energy_sigma) ** 2)
    pitch_response = np.exp(-0.5 * ((pitches - 180.0) / pitch_sigma) ** 2)
    observed = np.ones((len(energies), len(pitches)))
    observed += 6.0 * energy_response[:, None] * pitch_response[None, :]
    fitted_center, _, fitted_pitch_sigma, _ = phase_a3.fit_beam_2d(observed, energies)
    assert abs(fitted_center - center) < 1.0
    assert abs(fitted_pitch_sigma - pitch_sigma) < 1.0
