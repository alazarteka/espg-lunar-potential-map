"""Smoke tests for src/diagnostics/beam_detection.py.

These are not physics-accuracy tests. They just confirm the public
`detect_peak` entry point runs on realistic-shaped synthetic inputs
(built from the shared synthetic helpers) and returns a well-formed
`BeamDetectionResult`.
"""

from __future__ import annotations

import numpy as np

from src import config
from src.diagnostics.beam_detection import (
    BeamDetectionResult,
    _build_energy_profile,
    detect_peak,
)
from src.utils.synthetic import prepare_flux, prepare_phis, prepare_synthetic_er


def _energy_profile_and_grids() -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """Build a norm2d/pitches grid with an injected beam, then reduce it to the
    energy profile via the module's own `_build_energy_profile` helper (the
    same pipeline used by `scripts/diagnostics/losscone_peak_scan.py`)."""
    _flux, energy_centers = prepare_flux()
    energies = energy_centers.to(config.ureg.electron_volt).magnitude  # type: ignore[attr-defined]
    phis, _solid_angles = prepare_phis()
    n_pitch = max(len(phis), 40)

    pitches_1d = np.linspace(0.0, 180.0, n_pitch)
    pitches = np.broadcast_to(pitches_1d[None, :], (len(energies), n_pitch)).copy()

    # Background level everywhere, with a clear contiguous peak in the
    # backscatter pitch band (>150 deg) at one energy, bracketed by weaker
    # (but still present) signal in the neighboring energy bins.
    norm2d = np.full((len(energies), n_pitch), 0.3, dtype=np.float64)
    # Keep the peak within _build_energy_profile's default [20, 500] eV
    # energy window (energies are geomspaced 20 eV to 20 keV).
    peak_energy_idx = 3
    band_mask = pitches_1d >= 150.0
    norm2d[peak_energy_idx, band_mask] = 5.0
    if peak_energy_idx > 0:
        norm2d[peak_energy_idx - 1, band_mask] = 2.0
    if peak_energy_idx + 1 < len(energies):
        norm2d[peak_energy_idx + 1, band_mask] = 2.0

    energies_sorted, profile = _build_energy_profile(energies, pitches, norm2d)
    return profile, energies_sorted, norm2d, pitches


def test_prepare_synthetic_er_smoke() -> None:
    """Sanity-check the shared synthetic ER helper produces a usable dataset."""
    er = prepare_synthetic_er()
    assert not er.data.empty
    assert len(er.data) == config.SWEEP_ROWS


def test_detect_peak_finds_injected_beam() -> None:
    """detect_peak runs without error and flags the injected energy peak."""
    profile, energies, norm2d, pitches = _energy_profile_and_grids()

    result = detect_peak(
        profile,
        energies=energies,
        norm2d=norm2d,
        pitches=pitches,
    )

    assert isinstance(result, BeamDetectionResult)
    assert isinstance(result.has_peak, bool)
    assert result.has_peak is True
    assert result.peak_idx is not None
    assert 0 <= result.peak_idx < profile.size
    assert result.peak_value is not None
    assert np.isfinite(result.peak_value)
    assert result.beam_energy is not None
    assert np.isfinite(result.beam_energy)
    assert result.beam_energy == energies[result.peak_idx]


def test_detect_peak_no_peak_on_flat_profile() -> None:
    """A flat (no-contrast) profile should not report a beam."""
    profile = np.full(config.SWEEP_ROWS, 1.0, dtype=np.float64)

    result = detect_peak(profile)

    assert isinstance(result, BeamDetectionResult)
    assert result.has_peak is False
    assert result.peak_idx is None
    assert result.peak_value is None
    assert result.beam_energy is None


def test_detect_peak_empty_profile_returns_no_peak() -> None:
    """Empty input is handled gracefully rather than raising."""
    result = detect_peak(np.array([], dtype=np.float64))

    assert result == BeamDetectionResult(False, None, None, None)
