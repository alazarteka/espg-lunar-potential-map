"""Focused tests for Experiment 004 Phase A4."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


def _load_module():
    path = Path(__file__).with_name("phase_a4.py")
    spec = importlib.util.spec_from_file_location("experiment_004_phase_a4", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


phase_a4 = _load_module()


def test_transition_fit_recovers_synthetic_boundary() -> None:
    pitches = np.linspace(91.0, 179.0, 44)
    boundary = 137.0
    width = 7.0
    model = 0.18 + 0.72 / (1.0 + np.exp(-(boundary - pitches) / width))
    fitted = phase_a4.fit_transition(pitches, model, np.ones_like(pitches))
    assert fitted is not None
    parameters, rms = fitted
    assert abs(parameters[2] - boundary) < 0.1
    assert abs(parameters[3] - width) < 0.1
    assert rms < 1e-6


def test_published_panel_sampling_maps_known_corners() -> None:
    values = np.arange(89 * 44, dtype=float).reshape(89, 44)
    sampled, valid = phase_a4.sample_published_panel(
        values,
        np.zeros_like(values, dtype=bool),
        np.array([10.0, 10000.0]),
    )
    assert valid.all()
    assert sampled[0, 0] == values[2, 0]
    assert sampled[-1, -1] == values[86, 43]


def test_physical_boundary_fit_recovers_synthetic_voltage() -> None:
    energies = np.array([340.0, 590.0, 1020.0, 1770.0, 3060.0, 5300.0])
    boundary = phase_a4.predicted_boundary(
        energies,
        np.array([-160.0]),
        np.array([0.975]),
        11.0,
    )[0]
    points = [
        phase_a4.BoundaryPoint(
            spec_no=653,
            normalization="incident_mean",
            magnetic_field_mode="row_field",
            sector_weighting="equal",
            energy_ev=float(energy),
            boundary_pitch_deg=float(pitch),
            boundary_sigma_deg=5.625,
            transition_width_deg=5.0,
            cone_level=0.1,
            outside_contrast=0.8,
            n_sectors=40,
            bootstrap_successes=100,
            fit_rms_log_residual=0.0,
            nuisance_parameter_edge=False,
            accepted=True,
        )
        for energy, pitch in zip(energies, boundary, strict=True)
    ]
    fits, _ = phase_a4.fit_physical_boundary(points, 11.0)
    free = next(fit for fit in fits if fit.constraint_mode == "free")
    assert free.u_surface_v == -160.0
    assert np.isclose(free.bs_over_bm, 0.975)
