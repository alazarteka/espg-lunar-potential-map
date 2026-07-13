"""Focused construction tests for G2 scientific certification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

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
_load("beam_torch.py", "beam_torch")
_load("fit_reference.py", "fit_reference")
_load("start_parameters.py", "start_parameters")
_load("g2_de_torch.py", "g2_de_torch")
_load("g2_beam_fit.py", "g2_beam_fit")
_load("simulator.py", "simulator")
_load("c1_estimator.py", "c1_estimator")
certify = _load("certify_g2.py", "experiment_006_certify_g2")


def _support():
    energies = np.geomspace(30.0, 1500.0, 15)
    low = energies / 1.15
    high = energies * 1.15
    pitches = np.tile(np.linspace(5.0, 175.0, 32), (15, 1))
    return SimpleNamespace(
        spec_no=1,
        energies_ev=energies,
        channel_low_ev=low,
        channel_high_ev=high,
        pitches_deg=pitches,
    )


def test_matched_observation_recovers_manifest_truth_exactly() -> None:
    row = {
        "amplitude": "8.0",
        "true_center_ev": "220.0",
        "pitch_sigma_deg": "24.0",
        **{f"coefficient_{index}": "0.0" for index in range(12)},
    }
    observation = certify.matched_observation(_support(), row)
    parameters = np.array([8.0, 220.0, 24.0, *([0.0] * 12)])
    residual = certify.fit_reference._residual(observation, parameters)
    assert np.max(np.abs(residual)) < 1e-14


def test_frozen_corpus_loads_all_rows() -> None:
    rows = certify.load_rows()
    assert len(rows) == 192
    assert {row["arm"] for row in rows} == {
        "matched",
        "discrepancy",
        "calibration_geometry",
        "control",
    }
