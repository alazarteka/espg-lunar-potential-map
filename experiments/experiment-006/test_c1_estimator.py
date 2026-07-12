"""Focused tests for the executable Experiment 006 C1 estimator."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("beam_model.py", "beam_model")
_load("fit_reference.py", "fit_reference")
_load("start_parameters.py", "start_parameters")
estimator = _load("c1_estimator.py", "experiment_006_c1_estimator")


def _arrays():
    energies = np.array([100.0, 140.0, 220.0])
    low = energies * 0.8
    high = energies * 1.2
    pitch_row = np.array([10, 25, 40, 55, 70, 85, 100, 125, 145, 160, 170, 180.0])
    pitch = np.tile(pitch_row, (3, 1))
    flux = np.tile(np.arange(1.0, 13.0), (3, 1))
    valid = np.ones_like(flux, dtype=bool)
    mean = np.full_like(flux, np.nan)
    conjugate = np.full_like(flux, np.nan)
    mean[:, 6:] = flux[:, 6:] / np.mean(flux[:, :6], axis=1)[:, None]
    conjugate[:, 6:] = flux[:, 6:] / flux[:, 5::-1]
    return energies, low, high, pitch, flux, valid, mean, conjugate


def test_observation_construction_uses_only_qualified_reflected_samples() -> None:
    arrays = list(_arrays())
    arrays[5][1, 0] = False
    branches = estimator.construct_observations(*arrays)
    assert set(branches) == {estimator.INCIDENT_MEAN, estimator.CONJUGATE}
    assert branches[estimator.INCIDENT_MEAN].observation.observed.size == 12
    np.testing.assert_array_equal(
        branches[estimator.INCIDENT_MEAN].observation.row_ids,
        np.repeat([0, 2], 6),
    )


def test_fit_helper_requires_exactly_twelve_starts(monkeypatch) -> None:
    branch = estimator.construct_observations(*_arrays())[estimator.INCIDENT_MEAN]
    monkeypatch.setattr(
        estimator.start_parameters, "construct_starts", lambda *args: np.zeros((11, 15))
    )
    with pytest.raises(RuntimeError, match="exactly 12"):
        estimator._fit(branch)


def test_fit_helper_uses_certified_reference_and_solver_limit(monkeypatch) -> None:
    branch = estimator.construct_observations(*_arrays())[estimator.INCIDENT_MEAN]
    starts = np.zeros((12, 15))
    sentinel = object()
    seen = {}
    monkeypatch.setattr(
        estimator.start_parameters, "construct_starts", lambda *args: starts
    )

    def fake_fit(observation, supplied_starts, *, max_nfev):
        seen.update(observation=observation, starts=supplied_starts, limit=max_nfev)
        return sentinel

    monkeypatch.setattr(estimator.fit_reference, "fit_reference", fake_fit)
    assert estimator._fit(branch, max_nfev=37) is sentinel
    assert seen == {"observation": branch.observation, "starts": starts, "limit": 37}


def test_stable_reason_codes_and_base_valid_property() -> None:
    assessment = estimator.BranchAssessment(
        "x",
        None,
        (estimator.FIT_FAILED,),
        np.array([], dtype=int),
        np.array([], dtype=int),
    )
    result = estimator.C1Estimate(
        assessment, assessment, (), np.array([], dtype=int), (estimator.FIT_FAILED,)
    )
    assert not result.base_valid
    assert estimator.FIT_FAILED == "fit_failed"
    assert estimator.LEAVE_ONE_ROW_OUT_SHIFT == "leave_one_energy_row_out_shift"


def test_row_evidence_reports_contributing_and_axis_rows() -> None:
    reference = sys.modules["fit_reference"]
    branch = estimator.construct_observations(*_arrays())[estimator.INCIDENT_MEAN]
    parameters = np.array([8.0, 140.0, 20.0, *([0.0] * 12)])
    fit = type("Fit", (), {"parameters": parameters})()
    contributing, axis = estimator._row_evidence(
        branch, fit, estimator.load_configuration()["qc"]
    )
    assert contributing.ndim == axis.ndim == 1
    assert set(axis).issubset(set(contributing))
    assert reference.N_PARAMETERS == 15
