"""Focused tests for executable pre-C1 estimator certification."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


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
_load("fit_reference.py", "fit_reference")
_load("start_parameters.py", "start_parameters")
_load("c1_estimator.py", "c1_estimator")
_load("support_data.py", "support_data")
_load("simulator.py", "simulator")
certification = _load("certify_c1_estimator.py", "experiment_006_c1_certification")


def _support(spec_no: int = 1):
    energies = np.geomspace(35.0, 1400.0, 15)
    low, high = sys.modules["beam_model"].channel_edges_ev(energies)
    pitches = np.tile(np.linspace(5.0, 175.0, 24), (15, 1))
    return SimpleNamespace(
        spec_no=spec_no,
        energies_ev=energies,
        channel_low_ev=low,
        channel_high_ev=high,
        pitches_deg=pitches,
    )


def _config(count: int = 2) -> dict:
    config = json.loads(Path(__file__).with_name("configuration-c1.json").read_text())
    config["allocation"]["anchors_ev"] = [100.0, 220.0]
    config["allocation"]["calibration_support_count"] = count
    return config


def test_fixture_matrix_is_matched_and_covers_labels() -> None:
    fixtures = certification.make_fixtures([_support()], [100.0, 220.0])
    assert len(fixtures) == 4
    assert {fixture.normalization for fixture in fixtures} == set(
        certification.NORMALIZATION_LABELS
    )
    for fixture in fixtures:
        residual = sys.modules["fit_reference"]._residual(
            fixture.observation, fixture.true_parameters
        )
        np.testing.assert_allclose(residual, 0.0, atol=2e-15)


def test_complete_fit_rejects_non_twelve_start_constructor(monkeypatch) -> None:
    fixture = certification.make_fixture(_support(), 220.0, "incident_mean")
    monkeypatch.setattr(
        certification.start_parameters,
        "construct_starts",
        lambda *args: np.zeros((11, 15)),
    )
    with pytest.raises(RuntimeError, match="exactly 12"):
        certification._fit_complete(fixture.observation, max_nfev=1)


def test_certification_gates_with_fast_deterministic_fitter() -> None:
    supports = [_support(10), _support(20)]
    calls = []

    def fake_fit(observation, *, max_nfev):
        calls.append((observation, max_nfev))
        # The matched response peak identifies the anchor for this test double.
        centers = np.array([100.0, 220.0])
        objectives = []
        for center in centers:
            parameters = certification._parameters(center)
            modeled = sys.modules["beam_model"].model_values(
                observation.energies_ev,
                observation.channel_low_ev,
                observation.channel_high_ev,
                observation.row_ids,
                observation.pitches_deg,
                parameters,
                energy_domain_ev=observation.energy_domain_ev,
                fwhm_fraction=observation.fwhm_fraction,
                physical_lower=np.asarray(observation.physical_lower),
                physical_upper=np.asarray(observation.physical_upper),
            )
            objectives.append(
                np.linalg.norm(np.log(modeled) - np.log(observation.observed))
            )
        parameters = certification._parameters(centers[int(np.argmin(objectives))])
        return SimpleNamespace(parameters=parameters)

    summary = certification.certify(supports, _config(), fit=fake_fit)
    assert summary["cases"] == 8
    assert summary["fits_per_case"] == 12
    assert summary["confirmation_flux_opened"] is False
    assert summary["common_native_flux_scale_pass"]
    assert summary["full_gate_pass"]
    assert all(limit == 2000 for _, limit in calls)


def test_numerical_and_source_independence_gates() -> None:
    assert certification.quadrature_refinement_error() < 1e-8
    assert certification.simulator_source_independent()
    assert certification.common_native_flux_scale_invariant(_support())
