"""Focused tests for the frozen Experiment 006 certification protocol."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
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


beam_model = _load("beam_model.py", "beam_model")
reference = _load("fit_reference.py", "fit_reference")
_load("optimizer_torch.py", "optimizer_torch")
_load("start_parameters.py", "start_parameters")
support_data = _load("support_data.py", "support_data")
certification = _load("basin_certification.py", "experiment_006_basin_certification")


def configuration() -> dict:
    return json.loads(
        Path(__file__)
        .with_name("configuration-d1-certification.json")
        .read_text(encoding="ascii")
    )


def supports() -> list:
    energies = np.geomspace(25.0, 1800.0, 15)
    low, high = beam_model.channel_edges_ev(energies)
    pitches = np.tile(np.linspace(92.0, 178.0, 12), (15, 1))
    return [
        support_data.GeometrySupport(
            spec_no=1000 + index,
            energies_ev=energies.copy(),
            channel_low_ev=low.copy(),
            channel_high_ev=high.copy(),
            pitches_deg=pitches.copy(),
            polarity=np.ones(15, dtype=np.int8),
        )
        for index in range(12)
    ]


def _fit(parameters: np.ndarray, rank: int = 15) -> reference.FitResult:
    jacobian = np.eye(20, 15)
    if rank < 15:
        jacobian[:, rank:] = 0.0
    start_fit = reference.StartFit(
        0, parameters.copy(), parameters.copy(), 1.0, True, 1, "ok", jacobian
    )
    return reference.FitResult(
        (start_fit,),
        0,
        parameters.copy(),
        1.0,
        jacobian,
        rank,
        np.eye(15),
        np.ones(15),
    )


def test_corpus_has_frozen_512_case_quotas_and_order() -> None:
    first = certification.make_cases(supports(), configuration())
    second = certification.make_cases(supports(), configuration())
    assert len(first) == 512
    assert [case.case_id for case in first] == list(range(512))
    assert Counter(case.arm for case in first) == {
        "analytic_matched": 96,
        "generic_discrepancy": 160,
        "null_adversarial": 64,
        "calibration_support": 192,
    }
    adversarial = [case for case in first if case.arm == "null_adversarial"]
    assert set(Counter(case.challenge for case in adversarial).values()) == {8}
    calibration = [case for case in first if case.arm == "calibration_support"]
    assert set(Counter(case.support_spec_no for case in calibration).values()) == {16}
    assert np.array_equal(
        first[137].observation.observed, second[137].observation.observed
    )


def test_canonical_ranking_is_stable_and_nonfinite_last() -> None:
    ranking = certification.canonical_ranking(np.array([2.0, np.nan, 1.0, 1.0, np.inf]))
    assert ranking.tolist() == [2, 3, 0, 1, 4]
    ranking = certification.eligible_ranking(
        np.array([0.1, 0.2, 0.3]), np.array([False, True, True])
    )
    assert ranking.tolist() == [1, 2, 0]


def test_basin_matching_requires_center_only_when_identifiable() -> None:
    case = certification.make_cases(supports(), configuration())[0]
    parameters = np.array([1.0, 200.0, 20.0, *([0.0] * 12)])
    changed = parameters.copy()
    changed[1] = 300.0
    # A center change also changes the model, so use a permissive model gate here
    # to isolate the center-identifiability rule.
    config = configuration()
    config["basin_match"]["log_model_rms_max"] = 1e6
    assert not certification.basin_matches(
        case.observation, _fit(parameters), _fit(changed), True, config
    )[0]
    assert certification.basin_matches(
        case.observation, _fit(parameters, rank=14), _fit(changed), True, config
    )[0]
    assert certification.basin_matches(
        case.observation, _fit(parameters), _fit(changed), False, config
    )[0]


def test_summary_evaluates_all_k_and_final_gates() -> None:
    row = {
        "arm": "analytic_matched",
        "challenge": "matched",
        "gpu_finite_candidates": 12,
        "gpu_converged_candidates": 12,
        "reference_success": True,
        "used_fallback": False,
        "fallback_reason": "",
        "final_objective_relative_difference": 0.0,
        "final_log_model_rms": 0.0,
        "final_center_difference_response_fwhm": 0.0,
        "final_contacts_bound": False,
        "final_rank": 15,
        "starts_seconds": 1.0,
        "reference_seconds": 1.0,
        "gpu_seconds": 1.0,
        "all_gpu_polish_seconds": 1.0,
    }
    row.update({f"recall_k{k}": k >= 4 for k in range(1, 13)})
    summary = certification.summarize([row], configuration())
    assert list(summary["recall_by_k"]) == [str(k) for k in range(1, 13)]
    assert summary["minimum_full_recall_k_with_ceiling_6"] == 4
    assert summary["full_gate_pass"]
    failed = dict(row, final_log_model_rms=1.0)
    assert not certification.summarize([failed], configuration())["full_gate_pass"]
    failed = dict(row, gpu_converged_candidates=5)
    assert not certification.summarize([failed], configuration())["full_gate_pass"]
    failed = dict(row, reference_success=False)
    assert not certification.summarize([failed], configuration())["full_gate_pass"]
