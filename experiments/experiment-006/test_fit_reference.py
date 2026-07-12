"""Implementation-only tests for Experiment 006 CPU certification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from types import ModuleType


def _load_module(filename: str, name: str) -> ModuleType:
    path = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


beam_model = _load_module("beam_model.py", "beam_model")
reference = _load_module("fit_reference.py", "experiment_006_fit_reference")

TRUE = np.array(
    [
        8.0,
        225.0,
        24.0,
        0.2,
        -0.5,
        0.1,
        0.4,
        0.2,
        -0.3,
        0.1,
        0.6,
        -0.2,
        -0.4,
        0.3,
        0.5,
    ],
    dtype=np.float64,
)


def _observation(noise: float = 0.0) -> object:
    energies = np.array([45.0, 70.0, 110.0, 175.0, 275.0, 430.0])
    low, high = beam_model.channel_edges_ev(energies)
    pitch_grid = np.array([100.0, 130.0, 150.0, 164.0, 173.0, 179.0])
    rows = np.repeat(np.arange(energies.size), pitch_grid.size)
    pitches = np.tile(pitch_grid, energies.size)
    values = beam_model.model_values(
        energies,
        low,
        high,
        rows,
        pitches,
        TRUE,
        energy_domain_ev=(20.0, 2000.0),
        fwhm_fraction=0.48,
        physical_lower=np.array([0.01, 0.01, 90.0, 0.5]),
        physical_upper=np.array([2.0, 3.0, 180.0, 45.0]),
    )
    if noise:
        values = values * np.exp(noise * np.sin(np.arange(values.size)))
    return reference.Observation(energies, low, high, rows, pitches, values)


def _near(offset: float) -> np.ndarray:
    start = TRUE.copy()
    start[:3] += np.array([offset, 3.0 * offset, offset])
    start[3:] += offset * 0.01
    return start


def test_exact_and_noisy_fits() -> None:
    exact = reference.fit_reference(_observation(), TRUE[None, :])
    assert exact.objective < 1e-25
    np.testing.assert_allclose(exact.parameters, TRUE, rtol=0.0, atol=1e-12)

    noisy = reference.fit_reference(_observation(0.002), _near(0.2)[None, :])
    assert np.isfinite(noisy.objective)
    assert noisy.objective > 0.0
    assert np.linalg.norm(noisy.parameters - TRUE) < 2.0


def test_all_start_winner_and_trajectory_retention() -> None:
    starts = np.stack((_near(0.4), TRUE, _near(-0.3)))
    result = reference.fit_reference(_observation(), starts)
    assert len(result.fits) == 3
    assert result.winner_index == 1
    assert all(fit.initial_parameters.shape == (15,) for fit in result.fits)
    assert all(np.isfinite(fit.objective) for fit in result.fits)


def test_top_k_candidates_are_ranked_and_cpu_polished() -> None:
    candidates = np.stack((_near(0.5), _near(0.1), _near(-0.2), _near(0.3)))
    observation = _observation(0.001)
    gpu_objectives = np.array(
        [
            np.dot(
                reference._residual(observation, candidate),
                reference._residual(observation, candidate),
            )
            for candidate in candidates
        ]
    )
    certified = reference.certify_candidates(
        observation, candidates, gpu_objectives, candidates, top_k=2
    )
    expected = np.argsort(gpu_objectives, kind="stable")
    np.testing.assert_array_equal(certified.ranked_candidate_indices, expected)
    np.testing.assert_array_equal(certified.polished_candidate_indices, expected[:2])
    assert len(certified.result.fits) == 2
    assert not certified.used_fallback


def test_nonfinite_candidate_forces_complete_reference_fallback() -> None:
    starts = np.stack((_near(0.5), TRUE, _near(-0.2)))
    candidates = starts.copy()
    candidates[0, 0] = np.nan
    certified = reference.certify_candidates(
        _observation(), candidates, np.array([0.0, 1.0, 2.0]), starts, top_k=1
    )
    assert certified.used_fallback
    assert certified.fallback_reason == (
        "invalid, unconverged, or objective-disagreeing candidate"
    )
    assert len(certified.result.fits) == len(starts)
    np.testing.assert_allclose(certified.result.parameters, TRUE, atol=1e-12)


def test_out_of_bounds_candidate_forces_complete_reference_fallback() -> None:
    starts = np.stack((_near(0.5), TRUE, _near(-0.2)))
    candidates = starts.copy()
    candidates[0, 1] = 2001.0
    certified = reference.certify_candidates(
        _observation(), candidates, np.array([0.0, 1.0, 2.0]), starts, top_k=1
    )
    assert certified.used_fallback
    assert certified.fallback_reason == (
        "invalid, unconverged, or objective-disagreeing candidate"
    )
    assert len(certified.result.fits) == len(starts)


def test_unconverged_or_objective_disagreement_forces_fallback() -> None:
    starts = np.stack((_near(0.5), TRUE, _near(-0.2)))
    converged = np.array([True, False, True])
    certified = reference.certify_candidates(
        _observation(),
        starts,
        np.array([1e9, 1.0, 2.0]),
        starts,
        candidate_converged=converged,
        top_k=1,
    )
    assert certified.used_fallback
    assert certified.fallback_reason == (
        "invalid, unconverged, or objective-disagreeing candidate"
    )


def test_covariance_rank_and_standard_errors_are_cpu_products() -> None:
    result = reference.fit_reference(_observation(0.002), _near(0.1)[None, :])
    assert result.jacobian.shape == (36, 15)
    assert 0 < result.jacobian_rank <= 15
    assert result.covariance.shape == (15, 15)
    assert result.standard_errors.shape == (15,)
    assert np.isfinite(result.covariance).all()
    np.testing.assert_allclose(result.covariance, result.covariance.T, atol=1e-14)
    np.testing.assert_allclose(
        result.standard_errors, np.sqrt(np.maximum(np.diag(result.covariance), 0.0))
    )


def test_repeated_reference_fit_is_deterministic() -> None:
    starts = np.stack((_near(0.2), _near(-0.1)))
    first = reference.fit_reference(_observation(0.001), starts)
    second = reference.fit_reference(_observation(0.001), starts)
    assert first.winner_index == second.winner_index
    np.testing.assert_array_equal(first.parameters, second.parameters)
    np.testing.assert_array_equal(first.covariance, second.covariance)
    np.testing.assert_array_equal(
        [fit.objective for fit in first.fits],
        [fit.objective for fit in second.fits],
    )
