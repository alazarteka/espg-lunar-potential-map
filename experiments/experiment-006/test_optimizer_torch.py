"""Implementation-only tests for the Experiment 006 batched LM optimizer."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
import scipy.optimize
import torch

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
_load_module("beam_torch.py", "beam_torch")
optimizer = _load_module("optimizer_torch.py", "experiment_006_optimizer_torch")

MODEL_KWARGS = {
    "energy_domain_ev": (20.0, 2000.0),
    "fwhm_fraction": 0.48,
    "physical_lower": (0.01, 0.01, 90.0, 0.5),
    "physical_upper": (2.0, 3.0, 180.0, 45.0),
}
TRUE_PARAMETERS = np.array(
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
    ]
)


def _problem(starts: np.ndarray) -> tuple[torch.Tensor, ...]:
    energies = np.array([45.0, 65.0, 90.0, 125.0, 175.0, 245.0, 340.0, 475.0])
    low, high = beam_model.channel_edges_ev(energies)
    pitch_grid = np.array([96.0, 120.0, 142.0, 158.0, 169.0, 176.0, 179.0])
    row_ids = np.repeat(np.arange(energies.size), pitch_grid.size)
    pitches = np.tile(pitch_grid, energies.size)
    observed = beam_model.model_values(
        energies, low, high, row_ids, pitches, TRUE_PARAMETERS, **MODEL_KWARGS
    )

    def floating(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float64).unsqueeze(0)

    return (
        floating(energies),
        floating(low),
        floating(high),
        torch.as_tensor(row_ids, dtype=torch.long).unsqueeze(0),
        floating(pitches),
        floating(observed),
        torch.as_tensor(starts, dtype=torch.float64).unsqueeze(0),
    )


def _fit(starts: np.ndarray, **kwargs: float | int) -> object:
    return optimizer.projected_levenberg_marquardt(
        *_problem(starts), **MODEL_KWARGS, **kwargs
    )


def test_noiseless_local_recovery() -> None:
    offset = np.array([0.1, 1.0, 0.1, *([0.002] * 12)])
    result = _fit(TRUE_PARAMETERS[None, :] + offset, max_iterations=45)
    assert result.converged[0, 0]
    assert result.objectives[0, 0] < 1e-10
    torch.testing.assert_close(
        result.parameters[0, 0],
        torch.as_tensor(TRUE_PARAMETERS),
        rtol=2e-5,
        atol=2e-5,
    )


def test_trajectories_stop_independently() -> None:
    perturbed = TRUE_PARAMETERS.copy()
    perturbed[:3] += [0.3, 3.0, 0.3]
    starts = np.stack((TRUE_PARAMETERS, perturbed))
    result = _fit(starts, max_iterations=35)
    assert result.converged.all()
    assert result.diagnostics.iterations[0, 0] == 0
    assert result.diagnostics.termination_code[0, 0] == 4
    assert result.diagnostics.iterations[0, 1] > 0


def test_projection_respects_bounds() -> None:
    lower, upper = optimizer.parameter_bounds()
    starts = np.stack((lower.numpy() - 100.0, upper.numpy() + 100.0))
    result = _fit(starts, max_iterations=0)
    assert torch.all(result.parameters >= lower)
    assert torch.all(result.parameters <= upper)
    torch.testing.assert_close(result.parameters[0, 0], lower)
    torch.testing.assert_close(result.parameters[0, 1], upper)


def test_repeatability_is_deterministic() -> None:
    start = TRUE_PARAMETERS.copy()
    start[:3] += [0.2, -2.0, 0.2]
    first = _fit(start[None, :], max_iterations=25)
    second = _fit(start[None, :], max_iterations=25)
    torch.testing.assert_close(first.parameters, second.parameters, rtol=0.0, atol=0.0)
    torch.testing.assert_close(first.objectives, second.objectives, rtol=0.0, atol=0.0)
    assert torch.equal(first.converged, second.converged)
    assert torch.equal(first.diagnostics.iterations, second.diagnostics.iterations)


def test_small_case_matches_scipy_from_same_start() -> None:
    start = TRUE_PARAMETERS.copy()
    start[:3] += [0.25, 2.5, -0.2]
    start[3:] += 0.004
    torch_result = _fit(start[None, :], max_iterations=50)
    problem = _problem(start[None, :])
    energies, low, high, row_ids, pitches, observed, _ = problem
    lower, upper = optimizer.parameter_bounds()

    def residual(parameters: np.ndarray) -> np.ndarray:
        modeled = beam_model.model_values(
            energies[0].numpy(),
            low[0].numpy(),
            high[0].numpy(),
            row_ids[0].numpy(),
            pitches[0].numpy(),
            parameters,
            **MODEL_KWARGS,
        )
        return beam_model.log_residuals(observed[0].numpy(), modeled)

    scipy_result = scipy.optimize.least_squares(
        residual,
        start,
        bounds=(lower.numpy(), upper.numpy()),
        xtol=1e-13,
        ftol=1e-13,
        gtol=1e-13,
    )
    scipy_objective = np.sum(np.square(scipy_result.fun))
    np.testing.assert_allclose(
        torch_result.objectives[0, 0].item(), scipy_objective, atol=2e-13, rtol=1e-5
    )
    np.testing.assert_allclose(
        torch_result.parameters[0, 0].numpy(), scipy_result.x, atol=3e-4, rtol=3e-5
    )


def test_optimizer_defaults_match_frozen_configuration() -> None:
    configuration = json.loads(
        Path(__file__)
        .with_name("configuration-d1-optimizer.json")
        .read_text(encoding="ascii")
    )
    frozen = configuration["levenberg_marquardt"]
    defaults = inspect.signature(optimizer.projected_levenberg_marquardt).parameters
    mapping = {
        "max_iterations": "maximum_iterations",
        "initial_damping": "initial_damping",
        "minimum_damping": "minimum_damping",
        "maximum_damping": "maximum_damping",
        "accepted_step_damping_divisor": "accepted_step_damping_divisor",
        "rejected_step_damping_multiplier": "rejected_step_damping_multiplier",
        "gradient_tolerance": "gradient_infinity_tolerance",
        "step_tolerance": "scaled_step_tolerance",
        "relative_objective_tolerance": "relative_objective_tolerance",
        "exact_objective_tolerance": "exact_objective_tolerance",
    }
    for argument, key in mapping.items():
        assert defaults[argument].default == frozen[key]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_cuda_float64_local_recovery() -> None:
    offset = np.array([0.1, 1.0, 0.1, *([0.002] * 12)])
    problem = tuple(
        value.cuda() for value in _problem(TRUE_PARAMETERS[None, :] + offset)
    )
    result = optimizer.projected_levenberg_marquardt(
        *problem,
        **MODEL_KWARGS,
        max_iterations=45,
    )
    assert result.converged[0, 0]
    assert result.objectives[0, 0] < 1e-10
    torch.testing.assert_close(
        result.parameters[0, 0].cpu(),
        torch.as_tensor(TRUE_PARAMETERS),
        rtol=2e-5,
        atol=2e-5,
    )
