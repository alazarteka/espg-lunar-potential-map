"""Parity tests for the Experiment 006 PyTorch forward model."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pytest
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


numpy_model = _load_module("beam_model.py", "experiment_006_beam_model_for_torch")
torch_model = _load_module("beam_torch.py", "experiment_006_beam_torch")

ENERGY_DOMAIN = (20.0, 2000.0)
LOWER = np.array([0.01, 0.01, 90.0, 0.5])
UPPER = np.array([2.0, 3.0, 180.0, 45.0])


def _case(dtype: np.dtype) -> dict[str, np.ndarray]:
    energies = np.array([65.0, 113.0, 196.0, 340.0, 588.0], dtype=dtype)
    low, high = numpy_model.channel_edges_ev(energies)
    pitches = np.array([98.0, 132.0, 166.0, 178.0], dtype=dtype)
    row_ids = np.repeat(np.arange(energies.size), pitches.size)
    sample_pitches = np.tile(pitches, energies.size)
    parameters = np.array(
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
        dtype=dtype,
    )
    return {
        "energies": energies,
        "low": low.astype(dtype),
        "high": high.astype(dtype),
        "row_ids": row_ids,
        "pitches": sample_pitches,
        "parameters": parameters,
    }


@pytest.mark.parametrize(
    ("numpy_dtype", "torch_dtype", "rtol", "atol"),
    [
        (np.float64, torch.float64, 2e-12, 2e-12),
        (np.float32, torch.float32, 2e-5, 3e-6),
    ],
)
def test_component_and_model_numpy_parity(
    numpy_dtype: np.dtype,
    torch_dtype: torch.dtype,
    rtol: float,
    atol: float,
) -> None:
    case = _case(numpy_dtype)

    def tensor(value: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch_dtype).unsqueeze(0)

    expected_response = numpy_model.integrated_energy_response(
        case["low"], case["high"], 225.0, 0.48
    )
    actual_response = torch_model.integrated_energy_response(
        tensor(case["low"]),
        tensor(case["high"]),
        torch.tensor([225.0], dtype=torch_dtype),
        0.48,
    )
    np.testing.assert_allclose(
        actual_response[0].numpy(), expected_response, rtol=rtol, atol=atol
    )

    expected_basis = numpy_model.energy_basis(case["energies"], ENERGY_DOMAIN)
    actual_basis = torch_model.energy_basis(tensor(case["energies"]), ENERGY_DOMAIN)
    np.testing.assert_allclose(
        actual_basis[0].numpy(), expected_basis, rtol=rtol, atol=atol
    )

    expected_rows = numpy_model.continuum_row_parameters(
        case["energies"], case["parameters"][3:], ENERGY_DOMAIN, LOWER, UPPER
    )
    actual_rows = torch_model.continuum_row_parameters(
        tensor(case["energies"]),
        tensor(case["parameters"][3:]),
        ENERGY_DOMAIN,
        LOWER,
        UPPER,
    )
    np.testing.assert_allclose(
        actual_rows[0].numpy(), expected_rows, rtol=rtol, atol=atol
    )

    expected_model = numpy_model.model_values(
        case["energies"],
        case["low"],
        case["high"],
        case["row_ids"],
        case["pitches"],
        case["parameters"],
        energy_domain_ev=ENERGY_DOMAIN,
        fwhm_fraction=0.48,
        physical_lower=LOWER,
        physical_upper=UPPER,
    )
    actual_model = torch_model.model_values(
        tensor(case["energies"]),
        tensor(case["low"]),
        tensor(case["high"]),
        torch.as_tensor(case["row_ids"], dtype=torch.long).unsqueeze(0),
        tensor(case["pitches"]),
        tensor(case["parameters"]),
        energy_domain_ev=ENERGY_DOMAIN,
        fwhm_fraction=0.48,
        physical_lower=LOWER,
        physical_upper=UPPER,
    )
    np.testing.assert_allclose(
        actual_model[0].numpy(), expected_model, rtol=rtol, atol=atol
    )

    observed = expected_model * np.linspace(0.8, 1.2, expected_model.size)
    expected_residuals = numpy_model.log_residuals(observed, expected_model)
    actual_residuals = torch_model.log_residuals(
        torch.as_tensor(observed, dtype=torch_dtype).unsqueeze(0), actual_model
    )
    np.testing.assert_allclose(
        actual_residuals[0].numpy(), expected_residuals, rtol=rtol, atol=atol
    )
    expected_objective = np.sum(np.square(expected_residuals))
    actual_objective = torch_model.masked_log_objective(
        torch.as_tensor(observed, dtype=torch_dtype).unsqueeze(0),
        actual_model,
        torch.ones_like(actual_model, dtype=torch.bool),
    )
    np.testing.assert_allclose(
        actual_objective.numpy(), expected_objective, rtol=1e-4, atol=1e-6
    )


def _padded_batch(order: tuple[int, int] = (0, 1)) -> tuple[torch.Tensor, ...]:
    cases = [_case(np.float64), _case(np.float64)]
    cases[1] = {key: value.copy() for key, value in cases[1].items()}
    cases[1]["energies"] = cases[1]["energies"][:3]
    cases[1]["low"] = cases[1]["low"][:3]
    cases[1]["high"] = cases[1]["high"][:3]
    keep = cases[1]["row_ids"] < 3
    cases[1]["row_ids"] = cases[1]["row_ids"][keep]
    cases[1]["pitches"] = cases[1]["pitches"][keep]
    cases[1]["parameters"][0:3] = [5.0, 310.0, 32.0]
    cases = [cases[index] for index in order]

    rows, samples = 7, 24
    energies = torch.full((2, rows), -99.0, dtype=torch.float64)
    low = torch.full_like(energies, -88.0)
    high = torch.full_like(energies, -77.0)
    row_mask = torch.zeros((2, rows), dtype=torch.bool)
    row_ids = torch.full((2, samples), -1, dtype=torch.long)
    pitches = torch.full((2, samples), float("nan"), dtype=torch.float64)
    sample_mask = torch.zeros((2, samples), dtype=torch.bool)
    parameters = torch.empty((2, 15), dtype=torch.float64)
    for batch, case in enumerate(cases):
        n_rows = len(case["energies"])
        n_samples = len(case["row_ids"])
        energies[batch, :n_rows] = torch.from_numpy(case["energies"])
        low[batch, :n_rows] = torch.from_numpy(case["low"])
        high[batch, :n_rows] = torch.from_numpy(case["high"])
        row_mask[batch, :n_rows] = True
        row_ids[batch, :n_samples] = torch.from_numpy(case["row_ids"])
        pitches[batch, :n_samples] = torch.from_numpy(case["pitches"])
        sample_mask[batch, :n_samples] = True
        parameters[batch] = torch.from_numpy(case["parameters"])
    return energies, low, high, row_ids, pitches, parameters, row_mask, sample_mask


def _evaluate(batch: tuple[torch.Tensor, ...]) -> torch.Tensor:
    energies, low, high, row_ids, pitches, parameters, row_mask, sample_mask = batch
    return torch_model.model_values(
        energies,
        low,
        high,
        row_ids,
        pitches,
        parameters,
        energy_domain_ev=ENERGY_DOMAIN,
        fwhm_fraction=0.48,
        physical_lower=LOWER,
        physical_upper=UPPER,
        row_valid_mask=row_mask,
        sample_valid_mask=sample_mask,
    )


def test_padding_and_batch_order_invariance() -> None:
    forward = _evaluate(_padded_batch())
    reversed_result = _evaluate(_padded_batch((1, 0)))
    torch.testing.assert_close(forward[0], reversed_result[1])
    torch.testing.assert_close(forward[1], reversed_result[0])
    assert torch.count_nonzero(forward[1, 12:]) == 0


def test_masked_log_residuals_and_objective_ignore_padding() -> None:
    modeled = torch.tensor([[1.0, 4.0, 0.0], [3.0, 2.0, float("nan")]])
    observed = torch.tensor([[1.0, 2.0, -1.0], [1.5, 2.0, float("nan")]])
    valid = torch.tensor([[True, True, False], [True, True, False]])
    residuals = torch_model.masked_log_residuals(observed, modeled, valid)
    expected = modeled.new_tensor([[0.0, np.log(2.0), 0.0], [np.log(2.0), 0.0, 0.0]])
    torch.testing.assert_close(residuals, expected)
    torch.testing.assert_close(
        torch_model.masked_log_objective(observed, modeled, valid),
        modeled.new_tensor([np.log(2.0) ** 2, np.log(2.0) ** 2]),
    )


def test_gradients_are_finite_with_padded_rows_and_samples() -> None:
    batch = list(_padded_batch())
    batch[5].requires_grad_()
    modeled = _evaluate(tuple(batch))
    observed = torch.where(batch[7], modeled.detach() * 1.03, torch.nan)
    objective = torch_model.masked_log_objective(observed, modeled, batch[7]).sum()
    objective.backward()
    assert batch[5].grad is not None
    assert torch.isfinite(batch[5].grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cuda_parity(dtype: torch.dtype) -> None:
    cpu_batch = tuple(
        value.to(dtype=dtype) if value.is_floating_point() else value
        for value in _padded_batch()
    )
    cpu = _evaluate(cpu_batch)
    cuda = _evaluate(tuple(value.cuda() for value in cpu_batch)).cpu()
    torch.testing.assert_close(cuda, cpu, rtol=3e-5, atol=3e-6)
