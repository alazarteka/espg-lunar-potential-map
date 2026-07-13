"""Implementation tests for deterministic G2 island differential evolution."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch


def _load():
    path = Path(__file__).with_name("g2_de_torch.py")
    spec = importlib.util.spec_from_file_location("experiment_006_g2_de", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


de = _load()


def _run(keys=(11, 29)):
    targets = torch.tensor([[0.2, 0.7], [0.8, 0.3]], dtype=torch.float64)
    selected = targets[[0, 1] if keys == (11, 29) else [1, 0]]
    seeds = torch.stack((selected + 0.2, selected - 0.2), dim=1).clamp(0.0, 1.0)

    def objective(population):
        return torch.square(population - selected[:, None, None, :]).sum(dim=-1)

    return de.differential_evolution(
        objective,
        seeds,
        keys,
        [0.0, 0.0],
        [1.0, 1.0],
        islands=3,
        population=24,
        generations=60,
        migration_interval=15,
        migration_elites=1,
        root_seed=73,
    )


def test_sphere_recovery_and_bounds() -> None:
    result = _run()
    targets = torch.tensor([[0.2, 0.7], [0.8, 0.3]], dtype=torch.float64)
    assert result.parameters.shape == (2, 3, 2)
    assert torch.all((result.parameters >= 0.0) & (result.parameters <= 1.0))
    torch.testing.assert_close(
        result.parameters,
        targets[:, None, :].expand_as(result.parameters),
        atol=2e-4,
        rtol=0.0,
    )


def test_deterministic_replay() -> None:
    first = _run()
    second = _run()
    torch.testing.assert_close(first.parameters, second.parameters, atol=0.0, rtol=0.0)
    torch.testing.assert_close(first.objectives, second.objectives, atol=0.0, rtol=0.0)


def test_batch_order_invariance_by_problem_key() -> None:
    canonical = _run((11, 29))
    reversed_result = _run((29, 11))
    torch.testing.assert_close(
        canonical.parameters,
        reversed_result.parameters.flip(0),
        atol=0.0,
        rtol=0.0,
    )


def test_parent_indices_are_distinct_and_exclude_target() -> None:
    parents = de._parent_indices(np.random.default_rng(9), 32)
    target = np.arange(32)
    assert np.all(parents != target[:, None])
    assert np.all(parents[:, 0] != parents[:, 1])
    assert np.all(parents[:, 0] != parents[:, 2])
    assert np.all(parents[:, 1] != parents[:, 2])


def test_initialization_retains_every_seed_in_every_island() -> None:
    seeds = torch.tensor([[[0.1, 0.2], [0.3, 0.4]]], dtype=torch.float64)
    population = de.initialize_population(
        seeds, [5], islands=4, population=16, root_seed=3
    )
    torch.testing.assert_close(
        population[0, :, :2], seeds[0][None, :, :].expand(4, -1, -1)
    )
