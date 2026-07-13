"""Deterministic batched island differential evolution for Experiment 006 G2."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor
from torch.quasirandom import SobolEngine

if TYPE_CHECKING:
    from collections.abc import Sequence

Objective = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class IslandDEResult:
    """Best retained member and diagnostics for every problem and island."""

    parameters: Tensor
    objectives: Tensor
    population_objective_mean: Tensor
    population_objective_std: Tensor
    generations: int
    evaluations_per_problem: int


def _seed32(root_seed: int, problem_key: int, island: int) -> int:
    sequence = np.random.SeedSequence([root_seed, problem_key, island])
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _reflect_unit(values: Tensor) -> Tensor:
    folded = torch.remainder(values, 2.0)
    return torch.where(folded <= 1.0, folded, 2.0 - folded)


def _parent_indices(generator: np.random.Generator, population: int) -> np.ndarray:
    """Draw three distinct parents that also differ from each target."""
    target = np.arange(population)[:, None]
    parents = generator.integers(0, population - 1, size=(population, 3))
    parents += parents >= target
    duplicate = (
        (parents[:, 0] == parents[:, 1])
        | (parents[:, 0] == parents[:, 2])
        | (parents[:, 1] == parents[:, 2])
    )
    while np.any(duplicate):
        count = int(np.count_nonzero(duplicate))
        replacement = generator.integers(0, population - 1, size=(count, 3))
        replacement += replacement >= target[duplicate]
        parents[duplicate] = replacement
        duplicate = (
            (parents[:, 0] == parents[:, 1])
            | (parents[:, 0] == parents[:, 2])
            | (parents[:, 1] == parents[:, 2])
        )
    return parents


def initialize_population(
    initial_seeds: Tensor,
    problem_keys: Sequence[int] | np.ndarray,
    *,
    islands: int,
    population: int,
    root_seed: int,
) -> Tensor:
    """Build separately scrambled Sobol islands and insert all supplied seeds."""
    seeds = torch.as_tensor(initial_seeds)
    if seeds.ndim != 3:
        raise ValueError("Initial seeds must have shape [problems, seeds, parameters]")
    problems, seed_count, parameters = seeds.shape
    keys = np.asarray(problem_keys, dtype=np.int64)
    if keys.shape != (problems,) or np.unique(keys).size != problems:
        raise ValueError("Problem keys must be a unique vector matching the batch")
    if not 1 <= seed_count <= population:
        raise ValueError("Seed count must be between one and the population size")
    if islands < 1 or population < 4:
        raise ValueError("At least one island and four population members are required")

    batches = []
    for problem, key in enumerate(keys):
        problem_islands = []
        for island in range(islands):
            engine = SobolEngine(
                dimension=parameters,
                scramble=True,
                seed=_seed32(root_seed, int(key), island),
            )
            members = engine.draw(population).to(device=seeds.device, dtype=seeds.dtype)
            members[:seed_count] = seeds[problem]
            problem_islands.append(members)
        batches.append(torch.stack(problem_islands))
    return torch.stack(batches)


def differential_evolution(
    objective: Objective,
    initial_seeds: Tensor,
    problem_keys: Sequence[int] | np.ndarray,
    lower_bounds: Tensor | Sequence[float],
    upper_bounds: Tensor | Sequence[float],
    *,
    islands: int = 4,
    population: int = 128,
    generations: int = 400,
    mutation: tuple[float, float] = (0.5, 1.0),
    crossover: float = 0.9,
    migration_interval: int = 40,
    migration_elites: int = 2,
    root_seed: int = 0,
) -> IslandDEResult:
    """Minimize independent bounded objectives with deterministic DE/rand/1/bin."""
    seeds = torch.as_tensor(initial_seeds, dtype=torch.float64)
    if generations < 0:
        raise ValueError("Generations must be nonnegative")
    if not 0.0 <= crossover <= 1.0:
        raise ValueError("Crossover must lie in [0, 1]")
    if not 0.0 < mutation[0] <= mutation[1] <= 2.0:
        raise ValueError("Mutation bounds must satisfy 0 < low <= high <= 2")
    if migration_interval < 1 or not 0 <= migration_elites < population:
        raise ValueError("Invalid migration settings")

    lower = torch.as_tensor(lower_bounds, device=seeds.device, dtype=seeds.dtype)
    upper = torch.as_tensor(upper_bounds, device=seeds.device, dtype=seeds.dtype)
    if lower.shape != (seeds.shape[-1],) or upper.shape != lower.shape:
        raise ValueError("Bounds must match the parameter dimension")
    valid_bounds = torch.isfinite(lower) & torch.isfinite(upper) & (upper > lower)
    if not bool(torch.all(valid_bounds)):
        raise ValueError("Bounds must be finite and strictly ordered")
    width = upper - lower
    unit_seeds = ((seeds - lower) / width).clamp(0.0, 1.0)
    unit = initialize_population(
        unit_seeds,
        problem_keys,
        islands=islands,
        population=population,
        root_seed=root_seed,
    )
    keys = np.asarray(problem_keys, dtype=np.int64)
    generators = [
        [
            np.random.default_rng(_seed32(root_seed + 1, int(key), island))
            for island in range(islands)
        ]
        for key in keys
    ]

    def physical(values: Tensor) -> Tensor:
        return lower + width * values

    fitness = objective(physical(unit))
    expected = unit.shape[:-1]
    if fitness.shape != expected:
        raise ValueError(f"Objective must return shape {expected}, got {fitness.shape}")
    fitness = torch.where(torch.isfinite(fitness), fitness, torch.inf)
    dimensions = unit.shape[-1]

    for generation in range(generations):
        parents_np = np.empty((*expected, 3), dtype=np.int64)
        mutation_np = np.empty(expected, dtype=np.float64)
        crossover_np = np.empty((*expected, dimensions), dtype=bool)
        for problem in range(len(generators)):
            for island, generator in enumerate(generators[problem]):
                parents_np[problem, island] = _parent_indices(generator, population)
                mutation_np[problem, island] = generator.uniform(
                    mutation[0], mutation[1], size=population
                )
                mask = generator.random((population, dimensions)) < crossover
                forced = generator.integers(0, dimensions, population)
                mask[np.arange(population), forced] = True
                crossover_np[problem, island] = mask
        parents = torch.as_tensor(parents_np, device=unit.device)
        gather_shape = (*expected, dimensions)
        selected = [
            torch.gather(
                unit,
                -2,
                parents[..., index, None].expand(gather_shape),
            )
            for index in range(3)
        ]
        factor = torch.as_tensor(mutation_np, device=unit.device, dtype=unit.dtype)
        mutant = _reflect_unit(
            selected[0] + factor[..., None] * (selected[1] - selected[2])
        )
        mask = torch.as_tensor(crossover_np, device=unit.device)
        trial = torch.where(mask, mutant, unit)
        trial_fitness = objective(physical(trial))
        trial_fitness = torch.where(
            torch.isfinite(trial_fitness), trial_fitness, torch.inf
        )
        improved = trial_fitness < fitness
        unit = torch.where(improved[..., None], trial, unit)
        fitness = torch.where(improved, trial_fitness, fitness)

        if migration_elites and (generation + 1) % migration_interval == 0:
            order = torch.argsort(fitness, dim=-1)
            elite_indices = order[..., :migration_elites]
            worst_indices = order[..., -migration_elites:]
            elites = torch.gather(
                unit,
                -2,
                elite_indices[..., None].expand(*elite_indices.shape, dimensions),
            )
            elite_fitness = torch.gather(fitness, -1, elite_indices)
            incoming = torch.roll(elites, shifts=1, dims=1)
            incoming_fitness = torch.roll(elite_fitness, shifts=1, dims=1)
            unit.scatter_(
                -2,
                worst_indices[..., None].expand(*worst_indices.shape, dimensions),
                incoming,
            )
            fitness.scatter_(-1, worst_indices, incoming_fitness)

    best_index = torch.argmin(fitness, dim=-1)
    best = torch.gather(
        unit,
        -2,
        best_index[..., None, None].expand(*best_index.shape, 1, dimensions),
    ).squeeze(-2)
    best_fitness = torch.gather(fitness, -1, best_index[..., None]).squeeze(-1)
    return IslandDEResult(
        parameters=physical(best),
        objectives=best_fitness,
        population_objective_mean=fitness.mean(dim=-1),
        population_objective_std=fitness.std(dim=-1),
        generations=generations,
        evaluations_per_problem=islands * population * (generations + 1),
    )
