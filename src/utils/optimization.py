"""
Batched Differential Evolution optimizers for PyTorch.

This module provides GPU-accelerated Differential Evolution implementations
that can be used for various optimization tasks. The BatchedDifferentialEvolution
class serves as the base implementation, with specialized versions for
specific use cases.
"""

from __future__ import annotations

try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None  # type: ignore[misc, assignment]


def get_torch_device(device: str | None = None) -> torch.device:
    """
    Get the appropriate torch device.

    Args:
        device: Device specification ('cuda', 'cpu', 'mps', or None for auto)

    Returns:
        torch.device: The selected device
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for optimization utilities")

    if device is not None:
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BatchedDifferentialEvolution:
    """
    GPU-resident Differential Evolution optimizer.

    Supports two modes:
    1. Single-spectrum: Optimizes one objective function
    2. Multi-spectrum: Optimizes N independent objective functions in parallel

    Keeps the entire population on GPU and performs all operations
    without CPU-GPU transfers during iteration.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        n_spectra: int = 1,
        popsize: int = 50,
        mutation: float = 0.5,
        crossover: float = 0.9,
        maxiter: int = 1000,
        atol: float = 1e-3,
        seed: int = 42,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        Initialize DE optimizer.

        Args:
            bounds: List of (min, max) tuples for each parameter
            n_spectra: Number of independent spectra to optimize (1 for
                single-spectrum mode)
            popsize: Population size (per spectrum in multi-spectrum mode)
            mutation: Mutation factor F in [0, 2]
            crossover: Crossover probability CR in [0, 1]
            maxiter: Maximum iterations
            atol: Absolute tolerance for convergence
            seed: Random seed
            device: Torch device ('cuda', 'cpu', or None for auto)
            dtype: Data type (torch.float32 or torch.float64)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for BatchedDifferentialEvolution")

        self.bounds = bounds
        self.n_params = len(bounds)
        self.n_spectra = n_spectra
        self.popsize = popsize
        self.mutation = mutation
        self.crossover = crossover
        self.maxiter = maxiter
        self.atol = atol
        self.device = get_torch_device(device)
        self.seed = seed
        self.dtype = dtype if dtype is not None else torch.float64

        torch.manual_seed(seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed(seed)

        # Convert bounds to tensors
        self.lower = torch.tensor(
            [b[0] for b in bounds], device=self.device, dtype=self.dtype
        )
        self.upper = torch.tensor(
            [b[1] for b in bounds], device=self.device, dtype=self.dtype
        )

    def _init_population_single(self, lhs_samples: Tensor | None = None) -> Tensor:
        """
        Initialize population for single-spectrum mode using Sobol sequence.

        Args:
            lhs_samples: Optional pre-computed LHS samples to seed population

        Returns:
            (popsize, n_params) population tensor
        """
        if lhs_samples is not None and lhs_samples.size(0) >= self.popsize:
            pop = lhs_samples[: self.popsize].to(device=self.device, dtype=self.dtype)
        else:
            try:
                from torch.quasirandom import SobolEngine

                sobol = SobolEngine(
                    dimension=self.n_params, scramble=True, seed=self.seed
                )
                pop = sobol.draw(self.popsize).to(device=self.device, dtype=self.dtype)
            except ImportError:
                # Fallback to stratified random sampling
                pop = torch.zeros(
                    self.popsize, self.n_params, device=self.device, dtype=self.dtype
                )
                for i in range(self.n_params):
                    bins = torch.linspace(
                        0, 1, self.popsize + 1, device=self.device, dtype=self.dtype
                    )
                    for j in range(self.popsize):
                        pop[j, i] = bins[j] + torch.rand(
                            1, device=self.device, dtype=self.dtype
                        ) * (bins[j + 1] - bins[j])
                    perm = torch.randperm(self.popsize, device=self.device)
                    pop[:, i] = pop[perm, i]

        # Scale to bounds
        pop = self.lower + pop * (self.upper - self.lower)
        return pop

    def _init_population_multi(self, x0: Tensor | None = None) -> Tensor:
        """
        Initialize population for multi-spectrum mode using Sobol sequence.

        Args:
            x0: Optional (n_spectra, n_params) initial best guesses to seed population

        Returns:
            (n_spectra, popsize, n_params) population tensor
        """
        N, P = self.n_spectra, self.popsize

        try:
            from torch.quasirandom import SobolEngine

            sobol = SobolEngine(dimension=self.n_params, scramble=True)
            unit_samples = sobol.draw(P).to(device=self.device, dtype=self.dtype)
            unit_samples = unit_samples.unsqueeze(0).expand(N, -1, -1).clone()
        except ImportError:
            unit_samples = torch.rand(
                N, P, self.n_params, device=self.device, dtype=self.dtype
            )

        # Scale to bounds
        pop = self.lower + unit_samples * (self.upper - self.lower)

        # Seed first member of each population with x0 if provided
        if x0 is not None:
            pop[:, 0, :] = x0.to(self.device)

        return pop

    def _mutate_single(
        self, population: Tensor, best_idx: int, fitness: Tensor
    ) -> Tensor:
        """DE/best/1 mutation for single-spectrum mode."""
        pop_size = population.size(0)
        best = population[best_idx]

        rand_vals = torch.rand(pop_size, pop_size, device=self.device)
        rand_vals.fill_diagonal_(float("inf"))
        _, sorted_indices = rand_vals.sort(dim=1)
        r0_idx = sorted_indices[:, 0]
        r1_idx = sorted_indices[:, 1]

        r0 = population[r0_idx]
        r1 = population[r1_idx]

        mutant = best.unsqueeze(0) + self.mutation * (r0 - r1)
        mutant = torch.clamp(mutant, self.lower, self.upper)

        return mutant

    def _mutate_multi(self, population: Tensor, best: Tensor) -> Tensor:
        """DE/best/1 mutation for multi-spectrum mode."""
        N, P, D = population.shape

        idx = torch.randint(0, P, (N, P, 2), device=self.device)

        r0 = torch.gather(population, 1, idx[:, :, 0:1].expand(-1, -1, D))
        r1 = torch.gather(population, 1, idx[:, :, 1:2].expand(-1, -1, D))

        best_exp = best.unsqueeze(1).expand(-1, P, -1)
        mutant = best_exp + self.mutation * (r0 - r1)

        mutant = torch.clamp(mutant, self.lower, self.upper)
        return mutant

    def _crossover_single(self, population: Tensor, mutant: Tensor) -> Tensor:
        """Binomial crossover for single-spectrum mode."""
        pop_size, n_params = population.shape

        cross_mask = torch.rand(pop_size, n_params, device=self.device) < self.crossover

        j_rand = torch.randint(0, n_params, (pop_size,), device=self.device)
        for i in range(pop_size):
            cross_mask[i, j_rand[i]] = True

        trial = torch.where(cross_mask, mutant, population)
        return trial

    def _crossover_multi(self, population: Tensor, mutant: Tensor) -> Tensor:
        """Binomial crossover for multi-spectrum mode."""
        N, P, D = population.shape

        mask = torch.rand(N, P, D, device=self.device) < self.crossover

        j_rand = torch.randint(0, D, (N, P), device=self.device)
        for i in range(D):
            mask[:, :, i] = mask[:, :, i] | (j_rand == i)

        trial = torch.where(mask, mutant, population)
        return trial

    def optimize(
        self,
        objective_fn,
        x0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | float, int]:
        """
        Run DE optimization.

        Args:
            objective_fn: For single-spectrum mode: (popsize, n_params)
                -> (popsize,). For multi-spectrum mode: (n_spectra, popsize,
                n_params) -> (n_spectra, popsize)
            x0: Optional initial best guess
                - Single-spectrum: (n_params,) tensor
                - Multi-spectrum: (n_spectra, n_params) tensor

        Returns:
            best_params: (n_params,) or (n_spectra, n_params) best solution(s)
            best_fitness: scalar or (n_spectra,) best fitness value(s)
            n_iterations: number of iterations run
        """
        if self.n_spectra == 1:
            return self._optimize_single(objective_fn, x0)
        else:
            return self._optimize_multi(objective_fn, x0)

    def _optimize_single(
        self,
        objective_fn,
        x0: Tensor | None = None,
    ) -> tuple[Tensor, float, int]:
        """Run single-spectrum DE optimization."""
        population = self._init_population_single()

        if x0 is not None:
            population[0] = x0.to(self.device)

        fitness = objective_fn(population)

        best_idx = torch.argmin(fitness).item()
        best_fitness = fitness[best_idx].item()
        best_params = population[best_idx].clone()

        prev_best = best_fitness

        for iteration in range(self.maxiter):
            mutant = self._mutate_single(population, best_idx, fitness)
            trial = self._crossover_single(population, mutant)
            trial_fitness = objective_fn(trial)

            improved = trial_fitness < fitness
            population = torch.where(improved.unsqueeze(-1), trial, population)
            fitness = torch.where(improved, trial_fitness, fitness)

            current_best_idx = torch.argmin(fitness).item()
            current_best_fitness = fitness[current_best_idx].item()

            if current_best_fitness < best_fitness:
                best_fitness = current_best_fitness
                best_params = population[current_best_idx].clone()
                best_idx = current_best_idx

            if abs(prev_best - best_fitness) < self.atol and iteration > 10:
                break

            prev_best = best_fitness

        return best_params, best_fitness, iteration + 1

    def _optimize_multi(
        self,
        objective_fn,
        x0: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, int]:
        """Run multi-spectrum DE optimization."""
        N, _P = self.n_spectra, self.popsize

        population = self._init_population_multi(x0)
        fitness = objective_fn(population)

        best_idx = torch.argmin(fitness, dim=1)
        best_fitness = fitness.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        best_params = population.gather(
            1, best_idx.view(N, 1, 1).expand(-1, -1, self.n_params)
        ).squeeze(1)

        prev_best = best_fitness.clone()

        for iteration in range(self.maxiter):
            mutant = self._mutate_multi(population, best_params)
            trial = self._crossover_multi(population, mutant)
            trial_fitness = objective_fn(trial)

            improved = trial_fitness < fitness
            population = torch.where(improved.unsqueeze(-1), trial, population)
            fitness = torch.where(improved, trial_fitness, fitness)

            current_best_idx = torch.argmin(fitness, dim=1)
            current_best_fitness = fitness.gather(
                1, current_best_idx.unsqueeze(1)
            ).squeeze(1)

            improved_mask = current_best_fitness < best_fitness
            best_fitness = torch.where(
                improved_mask, current_best_fitness, best_fitness
            )

            new_best_params = population.gather(
                1, current_best_idx.view(N, 1, 1).expand(-1, -1, self.n_params)
            ).squeeze(1)
            best_params = torch.where(
                improved_mask.unsqueeze(-1), new_best_params, best_params
            )

            improvement = torch.abs(prev_best - best_fitness).mean()
            if improvement < self.atol and iteration > 10:
                break

            prev_best = best_fitness.clone()

        return best_params, best_fitness, iteration + 1


# Backwards compatibility aliases
GPUDifferentialEvolution = BatchedDifferentialEvolution
KappaDifferentialEvolution = BatchedDifferentialEvolution
