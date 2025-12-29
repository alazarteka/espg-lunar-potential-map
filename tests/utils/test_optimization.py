"""Tests for PyTorch-based Differential Evolution optimizer.

Tests both single-spectrum and multi-spectrum modes.
"""

from __future__ import annotations

import numpy as np
import pytest

try:
    import torch

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not installed")


class TestGetTorchDevice:
    """Tests for get_torch_device utility."""

    def test_explicit_cpu(self):
        """Test explicit CPU device selection."""
        from src.utils.optimization import get_torch_device

        device = get_torch_device("cpu")
        assert device == torch.device("cpu")

    def test_auto_detection(self):
        """Test auto device detection returns valid device."""
        from src.utils.optimization import get_torch_device

        device = get_torch_device(None)
        assert device.type in ("cpu", "cuda", "mps")

    @pytest.mark.skipif(
        not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available"
    )
    def test_explicit_cuda(self):
        """Test explicit CUDA device selection."""
        from src.utils.optimization import get_torch_device

        device = get_torch_device("cuda")
        assert device.type == "cuda"


class TestBatchedDEInitialization:
    """Tests for BatchedDifferentialEvolution initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-10, 10), (-5, 5)]
        de = BatchedDifferentialEvolution(bounds=bounds)

        assert de.n_params == 2
        assert de.n_spectra == 1
        assert de.popsize == 50
        assert de.device == torch.device("cpu") or de.device.type in ("cuda", "mps")

    def test_custom_params(self):
        """Test initialization with custom parameters."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-1, 1), (-1, 1), (-1, 1)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=10,
            popsize=30,
            mutation=0.7,
            crossover=0.8,
            maxiter=500,
            atol=1e-4,
            device="cpu",
        )

        assert de.n_params == 3
        assert de.n_spectra == 10
        assert de.popsize == 30
        assert de.mutation == 0.7
        assert de.crossover == 0.8
        assert de.maxiter == 500
        assert de.atol == 1e-4


class TestBatchedDESingleSpectrum:
    """Tests for single-spectrum (n_spectra=1) optimization."""

    def test_sphere_function(self):
        """Test optimization of sphere function f(x) = sum(x^2)."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-5, 5), (-5, 5)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            popsize=20,
            maxiter=100,
            device="cpu",
        )

        def sphere(x):
            return (x**2).sum(dim=1)

        best_params, best_fitness, n_iter = de.optimize(sphere)

        # Optimum is at (0, 0) with f=0
        assert best_fitness < 0.1
        assert torch.all(torch.abs(best_params) < 0.5)

    def test_rosenbrock_2d(self):
        """Test optimization of 2D Rosenbrock function."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-2, 2), (-2, 2)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            popsize=30,
            maxiter=200,
            device="cpu",
        )

        def rosenbrock(x):
            # f(x,y) = (1-x)^2 + 100*(y-x^2)^2
            return (1 - x[:, 0]) ** 2 + 100 * (x[:, 1] - x[:, 0] ** 2) ** 2

        best_params, best_fitness, n_iter = de.optimize(rosenbrock)

        # Optimum is at (1, 1) with f=0
        assert best_fitness < 1.0  # Rosenbrock is hard, allow some tolerance
        assert torch.abs(best_params[0] - 1.0) < 0.5
        assert torch.abs(best_params[1] - 1.0) < 0.5

    def test_with_initial_guess(self):
        """Test optimization with initial guess (x0)."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-10, 10), (-10, 10)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            popsize=20,
            maxiter=50,
            device="cpu",
        )

        def sphere(x):
            return (x**2).sum(dim=1)

        # Start near optimum
        x0 = torch.tensor([0.1, 0.1], dtype=torch.float64)
        best_params, best_fitness, n_iter = de.optimize(sphere, x0=x0)

        # Should converge quickly to optimum
        assert best_fitness < 0.01

    def test_returns_correct_types(self):
        """Test that optimize returns correct types."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-1, 1)]
        de = BatchedDifferentialEvolution(
            bounds=bounds, popsize=10, maxiter=10, device="cpu"
        )

        def simple(x):
            return x[:, 0] ** 2

        best_params, best_fitness, n_iter = de.optimize(simple)

        assert isinstance(best_params, torch.Tensor)
        assert isinstance(best_fitness, float)
        assert isinstance(n_iter, int)
        assert best_params.shape == (1,)


class TestBatchedDEMultiSpectrum:
    """Tests for multi-spectrum optimization."""

    def test_multi_spectrum_shape(self):
        """Test multi-spectrum optimization returns correct shapes."""
        from src.utils.optimization import BatchedDifferentialEvolution

        N_spectra = 5
        bounds = [(-5, 5), (-5, 5)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=N_spectra,
            popsize=15,
            maxiter=30,
            device="cpu",
        )

        def multi_sphere(x):
            # x: (N_spectra, popsize, n_params)
            return (x**2).sum(dim=2)  # -> (N_spectra, popsize)

        best_params, best_fitness, n_iter = de.optimize(multi_sphere)

        assert best_params.shape == (N_spectra, 2)
        assert best_fitness.shape == (N_spectra,)
        assert isinstance(n_iter, int)

    def test_independent_spectra_optimization(self):
        """Test that each spectrum is optimized independently."""
        from src.utils.optimization import BatchedDifferentialEvolution

        N_spectra = 3
        bounds = [(-10, 10)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=N_spectra,
            popsize=20,
            maxiter=50,
            device="cpu",
        )

        # Each spectrum has a different optimum
        targets = torch.tensor([2.0, -3.0, 5.0], dtype=torch.float64)

        def multi_objective(x):
            # x: (N_spectra, popsize, 1)
            # targets: (N_spectra,) -> (N_spectra, 1, 1)
            t = targets.view(N_spectra, 1, 1)
            return ((x - t) ** 2).sum(dim=2)  # -> (N_spectra, popsize)

        best_params, best_fitness, n_iter = de.optimize(multi_objective)

        # Each should converge to its target
        for i in range(N_spectra):
            assert torch.abs(best_params[i, 0] - targets[i]) < 0.5

    def test_multi_spectrum_with_x0(self):
        """Test multi-spectrum with initial guesses."""
        from src.utils.optimization import BatchedDifferentialEvolution

        N_spectra = 4
        bounds = [(-5, 5), (-5, 5)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=N_spectra,
            popsize=15,
            maxiter=30,
            device="cpu",
        )

        def multi_sphere(x):
            return (x**2).sum(dim=2)

        # Initial guesses near optimum
        x0 = torch.randn(N_spectra, 2, dtype=torch.float64) * 0.1

        best_params, best_fitness, n_iter = de.optimize(multi_sphere, x0=x0)

        assert best_params.shape == (N_spectra, 2)
        # Should all be near (0, 0)
        assert torch.all(best_fitness < 0.1)


class TestBatchedDEConvergence:
    """Tests for convergence behavior."""

    def test_early_stopping(self):
        """Test that optimization stops early when converged."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-1, 1)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            popsize=20,
            maxiter=1000,  # High max, but should stop early
            atol=1e-6,
            device="cpu",
        )

        def trivial(x):
            return x[:, 0] ** 2

        _, _, n_iter = de.optimize(trivial)

        # Should converge well before maxiter
        assert n_iter < 500

    def test_respects_maxiter(self):
        """Test that optimization respects maxiter limit."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-10, 10), (-10, 10)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            popsize=10,
            maxiter=20,
            atol=1e-20,  # Very tight tolerance to prevent early stopping
            device="cpu",
        )

        def hard_function(x):
            # Rastrigin-like function (hard to optimize)
            A = 10
            return A * 2 + (x**2 - A * torch.cos(2 * np.pi * x)).sum(dim=1)

        _, _, n_iter = de.optimize(hard_function)

        assert n_iter <= 20


class TestBatchedDEBoundsHandling:
    """Tests for parameter bounds handling."""

    def test_population_within_bounds(self):
        """Test that population stays within bounds during optimization."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-2, 3), (0, 10)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            popsize=30,
            maxiter=50,
            device="cpu",
        )

        populations_checked = []

        def objective_with_check(x):
            # Verify bounds during optimization
            assert torch.all(x[:, 0] >= -2) and torch.all(x[:, 0] <= 3)
            assert torch.all(x[:, 1] >= 0) and torch.all(x[:, 1] <= 10)
            populations_checked.append(True)
            return (x**2).sum(dim=1)

        de.optimize(objective_with_check)

        # Should have checked multiple times
        assert len(populations_checked) > 10

    def test_result_within_bounds(self):
        """Test that final result is within bounds."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-5, 5), (-3, 7)]
        de = BatchedDifferentialEvolution(
            bounds=bounds, popsize=20, maxiter=30, device="cpu"
        )

        def sphere(x):
            return (x**2).sum(dim=1)

        best_params, _, _ = de.optimize(sphere)

        assert best_params[0] >= -5 and best_params[0] <= 5
        assert best_params[1] >= -3 and best_params[1] <= 7


class TestBatchedDEDeterminism:
    """Tests for reproducibility with seeds."""

    def test_same_seed_same_result(self):
        """Test that same seed gives same result."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-5, 5), (-5, 5)]

        def sphere(x):
            return (x**2).sum(dim=1)

        # First run
        de1 = BatchedDifferentialEvolution(
            bounds=bounds, popsize=20, maxiter=50, seed=42, device="cpu"
        )
        params1, fitness1, _ = de1.optimize(sphere)

        # Second run with same seed
        de2 = BatchedDifferentialEvolution(
            bounds=bounds, popsize=20, maxiter=50, seed=42, device="cpu"
        )
        params2, fitness2, _ = de2.optimize(sphere)

        torch.testing.assert_close(params1, params2)
        assert fitness1 == fitness2

    def test_different_seed_different_result(self):
        """Test that different seeds give different results."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-5, 5), (-5, 5)]

        def sphere(x):
            return (x**2).sum(dim=1)

        de1 = BatchedDifferentialEvolution(
            bounds=bounds, popsize=20, maxiter=30, seed=42, device="cpu"
        )
        params1, _, _ = de1.optimize(sphere)

        de2 = BatchedDifferentialEvolution(
            bounds=bounds, popsize=20, maxiter=30, seed=123, device="cpu"
        )
        params2, _, _ = de2.optimize(sphere)

        # Results should differ (though both should be near optimum)
        # This is probabilistic but very likely with different seeds
        assert not torch.allclose(params1, params2, atol=1e-6)


class TestBackwardsCompatibilityAliases:
    """Tests for backwards compatibility aliases."""

    def test_gpu_differential_evolution_alias(self):
        """Test GPUDifferentialEvolution alias exists."""
        from src.utils.optimization import (
            BatchedDifferentialEvolution,
            GPUDifferentialEvolution,
        )

        assert GPUDifferentialEvolution is BatchedDifferentialEvolution

    def test_kappa_differential_evolution_alias(self):
        """Test KappaDifferentialEvolution alias exists."""
        from src.utils.optimization import (
            BatchedDifferentialEvolution,
            KappaDifferentialEvolution,
        )

        assert KappaDifferentialEvolution is BatchedDifferentialEvolution


@pytest.mark.skipif(
    not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available"
)
class TestBatchedDECUDA:
    """Tests for CUDA execution."""

    def test_cuda_optimization(self):
        """Test optimization runs on CUDA."""
        from src.utils.optimization import BatchedDifferentialEvolution

        bounds = [(-5, 5), (-5, 5)]
        de = BatchedDifferentialEvolution(
            bounds=bounds, popsize=20, maxiter=30, device="cuda"
        )

        def sphere(x):
            return (x**2).sum(dim=1)

        best_params, best_fitness, _ = de.optimize(sphere)

        assert best_params.device.type == "cuda"
        assert best_fitness < 0.5

    def test_cuda_multi_spectrum(self):
        """Test multi-spectrum optimization on CUDA."""
        from src.utils.optimization import BatchedDifferentialEvolution

        N_spectra = 10
        bounds = [(-5, 5)]
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=N_spectra,
            popsize=15,
            maxiter=30,
            device="cuda",
        )

        def multi_sphere(x):
            return (x**2).sum(dim=2)

        best_params, best_fitness, _ = de.optimize(multi_sphere)

        assert best_params.device.type == "cuda"
        assert best_params.shape == (N_spectra, 1)
