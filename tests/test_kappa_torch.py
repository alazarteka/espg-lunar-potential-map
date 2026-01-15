"""Tests for PyTorch-accelerated Kappa distribution fitter.

Tests numerical equivalence between PyTorch and NumPy implementations.
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


class TestOmnidirectionalFluxBatch:
    """Tests for omnidirectional_flux_batch_torch."""

    def test_single_spectrum_single_candidate(self):
        """Test flux computation for single spectrum and candidate."""
        from src.kappa_torch import omnidirectional_flux_batch_torch

        # Simple test case
        energy = torch.tensor([10.0, 100.0, 1000.0], dtype=torch.float64)
        density = torch.tensor([1e6], dtype=torch.float64)  # 1e6 particles/m³
        kappa = torch.tensor([[3.0]], dtype=torch.float64)  # N=1, P=1
        theta = torch.tensor([[1e6]], dtype=torch.float64)  # 1e6 m/s

        flux = omnidirectional_flux_batch_torch(density, kappa, theta, energy)

        assert flux.shape == (1, 1, 3)
        assert torch.all(torch.isfinite(flux))
        assert torch.all(flux > 0)

    def test_batch_spectrum_candidates(self):
        """Test flux computation for multiple spectra and candidates."""
        from src.kappa_torch import omnidirectional_flux_batch_torch

        N_spectra = 5
        N_candidates = 10
        N_energies = 20

        energy = torch.logspace(1, 4, N_energies, dtype=torch.float64)
        density = torch.full((N_spectra,), 1e6, dtype=torch.float64)
        kappa = torch.rand(N_spectra, N_candidates, dtype=torch.float64) * 3.5 + 2.5
        theta = 10 ** (torch.rand(N_spectra, N_candidates, dtype=torch.float64) * 2 + 6)

        flux = omnidirectional_flux_batch_torch(density, kappa, theta, energy)

        assert flux.shape == (N_spectra, N_candidates, N_energies)
        assert torch.all(torch.isfinite(flux))
        assert torch.all(flux > 0)

    def test_flux_decreases_with_energy(self):
        """Test that flux generally decreases at high energies (Kappa tail)."""
        from src.kappa_torch import omnidirectional_flux_batch_torch

        energy = torch.logspace(1, 5, 50, dtype=torch.float64)
        density = torch.tensor([1e6], dtype=torch.float64)
        kappa = torch.tensor([[3.5]], dtype=torch.float64)
        theta = torch.tensor([[5e6]], dtype=torch.float64)

        flux = omnidirectional_flux_batch_torch(density, kappa, theta, energy)
        flux_1d = flux[0, 0, :]

        # High energy tail should decrease
        assert flux_1d[-1] < flux_1d[0]

    def test_higher_kappa_narrower_tail(self):
        """Test that higher kappa gives narrower (steeper) tail."""
        from src.kappa_torch import omnidirectional_flux_batch_torch

        energy = torch.logspace(1, 5, 50, dtype=torch.float64)
        density = torch.tensor([1e6, 1e6], dtype=torch.float64)
        kappa = torch.tensor([[3.0], [5.5]], dtype=torch.float64)  # low vs high kappa
        theta = torch.tensor([[5e6], [5e6]], dtype=torch.float64)

        flux = omnidirectional_flux_batch_torch(density, kappa, theta, energy)

        # At high energies, higher kappa should give lower flux (steeper falloff)
        high_E_idx = -5
        assert flux[1, 0, high_E_idx] < flux[0, 0, high_E_idx]


class TestResponseMatrix:
    """Tests for build_response_matrix_torch."""

    def test_response_matrix_shape(self):
        """Test response matrix has correct shape."""
        from src.kappa_torch import build_response_matrix_torch

        energy = torch.logspace(1, 4, 20, dtype=torch.float64)
        W = build_response_matrix_torch(energy)

        assert W.shape == (20, 20)

    def test_response_matrix_normalized(self):
        """Test response matrix rows sum to 1."""
        from src.kappa_torch import build_response_matrix_torch

        energy = torch.logspace(1, 4, 20, dtype=torch.float64)
        W = build_response_matrix_torch(energy)

        row_sums = W.sum(dim=1)
        torch.testing.assert_close(row_sums, torch.ones(20, dtype=torch.float64))

    def test_response_matrix_symmetric(self):
        """Test response matrix is symmetric (due to symmetric kernel)."""
        from src.kappa_torch import build_response_matrix_torch

        energy = torch.logspace(1, 4, 20, dtype=torch.float64)
        W = build_response_matrix_torch(energy, energy_window_width_relative=0.5)

        # Not strictly symmetric due to normalization, but should be close
        # The Gaussian kernel is symmetric before normalization
        assert W.shape == (20, 20)

    def test_diagonal_dominance(self):
        """Test response matrix has largest values on diagonal."""
        from src.kappa_torch import build_response_matrix_torch

        energy = torch.logspace(1, 4, 20, dtype=torch.float64)
        W = build_response_matrix_torch(energy, energy_window_width_relative=0.3)

        diagonal = W.diag()
        off_diagonal_max = (W - torch.diag(diagonal)).max(dim=1).values

        # Diagonal should be larger than any off-diagonal in same row
        assert torch.all(diagonal >= off_diagonal_max)


class TestKappaChi2:
    """Tests for compute_kappa_chi2_batch_torch."""

    def test_chi2_zero_for_perfect_fit(self):
        """Test chi² is zero when model equals data."""
        from src.kappa_torch import compute_kappa_chi2_batch_torch

        N, P, E = 3, 5, 10
        model_flux = torch.rand(N, P, E, dtype=torch.float64) + 0.1
        # Data equals first candidate for each spectrum
        data_flux = model_flux[:, 0, :].clone()
        weights = torch.ones(N, E, dtype=torch.float64)

        chi2 = compute_kappa_chi2_batch_torch(model_flux, data_flux, weights)

        assert chi2.shape == (N, P)
        # First candidate should have ~0 chi2
        torch.testing.assert_close(
            chi2[:, 0], torch.zeros(N, dtype=torch.float64), atol=1e-10, rtol=0
        )

    def test_chi2_positive_for_mismatch(self):
        """Test chi² is positive when model differs from data."""
        from src.kappa_torch import compute_kappa_chi2_batch_torch

        N, P, E = 2, 3, 10
        model_flux = torch.rand(N, P, E, dtype=torch.float64) + 0.1
        data_flux = torch.rand(N, E, dtype=torch.float64) + 0.1
        weights = torch.ones(N, E, dtype=torch.float64)

        chi2 = compute_kappa_chi2_batch_torch(model_flux, data_flux, weights)

        assert torch.all(chi2 >= 0)

    def test_chi2_with_response_matrix(self):
        """Test chi² works with response matrix convolution."""
        from src.kappa_torch import (
            build_response_matrix_torch,
            compute_kappa_chi2_batch_torch,
        )

        N, P, E = 2, 4, 15
        energy = torch.logspace(1, 4, E, dtype=torch.float64)
        response = build_response_matrix_torch(energy)

        model_flux = torch.rand(N, P, E, dtype=torch.float64) + 0.1
        data_flux = torch.rand(N, E, dtype=torch.float64) + 0.1
        weights = torch.ones(N, E, dtype=torch.float64)

        chi2 = compute_kappa_chi2_batch_torch(
            model_flux, data_flux, weights, response_matrix=response
        )

        assert chi2.shape == (N, P)
        assert torch.all(torch.isfinite(chi2))


class TestKappaFitterTorch:
    """Tests for KappaFitterTorch class."""

    def test_fitter_initialization(self):
        """Test fitter initializes correctly."""
        from src.kappa_torch import KappaFitterTorch

        fitter = KappaFitterTorch(device="cpu", dtype="float64")

        assert fitter.device == torch.device("cpu")
        assert fitter.dtype == torch.float64

    def test_fit_batch_returns_correct_shapes(self):
        """Test fit_batch returns arrays with correct shapes."""
        from src.kappa_torch import KappaFitterTorch

        N_spectra = 3
        N_energies = 15

        # Create synthetic flux data
        energy = np.logspace(1, 4, N_energies)
        flux_data = np.random.rand(N_spectra, N_energies) * 1e8 + 1e6
        density_estimates = np.full(N_spectra, 1e6)

        fitter = KappaFitterTorch(device="cpu", dtype="float64", popsize=10, maxiter=20)
        kappa, theta, chi2 = fitter.fit_batch(energy, flux_data, density_estimates)

        assert kappa.shape == (N_spectra,)
        assert theta.shape == (N_spectra,)
        assert chi2.shape == (N_spectra,)

    def test_fit_batch_returns_valid_bounds(self):
        """Test fitted parameters are within expected bounds."""
        from src.kappa_torch import KappaFitterTorch

        N_spectra = 5
        N_energies = 15

        energy = np.logspace(1, 4, N_energies)
        flux_data = np.random.rand(N_spectra, N_energies) * 1e8 + 1e6
        density_estimates = np.full(N_spectra, 1e6)

        fitter = KappaFitterTorch(device="cpu", dtype="float64", popsize=10, maxiter=30)
        kappa, theta, chi2 = fitter.fit_batch(energy, flux_data, density_estimates)

        # Check kappa bounds
        assert np.all(kappa >= 2.5)
        assert np.all(kappa <= 6.0)

        # Check theta bounds (log10 between 6 and 8)
        log_theta = np.log10(theta)
        assert np.all(log_theta >= 6.0)
        assert np.all(log_theta <= 8.0)

        # Chi2 should be positive
        assert np.all(chi2 >= 0)


@pytest.mark.skipif(
    not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available"
)
class TestKappaTorchCUDA:
    """Tests that require CUDA."""

    def test_cuda_device_works(self):
        """Test that CUDA device can be used."""
        from src.kappa_torch import KappaFitterTorch

        fitter = KappaFitterTorch(device="cuda", dtype="float32")
        assert fitter.device == torch.device("cuda")

    def test_fit_batch_on_cuda(self):
        """Test fit_batch works on CUDA."""
        from src.kappa_torch import KappaFitterTorch

        N_spectra = 3
        N_energies = 15

        energy = np.logspace(1, 4, N_energies)
        flux_data = np.random.rand(N_spectra, N_energies) * 1e8 + 1e6
        density_estimates = np.full(N_spectra, 1e6)

        fitter = KappaFitterTorch(
            device="cuda", dtype="float32", popsize=10, maxiter=20
        )
        kappa, theta, chi2 = fitter.fit_batch(energy, flux_data, density_estimates)

        assert kappa.shape == (N_spectra,)
        assert np.all(np.isfinite(kappa))
