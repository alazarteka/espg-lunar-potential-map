"""Tests for PyTorch-accelerated loss-cone model.

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


class TestSynthLossconeBatchTorch:
    """Tests for synth_losscone_batch_torch."""

    def test_output_shape(self):
        """Test output has correct shape."""
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch = 20, 15
        n_params = 10

        energy_grid = torch.logspace(1, 4, nE, dtype=torch.float64)
        pitch_grid = torch.linspace(0, 180, nPitch).unsqueeze(0).expand(nE, -1)
        pitch_grid = pitch_grid.clone().to(dtype=torch.float64)
        U_surface = torch.linspace(-100, -10, n_params, dtype=torch.float64)

        model = synth_losscone_batch_torch(energy_grid, pitch_grid, U_surface)

        assert model.shape == (n_params, nE, nPitch)

    def test_loss_cone_structure(self):
        """Test that model has loss cone structure: 1.0 inside, background outside."""
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch = 20, 30
        energy_grid = torch.logspace(1, 3, nE, dtype=torch.float64)
        pitch_grid = torch.linspace(0, 180, nPitch).unsqueeze(0).expand(nE, -1)
        pitch_grid = pitch_grid.clone().to(dtype=torch.float64)

        # Moderate negative surface potential
        U_surface = torch.tensor([-50.0], dtype=torch.float64)

        model = synth_losscone_batch_torch(energy_grid, pitch_grid, U_surface)

        # Check we have both 1.0 (inside cone) and background (outside)
        assert torch.any(model == 1.0)
        assert torch.any(torch.abs(model - 0.05) < 0.01)  # background ~ 0.05

    def test_equivalence_with_numpy(self):
        """Test PyTorch implementation matches NumPy."""
        from src.model import synth_losscone_batch as synth_losscone_batch_np
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch = 15, 20
        n_params = 5

        # Create grids
        energy_np = np.logspace(1, 4, nE)
        pitch_np = np.tile(np.linspace(0, 180, nPitch), (nE, 1))

        energy_torch = torch.tensor(energy_np, dtype=torch.float64)
        pitch_torch = torch.tensor(pitch_np, dtype=torch.float64)

        # Random parameters
        np.random.seed(42)
        U_surface_np = np.random.uniform(-200, -10, n_params)
        bs_over_bm_np = np.random.uniform(0.3, 0.9, n_params)

        U_surface_torch = torch.tensor(U_surface_np, dtype=torch.float64)
        bs_over_bm_torch = torch.tensor(bs_over_bm_np, dtype=torch.float64)

        # Compute with both implementations
        model_np = synth_losscone_batch_np(
            energy_np, pitch_np, U_surface_np, bs_over_bm=bs_over_bm_np
        )
        model_torch = synth_losscone_batch_torch(
            energy_torch, pitch_torch, U_surface_torch, bs_over_bm=bs_over_bm_torch
        )

        np.testing.assert_allclose(
            model_torch.numpy(), model_np, rtol=1e-10, atol=1e-10
        )

    def test_equivalence_with_beam(self):
        """Test PyTorch matches NumPy with secondary electron beam."""
        from src.model import synth_losscone_batch as synth_losscone_batch_np
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch = 15, 20
        n_params = 3

        energy_np = np.logspace(1, 3, nE)
        pitch_np = np.tile(np.linspace(0, 180, nPitch), (nE, 1))

        energy_torch = torch.tensor(energy_np, dtype=torch.float64)
        pitch_torch = torch.tensor(pitch_np, dtype=torch.float64)

        U_surface_np = np.array([-50.0, -100.0, -150.0])
        bs_over_bm_np = np.array([0.5, 0.6, 0.7])
        beam_amp_np = np.array([2.0, 3.0, 1.5])
        beam_width_np = np.array([10.0, 15.0, 20.0])

        U_surface_torch = torch.tensor(U_surface_np, dtype=torch.float64)
        bs_over_bm_torch = torch.tensor(bs_over_bm_np, dtype=torch.float64)
        beam_amp_torch = torch.tensor(beam_amp_np, dtype=torch.float64)
        beam_width_torch = torch.tensor(beam_width_np, dtype=torch.float64)

        model_np = synth_losscone_batch_np(
            energy_np,
            pitch_np,
            U_surface_np,
            bs_over_bm=bs_over_bm_np,
            beam_amp=beam_amp_np,
            beam_width_eV=beam_width_np,
            beam_pitch_sigma_deg=10.0,
        )
        model_torch = synth_losscone_batch_torch(
            energy_torch,
            pitch_torch,
            U_surface_torch,
            bs_over_bm=bs_over_bm_torch,
            beam_amp=beam_amp_torch,
            beam_width_eV=beam_width_torch,
            beam_pitch_sigma_deg=10.0,
        )

        np.testing.assert_allclose(
            model_torch.numpy(), model_np, rtol=1e-10, atol=1e-10
        )

    def test_spacecraft_potential_scalar(self):
        """Test with scalar spacecraft potential."""
        from src.model import synth_losscone_batch as synth_losscone_batch_np
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch = 10, 15
        energy_np = np.logspace(1, 3, nE)
        pitch_np = np.tile(np.linspace(0, 180, nPitch), (nE, 1))

        energy_torch = torch.tensor(energy_np, dtype=torch.float64)
        pitch_torch = torch.tensor(pitch_np, dtype=torch.float64)

        U_surface_np = np.array([-50.0, -100.0])
        U_surface_torch = torch.tensor(U_surface_np, dtype=torch.float64)

        model_np = synth_losscone_batch_np(
            energy_np, pitch_np, U_surface_np, U_spacecraft=-5.0
        )
        model_torch = synth_losscone_batch_torch(
            energy_torch, pitch_torch, U_surface_torch, U_spacecraft=-5.0
        )

        np.testing.assert_allclose(
            model_torch.numpy(), model_np, rtol=1e-10, atol=1e-10
        )

    def test_spacecraft_potential_array(self):
        """Test with per-energy spacecraft potential."""
        from src.model import synth_losscone_batch as synth_losscone_batch_np
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch = 10, 15
        energy_np = np.logspace(1, 3, nE)
        pitch_np = np.tile(np.linspace(0, 180, nPitch), (nE, 1))
        U_sc_np = np.linspace(-10, -2, nE)

        energy_torch = torch.tensor(energy_np, dtype=torch.float64)
        pitch_torch = torch.tensor(pitch_np, dtype=torch.float64)
        U_sc_torch = torch.tensor(U_sc_np, dtype=torch.float64)

        U_surface_np = np.array([-50.0])
        U_surface_torch = torch.tensor(U_surface_np, dtype=torch.float64)

        model_np = synth_losscone_batch_np(
            energy_np, pitch_np, U_surface_np, U_spacecraft=U_sc_np
        )
        model_torch = synth_losscone_batch_torch(
            energy_torch, pitch_torch, U_surface_torch, U_spacecraft=U_sc_torch
        )

        np.testing.assert_allclose(
            model_torch.numpy(), model_np, rtol=1e-10, atol=1e-10
        )


class TestComputeChi2BatchTorch:
    """Tests for compute_chi2_batch_torch."""

    def test_chi2_zero_for_perfect_match(self):
        """Test chi² is zero when model equals data."""
        from src.model_torch import compute_chi2_batch_torch

        n_params, nE, nPitch = 3, 10, 15
        model = torch.rand(n_params, nE, nPitch, dtype=torch.float64) + 0.1
        data = model[0].clone()  # Perfect match for first param set
        mask = torch.ones(nE, nPitch, dtype=torch.bool)

        chi2 = compute_chi2_batch_torch(model, data, mask)

        assert chi2.shape == (n_params,)
        assert chi2[0].item() < 1e-10  # Should be ~0 for first

    def test_chi2_positive_for_mismatch(self):
        """Test chi² is positive for mismatched model/data."""
        from src.model_torch import compute_chi2_batch_torch

        n_params, nE, nPitch = 3, 10, 15
        model = torch.rand(n_params, nE, nPitch, dtype=torch.float64) + 0.1
        data = torch.rand(nE, nPitch, dtype=torch.float64) + 0.1
        mask = torch.ones(nE, nPitch, dtype=torch.bool)

        chi2 = compute_chi2_batch_torch(model, data, mask)

        assert torch.all(chi2 >= 0)

    def test_masked_points_ignored(self):
        """Test that masked points don't contribute to chi²."""
        from src.model_torch import compute_chi2_batch_torch

        n_params, nE, nPitch = 2, 5, 5
        model = torch.ones(n_params, nE, nPitch, dtype=torch.float64)
        data = torch.ones(nE, nPitch, dtype=torch.float64) * 10  # Very different

        # Mask everything
        mask = torch.zeros(nE, nPitch, dtype=torch.bool)

        chi2 = compute_chi2_batch_torch(model, data, mask)

        # With all masked, chi2 should be 0
        torch.testing.assert_close(
            chi2, torch.zeros(n_params, dtype=torch.float64), atol=1e-10, rtol=0
        )


class TestSynthLossconeMultiChunkTorch:
    """Tests for synth_losscone_multi_chunk_torch."""

    def test_output_shape(self):
        """Test multi-chunk output has correct shape."""
        from src.model_torch import synth_losscone_multi_chunk_torch

        N_chunks, nE, nPitch, n_pop = 4, 15, 20, 10

        energy_grids = torch.logspace(1, 4, nE, dtype=torch.float64).unsqueeze(0)
        energy_grids = energy_grids.expand(N_chunks, -1).clone()

        pitch_grid = torch.linspace(0, 180, nPitch).unsqueeze(0).expand(nE, -1)
        pitch_grids = pitch_grid.unsqueeze(0).expand(N_chunks, -1, -1).clone()
        pitch_grids = pitch_grids.to(dtype=torch.float64)

        U_surface = torch.rand(N_chunks, n_pop, dtype=torch.float64) * -100 - 10

        model = synth_losscone_multi_chunk_torch(energy_grids, pitch_grids, U_surface)

        assert model.shape == (N_chunks, n_pop, nE, nPitch)

    def test_multi_chunk_matches_single_chunk(self):
        """Test multi-chunk gives same results as single-chunk loop."""
        from src.model_torch import (
            synth_losscone_batch_torch,
            synth_losscone_multi_chunk_torch,
        )

        N_chunks, nE, nPitch, n_pop = 3, 10, 12, 5

        # Same energy/pitch for all chunks (simpler test)
        energy_1d = torch.logspace(1, 3, nE, dtype=torch.float64)
        pitch_2d = torch.linspace(0, 180, nPitch).unsqueeze(0).expand(nE, -1)
        pitch_2d = pitch_2d.clone().to(dtype=torch.float64)

        energy_grids = energy_1d.unsqueeze(0).expand(N_chunks, -1).clone()
        pitch_grids = pitch_2d.unsqueeze(0).expand(N_chunks, -1, -1).clone()

        # Different U_surface per chunk
        torch.manual_seed(42)
        U_surface = torch.rand(N_chunks, n_pop, dtype=torch.float64) * -100 - 10
        bs_over_bm = torch.rand(N_chunks, n_pop, dtype=torch.float64) * 0.5 + 0.3

        # Multi-chunk version
        model_multi = synth_losscone_multi_chunk_torch(
            energy_grids, pitch_grids, U_surface, bs_over_bm=bs_over_bm
        )

        # Single-chunk loop
        model_single = []
        for i in range(N_chunks):
            m = synth_losscone_batch_torch(
                energy_1d, pitch_2d, U_surface[i], bs_over_bm=bs_over_bm[i]
            )
            model_single.append(m)
        model_single = torch.stack(model_single)

        torch.testing.assert_close(model_multi, model_single, rtol=1e-10, atol=1e-10)


class TestComputeChi2MultiChunkTorch:
    """Tests for compute_chi2_multi_chunk_torch."""

    def test_output_shape(self):
        """Test multi-chunk chi² has correct shape."""
        from src.model_torch import compute_chi2_multi_chunk_torch

        N_chunks, n_pop, nE, nPitch = 4, 8, 10, 12

        models = torch.rand(N_chunks, n_pop, nE, nPitch, dtype=torch.float64) + 0.1
        data = torch.rand(N_chunks, nE, nPitch, dtype=torch.float64) + 0.1
        mask = torch.ones(N_chunks, nE, nPitch, dtype=torch.bool)

        chi2 = compute_chi2_multi_chunk_torch(models, data, mask)

        assert chi2.shape == (N_chunks, n_pop)

    def test_multi_chunk_chi2_matches_single(self):
        """Test multi-chunk chi² matches single-chunk loop."""
        from src.model_torch import (
            compute_chi2_batch_torch,
            compute_chi2_multi_chunk_torch,
        )

        N_chunks, n_pop, nE, nPitch = 3, 5, 8, 10

        torch.manual_seed(123)
        models = torch.rand(N_chunks, n_pop, nE, nPitch, dtype=torch.float64) + 0.1
        data = torch.rand(N_chunks, nE, nPitch, dtype=torch.float64) + 0.1
        mask = torch.rand(N_chunks, nE, nPitch) > 0.2  # Random mask

        # Multi-chunk
        chi2_multi = compute_chi2_multi_chunk_torch(models, data, mask)

        # Single-chunk loop
        chi2_single = []
        for i in range(N_chunks):
            c = compute_chi2_batch_torch(models[i], data[i], mask[i])
            chi2_single.append(c)
        chi2_single = torch.stack(chi2_single)

        torch.testing.assert_close(chi2_multi, chi2_single, rtol=1e-10, atol=1e-10)


class TestDeviceHandling:
    """Tests for device handling."""

    def test_cpu_device(self):
        """Test operations work on CPU."""
        from src.model_torch import synth_losscone_batch_torch

        energy = torch.logspace(1, 3, 10, dtype=torch.float64, device="cpu")
        pitch = torch.linspace(0, 180, 15).unsqueeze(0).expand(10, -1)
        pitch = pitch.clone().to(dtype=torch.float64, device="cpu")
        U_surface = torch.tensor([-50.0], dtype=torch.float64, device="cpu")

        model = synth_losscone_batch_torch(energy, pitch, U_surface)

        assert model.device == torch.device("cpu")

    @pytest.mark.skipif(
        not (HAS_TORCH and torch.cuda.is_available()), reason="CUDA not available"
    )
    def test_cuda_device(self):
        """Test operations work on CUDA."""
        from src.model_torch import synth_losscone_batch_torch

        energy = torch.logspace(1, 3, 10, dtype=torch.float32, device="cuda")
        pitch = torch.linspace(0, 180, 15, device="cuda").unsqueeze(0).expand(10, -1)
        pitch = pitch.clone().to(dtype=torch.float32)
        U_surface = torch.tensor([-50.0], dtype=torch.float32, device="cuda")

        model = synth_losscone_batch_torch(energy, pitch, U_surface)

        assert model.device.type == "cuda"


class TestFloat32Precision:
    """Tests for float32 precision (typical GPU dtype)."""

    def test_float32_equivalence(self):
        """Test float32 gives reasonable results compared to float64."""
        from src.model_torch import synth_losscone_batch_torch

        nE, nPitch, n_params = 15, 20, 5

        energy_64 = torch.logspace(1, 3, nE, dtype=torch.float64)
        pitch_64 = torch.linspace(0, 180, nPitch).unsqueeze(0).expand(nE, -1)
        pitch_64 = pitch_64.clone().to(dtype=torch.float64)
        U_surface_64 = torch.linspace(-100, -20, n_params, dtype=torch.float64)

        energy_32 = energy_64.to(torch.float32)
        pitch_32 = pitch_64.to(torch.float32)
        U_surface_32 = U_surface_64.to(torch.float32)

        model_64 = synth_losscone_batch_torch(energy_64, pitch_64, U_surface_64)
        model_32 = synth_losscone_batch_torch(energy_32, pitch_32, U_surface_32)

        # Should be close within float32 precision
        np.testing.assert_allclose(
            model_32.numpy(), model_64.numpy(), rtol=1e-5, atol=1e-5
        )
