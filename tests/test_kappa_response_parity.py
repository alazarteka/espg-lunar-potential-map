"""CPU/torch parity guards for the duplicated kappa response-matrix physics.

The response matrix and gamma-ratio formulas are intentionally implemented
twice: once for the NumPy/Numba CPU path
(``src.kappa.Kappa.build_log_energy_response_matrix``,
``src.physics.kappa._gamma_ratio``) and once for the PyTorch GPU hot path
(``src.kappa_torch.build_response_matrix_torch``,
``src.kappa_torch._torch_gamma_ratio``). They cannot share a kernel
(numba vs torch), so these tests pin the two implementations to each other
on identical inputs to catch numerical drift if either side changes.
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

# LP ER sweeps have 15 log-spaced energy bins spanning roughly 10 eV - 20 keV.
_ENERGY_GRID = np.geomspace(10.0, 2.0e4, 15)


class TestResponseMatrixParity:
    """Kappa.build_log_energy_response_matrix vs build_response_matrix_torch."""

    @pytest.mark.parametrize("width", [0.1, 0.25, 0.5, 1.0])
    def test_parity_float64_across_widths(self, width: float):
        """Both paths produce the same matrix on identical inputs (float64)."""
        from src.kappa import Kappa
        from src.kappa_torch import build_response_matrix_torch

        cpu = Kappa.build_log_energy_response_matrix(
            _ENERGY_GRID, energy_window_width_relative=width
        )
        torch_result = build_response_matrix_torch(
            torch.tensor(_ENERGY_GRID, dtype=torch.float64),
            energy_window_width_relative=width,
        )

        assert cpu.shape == tuple(torch_result.shape)
        np.testing.assert_allclose(
            cpu, torch_result.cpu().numpy(), rtol=1e-10, atol=1e-14
        )

    def test_parity_with_default_width(self):
        """Calling both with their *defaults* agrees (guards default drift)."""
        from src.kappa import Kappa
        from src.kappa_torch import build_response_matrix_torch

        cpu = Kappa.build_log_energy_response_matrix(_ENERGY_GRID)
        torch_result = build_response_matrix_torch(
            torch.tensor(_ENERGY_GRID, dtype=torch.float64)
        )

        np.testing.assert_allclose(
            cpu, torch_result.cpu().numpy(), rtol=1e-10, atol=1e-14
        )

    def test_parity_float32_gpu_dtype(self):
        """Torch float32 (typical GPU dtype) stays close to the CPU float64."""
        from src.kappa import Kappa
        from src.kappa_torch import build_response_matrix_torch

        cpu = Kappa.build_log_energy_response_matrix(_ENERGY_GRID)
        torch_result = build_response_matrix_torch(
            torch.tensor(_ENERGY_GRID, dtype=torch.float32)
        )

        np.testing.assert_allclose(
            cpu, torch_result.cpu().numpy().astype(np.float64), rtol=1e-5, atol=1e-7
        )

    def test_parity_on_dense_grid(self):
        """Parity also holds on a denser, wider energy grid."""
        from src import config
        from src.kappa import Kappa
        from src.kappa_torch import build_response_matrix_torch

        energy = np.geomspace(1.0, 1.0e5, 64)
        cpu = Kappa.build_log_energy_response_matrix(
            energy,
            energy_window_width_relative=config.ENERGY_WINDOW_WIDTH_RELATIVE,
        )
        torch_result = build_response_matrix_torch(
            torch.tensor(energy, dtype=torch.float64),
            energy_window_width_relative=config.ENERGY_WINDOW_WIDTH_RELATIVE,
        )

        np.testing.assert_allclose(
            cpu, torch_result.cpu().numpy(), rtol=1e-10, atol=1e-14
        )


class TestGammaRatioParity:
    """physics.kappa._gamma_ratio (numba) vs kappa_torch._torch_gamma_ratio."""

    def test_parity_over_kappa_range(self):
        """Both paths agree over the physical kappa range (float64)."""
        from src.kappa_torch import _torch_gamma_ratio
        from src.physics.kappa import _gamma_ratio

        # Span beyond the fit bounds [2.5, 6.0]; kappa must exceed 1.5.
        kappa_values = np.concatenate(
            [np.linspace(1.6, 10.0, 43), np.array([2.5, 6.0])]
        )

        cpu = np.array([_gamma_ratio(float(k)) for k in kappa_values])
        torch_result = _torch_gamma_ratio(
            torch.tensor(kappa_values, dtype=torch.float64)
        )

        np.testing.assert_allclose(cpu, torch_result.cpu().numpy(), rtol=1e-10)

    def test_parity_batched_shape(self):
        """The (N, P)-shaped torch path matches per-scalar numba results."""
        from src.kappa_torch import _torch_gamma_ratio
        from src.physics.kappa import _gamma_ratio

        kappa_grid = np.linspace(2.5, 6.0, 12).reshape(3, 4)

        cpu = np.array([[_gamma_ratio(float(k)) for k in row] for row in kappa_grid])
        torch_result = _torch_gamma_ratio(torch.tensor(kappa_grid, dtype=torch.float64))

        assert tuple(torch_result.shape) == kappa_grid.shape
        np.testing.assert_allclose(cpu, torch_result.cpu().numpy(), rtol=1e-10)
