"""
PyTorch-accelerated Kappa distribution fitter.

Provides batched fitting of kappa distribution parameters across multiple
spectra simultaneously using vectorized tensor operations.

The key insight is that we can evaluate the forward model for all spectra
and all parameter candidates in parallel, then use Differential Evolution
to find the best parameters for each spectrum independently.
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from src import config

try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None  # type: ignore[misc, assignment]

from src.utils.optimization import BatchedDifferentialEvolution


def _auto_detect_dtype(device: torch.device) -> torch.dtype:
    """
    Auto-detect optimal dtype for the given device.

    Returns float16 on modern CUDA GPUs (Volta+, compute 7.0+) which have
    tensor cores for fast half-precision. Returns float32 for older GPUs
    and CPU where float16 would be slower.
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required")

    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        # Volta+ (compute 7.0+) has tensor cores for fast float16
        if props.major >= 7:
            return torch.float16
        return torch.float32
    return torch.float32  # CPU: float32 is fastest


def _torch_lgamma(x: Tensor) -> Tensor:
    """Compute log-gamma function."""
    return torch.lgamma(x)


def _torch_gamma_ratio(kappa: Tensor) -> Tensor:
    """
    Compute Γ(κ+1) / (π^1.5 · κ^1.5 · Γ(κ-0.5)) in log space for stability.

    Args:
        kappa: (N, P) or (P,) kappa values

    Returns:
        Gamma ratio with same shape as kappa
    """
    # log(Γ(κ+1)) - log(π^1.5) - 1.5*log(κ) - log(Γ(κ-0.5))
    log_ratio = (
        _torch_lgamma(kappa + 1)
        - 1.5 * math.log(math.pi)
        - 1.5 * torch.log(kappa)
        - _torch_lgamma(kappa - 0.5)
    )
    return torch.exp(log_ratio)


def omnidirectional_flux_batch_torch(
    density: Tensor,
    kappa: Tensor,
    theta: Tensor,
    energy: Tensor,
) -> Tensor:
    """
    Compute omnidirectional flux for batched parameters.

    Args:
        density: (N,) density estimates per spectrum [particles/m³]
        kappa: (N, P) kappa values for each spectrum and candidate
        theta: (N, P) thermal speed values [m/s]
        energy: (E,) energy grid [eV]

    Returns:
        (N, P, E) omnidirectional flux [particles/(cm² s eV)]
    """
    N, P = kappa.shape
    E = energy.shape[0]

    # v = sqrt(2E/m_e)
    velocity = torch.sqrt(2.0 * energy / config.ELECTRON_MASS_EV_S2_M2)

    density_exp = density.view(N, 1, 1)
    kappa_exp = kappa.view(N, P, 1)
    theta_exp = theta.view(N, P, 1)
    velocity_exp = velocity.view(1, 1, E)

    # Γ(κ+1) / (π^1.5 · κ^1.5 · Γ(κ-0.5))
    prefactor = _torch_gamma_ratio(kappa_exp)
    core = density_exp / (theta_exp**3)

    # (1 + (v/θ)² / κ)^(-κ-1)
    velocity_ratio_sq = (velocity_exp / theta_exp) ** 2
    tail = torch.pow(1.0 + velocity_ratio_sq / kappa_exp, -kappa_exp - 1.0)

    distribution = prefactor * core * tail
    # j(E) = f(v) · v² / m_e; J(E) = 4π · j · cm²→m²
    directional_flux = distribution * (velocity_exp**2) / config.ELECTRON_MASS_EV_S2_M2
    omni_flux = 4.0 * math.pi * config.CM2_TO_M2 * directional_flux

    return omni_flux


def build_response_matrix_torch(
    energy: Tensor,
    energy_window_width_relative: float = config.ENERGY_WINDOW_WIDTH_RELATIVE,
) -> Tensor:
    """
    Build log-energy response matrix for instrumental resolution effects.

    This implements a Gaussian convolution in log-energy space to account for
    finite instrumental energy resolution. The sigma is computed from the
    relative energy width using: s = asinh(0.5 * width) / sqrt(2 * ln(2))

    Note: A NumPy/Numba equivalent exists in
    kappa.Kappa.build_log_energy_response_matrix for CPU execution. Both share
    the same physics formula.

    Args:
        energy: (E,) energy centers [eV]
        energy_window_width_relative: relative energy resolution (FWHM/E)

    Returns:
        (E, E) response matrix W
    """
    ln_energy = torch.log(energy)

    # Relative FWHM → Gaussian σ in log-energy space
    s = math.asinh(0.5 * energy_window_width_relative) / math.sqrt(2.0 * math.log(2.0))

    diff = ln_energy.unsqueeze(0) - ln_energy.unsqueeze(1)
    W = torch.exp(-0.5 * (diff / s) ** 2)
    W = W / W.sum(dim=1, keepdim=True)

    return W


def compute_kappa_chi2_batch_torch(
    model_flux: Tensor,
    data_flux: Tensor,
    weights: Tensor,
    response_matrix: Tensor | None = None,
    eps: float = config.EPS,
) -> Tensor:
    """
    Compute chi² for batched kappa model fits.

    Args:
        model_flux: (N, P, E) model omnidirectional flux
        data_flux: (N, E) measured flux
        weights: (N, E) fitting weights (1/σ)
        response_matrix: (E, E) optional convolution matrix
        eps: small value to avoid log(0)

    Returns:
        (N, P) chi² values
    """
    _N, _P, _E = model_flux.shape

    if response_matrix is not None:
        # Match CPU: W @ flux (W[i,j] maps input energy j → output i)
        model_flux = torch.einsum("fe,npe->npf", response_matrix, model_flux)

    log_model = torch.log(model_flux + eps)
    log_data = torch.log(data_flux.unsqueeze(1) + eps)
    weights_exp = weights.unsqueeze(1)

    diff = (log_model - log_data) * weights_exp
    chi2 = (diff**2).sum(dim=2)

    return chi2


# Use shared BatchedDifferentialEvolution for backwards compatibility
KappaDifferentialEvolution = BatchedDifferentialEvolution


class KappaFitterTorch:
    """
    PyTorch-accelerated batched Kappa fitter.

    Fits kappa distribution parameters for multiple spectra simultaneously
    using vectorized tensor operations.
    """

    DEFAULT_BOUNDS: ClassVar[list[tuple[float, float]]] = [
        (2.5, 6.0),  # kappa
        (6.0, 8.0),  # log10(theta) where theta is in m/s
    ]

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "auto",
        popsize: int = 30,
        maxiter: int = 100,
        use_convolution: bool = True,
        energy_window_width_relative: float = config.ENERGY_WINDOW_WIDTH_RELATIVE,
    ):
        """
        Initialize batched Kappa fitter.

        Args:
            device: torch device
            dtype: 'auto', 'float16', 'float32', or 'float64'.
                'auto' (default): float16 on modern GPUs (Volta+), float32 otherwise.
            popsize: DE population size
            maxiter: DE max iterations
            use_convolution: Apply energy response matrix
            energy_window_width_relative: Energy resolution (FWHM/E)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for KappaFitterTorch")

        self.device = torch.device(device)
        if dtype == "auto":
            self.dtype = _auto_detect_dtype(self.device)
        elif dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float32":
            self.dtype = torch.float32
        else:
            self.dtype = torch.float16
        self.popsize = popsize
        self.maxiter = maxiter
        self.use_convolution = use_convolution
        self.energy_window_width_relative = energy_window_width_relative

    def fit_batch(
        self,
        energy: np.ndarray,
        flux_data: np.ndarray,
        density_estimates: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit kappa parameters for multiple spectra in batch.

        Args:
            energy: (E,) energy grid [eV]
            flux_data: (N, E) measured omnidirectional flux
            density_estimates: (N,) density estimates [particles/m³]
            weights: (N, E) optional fitting weights

        Returns:
            kappa: (N,) fitted kappa values
            theta: (N,) fitted theta values [m/s]
            chi2: (N,) chi² values
        """
        N, _E = flux_data.shape

        energy_t = torch.tensor(energy, device=self.device, dtype=self.dtype)
        flux_t = torch.tensor(flux_data, device=self.device, dtype=self.dtype)
        density_t = torch.tensor(
            density_estimates, device=self.device, dtype=self.dtype
        )

        if weights is None:
            # Unweighted default; for CPU parity pass 1/(sigma_log_flux + EPS)
            weights = np.ones_like(flux_data)
        weights_t = torch.tensor(weights, device=self.device, dtype=self.dtype)

        response_matrix = None
        if self.use_convolution:
            response_matrix = build_response_matrix_torch(
                energy_t, self.energy_window_width_relative
            )

        de = KappaDifferentialEvolution(
            bounds=self.DEFAULT_BOUNDS,
            n_spectra=N,
            popsize=self.popsize,
            maxiter=self.maxiter,
            device=str(self.device),
            dtype=self.dtype,
        )

        def objective(params: Tensor) -> Tensor:
            """Evaluate chi² for all spectra and candidates. params: (N, P, 2)."""
            kappa = params[:, :, 0]
            log_theta = params[:, :, 1]
            theta = 10.0**log_theta

            model_flux = omnidirectional_flux_batch_torch(
                density_t, kappa, theta, energy_t
            )
            chi2 = compute_kappa_chi2_batch_torch(
                model_flux, flux_t, weights_t, response_matrix
            )
            chi2 = torch.where(
                torch.isfinite(chi2), chi2, torch.tensor(1e30, device=self.device)
            )
            return chi2

        best_params, best_chi2, _n_iter = de.optimize(objective)

        kappa = best_params[:, 0].cpu().numpy()
        log_theta = best_params[:, 1].cpu().numpy()
        theta = 10.0**log_theta
        chi2 = best_chi2.cpu().numpy()

        return kappa, theta, chi2
