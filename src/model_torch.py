"""
PyTorch-accelerated loss-cone model for lunar surface potential fitting.

This module provides PyTorch implementations of the loss-cone forward model
and optimizer, using vectorized tensor operations for ~5x speedup over scipy.

Works on CPU (default) or CUDA GPU if available and beneficial.

Physics basis from Halekas 2008 (doi:10.1029/2008JA013194), paragraph 37.
"""

from __future__ import annotations

import numpy as np

try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None  # type: ignore[misc, assignment]

from src.utils.optimization import (
    BatchedDifferentialEvolution,
    get_torch_device,
)

# Small epsilon to avoid division by zero
EPS = 1e-12

# Default flux value outside loss cone
DEFAULT_BACKGROUND = 0.05


def synth_losscone_batch_torch(
    energy_grid: Tensor,
    pitch_grid: Tensor,
    U_surface: Tensor,
    U_spacecraft: float | Tensor = 0.0,
    bs_over_bm: Tensor | None = None,
    beam_width_eV: Tensor | None = None,
    beam_amp: Tensor | None = None,
    beam_pitch_sigma_deg: float = 0.0,
    background: Tensor | None = None,
) -> Tensor:
    """
    PyTorch batch loss-cone model for GPU acceleration.

    Computes loss-cone models for multiple parameter combinations simultaneously.
    All tensor parameters must have the same length (n_params).

    This implementation uses hard masks (non-differentiable) to match
    the CPU implementation exactly.

    Args:
        energy_grid: (nE,) electron energies in eV
        pitch_grid: (nE, nPitch) pitch angles in degrees
        U_surface: (n_params,) lunar surface potentials in volts
        U_spacecraft: spacecraft potential in volts (scalar or (nE,))
        bs_over_bm: (n_params,) B_spacecraft/B_surface ratios (default: all 1.0)
        beam_width_eV: (n_params,) beam widths in eV (default: all 0.0)
        beam_amp: (n_params,) beam amplitudes (default: all 0.0)
        beam_pitch_sigma_deg: beam angular spread in degrees (scalar)
        background: (n_params,) flux outside loss cone (default: all 0.05)

    Returns:
        Model flux tensor of shape (n_params, nE, nPitch)
    """
    device = energy_grid.device
    dtype = energy_grid.dtype

    U_surface = U_surface.to(device=device, dtype=dtype)
    n_params = U_surface.size(0)

    # Set defaults for optional tensors
    if bs_over_bm is None:
        bs_over_bm = torch.ones(n_params, device=device, dtype=dtype)
    else:
        bs_over_bm = bs_over_bm.to(device=device, dtype=dtype)

    if beam_width_eV is None:
        beam_width_eV = torch.zeros(n_params, device=device, dtype=dtype)
    else:
        beam_width_eV = beam_width_eV.to(device=device, dtype=dtype)

    if beam_amp is None:
        beam_amp = torch.zeros(n_params, device=device, dtype=dtype)
    else:
        beam_amp = beam_amp.to(device=device, dtype=dtype)

    if background is None:
        background = torch.full(
            (n_params,), DEFAULT_BACKGROUND, device=device, dtype=dtype
        )
    else:
        background = background.to(device=device, dtype=dtype)

    # Reshape for broadcasting: params -> (nParams, 1, 1)
    U_surface = U_surface.view(-1, 1, 1)
    bs_over_bm = bs_over_bm.view(-1, 1, 1)
    beam_amp = beam_amp.view(-1, 1, 1)
    beam_width_eV = beam_width_eV.view(-1, 1, 1)
    background = background.view(-1, 1, 1)

    # Handle energy grid: guard against E <= 0
    valid_E = energy_grid > 0
    E_safe = torch.where(valid_E, energy_grid, torch.ones_like(energy_grid))

    # Reshape grids for broadcasting
    nE, nPitch = pitch_grid.shape
    pitch_exp = pitch_grid.unsqueeze(0)  # (1, nE, nPitch)
    E_exp = E_safe.unsqueeze(0).unsqueeze(-1)  # (1, nE, 1)
    valid_E_exp = valid_E.unsqueeze(0).unsqueeze(-1)  # (1, nE, 1)

    # Handle U_spacecraft: scalar -> (1,1,1), array(nE,) -> (1,nE,1)
    if isinstance(U_spacecraft, (int, float)):
        U_spacecraft_t = torch.tensor(U_spacecraft, device=device, dtype=dtype).view(
            1, 1, 1
        )
    else:
        U_spacecraft_t = (
            U_spacecraft.to(device=device, dtype=dtype).unsqueeze(0).unsqueeze(-1)
        )

    # Compute loss cone angle using Halekas 2008 formula
    # sin²(αc) = (BS/BM) × (1 + UM / (E - U_spacecraft))
    E_corrected = torch.clamp(E_exp - U_spacecraft_t, min=EPS)
    x = bs_over_bm * (1.0 + U_surface / E_corrected)
    x_clipped = torch.clamp(x, 0.0, 1.0)
    ac_deg = torch.rad2deg(torch.arcsin(torch.sqrt(x_clipped)))

    # Build model: background everywhere, 1.0 inside loss cone
    model = background.expand(n_params, nE, nPitch).clone()

    # Inside loss cone: pitch <= 180 - αc (hard mask, matches CPU)
    inside_cone = (pitch_exp <= (180.0 - ac_deg)) & valid_E_exp
    model = torch.where(inside_cone, torch.ones_like(model), model)

    # Add secondary electron beam if enabled
    has_beam = (beam_width_eV > 0).any() and (beam_amp > 0).any()
    if has_beam:
        beam_center = torch.clamp(
            torch.abs(U_surface - U_spacecraft_t), min=beam_width_eV
        )
        beam_width_safe = torch.clamp(beam_width_eV, min=EPS)
        energy_profile = beam_amp * torch.exp(
            -0.5 * ((E_exp - beam_center) / beam_width_safe) ** 2
        )

        if beam_pitch_sigma_deg > 0:
            pitch_profile = torch.exp(
                -0.5 * ((pitch_exp - 180.0) / beam_pitch_sigma_deg) ** 2
            )
        else:
            pitch_profile = 1.0

        beam = energy_profile * pitch_profile
        model = model + beam

    return model


def compute_chi2_batch_torch(
    model: Tensor,
    data: Tensor,
    data_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute chi-squared for batch of models against data.

    Args:
        model: (n_params, nE, nPitch) model predictions
        data: (nE, nPitch) observed normalized flux
        data_mask: (nE, nPitch) boolean mask for valid data points
        eps: small value to avoid log(0)

    Returns:
        (n_params,) chi-squared values
    """
    # Replace invalid data with 1.0 before log to avoid NaN propagation
    # (masked out anyway, but prevents NaN * 0 = NaN issue)
    data_safe = torch.where(data_mask, data, torch.ones_like(data))

    log_model = torch.log(model + eps)
    log_data = torch.log(data_safe + eps)

    # Broadcast data to match model shape
    log_data_exp = log_data.unsqueeze(0)  # (1, nE, nPitch)
    data_mask_exp = data_mask.unsqueeze(0)  # (1, nE, nPitch)

    # Compute diff only where mask is True
    diff = torch.where(
        data_mask_exp, log_data_exp - log_model, torch.zeros_like(log_model)
    )
    chi2 = (diff**2).sum(dim=(1, 2))

    return chi2


def synth_losscone_multi_chunk_torch(
    energy_grids: Tensor,
    pitch_grids: Tensor,
    U_surface: Tensor,
    U_spacecraft: Tensor | None = None,
    bs_over_bm: Tensor | None = None,
    beam_width_eV: Tensor | None = None,
    beam_amp: Tensor | None = None,
    beam_pitch_sigma_deg: float = 0.0,
    background: Tensor | None = None,
) -> Tensor:
    """
    PyTorch multi-chunk loss-cone model for batched GPU acceleration.

    Computes loss-cone models for multiple chunks × multiple parameter candidates.
    Each chunk has its own energy/pitch grids.

    Args:
        energy_grids: (N_chunks, nE) electron energies in eV per chunk
        pitch_grids: (N_chunks, nE, nPitch) pitch angles in degrees per chunk
        U_surface: (N_chunks, n_pop) lunar surface potentials in volts
        U_spacecraft: (N_chunks, nE) or (N_chunks,) spacecraft potential per chunk
        bs_over_bm: (N_chunks, n_pop) B_spacecraft/B_surface ratios
        beam_width_eV: (N_chunks, n_pop) beam widths in eV
        beam_amp: (N_chunks, n_pop) beam amplitudes
        beam_pitch_sigma_deg: beam angular spread in degrees (scalar)
        background: (N_chunks, n_pop) flux outside loss cone

    Returns:
        Model flux tensor of shape (N_chunks, n_pop, nE, nPitch)
    """
    device = energy_grids.device
    dtype = energy_grids.dtype

    N_chunks, nE = energy_grids.shape
    n_pop = U_surface.size(1)
    nPitch = pitch_grids.size(2)

    # Set defaults for optional tensors
    if bs_over_bm is None:
        bs_over_bm = torch.ones(N_chunks, n_pop, device=device, dtype=dtype)

    if beam_width_eV is None:
        beam_width_eV = torch.zeros(N_chunks, n_pop, device=device, dtype=dtype)

    if beam_amp is None:
        beam_amp = torch.zeros(N_chunks, n_pop, device=device, dtype=dtype)

    if background is None:
        background = torch.full(
            (N_chunks, n_pop), DEFAULT_BACKGROUND, device=device, dtype=dtype
        )

    # Reshape for broadcasting:
    # U_surface: (N, P) -> (N, P, 1, 1)
    # energy_grids: (N, E) -> (N, 1, E, 1)
    # pitch_grids: (N, E, A) -> (N, 1, E, A)
    U_surface_exp = U_surface.view(N_chunks, n_pop, 1, 1)
    bs_over_bm_exp = bs_over_bm.view(N_chunks, n_pop, 1, 1)
    beam_amp_exp = beam_amp.view(N_chunks, n_pop, 1, 1)
    beam_width_exp = beam_width_eV.view(N_chunks, n_pop, 1, 1)
    background_exp = background.view(N_chunks, n_pop, 1, 1)

    # Energy: (N, E) -> (N, 1, E, 1)
    valid_E = energy_grids > 0
    E_safe = torch.where(valid_E, energy_grids, torch.ones_like(energy_grids))
    E_exp = E_safe.view(N_chunks, 1, nE, 1)
    valid_E_exp = valid_E.view(N_chunks, 1, nE, 1)

    # Pitch: (N, E, A) -> (N, 1, E, A)
    pitch_exp = pitch_grids.unsqueeze(1)

    # Handle U_spacecraft: (N, E) -> (N, 1, E, 1) or (N,) -> (N, 1, 1, 1)
    if U_spacecraft is None:
        U_spacecraft_exp = torch.zeros(N_chunks, 1, 1, 1, device=device, dtype=dtype)
    elif U_spacecraft.dim() == 1:
        # Per-chunk scalar: (N,) -> (N, 1, 1, 1)
        U_spacecraft_exp = U_spacecraft.view(N_chunks, 1, 1, 1)
    else:
        # Per-chunk per-energy: (N, E) -> (N, 1, E, 1)
        U_spacecraft_exp = U_spacecraft.view(N_chunks, 1, nE, 1)

    # Compute loss cone angle using Halekas 2008 formula
    # sin²(αc) = (BS/BM) × (1 + UM / (E - U_spacecraft))
    E_corrected = torch.clamp(E_exp - U_spacecraft_exp, min=EPS)
    x = bs_over_bm_exp * (1.0 + U_surface_exp / E_corrected)
    x_clipped = torch.clamp(x, 0.0, 1.0)
    ac_deg = torch.rad2deg(torch.arcsin(torch.sqrt(x_clipped)))

    # Build model: background everywhere, 1.0 inside loss cone
    model = background_exp.expand(N_chunks, n_pop, nE, nPitch).clone()

    # Inside loss cone: pitch <= 180 - αc (hard mask)
    inside_cone = (pitch_exp <= (180.0 - ac_deg)) & valid_E_exp
    model = torch.where(inside_cone, torch.ones_like(model), model)

    # Add secondary electron beam if enabled
    has_beam = (beam_width_exp > 0).any() and (beam_amp_exp > 0).any()
    if has_beam:
        beam_center = torch.clamp(
            torch.abs(U_surface_exp - U_spacecraft_exp), min=beam_width_exp
        )
        beam_width_safe = torch.clamp(beam_width_exp, min=EPS)
        energy_profile = beam_amp_exp * torch.exp(
            -0.5 * ((E_exp - beam_center) / beam_width_safe) ** 2
        )

        if beam_pitch_sigma_deg > 0:
            pitch_profile = torch.exp(
                -0.5 * ((pitch_exp - 180.0) / beam_pitch_sigma_deg) ** 2
            )
        else:
            pitch_profile = 1.0

        beam = energy_profile * pitch_profile
        model = model + beam

    return model


def compute_chi2_multi_chunk_torch(
    models: Tensor,
    data: Tensor,
    data_mask: Tensor,
    eps: float = 1e-6,
) -> Tensor:
    """
    Compute chi-squared for multiple chunks × multiple candidates.

    Args:
        models: (N_chunks, n_pop, nE, nPitch) model predictions
        data: (N_chunks, nE, nPitch) observed normalized flux per chunk
        data_mask: (N_chunks, nE, nPitch) boolean mask for valid data points
        eps: small value to avoid log(0)

    Returns:
        (N_chunks, n_pop) chi-squared values
    """
    # data_safe: replace invalid with 1.0 to avoid log issues
    data_safe = torch.where(data_mask, data, torch.ones_like(data))

    log_model = torch.log(models + eps)
    log_data = torch.log(data_safe + eps)

    # Broadcast data: (N, E, A) -> (N, 1, E, A)
    log_data_exp = log_data.unsqueeze(1)
    data_mask_exp = data_mask.unsqueeze(1)

    # Compute diff only where mask is True
    diff = torch.where(
        data_mask_exp, log_data_exp - log_model, torch.zeros_like(log_model)
    )
    chi2 = (diff**2).sum(dim=(2, 3))  # Sum over (E, A) -> (N, P)

    return chi2


# Use shared BatchedDifferentialEvolution for backwards compatibility
GPUDifferentialEvolution = BatchedDifferentialEvolution


class LossConeFitterTorch:
    """
    PyTorch-accelerated loss cone fitter.

    Drop-in replacement for LossConeFitter using vectorized tensor operations.
    Provides ~5x speedup over scipy on CPU. Can use CUDA if available.
    """

    def __init__(
        self,
        er_data,
        thetas: str,
        pitch_angle=None,
        spacecraft_potential: np.ndarray | None = None,
        normalization_mode: str = "ratio",
        beam_amp_fixed: float | None = None,
        incident_flux_stat: str = "mean",
        loss_cone_background: float | None = None,
        device: str | None = None,
        dtype: str = "float64",
    ):
        """
        Initialize PyTorch-accelerated loss cone fitter.

        Args:
            er_data: ERData object
            thetas: Path to theta file
            pitch_angle: Optional pre-computed PitchAngle object
            spacecraft_potential: Optional per-row spacecraft potential [V]
            normalization_mode: Flux normalization mode
            beam_amp_fixed: Fixed beam amplitude (None to fit)
            incident_flux_stat: Statistic for incident flux ("mean" or "max")
            loss_cone_background: Background level outside loss cone
            device: Torch device ('cuda', 'cpu', or None for auto)
            dtype: Data type ('float32', 'float64', or 'float16').
                float32 recommended for GPU (good speed/precision balance).
                float16 saves memory, enabling larger batch sizes (use batch_size=200).
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch is required for GPU acceleration")

        # Import here to avoid circular imports
        from src import config
        from src.flux import PitchAngle

        self.er_data = er_data
        self.thetas = np.loadtxt(thetas, dtype=np.float64)
        self.pitch_angle = (
            pitch_angle if pitch_angle is not None else PitchAngle(er_data, thetas)
        )
        self.spacecraft_potential = spacecraft_potential
        self.device = get_torch_device(device)

        # Set dtype
        if dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(
                f"dtype must be 'float32', 'float64', or 'float16', got {dtype}"
            )

        self.beam_width_factor = config.LOSS_CONE_BEAM_WIDTH_FACTOR
        self.beam_amp_min = config.LOSS_CONE_BEAM_AMP_MIN
        self.beam_amp_max = config.LOSS_CONE_BEAM_AMP_MAX
        if beam_amp_fixed is not None:
            self.beam_amp_min = beam_amp_fixed
            self.beam_amp_max = beam_amp_fixed
        self.beam_pitch_sigma_deg = config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG

        if normalization_mode not in {"global", "ratio", "ratio2", "ratio_rescaled"}:
            raise ValueError(f"Unknown normalization_mode: {normalization_mode}")
        self.normalization_mode = normalization_mode

        if incident_flux_stat not in {"mean", "max"}:
            raise ValueError(f"Unknown incident_flux_stat: {incident_flux_stat}")
        self.incident_flux_stat = incident_flux_stat

        if loss_cone_background is None:
            loss_cone_background = config.LOSS_CONE_BACKGROUND
        if loss_cone_background <= 0:
            raise ValueError("loss_cone_background must be positive")
        self.background = float(loss_cone_background)

        self.config = config
        self._cpu_fitter = None  # Lazy-initialized for normalization

    def _get_cpu_fitter(self):
        """Get or create cached CPU fitter for normalization."""
        if self._cpu_fitter is None:
            from src.flux import LossConeFitter

            self._cpu_fitter = LossConeFitter(
                self.er_data,
                str(self.config.DATA_DIR / self.config.THETA_FILE),
                self.pitch_angle,
                self.spacecraft_potential,
                self.normalization_mode,
                beam_amp_fixed=self.beam_amp_min
                if self.beam_amp_min == self.beam_amp_max
                else None,
                incident_flux_stat=self.incident_flux_stat,
                loss_cone_background=self.background,
            )
        return self._cpu_fitter

    def _build_norm2d(self, measurement_chunk: int) -> np.ndarray:
        """Build normalized 2D flux array (delegates to cached CPU fitter)."""
        return self._get_cpu_fitter().build_norm2d(measurement_chunk)

    def _fit_surface_potential_torch(
        self, measurement_chunk: int
    ) -> tuple[float, float, float, float]:
        """
        Fit surface potential using GPU-accelerated two-phase optimization.

        Phase 1: LHS grid search (400 samples) to find good starting region
        Phase 2: DE refinement from best LHS point

        Args:
            measurement_chunk: Index of measurement chunk

        Returns:
            (U_surface, bs_over_bm, beam_amp, chi2)
        """
        eps = 1e-6
        norm2d = self._build_norm2d(measurement_chunk)

        if np.isnan(norm2d).all():
            return np.nan, np.nan, np.nan, np.nan

        s = measurement_chunk * self.config.SWEEP_ROWS
        e = (measurement_chunk + 1) * self.config.SWEEP_ROWS

        max_rows = len(self.er_data.data)
        if s >= max_rows:
            return np.nan, np.nan, np.nan, np.nan
        e = min(e, max_rows)

        energies = self.er_data.data[self.config.ENERGY_COLUMN].to_numpy(
            dtype=np.float64
        )[s:e]
        pitches = self.pitch_angle.pitch_angles[s:e]
        spacecraft_slice = (
            self.spacecraft_potential[s:e]
            if self.spacecraft_potential is not None
            else 0.0
        )

        actual_rows = e - s
        if norm2d.shape[0] > actual_rows:
            norm2d = norm2d[:actual_rows]

        data_mask = np.isfinite(norm2d) & (norm2d > 0)
        if not data_mask.any():
            return np.nan, np.nan, np.nan, np.nan

        # Convert to torch tensors
        energies_t = torch.tensor(energies, device=self.device, dtype=self.dtype)
        pitches_t = torch.tensor(pitches, device=self.device, dtype=self.dtype)
        norm2d_t = torch.tensor(norm2d, device=self.device, dtype=self.dtype)
        data_mask_t = torch.tensor(data_mask, device=self.device, dtype=torch.bool)

        if isinstance(spacecraft_slice, np.ndarray):
            spacecraft_t = torch.tensor(
                spacecraft_slice, device=self.device, dtype=self.dtype
            )
        else:
            spacecraft_t = float(spacecraft_slice)

        # Define objective function
        def objective(params: Tensor) -> Tensor:
            """Evaluate chi2 for population of parameters."""
            U_surface = params[:, 0]
            bs_over_bm = params[:, 1]
            beam_amp = params[:, 2]

            # Compute beam widths
            beam_width = torch.clamp(
                torch.abs(U_surface) * self.beam_width_factor,
                min=self.config.EPS,
            )

            # Evaluate models
            models = synth_losscone_batch_torch(
                energies_t,
                pitches_t,
                U_surface,
                U_spacecraft=spacecraft_t,
                bs_over_bm=bs_over_bm,
                beam_width_eV=beam_width,
                beam_amp=beam_amp,
                beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
                background=torch.full_like(U_surface, self.background),
            )

            # Compute chi2
            chi2 = compute_chi2_batch_torch(models, norm2d_t, data_mask_t, eps)

            # Penalize invalid models
            invalid = ~torch.isfinite(chi2)
            chi2 = torch.where(invalid, torch.tensor(1e30, device=self.device), chi2)

            return chi2

        # Bounds
        bounds = [
            (-2000.0, 2000.0),  # U_surface
            (0.1, 1.1),  # bs_over_bm
            (self.beam_amp_min, max(self.beam_amp_max, self.beam_amp_min + 1e-12)),
        ]

        # Phase 1: LHS grid search (matches CPU implementation)
        n_lhs = 400
        from torch.quasirandom import SobolEngine

        sobol = SobolEngine(dimension=3, scramble=True, seed=42)
        lhs_unit = sobol.draw(n_lhs).to(device=self.device, dtype=self.dtype)

        # Scale to bounds
        lower = torch.tensor(
            [b[0] for b in bounds], device=self.device, dtype=self.dtype
        )
        upper = torch.tensor(
            [b[1] for b in bounds], device=self.device, dtype=self.dtype
        )
        lhs_samples = lower + lhs_unit * (upper - lower)

        # Evaluate all LHS samples at once (GPU batch)
        lhs_chi2 = objective(lhs_samples)

        # Find best LHS point
        best_lhs_idx = torch.argmin(lhs_chi2).item()
        best_lhs_chi2 = lhs_chi2[best_lhs_idx].item()
        x0 = lhs_samples[best_lhs_idx]

        # Phase 2: DE refinement starting from best LHS point
        de_bounds = [
            (-2000.0, 2000.0),  # Wider bounds for DE exploration
            (0.1, 1.1),
            (self.beam_amp_min, max(self.beam_amp_max, self.beam_amp_min + 1e-12)),
        ]

        de = GPUDifferentialEvolution(
            bounds=de_bounds,
            popsize=50,
            mutation=0.5,
            crossover=0.9,
            maxiter=500,
            atol=1e-3,
            seed=42,
            device=str(self.device),
            dtype=self.dtype,
        )

        # Run optimization seeded with best LHS point
        best_params, best_chi2, _n_iter = de.optimize(objective, x0=x0)

        # Use LHS result if DE didn't improve
        if best_lhs_chi2 < best_chi2:
            best_params = x0
            best_chi2 = best_lhs_chi2

        U_surface = best_params[0].item()
        bs_over_bm = float(np.clip(best_params[1].item(), 0.1, 1.1))
        beam_amp = float(
            np.clip(best_params[2].item(), self.beam_amp_min, self.beam_amp_max)
        )

        return U_surface, bs_over_bm, beam_amp, best_chi2

    def fit_surface_potential(self) -> np.ndarray:
        """
        Fit surface potential for all measurement chunks.

        Returns:
            Array with columns [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
        """
        from tqdm import tqdm

        n_chunks = len(self.er_data.data) // self.config.SWEEP_ROWS
        results = np.zeros((n_chunks, 5))

        for i in tqdm(
            range(n_chunks),
            desc=f"Fitting chunks (GPU: {self.device})",
            unit="chunk",
            dynamic_ncols=True,
        ):
            U_surface, bs_over_bm, beam_amp, chi2 = self._fit_surface_potential_torch(i)
            results[i] = [U_surface, bs_over_bm, beam_amp, chi2, i]

        return results

    def _precompute_chunk_data(
        self, chunk_indices: list[int]
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, list[int]]:
        """
        Pre-compute and stack data for multiple chunks.

        Uses batched norm2d computation for significant speedup.

        Args:
            chunk_indices: List of chunk indices to precompute

        Returns:
            Tuple of:
                - energies: (N_valid, nE) energy grids
                - pitches: (N_valid, nE, nPitch) pitch angle grids
                - norm2d: (N_valid, nE, nPitch) normalized flux data
                - data_mask: (N_valid, nE, nPitch) valid data mask
                - sc_potential: (N_valid,) spacecraft potential per chunk
                - valid_indices: List of chunk indices that had valid data
        """
        if not chunk_indices:
            empty = torch.zeros(0, device=self.device, dtype=self.dtype)
            return empty, empty, empty, empty, empty, []

        nE = self.config.SWEEP_ROWS
        max_rows = len(self.er_data.data)

        # Get batched norm2d from CPU fitter (uses vectorized implementation)
        cpu_fitter = self._get_cpu_fitter()
        norm2d_all = cpu_fitter.build_norm2d_batch(chunk_indices)  # (n_chunks, nE, nP)

        # Pre-load all energy and pitch data
        energy_all = self.er_data.data[self.config.ENERGY_COLUMN].to_numpy(
            dtype=np.float64
        )
        pitch_all = self.pitch_angle.pitch_angles

        # Filter to valid chunks
        energies_list = []
        pitches_list = []
        norm2d_list = []
        mask_list = []
        sc_pot_list = []
        valid_indices = []

        for i, chunk_idx in enumerate(chunk_indices):
            norm2d = norm2d_all[i]

            # Skip if all NaN
            if np.isnan(norm2d).all():
                continue

            s = chunk_idx * nE
            e = min((chunk_idx + 1) * nE, max_rows)

            if s >= max_rows:
                continue

            actual_rows = e - s
            energies = energy_all[s:e]
            pitches = pitch_all[s:e]

            # Truncate norm2d if needed
            if norm2d.shape[0] > actual_rows:
                norm2d = norm2d[:actual_rows]

            # Pad to full nE if needed (for consistent tensor shapes)
            if actual_rows < nE:
                pad_rows = nE - actual_rows
                energies = np.pad(energies, (0, pad_rows), constant_values=np.nan)
                pitches = np.pad(
                    pitches, ((0, pad_rows), (0, 0)), constant_values=np.nan
                )
                norm2d = np.pad(norm2d, ((0, pad_rows), (0, 0)), constant_values=np.nan)

            data_mask = np.isfinite(norm2d) & (norm2d > 0)
            if not data_mask.any():
                continue

            # Get spacecraft potential for this chunk (per-row, not averaged)
            if self.spacecraft_potential is not None:
                sc_pot = self.spacecraft_potential[s:e].copy()
                # Pad to nE if needed
                if actual_rows < nE:
                    sc_pot = np.pad(
                        sc_pot, (0, nE - actual_rows), constant_values=np.nan
                    )
            else:
                sc_pot = np.zeros(nE)

            energies_list.append(energies)
            pitches_list.append(pitches)
            norm2d_list.append(norm2d)
            mask_list.append(data_mask)
            sc_pot_list.append(sc_pot)
            valid_indices.append(chunk_idx)

        if not valid_indices:
            empty = torch.zeros(0, device=self.device, dtype=self.dtype)
            return empty, empty, empty, empty, empty, []

        # Stack into tensors
        energies_t = torch.tensor(
            np.stack(energies_list), device=self.device, dtype=self.dtype
        )
        pitches_t = torch.tensor(
            np.stack(pitches_list), device=self.device, dtype=self.dtype
        )
        norm2d_t = torch.tensor(
            np.stack(norm2d_list), device=self.device, dtype=self.dtype
        )
        mask_t = torch.tensor(np.stack(mask_list), device=self.device, dtype=torch.bool)
        sc_pot_t = torch.tensor(
            np.array(sc_pot_list), device=self.device, dtype=self.dtype
        )

        return energies_t, pitches_t, norm2d_t, mask_t, sc_pot_t, valid_indices

    def _fit_batch_lhs(
        self,
        energies: Tensor,
        pitches: Tensor,
        norm2d: Tensor,
        data_mask: Tensor,
        sc_potential: Tensor,
        n_lhs: int = 400,
    ) -> tuple[Tensor, Tensor]:
        """
        Run LHS grid search for multiple chunks simultaneously.

        Args:
            energies: (N, nE) energy grids
            pitches: (N, nE, nPitch) pitch grids
            norm2d: (N, nE, nPitch) normalized flux
            data_mask: (N, nE, nPitch) valid data mask
            sc_potential: (N,) spacecraft potential per chunk
            n_lhs: Number of LHS samples

        Returns:
            best_params: (N, 3) best parameters per chunk
            best_chi2: (N,) best chi2 per chunk
        """
        N_chunks = energies.size(0)

        # Bounds
        bounds = [
            (-2000.0, 2000.0),  # U_surface
            (0.1, 1.1),  # bs_over_bm
            (self.beam_amp_min, max(self.beam_amp_max, self.beam_amp_min + 1e-12)),
        ]

        # Generate LHS samples (same for all chunks)
        from torch.quasirandom import SobolEngine

        sobol = SobolEngine(dimension=3, scramble=True, seed=42)
        lhs_unit = sobol.draw(n_lhs).to(device=self.device, dtype=self.dtype)

        lower = torch.tensor(
            [b[0] for b in bounds], device=self.device, dtype=self.dtype
        )
        upper = torch.tensor(
            [b[1] for b in bounds], device=self.device, dtype=self.dtype
        )
        lhs_samples = lower + lhs_unit * (upper - lower)  # (n_lhs, 3)

        # Expand to (N_chunks, n_lhs, 3)
        lhs_expanded = lhs_samples.unsqueeze(0).expand(N_chunks, -1, -1)

        # Extract parameters
        U_surface = lhs_expanded[:, :, 0]  # (N, n_lhs)
        bs_over_bm = lhs_expanded[:, :, 1]
        beam_amp = lhs_expanded[:, :, 2]

        # Compute beam widths
        beam_width = torch.clamp(
            torch.abs(U_surface) * self.beam_width_factor,
            min=self.config.EPS,
        )

        # Evaluate all models at once
        # sc_potential is (N, nE) per-row, model broadcasts appropriately
        models = synth_losscone_multi_chunk_torch(
            energies,
            pitches,
            U_surface,
            U_spacecraft=sc_potential,  # (N, nE) per-row
            bs_over_bm=bs_over_bm,
            beam_width_eV=beam_width,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
            background=torch.full_like(U_surface, self.background),
        )  # (N, n_lhs, nE, nPitch)

        # Compute chi2 for all
        chi2 = compute_chi2_multi_chunk_torch(models, norm2d, data_mask)  # (N, n_lhs)

        # Penalize invalid
        chi2 = torch.where(
            torch.isfinite(chi2), chi2, torch.tensor(1e30, device=self.device)
        )

        # Find best per chunk
        best_idx = torch.argmin(chi2, dim=1)  # (N,)
        best_chi2 = chi2.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # (N,)

        # Gather best params
        best_params = lhs_expanded.gather(
            1, best_idx.view(N_chunks, 1, 1).expand(-1, -1, 3)
        ).squeeze(1)  # (N, 3)

        return best_params, best_chi2

    def _fit_batch_de(
        self,
        energies: Tensor,
        pitches: Tensor,
        norm2d: Tensor,
        data_mask: Tensor,
        sc_potential: Tensor,
        x0: Tensor,
        popsize: int = 50,
        maxiter: int = 500,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Run DE optimization for multiple chunks simultaneously.

        Args:
            energies: (N, nE) energy grids
            pitches: (N, nE, nPitch) pitch grids
            norm2d: (N, nE, nPitch) normalized flux
            data_mask: (N, nE, nPitch) valid data mask
            sc_potential: (N,) spacecraft potential per chunk
            x0: (N, 3) initial guess from LHS
            popsize: Population size per chunk
            maxiter: Maximum iterations

        Returns:
            best_params: (N, 3) best parameters per chunk
            best_chi2: (N,) best chi2 per chunk
            n_iter: Number of iterations run
        """
        N_chunks = energies.size(0)

        bounds = [
            (-2000.0, 2000.0),
            (0.1, 1.1),
            (self.beam_amp_min, max(self.beam_amp_max, self.beam_amp_min + 1e-12)),
        ]

        def objective(params: Tensor) -> Tensor:
            """
            Evaluate chi2 for all chunks × all population members.

            Args:
                params: (N, popsize, 3) parameters

            Returns:
                chi2: (N, popsize)
            """
            U_surface = params[:, :, 0]
            bs_over_bm = params[:, :, 1]
            beam_amp = params[:, :, 2]

            beam_width = torch.clamp(
                torch.abs(U_surface) * self.beam_width_factor,
                min=self.config.EPS,
            )

            models = synth_losscone_multi_chunk_torch(
                energies,
                pitches,
                U_surface,
                U_spacecraft=sc_potential,
                bs_over_bm=bs_over_bm,
                beam_width_eV=beam_width,
                beam_amp=beam_amp,
                beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
                background=torch.full_like(U_surface, self.background),
            )

            chi2 = compute_chi2_multi_chunk_torch(models, norm2d, data_mask)
            chi2 = torch.where(
                torch.isfinite(chi2), chi2, torch.tensor(1e30, device=self.device)
            )

            return chi2

        # Create multi-spectrum DE optimizer
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=N_chunks,
            popsize=popsize,
            mutation=0.5,
            crossover=0.9,
            maxiter=maxiter,
            atol=1e-3,
            seed=42,
            device=str(self.device),
            dtype=self.dtype,
        )

        # Seed population with LHS results
        # Note: x0 is (N, 3), DE will use this as first member of each population
        best_params, best_chi2, n_iter = de.optimize(objective, x0=x0)

        return best_params, best_chi2, n_iter

    def fit_surface_potential_batched(
        self, batch_size: int = 100, n_lhs: int = 400
    ) -> np.ndarray:
        """
        Fit surface potential for all chunks using batched GPU optimization.

        Processes chunks in mega-batches to manage VRAM, with all chunks in a
        batch optimized simultaneously using multi-spectrum DE.

        Args:
            batch_size: Number of chunks to process simultaneously (tune for VRAM)
            n_lhs: Number of LHS samples for initial grid search

        Returns:
            Array with columns [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
        """
        from tqdm import tqdm

        n_chunks = len(self.er_data.data) // self.config.SWEEP_ROWS
        results = np.full((n_chunks, 5), np.nan)

        # Process in mega-batches
        for batch_start in tqdm(
            range(0, n_chunks, batch_size),
            desc=f"Fitting batches (GPU: {self.device})",
            unit="batch",
        ):
            batch_end = min(batch_start + batch_size, n_chunks)
            chunk_indices = list(range(batch_start, batch_end))

            # Pre-compute all chunk data
            energies, pitches, norm2d, mask, sc_pot, valid_indices = (
                self._precompute_chunk_data(chunk_indices)
            )

            if len(valid_indices) == 0:
                continue

            # Phase 1: LHS grid search
            lhs_params, lhs_chi2 = self._fit_batch_lhs(
                energies, pitches, norm2d, mask, sc_pot, n_lhs=n_lhs
            )

            # Phase 2: DE refinement
            de_params, de_chi2, _ = self._fit_batch_de(
                energies, pitches, norm2d, mask, sc_pot, x0=lhs_params
            )

            # Use LHS if it was better
            use_lhs = lhs_chi2 < de_chi2
            final_params = torch.where(use_lhs.unsqueeze(-1), lhs_params, de_params)
            final_chi2 = torch.where(use_lhs, lhs_chi2, de_chi2)

            # Extract and clip results
            final_params_np = final_params.cpu().numpy()
            final_chi2_np = final_chi2.cpu().numpy()

            for i, chunk_idx in enumerate(valid_indices):
                U_surface = final_params_np[i, 0]
                bs_over_bm = float(np.clip(final_params_np[i, 1], 0.1, 1.1))
                beam_amp = float(
                    np.clip(final_params_np[i, 2], self.beam_amp_min, self.beam_amp_max)
                )
                chi2 = final_chi2_np[i]

                results[chunk_idx] = [U_surface, bs_over_bm, beam_amp, chi2, chunk_idx]

        return results


# Backwards compatibility alias
LossConeFitterGPU = LossConeFitterTorch
