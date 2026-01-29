"""Torch loss-cone fitter.

Canonical torch implementation lives here; `src/losscone_torch.py` re-exports this
API (`src/model_torch.py` remains as a legacy alias).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor

from src import config
from src.losscone.cpu import LossConeFitter, PitchAngle
from src.losscone.fitter_base import LossConeFitterBase
from src.losscone.params import losscone_lhs_samples, losscone_optimizer_bounds
from src.losscone.torch.chi2 import (
    compute_chi2_batch_torch,
    compute_chi2_multi_chunk_torch,
    compute_lillis_chi2_batch_torch,
    compute_lillis_chi2_multi_chunk_torch,
    precompute_log_data_torch,
)
from src.losscone.torch.forward import (
    synth_losscone_batch_torch,
    synth_losscone_multi_chunk_torch,
)
from src.losscone.types import (
    ChunkFitResult,
    FitChunkData,
    FitMethod,
    parse_fit_method,
    parse_normalization_mode,
)
from src.utils import thetas as thetas_module
from src.utils.optimization import BatchedDifferentialEvolution, get_torch_device

if TYPE_CHECKING:
    from src.losscone.cpu import ERData
    from src.losscone.types import NormalizationMode


def _auto_detect_dtype(device: torch.device) -> torch.dtype:
    """Auto-detect an optimal dtype for the given device."""
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        if props.major >= 7:
            return torch.float16
        return torch.float32
    return torch.float32


def _auto_detect_batch_size(device: torch.device, dtype: torch.dtype) -> int:
    """Auto-detect a batch size based on available VRAM."""
    if device.type != "cuda":
        return 50

    try:
        vram_mb = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024

        bytes_per_elem = {torch.float16: 2, torch.float32: 4, torch.float64: 8}
        mb_per_chunk = 5.0 * bytes_per_elem.get(dtype, 4) / 2  # base is float16

        batch_size = int(vram_mb * 0.5 / mb_per_chunk)
        return max(25, min(batch_size, 1000))
    except Exception:
        return 100


# Use shared BatchedDifferentialEvolution for backwards compatibility
GPUDifferentialEvolution = BatchedDifferentialEvolution


class LossConeFitterTorch(LossConeFitterBase):
    """
    PyTorch-accelerated loss cone fitter.

    Drop-in replacement for LossConeFitter using vectorized tensor operations.
    Provides ~5x speedup over scipy on CPU. Can use CUDA if available.
    """

    def __init__(
        self,
        er_data: ERData,
        *,
        pitch_angle: PitchAngle | None = None,
        spacecraft_potential: np.ndarray | None = None,
        normalization_mode: str | NormalizationMode = "ratio",
        fit_method: str | FitMethod | None = None,
        beam_amp_fixed: float | None = None,
        incident_flux_stat: str = "mean",
        loss_cone_background: float | None = None,
        device: str | None = None,
        dtype: str = "auto",
    ):
        """
        Initialize PyTorch-accelerated loss cone fitter.

        Args:
            er_data: ERData object
            pitch_angle: Optional pre-computed PitchAngle object
            spacecraft_potential: Optional per-row spacecraft potential [V]
            normalization_mode: Flux normalization mode
            fit_method: Loss-cone fitting method ("halekas" or "lillis")
            beam_amp_fixed: Fixed beam amplitude (None to fit)
            incident_flux_stat: Statistic for incident flux ("mean" or "max")
            loss_cone_background: Background level outside loss cone
            device: Torch device ('cuda', 'cpu', or None for auto)
            dtype: Data type ('auto', 'float16', 'float32', or 'float64').
                'auto' (default): float16 on modern GPUs (Volta+), float32 otherwise.
                float16: fastest on modern GPUs, 4x less VRAM.
                float32: good balance of speed and precision.
                float64: maximum precision (slow).
        """
        self.er_data = er_data
        self.thetas = thetas_module.get_thetas()
        self.pitch_angle = pitch_angle or PitchAngle(er_data)
        self.spacecraft_potential = spacecraft_potential
        self.device = get_torch_device(device)

        # Set dtype (auto-detect if not specified)
        if dtype == "auto":
            self.dtype = _auto_detect_dtype(self.device)
        elif dtype == "float32":
            self.dtype = torch.float32
        elif dtype == "float64":
            self.dtype = torch.float64
        elif dtype == "float16":
            self.dtype = torch.float16
        else:
            raise ValueError(
                f"dtype must be 'auto', 'float16', 'float32', or 'float64', got {dtype}"
            )

        # Surface potential bounds from config
        self.u_surface_min = config.LOSS_CONE_U_SURFACE_MIN
        self.u_surface_max = config.LOSS_CONE_U_SURFACE_MAX
        self.bs_over_bm_min = config.LOSS_CONE_BS_OVER_BM_MIN
        self.bs_over_bm_max = config.LOSS_CONE_BS_OVER_BM_MAX

        # Beam parameters - use fixed width (not scaling with |U|)
        self.beam_width_ev = config.LOSS_CONE_BEAM_WIDTH_EV
        self.beam_amp_min = config.LOSS_CONE_BEAM_AMP_MIN
        self.beam_amp_max = config.LOSS_CONE_BEAM_AMP_MAX
        if beam_amp_fixed is not None:
            self.beam_amp_min = beam_amp_fixed
            self.beam_amp_max = beam_amp_fixed
        self.beam_pitch_sigma_deg = config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG

        self.normalization_mode = parse_normalization_mode(normalization_mode)

        if incident_flux_stat not in {"mean", "max"}:
            raise ValueError(f"Unknown incident_flux_stat: {incident_flux_stat}")
        self.incident_flux_stat = incident_flux_stat

        if loss_cone_background is None:
            loss_cone_background = config.LOSS_CONE_BACKGROUND
        if loss_cone_background <= 0:
            raise ValueError("loss_cone_background must be positive")
        self.background = float(loss_cone_background)

        self.fit_method = parse_fit_method(fit_method)

        self.config = config
        self._cpu_fitter = None  # Lazy-initialized for normalization

    def _get_cpu_fitter(self) -> LossConeFitter:
        """Get or create cached CPU fitter for normalization."""
        if self._cpu_fitter is None:
            self._cpu_fitter = LossConeFitter(
                self.er_data,
                pitch_angle=self.pitch_angle,
                spacecraft_potential=self.spacecraft_potential,
                normalization_mode=self.normalization_mode,
                beam_amp_fixed=self.beam_amp_min
                if self.beam_amp_min == self.beam_amp_max
                else None,
                incident_flux_stat=self.incident_flux_stat,
                loss_cone_background=self.background,
            )
        return self._cpu_fitter

    def build_norm2d(self, measurement_chunk: int) -> np.ndarray:
        """Build normalized 2D flux array (delegates to cached CPU fitter)."""
        return self._get_cpu_fitter().build_norm2d(measurement_chunk)

    def build_norm2d_batch(self, chunk_indices: list[int]) -> np.ndarray:
        """
        Build normalized 2D flux arrays for multiple chunks at once (CPU vectorized).
        """
        return self._get_cpu_fitter().build_norm2d_batch(chunk_indices)

    def fit_chunk_full(self, measurement_chunk: int) -> ChunkFitResult:
        """
        Fit surface potential using GPU-accelerated two-phase optimization.

        Phase 1: LHS grid search (config.LOSS_CONE_LHS_SAMPLES samples) to find
            a good starting region
        Phase 2: DE refinement from best LHS point

        Args:
            measurement_chunk: Index of measurement chunk

        Returns:
            ChunkFitResult: Fit result (unpackable as a 4-tuple for compatibility).
        """
        data = self._get_cpu_fitter()._prepare_chunk_data(measurement_chunk)
        if data is None:
            return ChunkFitResult.invalid(measurement_chunk)

        if not data.has_enough_valid_bins(self.fit_method):
            return ChunkFitResult.invalid(measurement_chunk)

        eps = float(config.EPS)
        norm2d = data.norm2d
        energies = data.energies
        pitches = data.pitches
        spacecraft_slice = data.spacecraft_slice
        data_mask = data.combined_mask(self.fit_method)

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

            # Use fixed beam width (not scaling with |U_surface|)
            beam_width = torch.full_like(U_surface, self.beam_width_ev)

            # Evaluate models
            models, model_masks = synth_losscone_batch_torch(
                energies_t,
                pitches_t,
                U_surface,
                U_spacecraft=spacecraft_t,
                bs_over_bm=bs_over_bm,
                beam_width_eV=beam_width,
                beam_amp=beam_amp,
                beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
                background=torch.full_like(U_surface, self.background),
                return_mask=True,
            )

            # Compute chi2
            if self.fit_method == FitMethod.LILLIS:
                chi2 = compute_lillis_chi2_batch_torch(
                    models, norm2d_t, data_mask_t, model_mask=model_masks
                )
            else:
                chi2 = compute_chi2_batch_torch(
                    models, norm2d_t, data_mask_t, eps, model_mask=model_masks
                )

            # Penalize invalid models
            invalid = ~torch.isfinite(chi2)
            chi2 = torch.where(invalid, torch.tensor(1e30, device=self.device), chi2)

            return chi2

        bounds = losscone_optimizer_bounds(
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
        )

        # Phase 1: LHS grid search (NumPy + SciPy sampler; shared with CPU)
        n_lhs = int(config.LOSS_CONE_LHS_SAMPLES)
        lhs_np = losscone_lhs_samples(
            n_samples=n_lhs,
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
            seed=config.LOSS_CONE_LHS_SEED,
        )
        lhs_samples = torch.tensor(lhs_np, device=self.device, dtype=self.dtype)

        # Evaluate all LHS samples at once (GPU batch)
        lhs_chi2 = objective(lhs_samples)

        # Find best LHS point
        best_lhs_idx = torch.argmin(lhs_chi2).item()
        best_lhs_chi2 = lhs_chi2[best_lhs_idx].item()
        x0 = lhs_samples[best_lhs_idx]

        de = GPUDifferentialEvolution(
            bounds=bounds,
            popsize=int(config.LOSS_CONE_DE_POPSIZE),
            mutation=float(config.LOSS_CONE_DE_MUTATION),
            crossover=float(config.LOSS_CONE_DE_CROSSOVER),
            maxiter=int(config.LOSS_CONE_DE_MAXITER),
            atol=float(config.LOSS_CONE_DE_ATOL),
            seed=int(config.LOSS_CONE_DE_SEED),
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
        bs_over_bm = float(
            np.clip(best_params[1].item(), self.bs_over_bm_min, self.bs_over_bm_max)
        )
        beam_amp = float(
            np.clip(best_params[2].item(), self.beam_amp_min, self.beam_amp_max)
        )

        if (
            self.fit_method == FitMethod.LILLIS
            and best_chi2 > self.config.LILLIS_CHI2_REDUCED_MAX
        ):
            return ChunkFitResult.invalid(measurement_chunk, chi2=float(best_chi2))

        return ChunkFitResult(
            u_surface=float(U_surface),
            bs_over_bm=bs_over_bm,
            beam_amp=beam_amp,
            chi2=float(best_chi2),
            chunk_index=measurement_chunk,
        )

    def _fit_surface_potential_torch(self, measurement_chunk: int) -> ChunkFitResult:
        """Backward-compatible alias for `fit_chunk_full`."""
        return self.fit_chunk_full(measurement_chunk)

    def _fit_surface_potential(self, measurement_chunk: int) -> ChunkFitResult:
        """Backward-compatible alias for `fit_chunk_full`."""
        return self.fit_chunk_full(measurement_chunk)

    def fit_surface_potential(self) -> np.ndarray:
        """
        Fit surface potential for all measurement chunks.

        Defaults to the batched implementation for performance.
        """
        return self.fit_surface_potential_batched()

    def fit_surface_potential_sequential(self) -> np.ndarray:
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
            results[i] = self.fit_chunk_full(i).as_row()

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

        # Pre-load all energy, pitch, and flux data
        energy_all = self.er_data.data[self.config.ENERGY_COLUMN].to_numpy(
            dtype=np.float64
        )
        pitch_all = self.pitch_angle.pitch_angles
        flux_all = self.er_data.data[self.config.FLUX_COLS].to_numpy(dtype=np.float64)

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

            flux_chunk = flux_all[s:e]
            if actual_rows < nE:
                flux_chunk = np.pad(
                    flux_chunk,
                    ((0, nE - actual_rows), (0, 0)),
                    constant_values=np.nan,
                )

            valid_energy = energies[:, None] >= sc_pot[:, None]
            valid_energy_mask = np.broadcast_to(valid_energy, pitches.shape)

            data = FitChunkData(
                norm2d=norm2d,
                energies=energies,
                pitches=pitches,
                raw_flux=flux_chunk,
                spacecraft_slice=sc_pot,
                valid_energy_mask=valid_energy_mask,
            )
            if not data.has_enough_valid_bins(self.fit_method):
                continue
            data_mask = data.combined_mask(self.fit_method)

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

    def fit_chunk_lhs(
        self,
        measurement_chunk: int,
        beam_width_ev: float | None = None,
        u_spacecraft: float | None = None,
        n_samples: int | None = None,
    ) -> ChunkFitResult:
        """
        LHS-only fit for a single chunk (mirrors CPU implementation).
        """
        cpu_fitter = self._get_cpu_fitter()
        data = cpu_fitter._prepare_chunk_data(measurement_chunk)
        if data is None:
            return ChunkFitResult.invalid(measurement_chunk)

        if u_spacecraft is not None:
            data = data.with_spacecraft_slice(float(u_spacecraft))

        norm2d = data.norm2d
        energies = data.energies
        pitches = data.pitches
        spacecraft_slice = data.spacecraft_slice
        if not data.has_enough_valid_bins(self.fit_method):
            return ChunkFitResult.invalid(measurement_chunk)
        data_mask = data.combined_mask(self.fit_method)

        if n_samples is None:
            n_samples = int(config.LOSS_CONE_LHS_SAMPLES)
        samples = losscone_lhs_samples(
            n_samples=int(n_samples),
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
            seed=config.LOSS_CONE_LHS_SEED,
        )
        samples_t = torch.tensor(samples, device=self.device, dtype=self.dtype)
        u_surface = samples_t[:, 0]
        bs_over_bm = samples_t[:, 1]
        beam_amp = samples_t[:, 2]

        width = self.beam_width_ev if beam_width_ev is None else beam_width_ev
        beam_width = torch.full_like(u_surface, width)
        background = torch.full_like(u_surface, self.background)

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

        models, model_masks = synth_losscone_batch_torch(
            energy_grid=energies_t,
            pitch_grid=pitches_t,
            U_surface=u_surface,
            U_spacecraft=spacecraft_t,
            bs_over_bm=bs_over_bm,
            beam_width_eV=beam_width,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
            background=background,
            return_mask=True,
        )

        if self.fit_method == FitMethod.LILLIS:
            chi2 = compute_lillis_chi2_batch_torch(
                models, norm2d_t, data_mask_t, model_mask=model_masks
            )
        else:
            chi2 = compute_chi2_batch_torch(
                models, norm2d_t, data_mask_t, eps=config.EPS, model_mask=model_masks
            )

        chi2 = torch.where(
            torch.isfinite(chi2), chi2, torch.tensor(1e30, device=self.device)
        )
        best_idx = int(torch.argmin(chi2).item())

        return ChunkFitResult(
            u_surface=float(u_surface[best_idx].item()),
            bs_over_bm=float(bs_over_bm[best_idx].item()),
            beam_amp=float(beam_amp[best_idx].item()),
            chi2=float(chi2[best_idx].item()),
            chunk_index=measurement_chunk,
        )

    def _fit_batch_lhs(
        self,
        energies: Tensor,
        pitches: Tensor,
        norm2d: Tensor,
        data_mask: Tensor,
        sc_potential: Tensor,
        n_lhs: int | None = None,
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
        if n_lhs is None:
            n_lhs = int(config.LOSS_CONE_LHS_SAMPLES)
        n_lhs = int(n_lhs)
        N_chunks = energies.size(0)

        # Precompute log(data) once to avoid redundant computation in chi2
        # (Halekas only).
        log_data_precomputed = None
        if self.fit_method != FitMethod.LILLIS:
            log_data_precomputed = precompute_log_data_torch(norm2d, data_mask)

        # Generate LHS samples (same for all chunks; shared with CPU)
        lhs_np = losscone_lhs_samples(
            n_samples=n_lhs,
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
            seed=config.LOSS_CONE_LHS_SEED,
        )
        lhs_samples = torch.tensor(
            lhs_np, device=self.device, dtype=self.dtype
        )  # (n_lhs, 3)

        # Expand to (N_chunks, n_lhs, 3)
        lhs_expanded = lhs_samples.unsqueeze(0).expand(N_chunks, -1, -1)

        # Extract parameters
        U_surface = lhs_expanded[:, :, 0]  # (N, n_lhs)
        bs_over_bm = lhs_expanded[:, :, 1]
        beam_amp = lhs_expanded[:, :, 2]

        # Use fixed beam width (not scaling with |U_surface|)
        beam_width = torch.full_like(U_surface, self.beam_width_ev)

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

        # Valid-energy mask (E >= U_spacecraft); broadcastable to
        # (N, n_pop, nE, nPitch).
        if sc_potential.dim() == 1:
            valid_energy = energies >= sc_potential.view(N_chunks, 1)
        else:
            valid_energy = energies >= sc_potential
        model_mask = valid_energy.view(N_chunks, 1, energies.size(1), 1)

        # Compute chi2 for all
        if self.fit_method == FitMethod.LILLIS:
            chi2 = compute_lillis_chi2_multi_chunk_torch(
                models, norm2d, data_mask, model_mask=model_mask
            )
        else:
            chi2 = compute_chi2_multi_chunk_torch(
                models,
                norm2d,
                data_mask,
                log_data_precomputed=log_data_precomputed,
                model_mask=model_mask,
            )  # (N, n_lhs)

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
        popsize: int | None = None,
        maxiter: int | None = None,
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
        if popsize is None:
            popsize = int(config.LOSS_CONE_DE_POPSIZE)
        if maxiter is None:
            maxiter = int(config.LOSS_CONE_DE_MAXITER)
        popsize = int(popsize)
        maxiter = int(maxiter)

        N_chunks = energies.size(0)

        # Precompute log(data) once to avoid redundant computation in chi2
        # (Halekas only).
        log_data_precomputed = None
        if self.fit_method != FitMethod.LILLIS:
            log_data_precomputed = precompute_log_data_torch(norm2d, data_mask)

        bounds = losscone_optimizer_bounds(
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
        )

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

            # Use fixed beam width (not scaling with |U_surface|)
            beam_width = torch.full_like(U_surface, self.beam_width_ev)

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

            if sc_potential.dim() == 1:
                valid_energy = energies >= sc_potential.view(N_chunks, 1)
            else:
                valid_energy = energies >= sc_potential
            model_mask = valid_energy.view(N_chunks, 1, energies.size(1), 1)

            if self.fit_method == FitMethod.LILLIS:
                chi2 = compute_lillis_chi2_multi_chunk_torch(
                    models, norm2d, data_mask, model_mask=model_mask
                )
            else:
                chi2 = compute_chi2_multi_chunk_torch(
                    models,
                    norm2d,
                    data_mask,
                    log_data_precomputed=log_data_precomputed,
                    model_mask=model_mask,
                )
            chi2 = torch.where(
                torch.isfinite(chi2), chi2, torch.tensor(1e30, device=self.device)
            )

            return chi2

        # Create multi-spectrum DE optimizer
        de = BatchedDifferentialEvolution(
            bounds=bounds,
            n_spectra=N_chunks,
            popsize=popsize,
            mutation=float(config.LOSS_CONE_DE_MUTATION),
            crossover=float(config.LOSS_CONE_DE_CROSSOVER),
            maxiter=maxiter,
            atol=float(config.LOSS_CONE_DE_ATOL),
            seed=int(config.LOSS_CONE_DE_SEED),
            device=str(self.device),
            dtype=self.dtype,
        )

        # Seed population with LHS results
        # Note: x0 is (N, 3), DE will use this as first member of each population
        best_params, best_chi2, n_iter = de.optimize(objective, x0=x0)

        return best_params, best_chi2, n_iter

    def fit_surface_potential_batched(
        self, batch_size: int | None = None, n_lhs: int | None = None
    ) -> np.ndarray:
        """
        Fit surface potential for all chunks using batched GPU optimization.

        Processes chunks in mega-batches to manage VRAM, with all chunks in a
        batch optimized simultaneously using multi-spectrum DE.

        Args:
            batch_size: Number of chunks to process simultaneously. If None,
                auto-detects based on available VRAM and dtype.
            n_lhs: Number of LHS samples for initial grid search (defaults to config)

        Returns:
            Array with columns [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
        """
        from tqdm import tqdm

        # Auto-detect batch size if not specified
        if batch_size is None:
            batch_size = _auto_detect_batch_size(self.device, self.dtype)

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
                bs_over_bm = float(
                    np.clip(
                        final_params_np[i, 1],
                        self.bs_over_bm_min,
                        self.bs_over_bm_max,
                    )
                )
                beam_amp = float(
                    np.clip(final_params_np[i, 2], self.beam_amp_min, self.beam_amp_max)
                )
                chi2 = final_chi2_np[i]

                if (
                    self.fit_method == FitMethod.LILLIS
                    and chi2 > self.config.LILLIS_CHI2_REDUCED_MAX
                ):
                    results[chunk_idx] = [np.nan, np.nan, np.nan, chi2, chunk_idx]
                    continue

                results[chunk_idx] = [U_surface, bs_over_bm, beam_amp, chi2, chunk_idx]

        return results


# Backwards compatibility alias
LossConeFitterGPU = LossConeFitterTorch
