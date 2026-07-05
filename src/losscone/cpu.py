from collections.abc import Callable

import numpy as np
from tqdm import tqdm

from src import config
from src.losscone.chi2 import (
    compute_halekas_chi2,
    compute_halekas_chi2_batch,
    compute_lillis_chi2,
    compute_lillis_chi2_batch,
)
from src.losscone.er_data import ERData
from src.losscone.fitter_base import LossConeFitterBase
from src.losscone.masks import build_lillis_mask
from src.losscone.model import synth_losscone, synth_losscone_batch
from src.losscone.params import losscone_lhs_samples, losscone_optimizer_bounds
from src.losscone.pitch_angle import PitchAngle
from src.losscone.types import (
    ChunkFitResult,
    FitChunkData,
    FitMethod,
    NormalizationMode,
    parse_fit_method,
    parse_normalization_mode,
)
from src.utils import thetas as thetas_module

__all__ = [
    "ChunkFitResult",
    "ERData",
    "FitChunkData",
    "FitMethod",
    "LossConeFitter",
    "NormalizationMode",
    "PitchAngle",
    "build_lillis_mask",
    "compute_halekas_chi2",
    "compute_lillis_chi2",
]


class LossConeFitter(LossConeFitterBase):
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
    ):
        """
        Initialize the LossConeFitter class with the ER data and theta values.

        Args:
            er_data (ERData): The ER data object.
            pitch_angle (PitchAngle, optional): Pre-computed pitch angle
                object. If None, creates a new one.
            spacecraft_potential (np.ndarray | None): Optional per-row spacecraft
                potential [V] aligned with `er_data.data`; used in synthetic model.
            normalization_mode (str): How to normalize flux for fitting.
                - "ratio": per-energy normalization by an incident statistic
                  (default; approximates reflected/incident normalization)
                - "ratio2": pairwise normalization (incident→1,
                  reflected→reflected/incident) (closest to Halekas 2008 wording)
            fit_method (str | None): Loss-cone fitting method ("halekas" or "lillis").
            beam_amp_fixed (float | None): If set, fix the Gaussian beam amplitude
                to this value instead of fitting it.
            incident_flux_stat (str): Statistic for incident flux normalization
                ("mean" or "max").
            loss_cone_background (float | None): Baseline model value outside the
                loss cone to stabilise log-space chi2 (defaults to config value).
        """
        self.er_data = er_data
        self.thetas = thetas_module.get_thetas()
        self.pitch_angle = pitch_angle or PitchAngle(er_data)
        self.spacecraft_potential = spacecraft_potential

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

        self.lhs = self._generate_latin_hypercube()

    def _generate_latin_hypercube(self, n_samples: int | None = None) -> np.ndarray:
        """
        Generate a Latin Hypercube sample.

        Returns:
            np.ndarray: The Latin Hypercube sample.
        """
        if n_samples is None:
            n_samples = int(config.LOSS_CONE_LHS_SAMPLES)
        return losscone_lhs_samples(
            n_samples=n_samples,
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
            seed=config.LOSS_CONE_LHS_SEED,
        )

    def _run_differential_evolution(
        self,
        objective_fn: Callable[[np.ndarray], float],
        bounds: list[tuple[float, float]],
        *,
        x0: np.ndarray,
        best_lhs_chi2: float,
    ) -> tuple[np.ndarray, float]:
        """
        Run SciPy differential evolution with shared (CPU/torch) optimizer settings.

        SciPy defines popsize as a multiplier of dimension (popsize * n_params),
        while our torch DE uses an absolute population size. We convert here so
        config.LOSS_CONE_DE_POPSIZE has consistent semantics across backends.
        """
        from scipy.optimize import differential_evolution

        n_params = len(bounds)
        popsize_mult = max(
            1, (int(config.LOSS_CONE_DE_POPSIZE) + n_params - 1) // n_params
        )
        result = differential_evolution(
            objective_fn,
            bounds,
            strategy="best1bin",
            popsize=popsize_mult,
            mutation=float(config.LOSS_CONE_DE_MUTATION),
            recombination=float(config.LOSS_CONE_DE_CROSSOVER),
            seed=int(config.LOSS_CONE_DE_SEED),
            maxiter=int(config.LOSS_CONE_DE_MAXITER),
            atol=float(config.LOSS_CONE_DE_ATOL),
            tol=float(config.LOSS_CONE_DE_TOL),
            polish=bool(config.LOSS_CONE_DE_POLISH),
            workers=1,  # Single-threaded for reproducibility
            updating="deferred",  # Faster convergence
            x0=np.asarray(x0, dtype=float),
        )

        if result.success and np.isfinite(result.fun):
            best_params = np.asarray(result.x, dtype=float)
            best_chi2 = float(result.fun)
        else:
            best_params = np.asarray(x0, dtype=float)
            best_chi2 = float(best_lhs_chi2)

        if best_lhs_chi2 < best_chi2:
            best_params = np.asarray(x0, dtype=float)
            best_chi2 = float(best_lhs_chi2)

        return best_params, best_chi2

    def _get_normalized_flux(
        self, energy_bin: int, measurement_chunk: int
    ) -> np.ndarray:
        """
        Get the normalized flux for a specific energy bin and measurement chunk.

        Args:
            energy_bin (int): The index of the energy bin.
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            np.ndarray: The normalized flux for the specified energy bin and
                measurement chunk.
        """
        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )

        index = measurement_chunk * config.SWEEP_ROWS + energy_bin

        if index >= len(self.er_data.data):
            return np.full(config.CHANNELS, np.nan)

        electron_flux = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[
            index
        ]
        if len(self.pitch_angle.pitch_angles) == 0:
            return np.full(config.CHANNELS, np.nan)
        angles = self.pitch_angle.pitch_angles[index]
        incident_mask = angles < 90
        # TODO: Check reconsider the reflected mask
        # reflected_mask = ~incident_mask

        # Check if the electron flux is valid
        if not incident_mask.any():
            return np.full_like(electron_flux, np.nan)

        incident_vals = electron_flux[incident_mask]
        incident_vals = incident_vals[np.isfinite(incident_vals)]
        incident_vals = incident_vals[incident_vals > 0]
        if len(incident_vals) == 0:
            return np.full_like(electron_flux, np.nan)

        if self.incident_flux_stat == "mean":
            incident_flux = float(np.mean(incident_vals))
        else:
            incident_flux = float(np.max(incident_vals))

        incident_flux = max(config.EPS, incident_flux)
        return electron_flux / incident_flux

    def build_norm2d(self, measurement_chunk: int) -> np.ndarray:
        """
        Build a 2D normalized flux distribution for a specific measurement chunk.

        Normalization modes:
        - 'ratio': per-energy normalization by incident flux
        - 'ratio2': pairwise normalization (incident→1.0, reflected→reflected/incident)

        Args:
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            np.ndarray: The 2D normalized flux distribution.
        """
        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )

        if self.normalization_mode == NormalizationMode.RATIO:
            # Per-energy normalization: divide each energy by its own incident flux
            norm2d = np.vstack(
                [
                    self._get_normalized_flux(energy_bin, measurement_chunk)
                    for energy_bin in range(config.SWEEP_ROWS)
                ]
            )
        elif self.normalization_mode == NormalizationMode.RATIO2:
            # Pairwise normalization: mirror incident/reflected angles around 90°
            # Each reflected angle normalized by its closest mirrored incident angle
            s = measurement_chunk * config.SWEEP_ROWS
            e = min((measurement_chunk + 1) * config.SWEEP_ROWS, len(self.er_data.data))

            flux_2d = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[
                s:e
            ]
            pitches_2d = self.pitch_angle.pitch_angles[s:e]

            nE, nPitch = flux_2d.shape
            norm2d = np.full((nE, nPitch), np.nan, dtype=np.float64)

            for row in range(nE):
                pitch_row = pitches_2d[row]
                flux_row = flux_2d[row]

                incident_mask = pitch_row < 90.0
                reflected_mask = ~incident_mask

                incident_idx = np.nonzero(incident_mask)[0]
                reflected_idx = np.nonzero(reflected_mask)[0]

                if len(incident_idx) == 0 or len(reflected_idx) == 0:
                    continue

                incident_flux = flux_row[incident_idx]
                valid_incident = (incident_flux > 0) & np.isfinite(incident_flux)
                if not valid_incident.any():
                    continue

                valid_inc_indices = incident_idx[valid_incident]
                norm2d[row, valid_inc_indices] = 1.0

                for i_ref in reflected_idx:
                    ref_flux = flux_row[i_ref]
                    if ref_flux <= 0 or not np.isfinite(ref_flux):
                        continue

                    target_angle = 180.0 - pitch_row[i_ref]
                    mirror_idx = valid_inc_indices[
                        np.argmin(np.abs(pitch_row[valid_inc_indices] - target_angle))
                    ]
                    denom = flux_row[mirror_idx]
                    if denom <= 0 or not np.isfinite(denom):
                        continue

                    norm2d[row, i_ref] = ref_flux / denom

                # Ensure the bin closest to 90° is defined
                mid = int(np.argmin(np.abs(pitch_row - 90.0)))
                norm2d[row, mid] = 1.0

        else:
            raise ValueError(
                f"Unsupported normalization_mode: {self.normalization_mode}"
            )

        return norm2d

    def _prepare_chunk_data(self, measurement_chunk: int) -> FitChunkData | None:
        if self.er_data.data.empty:
            return None

        norm2d = self.build_norm2d(measurement_chunk)
        if np.isnan(norm2d).all():
            return None

        s = measurement_chunk * config.SWEEP_ROWS
        e = (measurement_chunk + 1) * config.SWEEP_ROWS
        max_rows = len(self.er_data.data)
        if s >= max_rows:
            return None
        e = min(e, max_rows)

        energies = self.er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[
            s:e
        ]
        if self.pitch_angle.pitch_angles is None or s >= len(
            self.pitch_angle.pitch_angles
        ):
            return None
        pitches = self.pitch_angle.pitch_angles[s:e]

        spacecraft_slice = (
            self.spacecraft_potential[s:e]
            if self.spacecraft_potential is not None
            else 0.0
        )

        actual_rows = e - s
        if norm2d.shape[0] > actual_rows:
            norm2d = norm2d[:actual_rows]
        if pitches.shape[0] > actual_rows:
            pitches = pitches[:actual_rows]

        raw_flux = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[s:e]
        if raw_flux.shape[0] > actual_rows:
            raw_flux = raw_flux[:actual_rows]

        if isinstance(spacecraft_slice, np.ndarray) and (
            spacecraft_slice.shape[0] > actual_rows
        ):
            spacecraft_slice = spacecraft_slice[:actual_rows]

        if isinstance(spacecraft_slice, np.ndarray):
            valid_energy = energies[:, None] >= spacecraft_slice[:, None]
        else:
            valid_energy = energies[:, None] >= float(spacecraft_slice)
        valid_energy_mask = np.broadcast_to(valid_energy, pitches.shape)

        return FitChunkData(
            norm2d=norm2d,
            energies=energies,
            pitches=pitches,
            raw_flux=raw_flux,
            spacecraft_slice=spacecraft_slice,
            valid_energy_mask=valid_energy_mask,
        )

    def build_norm2d_batch(self, chunk_indices: list[int]) -> np.ndarray:
        """
        Build normalized 2D flux distributions for multiple chunks at once.

        Vectorized implementation for significant speedup over calling
        build_norm2d() in a loop.

        Args:
            chunk_indices: List of measurement chunk indices to process

        Returns:
            np.ndarray: Shape (n_chunks, SWEEP_ROWS, CHANNELS) normalized flux.
                        Invalid chunks are filled with NaN.
        """
        if not chunk_indices:
            return np.zeros((0, config.SWEEP_ROWS, config.CHANNELS), dtype=np.float64)

        n_chunks = len(chunk_indices)
        n_rows = len(self.er_data.data)
        nE = config.SWEEP_ROWS
        nP = config.CHANNELS

        # Load all flux and pitch data once
        flux_all = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)
        pitches_all = self.pitch_angle.pitch_angles

        # Build index arrays for all chunks
        chunk_indices_arr = np.array(chunk_indices, dtype=np.int64)
        start_indices = chunk_indices_arr * nE
        valid_chunks = start_indices < n_rows

        # Pre-allocate output
        result = np.full((n_chunks, nE, nP), np.nan, dtype=np.float64)

        if not valid_chunks.any():
            return result

        # Get valid chunk data
        valid_chunk_idx = np.where(valid_chunks)[0]

        if self.normalization_mode == NormalizationMode.RATIO:
            # Fully vectorized per-energy normalization
            for i in valid_chunk_idx:
                chunk_idx = chunk_indices[i]
                s = chunk_idx * nE
                e = min(s + nE, n_rows)
                actual_rows = e - s

                flux_chunk = flux_all[s:e]  # (actual_rows, nP)
                pitch_chunk = pitches_all[s:e]  # (actual_rows, nP)

                # Incident mask per row
                incident_mask = pitch_chunk < 90.0  # (actual_rows, nP)

                # Valid flux mask
                valid_flux = np.isfinite(flux_chunk) & (flux_chunk > 0)

                # Combined mask for valid incident flux
                valid_incident = incident_mask & valid_flux  # (actual_rows, nP)

                # Compute normalization factor per row using masked operations
                # Replace non-incident values with NaN for aggregation
                flux_for_norm = np.where(valid_incident, flux_chunk, np.nan)

                if self.incident_flux_stat == "mean":
                    # nanmean per row
                    norm_factors = np.nanmean(flux_for_norm, axis=1)  # (actual_rows,)
                else:
                    # nanmax per row
                    norm_factors = np.nanmax(flux_for_norm, axis=1)  # (actual_rows,)

                # Handle rows with no valid incident flux
                norm_factors = np.where(
                    np.isfinite(norm_factors) & (norm_factors > 0), norm_factors, np.nan
                )
                norm_factors = np.maximum(norm_factors, config.EPS)

                # Normalize: flux / norm_factor (broadcast over columns)
                result[i, :actual_rows, :] = flux_chunk / norm_factors[:, np.newaxis]

        elif self.normalization_mode == NormalizationMode.RATIO2:
            # ratio2 - complex pairwise normalization; fall back to per-chunk
            for i in valid_chunk_idx:
                result[i] = self.build_norm2d(chunk_indices[i])

        else:
            raise ValueError(
                f"Unsupported normalization_mode: {self.normalization_mode}"
            )

        return result

    def _fit_surface_potential(self, measurement_chunk: int) -> ChunkFitResult:
        """
        Fit surface potential (U_surface) and B_s/B_m for one 15-row measurement chunk
        using chi2 minimisation. Method is controlled by fit_method.

        Args:
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            ChunkFitResult: Fit result (unpackable as a 4-tuple for compatibility).
        """
        if self.fit_method == FitMethod.LILLIS:
            return self._fit_surface_potential_lillis(measurement_chunk)

        data = self._prepare_chunk_data(measurement_chunk)
        if data is None:
            return ChunkFitResult.invalid(measurement_chunk)

        eps = config.EPS
        norm2d = data.norm2d
        energies = data.energies
        pitches = data.pitches
        spacecraft_slice = data.spacecraft_slice

        if not data.has_enough_valid_bins(FitMethod.HALEKAS):
            return ChunkFitResult.invalid(measurement_chunk)

        # self.lhs is (N_samples, 3) -> [U_surface, bs_over_bm, beam_amp]
        lhs_U_surface = self.lhs[:, 0]
        lhs_bs_over_bm = self.lhs[:, 1]
        lhs_beam_amp = self.lhs[:, 2]

        # Use fixed beam width (not scaling with |U_surface| to prevent runaway)
        lhs_beam_width = np.full_like(lhs_U_surface, self.beam_width_ev)

        # Evaluate models in batch: (N_samples, nE, nPitch)
        models, model_masks = synth_losscone_batch(
            energy_grid=energies,
            pitch_grid=pitches,
            U_surface=lhs_U_surface,
            U_spacecraft=spacecraft_slice,
            bs_over_bm=lhs_bs_over_bm,
            beam_width_eV=lhs_beam_width,
            beam_amp=lhs_beam_amp,
            beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
            background=self.background,
            return_mask=True,
        )
        chi2_vals = compute_halekas_chi2_batch(
            norm2d=norm2d, models=models, model_mask=model_masks, eps=eps
        )

        bad_mask = ~np.isfinite(chi2_vals)
        chi2_vals[bad_mask] = 1e30

        best_idx = int(np.argmin(chi2_vals))
        best_lhs_chi2 = chi2_vals[best_idx]
        x0 = self.lhs[best_idx]

        # 2) Global optimization with differential_evolution
        # Objective for optimizer (scalar)
        def chi2_scalar(params):
            U_surface, bs_over_bm, beam_amp = params
            beam_amp = float(np.clip(beam_amp, self.beam_amp_min, self.beam_amp_max))
            # Use fixed beam width (not scaling with |U_surface|)
            beam_width = self.beam_width_ev
            model = synth_losscone(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=U_surface,
                U_spacecraft=spacecraft_slice,
                bs_over_bm=bs_over_bm,
                beam_width_eV=beam_width,
                beam_amp=beam_amp,
                beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
                background=self.background,
            )

            if not np.all(np.isfinite(model)):
                return 1e30  # big penalty

            chi2 = compute_halekas_chi2(
                norm2d=norm2d,
                model=model,
                model_mask=data.valid_energy_mask,
                eps=eps,
            )

            if not np.isfinite(chi2):
                return 1e30
            return chi2

        # Use constrained bounds: U_surface capped at detection threshold (~+20V)
        # because electron reflectometry cannot measure positive potentials reliably
        bounds = losscone_optimizer_bounds(
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
        )
        best_params, best_chi2 = self._run_differential_evolution(
            chi2_scalar,
            bounds,
            x0=x0,
            best_lhs_chi2=float(best_lhs_chi2),
        )

        U_surface, bs_over_bm, beam_amp = best_params

        # Clip to ensure exact bounds (DE should respect them, but be safe)
        bs_over_bm = float(
            np.clip(bs_over_bm, self.bs_over_bm_min, self.bs_over_bm_max)
        )
        beam_amp = float(np.clip(beam_amp, self.beam_amp_min, self.beam_amp_max))

        return ChunkFitResult(
            u_surface=float(U_surface),
            bs_over_bm=bs_over_bm,
            beam_amp=beam_amp,
            chi2=float(best_chi2),
            chunk_index=measurement_chunk,
        )

    def fit_chunk_full(self, measurement_chunk: int) -> ChunkFitResult:
        return self._fit_surface_potential(measurement_chunk)

    def fit_chunk_lhs(
        self,
        measurement_chunk: int,
        beam_width_ev: float | None = None,
        u_spacecraft: float | None = None,
        n_samples: int | None = None,
    ) -> ChunkFitResult:
        data = self._prepare_chunk_data(measurement_chunk)
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

        if n_samples is None:
            n_samples = len(self.lhs)
        samples = (
            self.lhs
            if n_samples == len(self.lhs)
            else self._generate_latin_hypercube(n_samples)
        )
        u_surface = samples[:, 0]
        bs_over_bm = samples[:, 1]
        beam_amp = samples[:, 2]
        width = self.beam_width_ev if beam_width_ev is None else beam_width_ev
        beam_width = np.full_like(u_surface, width)
        background = np.full_like(u_surface, self.background)

        models, model_masks = synth_losscone_batch(
            energy_grid=energies,
            pitch_grid=pitches,
            U_surface=u_surface,
            U_spacecraft=spacecraft_slice,
            bs_over_bm=bs_over_bm,
            beam_width_eV=beam_width,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
            background=background,
            return_mask=True,
        )
        if self.fit_method == FitMethod.LILLIS:
            chi2 = compute_lillis_chi2_batch(
                norm2d=norm2d,
                models=models,
                raw_flux=data.raw_flux,
                pitches=pitches,
                model_mask=model_masks,
            )
        else:
            chi2 = compute_halekas_chi2_batch(
                norm2d=norm2d, models=models, model_mask=model_masks, eps=config.EPS
            )

        chi2[~np.isfinite(chi2)] = 1e30
        best_idx = int(np.argmin(chi2))
        return ChunkFitResult(
            u_surface=float(u_surface[best_idx]),
            bs_over_bm=float(bs_over_bm[best_idx]),
            beam_amp=float(beam_amp[best_idx]),
            chi2=float(chi2[best_idx]),
            chunk_index=measurement_chunk,
        )

    def _fit_surface_potential_lillis_with_u_width_qc(
        self, measurement_chunk: int, *, delta_reduced: float = 0.001
    ) -> tuple[ChunkFitResult, float]:
        """
        Fit surface potential using Lillis-style masked linear chi2 + DE refinement.

        Uses a relative-flux mask to exclude low/zero bins, then minimizes
        linear-space chi2. Returns reduced chi2 for quality control.
        """
        delta_reduced = float(delta_reduced)
        if delta_reduced <= 0:
            raise ValueError("delta_reduced must be > 0")
        data = self._prepare_chunk_data(measurement_chunk)
        if data is None:
            return ChunkFitResult.invalid(measurement_chunk), float("nan")

        norm2d = data.norm2d
        energies = data.energies
        pitches = data.pitches
        spacecraft_slice = data.spacecraft_slice

        if not data.has_enough_valid_bins(FitMethod.LILLIS):
            return ChunkFitResult.invalid(measurement_chunk), float("nan")

        # LHS samples provide a robust starting point
        lhs_U_surface = self.lhs[:, 0]
        lhs_bs_over_bm = self.lhs[:, 1]
        lhs_beam_amp = self.lhs[:, 2]
        lhs_beam_width = np.full_like(lhs_U_surface, self.beam_width_ev)

        models, model_masks = synth_losscone_batch(
            energy_grid=energies,
            pitch_grid=pitches,
            U_surface=lhs_U_surface,
            U_spacecraft=spacecraft_slice,
            bs_over_bm=lhs_bs_over_bm,
            beam_width_eV=lhs_beam_width,
            beam_amp=lhs_beam_amp,
            beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
            background=self.background,
            return_mask=True,
        )
        chi2_vals = compute_lillis_chi2_batch(
            norm2d=norm2d,
            models=models,
            raw_flux=data.raw_flux,
            pitches=pitches,
            model_mask=model_masks,
        )
        chi2_vals[~np.isfinite(chi2_vals)] = 1e30

        best_idx = int(np.argmin(chi2_vals))
        best_lhs_chi2 = float(chi2_vals[best_idx])
        x0 = self.lhs[best_idx]

        bounds = losscone_optimizer_bounds(
            u_surface_min=self.u_surface_min,
            u_surface_max=self.u_surface_max,
            bs_over_bm_min=self.bs_over_bm_min,
            bs_over_bm_max=self.bs_over_bm_max,
            beam_amp_min=self.beam_amp_min,
            beam_amp_max=self.beam_amp_max,
        )

        def chi2_scalar(params):
            U_surface, bs_over_bm, beam_amp = params
            if not np.isfinite(params).all():
                return 1e30
            beam_amp = float(np.clip(beam_amp, self.beam_amp_min, self.beam_amp_max))

            model = synth_losscone(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=U_surface,
                U_spacecraft=spacecraft_slice,
                bs_over_bm=bs_over_bm,
                beam_width_eV=self.beam_width_ev,
                beam_amp=beam_amp,
                beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
                background=self.background,
            )
            if not np.all(np.isfinite(model)):
                return 1e30
            chi2_val = compute_lillis_chi2(
                norm2d=norm2d,
                model=model,
                raw_flux=data.raw_flux,
                pitches=pitches,
                model_mask=data.valid_energy_mask,
            )
            if not np.isfinite(chi2_val):
                return 1e30
            return float(chi2_val)

        best_params, best_chi2 = self._run_differential_evolution(
            chi2_scalar,
            bounds,
            x0=x0,
            best_lhs_chi2=best_lhs_chi2,
        )

        U_surface, bs_over_bm, beam_amp = best_params
        # Clip to bounds defensively
        U_surface = float(np.clip(U_surface, bounds[0][0], bounds[0][1]))
        bs_over_bm = float(np.clip(bs_over_bm, bounds[1][0], bounds[1][1]))
        beam_amp = float(np.clip(beam_amp, bounds[2][0], bounds[2][1]))

        if not np.isfinite(best_chi2) or best_chi2 > config.LILLIS_CHI2_REDUCED_MAX:
            return (
                ChunkFitResult.invalid(measurement_chunk, chi2=float(best_chi2)),
                float("nan"),
            )

        keep = chi2_vals <= (best_lhs_chi2 + delta_reduced)
        keep = keep & np.isfinite(lhs_U_surface) & np.isfinite(chi2_vals)
        u_width_lhs = float("nan")
        if np.any(keep):
            u_min = float(np.min(lhs_U_surface[keep]))
            u_max = float(np.max(lhs_U_surface[keep]))
            u_width_lhs = u_max - u_min

        return (
            ChunkFitResult(
                u_surface=U_surface,
                bs_over_bm=bs_over_bm,
                beam_amp=beam_amp,
                chi2=float(best_chi2),
                chunk_index=measurement_chunk,
            ),
            u_width_lhs,
        )

    def _fit_surface_potential_lillis(self, measurement_chunk: int) -> ChunkFitResult:
        result, _u_width = self._fit_surface_potential_lillis_with_u_width_qc(
            measurement_chunk, delta_reduced=0.001
        )
        return result

    def fit_surface_potential(self) -> np.ndarray:
        """
        Fit surface potential (U_surface) and B_s/B_m for all 15-row measurement chunks
        using chi2 minimisation with scipy.optimize.minimize.

        Returns:
            np.ndarray: Array with columns
                [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
                - U_surface: best-fit surface potential in volts
                - bs_over_bm: best-fit B_s/B_m ratio
                - beam_amp: best-fit Gaussian beam amplitude
                - chi2: final chi2 value
                - chunk_index: measurement chunk index
        """
        assert not self.er_data.data.empty, "Data not loaded."

        # Fit for each chunk independently (no warm-starting)
        n_chunks = len(self.er_data.data) // config.SWEEP_ROWS
        results = np.zeros((n_chunks, 5))

        for i in tqdm(
            range(n_chunks), desc="Fitting chunks", unit="chunk", dynamic_ncols=True
        ):
            result = self.fit_chunk_full(i)
            results[i] = result.as_row()

        return results

    def fit_surface_potential_with_u_width_qc(
        self, *, delta_reduced: float = 0.001
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit surface potential and return a per-chunk identifiability proxy for
        U_surface.

        The proxy is computed from the LHS phase for the Lillis fitter:
        U-width over LHS samples with chi2_red <= chi2_red_min + delta_reduced.

        Returns:
            results: (n_chunks, 5) array [U_surface, bs_over_bm, beam_amp, chi2,
                chunk_index]
            u_width_lhs: (n_chunks,) U-width [V] (NaN for non-Lillis or failed fits)
        """
        assert not self.er_data.data.empty, "Data not loaded."
        delta_reduced = float(delta_reduced)
        if delta_reduced <= 0:
            raise ValueError("delta_reduced must be > 0")

        n_chunks = len(self.er_data.data) // config.SWEEP_ROWS
        results = np.zeros((n_chunks, 5))
        u_width = np.full(n_chunks, np.nan, dtype=np.float64)

        for i in tqdm(
            range(n_chunks),
            desc="Fitting chunks (+U QC)",
            unit="chunk",
            dynamic_ncols=True,
        ):
            if self.fit_method == FitMethod.LILLIS:
                fit, u_w = self._fit_surface_potential_lillis_with_u_width_qc(
                    i, delta_reduced=delta_reduced
                )
                u_width[i] = u_w
            else:
                fit = self.fit_chunk_full(i)
            results[i] = fit.as_row()

        return results, u_width
