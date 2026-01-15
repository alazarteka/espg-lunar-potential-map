from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

from src import config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.model import synth_losscone, synth_losscone_batch

try:  # Optional GPU path
    import torch

    from src.model_torch import (
        HAS_TORCH,
        _auto_detect_dtype,
        synth_losscone_batch_torch,
    )
except Exception:  # pragma: no cover - optional GPU path
    HAS_TORCH = False
    torch = None  # type: ignore[assignment]
    _auto_detect_dtype = None  # type: ignore[assignment]
    synth_losscone_batch_torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChunkData:
    energies: np.ndarray
    pitches: np.ndarray
    flux: np.ndarray
    spec_no: int
    timestamp: str


def interpolate_to_regular_grid(
    energies: np.ndarray,
    pitches: np.ndarray,
    flux_data: np.ndarray,
    n_pitch_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pitch_min = np.nanmin(pitches)
    pitch_max = np.nanmax(pitches)
    pitches_reg = np.linspace(pitch_min, pitch_max, n_pitch_bins)

    flux_reg = np.zeros((len(energies), n_pitch_bins))
    for i in range(len(energies)):
        valid_mask = np.isfinite(flux_data[i]) & np.isfinite(pitches[i])
        if np.sum(valid_mask) > 1:
            pitch_pts = pitches[i, valid_mask]
            flux_pts = flux_data[i, valid_mask]
            sort_idx = np.argsort(pitch_pts)
            pitch_pts_sorted = pitch_pts[sort_idx]
            flux_pts_sorted = flux_pts[sort_idx]
            flux_reg[i] = np.interp(
                pitches_reg,
                pitch_pts_sorted,
                flux_pts_sorted,
                left=np.nan,
                right=np.nan,
            )
        else:
            flux_reg[i] = np.nan

    return energies, pitches_reg, flux_reg


def compute_loss_cone_boundary(
    energies: np.ndarray,
    u_surface: float,
    bs_over_bm: float,
    u_spacecraft: float,
) -> np.ndarray:
    e_corr = energies - u_spacecraft
    loss_cone = np.full_like(energies, np.nan, dtype=float)
    valid = e_corr > 0
    if not np.any(valid):
        return loss_cone
    x = bs_over_bm * (1.0 + u_surface / np.maximum(e_corr, config.EPS))
    x = np.clip(x, 0.0, 1.0)
    ac_deg = np.degrees(np.arcsin(np.sqrt(x)))
    loss_cone[valid] = 180.0 - ac_deg[valid]
    return loss_cone


class LossConeSession:
    def __init__(
        self,
        er_file: Path,
        theta_file: Path | None = None,
        normalization_mode: str = "ratio",
        incident_flux_stat: str = "mean",
        loss_cone_background: float | None = None,
        use_torch: bool = False,
        torch_device: str | None = None,
        use_polarity: bool = True,
    ) -> None:
        self.er_file = Path(er_file)
        self.theta_file = (
            Path(theta_file)
            if theta_file is not None
            else config.DATA_DIR / config.THETA_FILE
        )
        self.normalization_mode = normalization_mode
        self.incident_flux_stat = incident_flux_stat
        self.background = (
            float(loss_cone_background)
            if loss_cone_background is not None
            else float(config.LOSS_CONE_BACKGROUND)
        )

        if not self.er_file.exists():
            raise FileNotFoundError(f"ER file not found: {self.er_file}")
        if not self.theta_file.exists():
            raise FileNotFoundError(f"Theta file not found: {self.theta_file}")

        self.er_data = ERData(str(self.er_file))
        if self.er_data.data.empty:
            raise ValueError(f"No data loaded from {self.er_file}")
        self.polarity = self._compute_polarity() if use_polarity else None
        if self.polarity is not None:
            self.er_data.data, self.polarity = self._filter_zero_polarity(
                self.er_data.data, self.polarity
            )
        self.pitch_angle = PitchAngle(self.er_data, polarity=self.polarity)

        self._norm_cache: dict[tuple[int, str, str], np.ndarray] = {}
        self._raw_cache: dict[int, ChunkData] = {}
        self._spec_to_chunk = self._build_spec_map()

        self.use_torch = bool(use_torch and HAS_TORCH)
        self._torch_device = None
        self._torch_dtype = None
        if self.use_torch and torch is not None:
            self._torch_device = torch.device(
                torch_device
                if torch_device
                else ("cuda" if torch.cuda.is_available() else "cpu")
            )
            self._torch_dtype = _auto_detect_dtype(self._torch_device)

        self._init_fitter()

    def _filter_zero_polarity(self, data, polarity: np.ndarray):
        if len(polarity) != len(data):
            logger.warning(
                "Polarity length mismatch; skipping zero-polarity filtering."
            )
            return data, polarity

        zero_mask = polarity == 0
        if not np.any(zero_mask):
            return data, polarity

        spec_vals = data[config.SPEC_NO_COLUMN].to_numpy()
        bad_specs = set(spec_vals[zero_mask])
        if not bad_specs:
            return data, polarity

        keep_mask = ~data[config.SPEC_NO_COLUMN].isin(bad_specs)
        filtered = data[keep_mask].reset_index(drop=True)
        filtered_polarity = polarity[keep_mask]

        logger.info(
            "Filtered %d sweeps with zero polarity (%d rows removed).",
            len(bad_specs),
            len(data) - len(filtered),
        )
        return filtered, filtered_polarity

    def _compute_polarity(self) -> np.ndarray | None:
        try:
            from src.potential_mapper.coordinates import (
                CoordinateCalculator,
                find_surface_intersection_with_polarity,
                project_magnetic_fields,
            )
            from src.potential_mapper.spice import load_spice_files
            from src.utils.attitude import load_attitude_data
        except Exception as exc:
            logger.warning(
                "Polarity support unavailable; falling back to legacy pitch angles: %s",
                exc,
            )
            return None

        try:
            load_spice_files()
            et_spin, ra_vals, dec_vals = load_attitude_data(
                config.DATA_DIR / config.ATTITUDE_FILE
            )
            if (
                et_spin is None
                or ra_vals is None
                or dec_vals is None
                or len(et_spin) == 0
                or len(ra_vals) == 0
                or len(dec_vals) == 0
            ):
                raise RuntimeError("Attitude data unavailable or empty.")

            coord_calc = CoordinateCalculator(et_spin, ra_vals, dec_vals)
            coord_arrays = coord_calc.calculate_coordinate_transformation(self.er_data)
            projected_b = project_magnetic_fields(self.er_data, coord_arrays)
            _points, _mask, polarity = find_surface_intersection_with_polarity(
                coord_arrays, projected_b
            )
            return polarity.astype(np.int8, copy=False)
        except Exception as exc:
            logger.warning(
                "Polarity calculation failed; falling back to legacy pitch angles: %s",
                exc,
            )
            return None

    def _init_fitter(self) -> None:
        self._norm_cache.clear()
        if self.use_torch:
            try:
                from src.model_torch import LossConeFitterTorch

                self.fitter = LossConeFitterTorch(
                    self.er_data,
                    pitch_angle=self.pitch_angle,
                    normalization_mode=self.normalization_mode,
                    incident_flux_stat=self.incident_flux_stat,
                    loss_cone_background=self.background,
                    device=str(self._torch_device) if self._torch_device else None,
                )
                return
            except Exception:
                self.use_torch = False

        self.fitter = LossConeFitter(
            self.er_data,
            pitch_angle=self.pitch_angle,
            normalization_mode=self.normalization_mode,
            incident_flux_stat=self.incident_flux_stat,
            loss_cone_background=self.background,
        )

    def _build_spec_map(self) -> dict[int, int]:
        spec_vals = self.er_data.data[config.SPEC_NO_COLUMN].to_numpy()
        mapping: dict[int, int] = {}
        for idx, val in enumerate(spec_vals):
            try:
                spec_no = int(val)
            except (TypeError, ValueError):
                continue
            if spec_no not in mapping:
                mapping[spec_no] = idx // config.SWEEP_ROWS
        return mapping

    def set_normalization(self, mode: str, incident_stat: str) -> None:
        if mode == self.normalization_mode and incident_stat == self.incident_flux_stat:
            return
        self.normalization_mode = mode
        self.incident_flux_stat = incident_stat
        self._init_fitter()

    def chunk_count(self) -> int:
        return len(self.er_data.data) // config.SWEEP_ROWS

    def spec_to_chunk(self, spec_no: int) -> int | None:
        return self._spec_to_chunk.get(int(spec_no))

    def get_chunk_data(self, chunk_idx: int) -> ChunkData:
        if chunk_idx in self._raw_cache:
            return self._raw_cache[chunk_idx]

        total_rows = len(self.er_data.data)
        start = chunk_idx * config.SWEEP_ROWS
        end = min(start + config.SWEEP_ROWS, total_rows)
        if start >= total_rows:
            raise IndexError(f"Chunk {chunk_idx} out of range for {total_rows} rows.")

        chunk = self.er_data.data.iloc[start:end]
        energies = chunk[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
        flux = chunk[config.FLUX_COLS].to_numpy(dtype=np.float64)
        pitches = self.pitch_angle.pitch_angles[start:end]
        spec_no = int(chunk.iloc[0][config.SPEC_NO_COLUMN])
        timestamp = str(chunk.iloc[0][config.TIME_COLUMN])

        data = ChunkData(
            energies=energies,
            pitches=pitches,
            flux=flux,
            spec_no=spec_no,
            timestamp=timestamp,
        )
        self._raw_cache[chunk_idx] = data
        return data

    def get_norm2d(self, chunk_idx: int) -> np.ndarray:
        cache_key = (chunk_idx, self.normalization_mode, self.incident_flux_stat)
        if cache_key in self._norm_cache:
            return self._norm_cache[cache_key]
        if self.use_torch and hasattr(self.fitter, "_build_norm2d"):
            norm2d = self.fitter._build_norm2d(chunk_idx)
        else:
            norm2d = self.fitter.build_norm2d(chunk_idx)
        self._norm_cache[cache_key] = norm2d
        return norm2d

    def compute_model(
        self,
        energies: np.ndarray,
        pitches: np.ndarray,
        u_surface: float,
        bs_over_bm: float,
        beam_amp: float,
        beam_width_ev: float,
        beam_pitch_sigma_deg: float,
        u_spacecraft: float,
        return_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if (
            self.use_torch
            and synth_losscone_batch_torch is not None
            and torch is not None
        ):
            device = self._torch_device or torch.device("cpu")
            dtype = self._torch_dtype or torch.float32
            energies_t = torch.tensor(energies, device=device, dtype=dtype)
            pitches_t = torch.tensor(pitches, device=device, dtype=dtype)
            u_t = torch.tensor([u_surface], device=device, dtype=dtype)
            bs_t = torch.tensor([bs_over_bm], device=device, dtype=dtype)
            beam_amp_t = torch.tensor([beam_amp], device=device, dtype=dtype)
            beam_width_t = torch.tensor([beam_width_ev], device=device, dtype=dtype)
            background_t = torch.tensor([self.background], device=device, dtype=dtype)
            result_t = synth_losscone_batch_torch(
                energy_grid=energies_t,
                pitch_grid=pitches_t,
                U_surface=u_t,
                U_spacecraft=float(u_spacecraft),
                bs_over_bm=bs_t,
                beam_width_eV=beam_width_t,
                beam_amp=beam_amp_t,
                beam_pitch_sigma_deg=beam_pitch_sigma_deg,
                background=background_t,
                return_mask=return_mask,
            )
            if return_mask:
                model_t, mask_t = result_t
                return model_t[0].detach().cpu().numpy(), mask_t[
                    0
                ].detach().cpu().numpy()
            else:
                return result_t[0].detach().cpu().numpy()

        return synth_losscone(
            energy_grid=energies,
            pitch_grid=pitches,
            U_surface=u_surface,
            U_spacecraft=float(u_spacecraft),
            bs_over_bm=bs_over_bm,
            beam_width_eV=beam_width_ev,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=beam_pitch_sigma_deg,
            background=self.background,
            return_mask=return_mask,
        )

    def compute_chi2(
        self,
        norm2d: np.ndarray,
        model: np.ndarray,
        model_mask: np.ndarray | None = None,
    ) -> float:
        eps = config.EPS
        data_mask = np.isfinite(norm2d) & (norm2d > 0)

        # Combine data mask with model validity mask if provided
        combined_mask = data_mask & model_mask if model_mask is not None else data_mask

        if not combined_mask.any():
            return float("nan")
        log_data = np.zeros_like(norm2d, dtype=float)
        log_data[combined_mask] = np.log(norm2d[combined_mask] + eps)
        log_model = np.log(model + eps)
        diff = (log_data - log_model) * combined_mask
        chi2 = np.sum(diff * diff)
        return float(chi2)

    def _generate_lhs(self, n_samples: int = 400) -> np.ndarray:
        u_min = max(config.LOSS_CONE_U_SURFACE_MIN, -1000.0)
        u_max = min(config.LOSS_CONE_U_SURFACE_MAX, 0.0)
        lower = np.array(
            [u_min, config.LOSS_CONE_BS_OVER_BM_MIN, config.LOSS_CONE_BEAM_AMP_MIN],
            dtype=float,
        )
        upper = np.array(
            [u_max, config.LOSS_CONE_BS_OVER_BM_MAX, config.LOSS_CONE_BEAM_AMP_MAX],
            dtype=float,
        )
        if upper[2] <= lower[2]:
            upper[2] = lower[2] + 1e-12
        sampler = LatinHypercube(
            d=len(lower), scramble=False, seed=config.LOSS_CONE_LHS_SEED
        )
        lhs = sampler.random(n=n_samples)
        return scale(lhs, lower, upper)

    def fit_chunk_lhs(
        self,
        chunk_idx: int,
        beam_width_ev: float,
        u_spacecraft: float,
        n_samples: int = 400,
    ) -> tuple[float, float, float, float]:
        norm2d = self.get_norm2d(chunk_idx)
        if np.isnan(norm2d).all():
            return np.nan, np.nan, np.nan, np.nan

        chunk = self.get_chunk_data(chunk_idx)
        energies = chunk.energies
        pitches = chunk.pitches

        data_mask = np.isfinite(norm2d) & (norm2d > 0)
        if not data_mask.any():
            return np.nan, np.nan, np.nan, np.nan

        log_data = np.zeros_like(norm2d, dtype=float)
        log_data[data_mask] = np.log(norm2d[data_mask] + config.EPS)
        data_mask_3d = data_mask[None, :, :]

        samples = self._generate_lhs(n_samples)
        u_surface = samples[:, 0]
        bs_over_bm = samples[:, 1]
        beam_amp = samples[:, 2]
        beam_width = np.full_like(u_surface, beam_width_ev)
        background = np.full_like(u_surface, self.background)

        models, model_masks = synth_losscone_batch(
            energy_grid=energies,
            pitch_grid=pitches,
            U_surface=u_surface,
            U_spacecraft=float(u_spacecraft),
            bs_over_bm=bs_over_bm,
            beam_width_eV=beam_width,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
            background=background,
            return_mask=True,
        )
        log_models = np.log(models + config.EPS)
        combined_mask = data_mask_3d & model_masks
        diff = (log_data[None, :, :] - log_models) * combined_mask
        chi2 = np.sum(diff * diff, axis=(1, 2))
        chi2[~np.isfinite(chi2)] = 1e30
        best_idx = int(np.argmin(chi2))
        return (
            float(u_surface[best_idx]),
            float(bs_over_bm[best_idx]),
            float(beam_amp[best_idx]),
            float(chi2[best_idx]),
        )

    def fit_chunk_full(self, chunk_idx: int) -> tuple[float, float, float, float]:
        if self.use_torch:
            try:
                return self.fitter._fit_surface_potential_torch(chunk_idx)
            except Exception:
                pass
        return self.fitter._fit_surface_potential(chunk_idx)
