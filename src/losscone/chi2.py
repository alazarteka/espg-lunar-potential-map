from __future__ import annotations

import numpy as np

from src import config
from src.losscone.masks import build_lillis_mask


def compute_halekas_chi2(
    norm2d: np.ndarray,
    model: np.ndarray,
    model_mask: np.ndarray | None = None,
    eps: float | None = None,
) -> float:
    if eps is None:
        eps = config.EPS if hasattr(config, "EPS") else 1e-6
    data_mask = np.isfinite(norm2d) & (norm2d > 0)
    combined_mask = data_mask & model_mask if model_mask is not None else data_mask
    if not combined_mask.any():
        return float("nan")
    log_data = np.zeros_like(norm2d, dtype=float)
    log_data[combined_mask] = np.log(norm2d[combined_mask] + eps)
    log_model = np.log(model + eps)
    diff = (log_data - log_model) * combined_mask
    chi2 = np.sum(diff * diff)
    return float(chi2)


def compute_lillis_chi2(
    norm2d: np.ndarray,
    model: np.ndarray,
    raw_flux: np.ndarray,
    pitches: np.ndarray,
    model_mask: np.ndarray | None = None,
) -> float:
    data_mask = build_lillis_mask(raw_flux, pitches)
    combined_mask = data_mask & model_mask if model_mask is not None else data_mask
    n_valid = int(np.count_nonzero(combined_mask))
    if n_valid < config.LILLIS_MIN_VALID_BINS:
        return float("nan")
    diff = (norm2d - model) * combined_mask
    chi2 = np.sum(diff * diff)
    dof = max(n_valid - 3, 1)
    return float(chi2) / float(dof)
