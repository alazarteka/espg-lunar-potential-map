"""Shared utilities for loss-cone Latin hypercube sampling."""

from __future__ import annotations

import numpy as np
from scipy.stats.qmc import LatinHypercube, scale

from src import config


def generate_losscone_lhs(
    n_samples: int,
    u_surface_min: float,
    u_surface_max: float,
    bs_over_bm_min: float,
    bs_over_bm_max: float,
    beam_amp_min: float,
    beam_amp_max: float,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate Latin hypercube samples for loss-cone parameters.

    Uses a narrowed U_surface range for sampling efficiency.
    """
    u_lhs_min = max(u_surface_min, -1000.0)
    u_lhs_max = min(u_surface_max, 0.0)
    lower_bounds = np.array([u_lhs_min, bs_over_bm_min, beam_amp_min], dtype=float)
    upper_bounds = np.array([u_lhs_max, bs_over_bm_max, beam_amp_max], dtype=float)
    if upper_bounds[2] <= lower_bounds[2]:
        upper_bounds[2] = lower_bounds[2] + 1e-12

    sampler = LatinHypercube(
        d=len(lower_bounds),
        scramble=False,
        seed=config.LOSS_CONE_LHS_SEED if seed is None else seed,
    )
    lhs = sampler.random(n=n_samples)
    scaled = scale(lhs, lower_bounds, upper_bounds)
    if beam_amp_max <= beam_amp_min:
        scaled[:, 2] = beam_amp_min
    return scaled
