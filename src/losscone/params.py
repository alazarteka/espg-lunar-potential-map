"""Shared parameter helpers for loss-cone fitting."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.utils.losscone_lhs import generate_losscone_lhs

if TYPE_CHECKING:
    import numpy as np

_EPS = 1e-12


def losscone_optimizer_bounds(
    *,
    u_surface_min: float,
    u_surface_max: float,
    bs_over_bm_min: float,
    bs_over_bm_max: float,
    beam_amp_min: float,
    beam_amp_max: float,
) -> list[tuple[float, float]]:
    """
    Return bounds for (U_surface, bs_over_bm, beam_amp) suitable for optimizers.

    Some optimizers require strict (low < high) bounds. We keep beam_amp fixed
    by clipping later, but still return a tiny positive width when min==max.
    """
    beam_hi = float(max(beam_amp_max, beam_amp_min + _EPS))
    return [
        (float(u_surface_min), float(u_surface_max)),
        (float(bs_over_bm_min), float(bs_over_bm_max)),
        (float(beam_amp_min), beam_hi),
    ]


def losscone_lhs_samples(
    *,
    n_samples: int,
    u_surface_min: float,
    u_surface_max: float,
    bs_over_bm_min: float,
    bs_over_bm_max: float,
    beam_amp_min: float,
    beam_amp_max: float,
    seed: int | None = None,
) -> np.ndarray:
    """Generate Phase-1 samples for loss-cone fitting (NumPy)."""
    return generate_losscone_lhs(
        n_samples=int(n_samples),
        u_surface_min=float(u_surface_min),
        u_surface_max=float(u_surface_max),
        bs_over_bm_min=float(bs_over_bm_min),
        bs_over_bm_max=float(bs_over_bm_max),
        beam_amp_min=float(beam_amp_min),
        beam_amp_max=float(beam_amp_max),
        seed=seed,
    )
