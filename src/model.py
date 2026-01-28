"""Backward-compatible shim for the NumPy loss-cone forward model.

The canonical implementation lives in `src.losscone.model`.
"""

from src.losscone.model import (  # noqa: F401
    DEFAULT_BACKGROUND,
    EPS,
    LossConeParams,
    _compute_beam,
    _compute_loss_cone_angle,
    synth_losscone,
    synth_losscone_batch,
)

__all__ = [
    "DEFAULT_BACKGROUND",
    "EPS",
    "LossConeParams",
    "synth_losscone",
    "synth_losscone_batch",
]
