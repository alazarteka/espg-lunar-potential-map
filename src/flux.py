"""Backward-compatible shim for ER flux + loss-cone fitting APIs.

The canonical implementations live under `src.losscone.*`.
"""

from src.losscone.chi2 import compute_halekas_chi2, compute_lillis_chi2
from src.losscone.cpu import ERData, FluxData, LossConeFitter, PitchAngle
from src.losscone.masks import build_lillis_mask
from src.losscone.types import (
    ChunkFitResult,
    FitChunkData,
    FitMethod,
    NormalizationMode,
)
from src.utils.thetas import get_thetas  # noqa: F401

__all__ = [
    "ChunkFitResult",
    "ERData",
    "FitChunkData",
    "FitMethod",
    "FluxData",
    "LossConeFitter",
    "NormalizationMode",
    "PitchAngle",
    "build_lillis_mask",
    "compute_halekas_chi2",
    "compute_lillis_chi2",
]
