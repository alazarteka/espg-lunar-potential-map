"""Loss-cone fitting helpers shared by CPU/GPU implementations."""

from src.losscone.chi2 import compute_halekas_chi2, compute_lillis_chi2
from src.losscone.cpu import ERData, LossConeFitter, PitchAngle
from src.losscone.fitter_base import LossConeFitterBase
from src.losscone.masks import build_lillis_mask
from src.losscone.model import synth_losscone, synth_losscone_batch
from src.losscone.types import (
    ChunkFitResult,
    FitChunkData,
    FitMethod,
    NormalizationMode,
    parse_fit_method,
    parse_normalization_mode,
)

__all__ = [
    "ChunkFitResult",
    "ERData",
    "FitChunkData",
    "FitMethod",
    "LossConeFitter",
    "LossConeFitterBase",
    "NormalizationMode",
    "PitchAngle",
    "build_lillis_mask",
    "compute_halekas_chi2",
    "compute_lillis_chi2",
    "parse_fit_method",
    "parse_normalization_mode",
    "synth_losscone",
    "synth_losscone_batch",
]
