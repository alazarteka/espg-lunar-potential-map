"""Loss-cone fitting helpers shared by CPU/GPU implementations."""

from src.losscone.chi2 import compute_halekas_chi2, compute_lillis_chi2
from src.losscone.masks import build_lillis_mask
from src.losscone.types import (
    FitChunkData,
    FitMethod,
    NormalizationMode,
    parse_fit_method,
    parse_normalization_mode,
)

__all__ = [
    "FitChunkData",
    "FitMethod",
    "NormalizationMode",
    "build_lillis_mask",
    "compute_halekas_chi2",
    "compute_lillis_chi2",
    "parse_fit_method",
    "parse_normalization_mode",
]
