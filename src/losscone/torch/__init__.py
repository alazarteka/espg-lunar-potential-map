"""Torch loss-cone implementations.

This subpackage requires PyTorch. For optional import shims, prefer
`src.losscone_torch`. `src.model_torch` remains as a legacy alias.
"""

from src.losscone.torch.chi2 import (
    compute_chi2_batch_torch,
    compute_chi2_multi_chunk_torch,
    compute_lillis_chi2_batch_torch,
    compute_lillis_chi2_multi_chunk_torch,
    precompute_log_data_torch,
)
from src.losscone.torch.fitter import (
    GPUDifferentialEvolution,
    LossConeFitterTorch,
    _auto_detect_batch_size,
    _auto_detect_dtype,
)
from src.losscone.torch.forward import (
    DEFAULT_BACKGROUND,
    EPS,
    synth_losscone_batch_torch,
    synth_losscone_multi_chunk_torch,
)

__all__ = [
    "DEFAULT_BACKGROUND",
    "EPS",
    "GPUDifferentialEvolution",
    "LossConeFitterTorch",
    "_auto_detect_batch_size",
    "_auto_detect_dtype",
    "compute_chi2_batch_torch",
    "compute_chi2_multi_chunk_torch",
    "compute_lillis_chi2_batch_torch",
    "compute_lillis_chi2_multi_chunk_torch",
    "precompute_log_data_torch",
    "synth_losscone_batch_torch",
    "synth_losscone_multi_chunk_torch",
]
