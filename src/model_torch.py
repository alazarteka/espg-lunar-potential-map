"""Backward-compatible shim for torch loss-cone APIs.

The canonical PyTorch implementation lives under `src.losscone.torch.*`.
This module conditionally imports those implementations when PyTorch is available.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

try:
    from torch import Tensor

    HAS_TORCH = True
except ImportError:  # pragma: no cover - exercised via skipif in tests
    HAS_TORCH = False
    Tensor = None  # type: ignore[misc, assignment]


def _torch_required(*_args: Any, **_kwargs: Any) -> Any:
    raise ImportError("PyTorch is required for `src.model_torch`.")


if HAS_TORCH:
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
else:  # pragma: no cover - exercised via skipif in tests
    DEFAULT_BACKGROUND = 0.05
    EPS = 1e-12

    _auto_detect_dtype: Callable[..., Any] = _torch_required
    _auto_detect_batch_size: Callable[..., Any] = _torch_required

    synth_losscone_batch_torch: Callable[..., Any] = _torch_required
    synth_losscone_multi_chunk_torch: Callable[..., Any] = _torch_required

    precompute_log_data_torch: Callable[..., Any] = _torch_required
    compute_chi2_batch_torch: Callable[..., Any] = _torch_required
    compute_lillis_chi2_batch_torch: Callable[..., Any] = _torch_required
    compute_chi2_multi_chunk_torch: Callable[..., Any] = _torch_required
    compute_lillis_chi2_multi_chunk_torch: Callable[..., Any] = _torch_required

    GPUDifferentialEvolution: Any = _torch_required

    class LossConeFitterTorch:  # type: ignore[no-redef]
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            _torch_required()


__all__ = [
    "DEFAULT_BACKGROUND",
    "EPS",
    "HAS_TORCH",
    "GPUDifferentialEvolution",
    "LossConeFitterTorch",
    "Tensor",
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
