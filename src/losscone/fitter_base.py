from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class LossConeFitterBase(ABC):
    """
    Shared public API for CPU and Torch loss-cone fitters.

    This intentionally standardizes method names to reduce drift between
    implementations and to avoid downstream callers reaching into private
    methods like `_build_norm2d()` or `_fit_surface_potential_torch()`.
    """

    @abstractmethod
    def build_norm2d(self, measurement_chunk: int) -> np.ndarray:
        """Return normalized flux array with shape (nE, nPitch)."""

    @abstractmethod
    def build_norm2d_batch(self, chunk_indices: list[int]) -> np.ndarray:
        """Return normalized flux array with shape (n_chunks, nE, nPitch)."""

    @abstractmethod
    def fit_chunk_lhs(
        self,
        measurement_chunk: int,
        beam_width_ev: float | None = None,
        u_spacecraft: float | None = None,
        n_samples: int = 400,
    ) -> tuple[float, float, float, float]:
        """Return (U_surface, bs_over_bm, beam_amp, chi2) from LHS-only search."""

    @abstractmethod
    def fit_chunk_full(
        self, measurement_chunk: int
    ) -> tuple[float, float, float, float]:
        """Return (U_surface, bs_over_bm, beam_amp, chi2) from full optimization."""

    @abstractmethod
    def fit_surface_potential(self) -> np.ndarray:
        """Return fit matrix with columns [U, bs/bm, beam_amp, chi2, chunk_index]."""
