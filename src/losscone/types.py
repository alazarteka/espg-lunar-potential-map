from __future__ import annotations

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from src import config
from src.losscone.masks import build_lillis_mask

if TYPE_CHECKING:
    from collections.abc import Iterator


class FitMethod(str, Enum):
    """Loss-cone fitting method."""

    HALEKAS = "halekas"
    LILLIS = "lillis"


class NormalizationMode(str, Enum):
    """Flux normalization strategy."""

    # NOTE: `GLOBAL` and `RATIO_RESCALED` are deprecated. The Halekas (2008)
    # methodology describes fitting reflected/incident ratios, which is most
    # directly represented by `RATIO2` (pairwise) and approximately by `RATIO`
    # (per-energy normalization).
    GLOBAL = "global"
    RATIO = "ratio"
    RATIO2 = "ratio2"
    RATIO_RESCALED = "ratio_rescaled"


@dataclass(frozen=True)
class FitChunkData:
    norm2d: np.ndarray
    energies: np.ndarray
    pitches: np.ndarray
    raw_flux: np.ndarray
    spacecraft_slice: float | np.ndarray
    valid_energy_mask: np.ndarray

    def combined_mask(self, fit_method: FitMethod) -> np.ndarray:
        """
        Combined validity mask for fitting / chi² evaluation.

        Includes both data validity and E >= U_spacecraft validity.
        """
        if fit_method == FitMethod.LILLIS:
            return (
                build_lillis_mask(self.raw_flux, self.pitches) & self.valid_energy_mask
            )

        data_mask = np.isfinite(self.norm2d) & (self.norm2d > 0)
        return data_mask & self.valid_energy_mask

    def with_spacecraft_slice(
        self, spacecraft_slice: float | np.ndarray
    ) -> FitChunkData:
        """Return a copy with updated spacecraft potential + valid-energy mask."""
        if isinstance(spacecraft_slice, np.ndarray):
            valid_energy = self.energies[:, None] >= spacecraft_slice[:, None]
        else:
            valid_energy = self.energies[:, None] >= float(spacecraft_slice)
        valid_energy_mask = np.broadcast_to(valid_energy, self.pitches.shape)
        return FitChunkData(
            norm2d=self.norm2d,
            energies=self.energies,
            pitches=self.pitches,
            raw_flux=self.raw_flux,
            spacecraft_slice=spacecraft_slice,
            valid_energy_mask=valid_energy_mask,
        )

    def has_enough_valid_bins(self, fit_method: FitMethod) -> bool:
        """Return True if the chunk has enough usable bins for the fit method."""
        mask = self.combined_mask(fit_method)
        if fit_method == FitMethod.LILLIS:
            return int(np.count_nonzero(mask)) >= config.LILLIS_MIN_VALID_BINS
        return bool(mask.any())


@dataclass(frozen=True)
class ChunkFitResult:
    """
    Result for a single sweep (15-row measurement chunk) fit.

    Designed to replace the old tuple-based return values while remaining
    unpackable in existing call sites.
    """

    u_surface: float
    bs_over_bm: float
    beam_amp: float
    chi2: float
    chunk_index: int

    @classmethod
    def invalid(cls, chunk_index: int, *, chi2: float = float("nan")) -> ChunkFitResult:
        nan = float("nan")
        return cls(
            u_surface=nan,
            bs_over_bm=nan,
            beam_amp=nan,
            chi2=float(chi2),
            chunk_index=int(chunk_index),
        )

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.u_surface, self.bs_over_bm, self.beam_amp, self.chi2)

    def as_row(self) -> np.ndarray:
        return np.array(
            [
                self.u_surface,
                self.bs_over_bm,
                self.beam_amp,
                self.chi2,
                float(self.chunk_index),
            ],
            dtype=float,
        )

    def __iter__(self) -> Iterator[float]:  # pragma: no cover - trivial
        yield from self.as_tuple()


def parse_fit_method(value: FitMethod | str | None) -> FitMethod:
    if value is None:
        value = config.LOSS_CONE_FIT_METHOD
    if isinstance(value, FitMethod):
        return value
    try:
        return FitMethod(str(value))
    except ValueError:
        raise ValueError(f"Unknown fit_method: {value}") from None


def parse_normalization_mode(value: NormalizationMode | str) -> NormalizationMode:
    if isinstance(value, NormalizationMode):
        mode = value
    else:
        try:
            mode = NormalizationMode(str(value))
        except ValueError:
            raise ValueError(f"Unknown normalization_mode: {value}") from None
    if mode in {NormalizationMode.GLOBAL, NormalizationMode.RATIO_RESCALED}:
        warnings.warn(
            (
                f"Normalization mode '{mode.value}' is deprecated and will be "
                "removed in a future release. Use 'ratio' or 'ratio2' instead."
            ),
            FutureWarning,
            stacklevel=2,
        )
    return mode
