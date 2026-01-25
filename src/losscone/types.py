from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from src import config

if TYPE_CHECKING:
    import numpy as np


class FitMethod(str, Enum):
    """Loss-cone fitting method."""

    HALEKAS = "halekas"
    LILLIS = "lillis"


class NormalizationMode(str, Enum):
    """Flux normalization strategy."""

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
        return value
    try:
        return NormalizationMode(str(value))
    except ValueError:
        raise ValueError(f"Unknown normalization_mode: {value}") from None
