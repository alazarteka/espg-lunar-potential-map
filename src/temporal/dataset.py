"""Utilities for loading temporal harmonic coefficient bundles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(slots=True)
class TemporalDataset:
    """Container for time-dependent spherical harmonic coefficients."""

    times: np.ndarray
    lmax: int
    coeffs: np.ndarray
    n_samples: np.ndarray | None = None
    spatial_coverage: np.ndarray | None = None
    rms_residuals: np.ndarray | None = None


def load_temporal_coefficients(path: Path) -> TemporalDataset:
    """Load temporal coefficient dataset saved as NPZ."""
    with np.load(path) as data:
        return TemporalDataset(
            times=data["times"],
            lmax=int(data["lmax"]),
            coeffs=data["coeffs"],
            n_samples=data.get("n_samples"),
            spatial_coverage=data.get("spatial_coverage"),
            rms_residuals=data.get("rms_residuals"),
        )
