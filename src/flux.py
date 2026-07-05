"""Backward-compatible shim for ER flux + loss-cone fitting APIs.

The canonical implementations live under `src.losscone.*`.
"""

import numpy as np

from src.losscone.chi2 import compute_halekas_chi2, compute_lillis_chi2
from src.losscone.cpu import ERData, LossConeFitter, PitchAngle
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


class FluxData:
    def __init__(self, er_data_file: str):
        """
        Backwards-compatible orchestrator for ER data + pitch angles + fitter.

        Args:
            er_data_file (str): Path to the ER data file
        """
        # Use the new class structure
        self.er_data = ERData(er_data_file)
        self.pitch_angle = PitchAngle(self.er_data)
        self.loss_cone_fitter = LossConeFitter(
            self.er_data, pitch_angle=self.pitch_angle
        )

        # Expose data for backward compatibility
        self.data = self.er_data.data

    def get_normalized_flux(
        self, energy_bin: int, measurement_chunk: int
    ) -> np.ndarray:
        """
        Get the normalized flux for a specific energy bin and measurement chunk.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter._get_normalized_flux(energy_bin, measurement_chunk)

    def build_norm2d(self, measurement_chunk: int):
        """
        Build a 2D normalized flux distribution for a specific measurement chunk.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter.build_norm2d(measurement_chunk)

    def _fit_surface_potential(self, measurement_chunk: int):
        """
        Fit surface potential for one measurement chunk.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter._fit_surface_potential(measurement_chunk)

    def fit_surface_potential(self):
        """
        Fit surface potential for all measurement chunks.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter.fit_surface_potential()
