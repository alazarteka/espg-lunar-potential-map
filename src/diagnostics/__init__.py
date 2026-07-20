"""Diagnostics utilities for interactive loss-cone analysis."""

from src.diagnostics.beam_detection import (
    BeamDetectionResult,
    PeakCriteria,
    _build_energy_profile,
    detect_peak,
)
from src.diagnostics.losscone_session import (
    LossConeSession,
    compute_loss_cone_boundary,
    interpolate_to_regular_grid,
)
from src.diagnostics.ui_utils import finite_range

__all__ = [
    "BeamDetectionResult",
    "LossConeSession",
    "PeakCriteria",
    "_build_energy_profile",
    "compute_loss_cone_boundary",
    "detect_peak",
    "finite_range",
    "interpolate_to_regular_grid",
]
