from dataclasses import dataclass

import numpy as np


@dataclass()
class PotentialResults:
    """
    Row-aligned outputs from processing ER data into a potential map.

    Arrays share the same length N (number of ER rows after any filtering):
    - spacecraft_latitude/longitude: Spacecraft geodetic in degrees (IAU_MOON frame).
    - projection_latitude/longitude: Surface intersection geodetic in degrees.
    - spacecraft_potential: Floating potential of spacecraft (V) per row.
    - projected_potential: Surface potential Φ_surface (V) from loss-cone fit; NaN when not fit.
    - spacecraft_in_sun: True if LP→Sun line-of-sight does not intersect Moon.
    - projection_in_sun: True if surface normal · Moon→Sun > 0 at intersection.
    """

    spacecraft_latitude: np.ndarray
    spacecraft_longitude: np.ndarray
    projection_latitude: np.ndarray
    projection_longitude: np.ndarray
    spacecraft_potential: np.ndarray
    projected_potential: np.ndarray
    spacecraft_in_sun: np.ndarray
    projection_in_sun: np.ndarray


def _concat_results(results: list[PotentialResults]) -> PotentialResults:
    """Concatenate fields from multiple PotentialResults objects (row-wise)."""

    def cat(attr: str):
        return np.concatenate([getattr(r, attr) for r in results])

    return PotentialResults(
        spacecraft_latitude=cat("spacecraft_latitude"),
        spacecraft_longitude=cat("spacecraft_longitude"),
        projection_latitude=cat("projection_latitude"),
        projection_longitude=cat("projection_longitude"),
        spacecraft_potential=cat("spacecraft_potential"),
        projected_potential=cat("projected_potential"),
        spacecraft_in_sun=cat("spacecraft_in_sun"),
        projection_in_sun=cat("projection_in_sun"),
    )
