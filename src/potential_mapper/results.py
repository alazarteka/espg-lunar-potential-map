from dataclasses import dataclass

import numpy as np

@dataclass()
class PotentialResults:
    """
    Holds the results of the potential mapping.

    Attributes:
        - spacecraft_latitude: Latitude of the spacecraft.
        - spacecraft_longitude: Longitude of the spacecraft.
        - projection_latitude: Latitude of the projection.
        - projection_longitude: Longitude of the projection.
        - spacecraft_potential: Electric potential of the spacecraft.
        - projected_potential: Electric potential measured at the projection point.
        - spacecraft_in_sun: Boolean array indicating if the spacecraft is in sunlight.
        - projection_in_sun: Boolean array indicating if the projection point is in
        sunlight.

    """
    spacecraft_latitude: np.ndarray
    spacecraft_longitude: np.ndarray
    projection_latitude: np.ndarray
    projection_longitude: np.ndarray
    spacecraft_potential: np.ndarray
    projected_potential: np.ndarray
    spacecraft_in_sun: np.ndarray
    projection_in_sun: np.ndarray
