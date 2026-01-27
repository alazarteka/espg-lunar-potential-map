from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class PlasmaEnvironment(IntEnum):
    """
    Plasma environment classification based on electron temperature and illumination.

    Thresholds from Halekas et al. 2008 and GPT-5 Pro consultation:
    - UNKNOWN: No valid temperature measurement
    - SOLAR_WIND: Te < 30 eV (typical upstream solar wind)
    - MAGNETOSHEATH: Te 30-80 eV (shocked solar wind)
    - TAIL_LOBES: Te 80-150 eV (magnetotail lobes, moderate charging)
    - PLASMA_SHEET: Te > 150 eV (plasma sheet, can cause extreme charging to -kV)
    - WAKE: Nightside shadow (SZA > 90°), any Te (lunar wake region)
    """

    UNKNOWN = 0
    SOLAR_WIND = 1
    MAGNETOSHEATH = 2
    TAIL_LOBES = 3
    PLASMA_SHEET = 4
    WAKE = 5

    @classmethod
    def from_temperature(cls, te_ev: float) -> "PlasmaEnvironment":
        """Classify environment from electron temperature in eV (legacy method)."""
        if not np.isfinite(te_ev) or te_ev <= 0:
            return cls.UNKNOWN
        if te_ev < 30:
            return cls.SOLAR_WIND
        if te_ev < 80:
            return cls.MAGNETOSHEATH
        if te_ev < 150:
            return cls.TAIL_LOBES
        return cls.PLASMA_SHEET

    @classmethod
    def from_temperature_and_illumination(
        cls, te_ev: float, projection_in_sun: bool
    ) -> "PlasmaEnvironment":
        """
        Classify environment from electron temperature and illumination state.

        Args:
            te_ev: Electron temperature in eV
            projection_in_sun: True if footprint is sunlit (SZA < 90°)

        Returns:
            PlasmaEnvironment classification

        Classification priority:
        1. Shadow (projection_in_sun=False) → WAKE
        2. Temperature-based classification for sunlit regions
        """
        if not np.isfinite(te_ev) or te_ev <= 0:
            return cls.UNKNOWN
        # Shadow regions are classified as WAKE regardless of temperature
        if not projection_in_sun:
            return cls.WAKE
        # Sunlit regions: temperature-based classification
        if te_ev < 30:
            return cls.SOLAR_WIND
        if te_ev < 80:
            return cls.MAGNETOSHEATH
        if te_ev < 150:
            return cls.TAIL_LOBES
        return cls.PLASMA_SHEET


@dataclass()
class PotentialResults:
    """
    Row-aligned outputs from processing ER data into a potential map.

    Arrays share the same length N (number of ER rows after any filtering):

    Coordinates:
    - spacecraft_latitude/longitude: Spacecraft geodetic in degrees (IAU_MOON frame).
    - projection_latitude/longitude: Surface intersection geodetic in degrees.
    - projection_polarity: +1 if footprint is along +B, -1 if along -B, 0 if none.

    Potentials:
    - spacecraft_potential: Floating potential of spacecraft (V) per row.
    - projected_potential: Surface potential U_surface (V) from loss-cone fit;
      NaN if fit failed (finite values may include poor fits; use fit_chi2).

    Loss-cone fit parameters:
    - bs_over_bm: Fitted B_surface/B_spacecraft ratio; NaN if not fit.
    - beam_amp: Fitted secondary beam amplitude; NaN if not fit.
    - fit_chi2: Chi-squared of the loss-cone fit; NaN if not fit.

    Kappa fit parameters (from spacecraft potential calculation):
    - electron_temperature: Electron temperature Te in eV; NaN if not fit.
    - electron_density: Electron density ne in particles/m³; NaN if not fit.
    - kappa_value: Kappa parameter from distribution fit; NaN if not fit.

    Environment:
    - spacecraft_in_sun: True if LP→Sun line-of-sight does not intersect Moon.
    - projection_in_sun: True if surface normal · Moon→Sun > 0 at intersection.
    - environment_class: PlasmaEnvironment classification (0=unknown, 1=SW,
      2=sheath, 3=lobes, 4=PS, 5=wake).
    """

    # Coordinates
    spacecraft_latitude: np.ndarray
    spacecraft_longitude: np.ndarray
    projection_latitude: np.ndarray
    projection_longitude: np.ndarray

    # Potentials
    spacecraft_potential: np.ndarray
    projected_potential: np.ndarray

    # Illumination
    spacecraft_in_sun: np.ndarray
    projection_in_sun: np.ndarray
    projection_polarity: np.ndarray = field(default_factory=lambda: np.array([]))

    # Loss-cone fit parameters (new)
    bs_over_bm: np.ndarray = field(default_factory=lambda: np.array([]))
    beam_amp: np.ndarray = field(default_factory=lambda: np.array([]))
    fit_chi2: np.ndarray = field(default_factory=lambda: np.array([]))

    # Kappa fit parameters (new)
    electron_temperature: np.ndarray = field(default_factory=lambda: np.array([]))
    electron_density: np.ndarray = field(default_factory=lambda: np.array([]))
    kappa_value: np.ndarray = field(default_factory=lambda: np.array([]))

    # Environment classification (new)
    environment_class: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        """Initialize empty arrays to proper size if not provided."""
        n = len(self.spacecraft_latitude)
        if len(self.bs_over_bm) == 0:
            self.bs_over_bm = np.full(n, np.nan)
        if len(self.beam_amp) == 0:
            self.beam_amp = np.full(n, np.nan)
        if len(self.fit_chi2) == 0:
            self.fit_chi2 = np.full(n, np.nan)
        if len(self.electron_temperature) == 0:
            self.electron_temperature = np.full(n, np.nan)
        if len(self.electron_density) == 0:
            self.electron_density = np.full(n, np.nan)
        if len(self.kappa_value) == 0:
            self.kappa_value = np.full(n, np.nan)
        if len(self.environment_class) == 0:
            self.environment_class = np.zeros(n, dtype=np.int8)
        if len(self.projection_polarity) == 0:
            self.projection_polarity = np.zeros(n, dtype=np.int8)

    def classify_environments(self) -> None:
        """Classify plasma environment for each row by temperature/illumination."""
        for i, (te, in_sun) in enumerate(
            zip(self.electron_temperature, self.projection_in_sun, strict=True)
        ):
            self.environment_class[i] = (
                PlasmaEnvironment.from_temperature_and_illumination(te, bool(in_sun))
            )


def _concat_results(results: list[PotentialResults]) -> PotentialResults:
    """Concatenate fields from multiple PotentialResults objects (row-wise)."""

    def cat(attr: str):
        arrays = [getattr(r, attr) for r in results]
        # Handle case where some results may have empty arrays
        non_empty = [a for a in arrays if len(a) > 0]
        if not non_empty:
            return np.array([])
        return np.concatenate(non_empty)

    return PotentialResults(
        spacecraft_latitude=cat("spacecraft_latitude"),
        spacecraft_longitude=cat("spacecraft_longitude"),
        projection_latitude=cat("projection_latitude"),
        projection_longitude=cat("projection_longitude"),
        spacecraft_potential=cat("spacecraft_potential"),
        projected_potential=cat("projected_potential"),
        spacecraft_in_sun=cat("spacecraft_in_sun"),
        projection_in_sun=cat("projection_in_sun"),
        projection_polarity=cat("projection_polarity"),
        bs_over_bm=cat("bs_over_bm"),
        beam_amp=cat("beam_amp"),
        fit_chi2=cat("fit_chi2"),
        electron_temperature=cat("electron_temperature"),
        electron_density=cat("electron_density"),
        kappa_value=cat("kappa_value"),
        environment_class=cat("environment_class"),
    )
