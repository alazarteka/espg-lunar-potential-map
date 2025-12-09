"""Define sites of interest for ESPG analysis."""

from dataclasses import dataclass


@dataclass(slots=True)
class Site:
    """A site of interest on the lunar surface."""

    name: str
    lat: float  # degrees
    lon: float  # degrees
    description: str


# List of example sites
SITES_OF_INTEREST = [
    Site(
        name="Apollo 11",
        lat=0.67,
        lon=23.47,
        description="Mare Tranquillitatis (Equatorial Mare)",
    ),
    Site(
        name="Apollo 16",
        lat=-8.97,
        lon=15.50,
        description="Descartes Highlands (Equatorial Highlands)",
    ),
    Site(
        name="Tycho",
        lat=-43.3,
        lon=-11.2,
        description="Tycho Crater (Mid-latitude)",
    ),
    Site(
        name="Aristarchus",
        lat=23.7,
        lon=-47.4,
        description="Aristarchus Plateau (Anomaly/Resources)",
    ),
    Site(
        name="South Pole",
        lat=-90.0,
        lon=0.0,
        description="South Pole (Artemis/ISR)",
    ),
    Site(
        name="Reiner Gamma",
        lat=7.5,
        lon=-59.0,
        description="Reiner Gamma Swirl (Magnetic Anomaly)",
    ),
]
