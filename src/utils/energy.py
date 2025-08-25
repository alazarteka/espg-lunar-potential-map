import numpy as np

from src import config
from src.utils.units import EnergyType


def make_relative_energy_bounds(
    energy_centers: EnergyType, rel_width: float | None = None
) -> EnergyType:
    """
    Build symmetric energy bounds around centers using a relative width.

    Args:
        energy_centers: Energy values (Quantity with energy units).
        rel_width: Total relative width (FWHM/E), e.g., 0.5 => [0.75E, 1.25E].
                   If None, uses config.ENERGY_WINDOW_WIDTH_RELATIVE.

    Returns:
        Quantity array with shape (n, 2): [lower, upper] energy bounds.
    """
    if rel_width is None:
        rel_width = config.ENERGY_WINDOW_WIDTH_RELATIVE

    half = 0.5 * float(rel_width)
    lower = (1.0 - half) * energy_centers
    upper = (1.0 - (-half)) * energy_centers  # (1 + half) * centers, preserves units
    return np.column_stack([lower, upper])

