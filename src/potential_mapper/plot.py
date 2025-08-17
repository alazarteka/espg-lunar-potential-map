import logging
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import src.config as config
from src.potential_mapper.results import PotentialResults


def plot_map(
    results: PotentialResults,
    ax=None,
    *,
    illumination: str | None = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter plot of surface potential ΔU by projection lat/lon.

    - Adds moon map background if `config.MOON_MAP_FILE` exists under DATA_DIR.
    - Filters by finite potentials; optional day/night filtering via `illumination`.
    - Returns (Figure, Axes); caller decides to show/save.
    """
    fig: plt.Figure
    axes: plt.Axes
    if ax is None:
        fig, axes = plt.subplots(figsize=(8, 4))
    else:
        axes = ax
        fig = axes.figure

    # Optional moon map background (if available)
    try:
        moon_map_path = Path(config.DATA_DIR) / config.MOON_MAP_FILE
        if moon_map_path.exists():
            img = plt.imread(str(moon_map_path))
            axes.imshow(
                img,
                extent=(-180, 180, -90, 90),
                aspect="equal",
                zorder=-1,
            )
    except Exception as e:
        logging.debug(f"Could not load moon map background: {e}")

    # Base mask: finite potentials
    mask = np.isfinite(results.projected_potential)
    # Optional illumination filter
    if illumination == 'day':
        mask &= results.projection_in_sun.astype(bool)
    elif illumination == 'night':
        mask &= ~results.projection_in_sun.astype(bool)
    x = results.projection_longitude[mask]
    y = results.projection_latitude[mask]
    c = results.projected_potential[mask]

    try:
        sc = axes.scatter(x, y, c=c, s=8, cmap="viridis")
        fig.colorbar(sc, ax=axes, label="Surface potential ΔU (V)")
    except Exception:
        axes.scatter(x, y, s=8)

    axes.set_xlabel("Longitude (deg)")
    axes.set_ylabel("Latitude (deg)")
    title = "Lunar Surface Potential"
    if illumination in ("day", "night"):
        title += f" ({illumination}side)"
    axes.set_title(title)
    axes.grid(True, alpha=0.2)

    return fig, axes
