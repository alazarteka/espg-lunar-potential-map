from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt

from src.potential_mapper.results import PotentialResults


def plot_map(results: PotentialResults, ax=None) -> tuple[plt.Figure, plt.Axes]:
    """
    Minimal plotting of potential results. Returns (Figure, Axes) without showing.

    Currently a placeholder: scatters projection points colored by projected_potential
    if available, otherwise a simple scatter without color.
    """
    fig: plt.Figure
    axes: plt.Axes
    if ax is None:
        fig, axes = plt.subplots(figsize=(8, 4))
    else:
        axes = ax
        fig = axes.figure

    x = results.projection_longitude
    y = results.projection_latitude
    c = results.projected_potential

    try:
        sc = axes.scatter(x, y, c=c, s=8, cmap="viridis")
        fig.colorbar(sc, ax=axes, label="Projected potential (V)")
    except Exception:
        axes.scatter(x, y, s=8)

    axes.set_xlabel("Longitude (deg)")
    axes.set_ylabel("Latitude (deg)")
    axes.set_title("Lunar Surface Potential (preview)")
    axes.grid(True, alpha=0.2)

    return fig, axes
