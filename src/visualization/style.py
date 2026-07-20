"""Shared style definitions for plots."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.axes

# Colors
COLOR_SUNLIT = "#FDB462"  # Orange
COLOR_SHADOWED = "#8DD3C7"  # Teal
COLOR_TERMINATOR = "red"
COLOR_SUBSOLAR = "orange"
COLOR_ANTISOLAR = "navy"

# Colormaps - semantic categories
CMAP_MEASUREMENT = "viridis"  # Fluxes, voltages, harmonic reconstructions
CMAP_MAGNITUDE = "plasma"  # Engineering power, potential magnitudes
CMAP_RESIDUAL = "RdBu_r"  # Residuals, errors (diverging, centered at 0)

# Font Sizes
FONT_SIZE_TITLE = 13
FONT_SIZE_LABEL = 12
FONT_SIZE_TEXT = 9

# Grid Style
GRID_STYLE = {
    "alpha": 0.3,
    "linestyle": "--",
    "linewidth": 0.4,
}

# Text Box Style
BBOX_STYLE = {
    "boxstyle": "round",
    "facecolor": "wheat",
    "alpha": 0.8,
}


def apply_paper_style(ax: matplotlib.axes.Axes, grid: bool = True) -> None:
    """Apply standard paper style to a Matplotlib axes."""
    if grid:
        ax.grid(True, **GRID_STYLE)
    ax.tick_params(labelsize=FONT_SIZE_TEXT)
