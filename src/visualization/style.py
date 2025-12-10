"""Shared style definitions for plots."""

from typing import Any

# Colors
COLOR_SUNLIT = "#FDB462"  # Orange
COLOR_SHADOWED = "#8DD3C7"  # Teal
COLOR_TERMINATOR = "red"
COLOR_SUBSOLAR = "orange"
COLOR_ANTISOLAR = "navy"

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

# Dark Mode Theme (for flux plots)
DARK_THEME: dict[str, Any] = {
    "facecolor": "black",
    "text_color": "white",
    "spine_color": "white",
    "cmap": "inferno",
}


def apply_paper_style(ax: Any, grid: bool = True) -> None:
    """Apply standard paper style to a Matplotlib axes."""
    if grid:
        ax.grid(True, **GRID_STYLE)

    # Set label sizes if not already set (this is handled by matplotlib typically,
    # but we can enforce it if needed. For now, rely on script setting it or defaults).
    # We could set tick params here.
    ax.tick_params(labelsize=FONT_SIZE_TEXT)
