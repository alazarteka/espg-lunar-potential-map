"""Visualization package for ESPG Lunar Potential Map."""

from src.visualization.loaders import load_date_range_data_with_sza, load_measurements
from src.visualization.style import (
    BBOX_STYLE,
    CMAP_DEFAULT,
    CMAP_MAGNITUDE,
    CMAP_MEASUREMENT,
    CMAP_RESIDUAL,
    COLOR_ANTISOLAR,
    COLOR_SHADOWED,
    COLOR_SUBSOLAR,
    COLOR_SUNLIT,
    COLOR_TERMINATOR,
    DARK_THEME,
    FONT_SIZE_LABEL,
    FONT_SIZE_TEXT,
    FONT_SIZE_TITLE,
    GRID_STYLE,
    apply_paper_style,
)
from src.visualization.utils import add_stats_box, date_range, parse_iso_date

__all__ = [
    "BBOX_STYLE",
    "CMAP_DEFAULT",
    "CMAP_MAGNITUDE",
    "CMAP_MEASUREMENT",
    "CMAP_RESIDUAL",
    "COLOR_ANTISOLAR",
    "COLOR_SHADOWED",
    "COLOR_SUBSOLAR",
    "COLOR_SUNLIT",
    "COLOR_TERMINATOR",
    "DARK_THEME",
    "FONT_SIZE_LABEL",
    "FONT_SIZE_TEXT",
    "FONT_SIZE_TITLE",
    "GRID_STYLE",
    # utils
    "add_stats_box",
    # style
    "apply_paper_style",
    "date_range",
    "load_date_range_data_with_sza",
    # loaders
    "load_measurements",
    "parse_iso_date",
]
