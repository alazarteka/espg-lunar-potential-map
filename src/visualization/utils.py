"""Shared plotting utilities."""

import argparse
from datetime import date, datetime, timedelta
from typing import Any

from src.visualization.style import BBOX_STYLE, FONT_SIZE_TEXT


def add_stats_box(ax: Any, text: str, loc: str = "upper left") -> None:
    """Add a statistics text box to the axes."""
    # Simple location mapping
    x, y = 0.02, 0.98
    va, ha = "top", "left"

    if loc == "upper right":
        x, y = 0.98, 0.98
        ha = "right"
    elif loc == "lower left":
        x, y = 0.02, 0.02
        va = "bottom"
    elif loc == "lower right":
        x, y = 0.98, 0.02
        va = "bottom"
        ha = "right"

    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        fontsize=FONT_SIZE_TEXT,
        verticalalignment=va,
        horizontalalignment=ha,
        bbox=BBOX_STYLE,
    )


def parse_iso_date(value: str) -> date:
    """Parse YYYY-MM-DD string into a Python date."""
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        msg = "Dates must be provided as YYYY-MM-DD"
        raise argparse.ArgumentTypeError(msg) from exc


def date_range(start_day: date, end_day: date) -> list[date]:
    """Inclusive list of days between start_day and end_day."""
    if end_day < start_day:
        raise ValueError("end must be >= start")
    span = (end_day - start_day).days
    return [start_day + timedelta(days=offset) for offset in range(span + 1)]
