"""Helpers for locating ER flux files used by analysis scripts."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type hints only
    from src.potential_mapper.pipeline import DataLoader

__all__ = ["select_flux_day_file"]


def select_flux_day_file(year: int, month: int, day: int) -> Path:
    """Return the first ER flux file matching a given date.

    Parameters
    ----------
    year, month, day
        Date tokens passed to :meth:`DataLoader.discover_flux_files`.

    Returns
    -------
    pathlib.Path
        Path to the first matching flux file.

    Notes
    -----
    Emits a warning if multiple files satisfy the filters; callers should
    inspect the returned path if deterministic selection matters.

    Raises
    ------
    FileNotFoundError
        If no files match the requested date.
    """

    from src.potential_mapper.pipeline import DataLoader

    files = DataLoader.discover_flux_files(year=year, month=month, day=day)
    if not files:
        raise FileNotFoundError("No ER file found for the requested date")
    if len(files) > 1:
        print(f"Warning: multiple files matched; using {files[0]}")
    return Path(files[0])
