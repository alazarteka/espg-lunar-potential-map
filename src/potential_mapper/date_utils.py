"""Month name mappings and ER flux file discovery helpers."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import src.config as config

__all__ = [
    "MONTH_ABBREV_TO_NUM",
    "MONTH_INT_TO_ABBREV",
    "NUM_STR_TO_MONTH_ABBREV",
    "discover_flux_files",
    "parse_3d_filename",
]

_MONTH_ABBREVS = (
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC",
)

MONTH_ABBREV_TO_NUM = {
    abbrev: f"{month_num:02d}"
    for month_num, abbrev in enumerate(_MONTH_ABBREVS, start=1)
}
NUM_STR_TO_MONTH_ABBREV = {
    num_str: abbrev for abbrev, num_str in MONTH_ABBREV_TO_NUM.items()
}
MONTH_INT_TO_ABBREV = dict(enumerate(_MONTH_ABBREVS, start=1))


def parse_3d_filename(filename: str) -> tuple[int, int, int]:
    """Parse 3D date filename format (e.g., '3D980415') into (year, month, day).

    Args:
        filename: Filename with or without extension (e.g., "3D980415" or
            "3D980415.TAB")

    Returns:
        Tuple of (year, month, day) where year is full 4-digit year
    """
    stem = Path(filename).stem
    yy = int(stem[2:4])
    mm = int(stem[4:6])
    dd = int(stem[6:8])
    year = 1900 + yy if yy > 50 else 2000 + yy
    return year, mm, dd


def discover_flux_files(
    year: int | None = None,
    month: int | None = None,
    day: int | None = None,
) -> list[Path]:
    """Discover ER flux ``.TAB`` files under ``config.DATA_DIR``."""
    data_dir = config.DATA_DIR
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory {data_dir} not found")

    if month is not None and not 1 <= month <= 12:
        logging.warning("Month %s outside 1-12; no files processed.", month)
        return []
    if day is not None and not 1 <= day <= 31:
        logging.warning("Day %s outside 1-31; no files processed.", day)
        return []

    exclude_basenames = {
        config.ATTITUDE_FILE.lower(),
        config.SOLID_ANGLES_FILE.lower(),
        config.THETA_FILE.lower(),
        "areas.tab",
    }

    candidates: list[Path] = []
    for root, _dirs, files in os.walk(data_dir, followlinks=True):
        for filename in files:
            if not filename.endswith(config.EXT_TAB):
                continue
            if filename.lower() in exclude_basenames:
                continue
            candidates.append(Path(root) / filename)

    def matches_date(path: Path) -> bool:
        if year is None and month is None and day is None:
            return True
        path_str = str(path)
        ok = True
        if year is not None:
            ok &= str(year) in path_str
        if month is not None:
            mm = NUM_STR_TO_MONTH_ABBREV.get(f"{month:02d}")
            if mm is None:
                logging.warning("Month %s not recognized; skipping discovery.", month)
                return False
            ok &= mm in path_str
        if day is not None:
            dd = f"{day:02d}"
            ok &= f"{dd}{config.EXT_TAB}" in path_str
        return ok

    flux_files = sorted(path for path in candidates if matches_date(path))
    logging.debug(
        "Discovered %d candidate ER files under %s", len(flux_files), data_dir
    )
    if (year or month or day) and not flux_files:
        logging.warning(
            "No ER files matched the provided date filters; returning empty list"
        )
    return flux_files
