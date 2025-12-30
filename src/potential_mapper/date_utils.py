"""Month name mappings for date-related helpers."""

from pathlib import Path

__all__ = [
    "MONTH_ABBREV_TO_NUM",
    "MONTH_INT_TO_ABBREV",
    "NUM_STR_TO_MONTH_ABBREV",
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
        filename: Filename with or without extension (e.g., '3D980415' or '3D980415.TAB')

    Returns:
        Tuple of (year, month, day) where year is full 4-digit year
    """
    stem = Path(filename).stem
    yy = int(stem[2:4])
    mm = int(stem[4:6])
    dd = int(stem[6:8])
    year = 1900 + yy if yy > 50 else 2000 + yy
    return year, mm, dd
