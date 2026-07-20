"""Shared argparse flag definitions for the potential-mapper CLIs.

``batch``, ``parallel_batch``, and the top-level ``python -m
src.potential_mapper`` CLI all accept the same date-filter and verbosity
flags. Defining them once here keeps the common flags from drifting between
the three entry points; each CLI still adds its own unique flags directly.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import argparse


def validate_date_filters(
    year: int | None,
    month: int | None,
    day: int | None,
) -> str | None:
    """Return an error message if year/month/day filters are not a valid date.

    Partial filters are allowed (e.g. year-only or year+month). When ``day`` is
    set, ``year`` and ``month`` must also be set so the calendar date can be
    checked.
    """
    if month is not None and not 1 <= month <= 12:
        return f"month must be in 1..12 (got {month})"
    if day is not None and not 1 <= day <= 31:
        return f"day must be in 1..31 (got {day})"
    if day is not None and (year is None or month is None):
        return "day filter requires --year and --month"
    if year is not None and month is not None and day is not None:
        try:
            date(year, month, day)
        except ValueError as exc:
            return f"invalid calendar date {year:04d}-{month:02d}-{day:02d}: {exc}"
    return None


def add_common_batch_args(
    parser: argparse.ArgumentParser,
    *,
    year_required: bool = False,
    include_day: bool = True,
    include_overwrite: bool = True,
) -> None:
    """
    Add the date/overwrite/verbosity flags shared by the mapper CLIs.

    Adds ``--year``, ``--month``, optionally ``--day`` and ``--overwrite``,
    and ``-v/--verbose`` to ``parser`` in place.

    Args:
        parser: Parser to extend in place.
        year_required: Make ``--year`` mandatory (used by ``parallel_batch``)
            instead of an optional filter.
        include_day: Add the ``--day`` filter (``parallel_batch`` has no
            day-level filter).
        include_overwrite: Add ``--overwrite`` (the plotting CLI has none).
    """
    if year_required:
        parser.add_argument("--year", type=int, required=True, help="Year to process")
    else:
        parser.add_argument(
            "--year",
            type=int,
            default=None,
            help="Optional year filter (e.g. 1998, 1999)",
        )
    parser.add_argument(
        "--month", type=int, default=None, help="Optional month filter (1-12)"
    )
    if include_day:
        parser.add_argument(
            "--day", type=int, default=None, help="Optional day filter (1-31)"
        )
    if include_overwrite:
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing output file(s)",
        )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose logging"
    )
