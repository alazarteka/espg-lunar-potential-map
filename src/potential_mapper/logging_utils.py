"""Shared logging configuration helpers."""

from __future__ import annotations

import logging

__all__ = ["setup_logging"]

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(verbose: bool = False) -> None:
    """Configure root logger with consistent format.

    Args:
        verbose: If True, use DEBUG level; otherwise INFO.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATEFMT,
    )
