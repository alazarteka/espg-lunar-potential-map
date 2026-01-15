"""
Cached theta value loader.

Provides a singleton-style loader for theta values (detector look directions)
to avoid repeated np.loadtxt calls across PitchAngle and LossConeFitter.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import numpy as np

from src import config


@lru_cache(maxsize=1)
def get_thetas(theta_path: str | None = None) -> np.ndarray:
    """
    Load theta values from disk, caching the result.

    Args:
        theta_path: Path to theta file. Defaults to config.DATA_DIR / config.THETA_FILE.

    Returns:
        1D array of theta values in degrees.

    Note:
        Uses functools.lru_cache for process-wide caching. The first call
        loads from disk; subsequent calls return the cached array. If
        config.DATA_DIR or THETA_FILE changes at runtime, call
        clear_theta_cache() before calling get_thetas() again.
    """
    if theta_path is None:
        path = config.DATA_DIR / config.THETA_FILE
    else:
        path = Path(theta_path)

    if not path.exists():
        raise FileNotFoundError(f"Theta file not found: {path}")

    logging.debug(f"Loading theta values from {path}")
    return np.loadtxt(path, dtype=np.float64)


def clear_theta_cache() -> None:
    """Clear the theta cache (useful for testing or reloading)."""
    get_thetas.cache_clear()
