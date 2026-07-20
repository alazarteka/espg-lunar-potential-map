"""Small shared helpers for diagnostic UIs."""

from __future__ import annotations

import numpy as np


def finite_range(
    data: np.ndarray,
    fallback: tuple[float, float],
    pct: tuple[float, float],
) -> tuple[float, float]:
    """Percentile range over finite values, or ``fallback`` if empty/degenerate."""
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return fallback
    vmin = float(np.nanpercentile(finite, pct[0]))
    vmax = float(np.nanpercentile(finite, pct[1]))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        return fallback
    return vmin, vmax
