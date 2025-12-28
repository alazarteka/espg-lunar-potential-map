"""Shared harmonic helpers for temporal models."""

from __future__ import annotations

from scipy.special import sph_harm_y


def _sph_harm(m: int, l: int, phi, theta):  # noqa: E741
    """Evaluate spherical harmonics using SciPy sph_harm_y (theta=colat, phi=azimuth)."""
    return sph_harm_y(l, m, theta, phi)
