"""Spherical harmonic expansion utilities for lunar surface potential."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
try:
    from scipy.special import sph_harm_y as _scipy_sph_harm
    _SPH_HARM_NEW_SIGNATURE = True
except ImportError:  # pragma: no cover - fallback for older SciPy
    from scipy.special import sph_harm as _scipy_sph_harm
    _SPH_HARM_NEW_SIGNATURE = False

from src.potential_mapper.results import PotentialResults


@dataclass(frozen=True)
class HarmonicMode:
    """Descriptor for a single real-valued spherical harmonic basis function."""

    ell: int
    m: int
    kind: str  # 'm0', 'cos', or 'sin'


@dataclass
class HarmonicFit:
    """Result of fitting real spherical harmonics to surface potential data."""

    l_max: int
    coefficients: np.ndarray
    modes: tuple[HarmonicMode, ...]
    valid_mask: np.ndarray
    residuals: np.ndarray
    rms: float

    def predict(self, lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
        """Evaluate the fitted expansion at the provided latitude/longitude."""
        theta, phi = _to_radians(lat_deg, lon_deg)
        design = build_real_spherical_harmonic_matrix(theta, phi, self.modes)
        return design @ self.coefficients


def fit_surface_harmonics(
    results: PotentialResults,
    l_max: int,
    *,
    weights: np.ndarray | None = None,
    regularization: float | None = None,
) -> HarmonicFit:
    """Fit a real spherical-harmonic expansion to projected potentials."""
    if l_max < 0:
        raise ValueError("l_max must be non-negative")

    lat = results.projection_latitude
    lon = results.projection_longitude
    potential = results.projected_potential

    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(potential)
    if not np.any(valid):
        raise ValueError("No valid potential samples available for fitting")

    lat = lat[valid]
    lon = lon[valid]
    potential = potential[valid]

    theta, phi = _to_radians(lat, lon)
    modes = tuple(_enumerate_modes(l_max))
    design = build_real_spherical_harmonic_matrix(theta, phi, modes)

    target = potential.astype(float)
    y = target
    w = None
    if weights is not None:
        if weights.shape != results.projected_potential.shape:
            raise ValueError("weights must match projected_potential shape")
        weights_valid = weights[valid]
        w = np.sqrt(weights_valid.astype(float))
        design = design * w[:, None]
        y = y * w

    if regularization is not None:
        if regularization < 0:
            raise ValueError("regularization must be non-negative")
        if regularization > 0:
            ridge = np.sqrt(regularization) * np.eye(design.shape[1])
            design = np.vstack([design, ridge])
            y = np.concatenate([y, np.zeros(ridge.shape[0])])

    coeffs, residuals, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    if residuals.size == 0:
        residuals = np.array([], dtype=float)

    design_unweighted = build_real_spherical_harmonic_matrix(theta, phi, modes)
    predictions = design_unweighted @ coeffs
    residual_vector = target - predictions
    rms = float(np.sqrt(np.mean(residual_vector**2)))

    return HarmonicFit(
        l_max=l_max,
        coefficients=coeffs,
        modes=modes,
        valid_mask=valid,
        residuals=residual_vector,
        rms=rms,
    )


def build_real_spherical_harmonic_matrix(
    theta: np.ndarray,
    phi: np.ndarray,
    modes: Sequence[HarmonicMode],
) -> np.ndarray:
    """Construct the design matrix for real spherical harmonics up to `l_max`."""
    if theta.shape != phi.shape:
        raise ValueError("theta and phi must have the same shape")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)

    n = theta.size
    design = np.empty((n, len(modes)), dtype=float)

    for idx, mode in enumerate(modes):
        l, m, kind = mode.ell, mode.m, mode.kind
        Y = _compute_sph_harm(l, m, phi, theta)
        if kind == "m0":
            design[:, idx] = Y.real
        elif kind == "cos":
            design[:, idx] = np.sqrt(2.0) * Y.real
        elif kind == "sin":
            design[:, idx] = np.sqrt(2.0) * Y.imag
        else:
            raise ValueError(f"Unknown harmonic kind: {kind}")

    return design


def evaluate_modes(
    coefficients: Sequence[float],
    modes: Sequence[HarmonicMode],
    theta: np.ndarray,
    phi: np.ndarray,
) -> np.ndarray:
    """Evaluate a set of real spherical harmonics with precomputed coefficients."""
    design = build_real_spherical_harmonic_matrix(theta, phi, modes)
    return design @ np.asarray(coefficients, dtype=float)


def _to_radians(lat_deg: Iterable[float], lon_deg: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    lat = np.asarray(lat_deg, dtype=float)
    lon = np.asarray(lon_deg, dtype=float)
    theta = np.deg2rad(90.0 - lat)
    phi = np.deg2rad(lon)
    phi = np.mod(phi + 2 * np.pi, 2 * np.pi)
    return theta, phi


def _enumerate_modes(l_max: int) -> Iterable[HarmonicMode]:
    for l in range(l_max + 1):
        yield HarmonicMode(l, 0, "m0")
        for m in range(1, l + 1):
            yield HarmonicMode(l, m, "cos")
            yield HarmonicMode(l, m, "sin")


def _compute_sph_harm(l: int, m: int, phi: np.ndarray, theta: np.ndarray):
    if _SPH_HARM_NEW_SIGNATURE:
        return _scipy_sph_harm(l, m, theta, phi)
    return _scipy_sph_harm(m, l, phi, theta)
