"""Temporal basis function fitting for spherical harmonic coefficients.

Implements the expansion:
    a_lm(t) = Σ_k b_lmk × T_k(t)

where T_k are temporal basis functions (constant, linear, sinusoidal, etc.)
and b_lmk are the parameters fitted to all data jointly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.linalg import lstsq

from .coefficients import (
    DEFAULT_SYNODIC_PERIOD_DAYS,
    HarmonicCoefficients,
    _build_harmonic_design,
    _enforce_reality_condition,
    _harmonic_coefficient_count,
)

# Sidereal period for moon-fixed oscillations
SIDEREAL_PERIOD_DAYS = 27.321661


@dataclass
class BasisFunction:
    """A single temporal basis function."""

    name: str
    func: Callable[[np.ndarray], np.ndarray]


def _make_synodic_basis(period_days: float = DEFAULT_SYNODIC_PERIOD_DAYS) -> list[BasisFunction]:
    """Create cosine and sine bases at the synodic period."""
    period_hours = period_days * 24.0
    return [
        BasisFunction(
            name=f"synodic_cos",
            func=lambda t, p=period_hours: np.cos(2 * np.pi * t / p),
        ),
        BasisFunction(
            name=f"synodic_sin",
            func=lambda t, p=period_hours: np.sin(2 * np.pi * t / p),
        ),
    ]


def _make_sidereal_basis(period_days: float = SIDEREAL_PERIOD_DAYS) -> list[BasisFunction]:
    """Create cosine and sine bases at the sidereal period."""
    period_hours = period_days * 24.0
    return [
        BasisFunction(
            name=f"sidereal_cos",
            func=lambda t, p=period_hours: np.cos(2 * np.pi * t / p),
        ),
        BasisFunction(
            name=f"sidereal_sin",
            func=lambda t, p=period_hours: np.sin(2 * np.pi * t / p),
        ),
    ]


# Available basis function factories (for parsing user input)
AVAILABLE_BASES: dict[str, Callable[[], list[BasisFunction]]] = {
    "constant": lambda: [BasisFunction("constant", lambda t: np.ones_like(t))],
    "linear": lambda: [BasisFunction("linear", lambda t: t / max(t.max(), 1.0))],
    "synodic": _make_synodic_basis,
    "sidereal": _make_sidereal_basis,
}


def _get_basis_func_by_name(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Get a basis function by its expanded name (e.g., 'synodic_cos')."""
    synodic_hours = DEFAULT_SYNODIC_PERIOD_DAYS * 24.0
    sidereal_hours = SIDEREAL_PERIOD_DAYS * 24.0
    
    EXPANDED_BASES = {
        "constant": lambda t: np.ones_like(t),
        "linear": lambda t: t / max(t.max(), 1.0),
        "synodic_cos": lambda t: np.cos(2 * np.pi * t / synodic_hours),
        "synodic_sin": lambda t: np.sin(2 * np.pi * t / synodic_hours),
        "sidereal_cos": lambda t: np.cos(2 * np.pi * t / sidereal_hours),
        "sidereal_sin": lambda t: np.sin(2 * np.pi * t / sidereal_hours),
    }
    if name not in EXPANDED_BASES:
        raise ValueError(f"Unknown expanded basis name: {name}")
    return EXPANDED_BASES[name]


def parse_basis_spec(spec: str) -> list[BasisFunction]:
    """
    Parse a comma-separated basis specification.

    Examples:
        "constant,synodic" -> [constant, synodic_cos, synodic_sin]
        "constant,linear,synodic" -> [constant, linear, synodic_cos, synodic_sin]
    """
    basis_funcs: list[BasisFunction] = []
    for name in spec.split(","):
        name = name.strip().lower()
        if name not in AVAILABLE_BASES:
            raise ValueError(
                f"Unknown basis '{name}'. Available: {list(AVAILABLE_BASES.keys())}"
            )
        basis_funcs.extend(AVAILABLE_BASES[name]())
    return basis_funcs


def build_temporal_design(t_hours: np.ndarray, bases: list[BasisFunction]) -> np.ndarray:
    """
    Build temporal design matrix.

    Args:
        t_hours: Time in hours since reference (shape: N_measurements)
        bases: List of basis functions

    Returns:
        Matrix of shape (N_measurements, K) where K = len(bases)
    """
    K = len(bases)
    T = np.empty((t_hours.size, K), dtype=np.float64)
    for k, basis in enumerate(bases):
        T[:, k] = basis.func(t_hours)
    return T


@dataclass
class BasisFitResult:
    """Result of temporal basis fitting."""

    basis_names: list[str]
    basis_coeffs: np.ndarray  # Shape: (K, n_sph_coeffs), complex
    lmax: int
    n_samples: int
    rms_residual: float


def fit_temporal_basis(
    utc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    potential: np.ndarray,
    lmax: int,
    basis_spec: str = "constant,synodic",
    l2_penalty: float = 0.0,
) -> BasisFitResult:
    """
    Fit spherical harmonic coefficients with temporal basis expansion.

    The model is:
        Φ(lat, lon, t) = Σ_lm Σ_k b_lmk × T_k(t) × Y_lm(lat, lon)

    Args:
        utc: Timestamps (datetime64)
        lat: Latitudes in degrees
        lon: Longitudes in degrees
        potential: Observed potentials
        lmax: Maximum spherical harmonic degree
        basis_spec: Comma-separated basis names (e.g., "constant,synodic")
        l2_penalty: Ridge regularization strength

    Returns:
        BasisFitResult with fitted coefficients
    """
    # Parse basis specification
    bases = parse_basis_spec(basis_spec)
    K = len(bases)
    n_coeffs = _harmonic_coefficient_count(lmax)

    logging.info(
        "Fitting temporal basis: %d bases × %d spherical harmonics = %d parameters",
        K,
        n_coeffs,
        K * n_coeffs,
    )
    logging.info("Bases: %s", [b.name for b in bases])

    # Convert times to hours since start
    t_hours = (utc - utc.min()).astype("timedelta64[s]").astype(np.float64) / 3600.0

    # Build temporal design matrix: (N, K)
    T = build_temporal_design(t_hours, bases)

    # Build spatial design matrix: (N, n_coeffs)
    Y = _build_harmonic_design(lat, lon, lmax)

    # Build full design matrix: (N, K * n_coeffs)
    # Each column is Y_lm × T_k for all (l,m,k) combinations
    # Layout: [Y_00*T_0, Y_00*T_1, ..., Y_lmax,lmax*T_{K-1}]
    N = len(potential)
    design = np.empty((N, K * n_coeffs), dtype=np.complex128)
    for k in range(K):
        design[:, k * n_coeffs : (k + 1) * n_coeffs] = Y * T[:, k : k + 1]

    potential_complex = potential.astype(np.complex128)

    # Solve with optional ridge regularization
    if l2_penalty > 0.0:
        lam = np.sqrt(l2_penalty)
        identity = np.eye(K * n_coeffs, dtype=design.dtype)
        design_aug = np.vstack([design, lam * identity])
        rhs_aug = np.concatenate([potential_complex, np.zeros(K * n_coeffs, dtype=np.complex128)])
        b_flat, *_ = lstsq(design_aug, rhs_aug, rcond=None)
    else:
        b_flat, *_ = lstsq(design, potential_complex, rcond=None)

    # Reshape to (K, n_coeffs)
    b = b_flat.reshape(K, n_coeffs)

    # Enforce reality condition on each temporal slice
    for k in range(K):
        b[k] = _enforce_reality_condition(b[k], lmax)

    # Compute RMS residual
    predicted = np.real(design @ b_flat)
    residuals = potential - predicted
    rms_residual = float(np.sqrt(np.mean(residuals**2)))

    logging.info("Fit complete: RMS residual = %.2f V", rms_residual)

    return BasisFitResult(
        basis_names=[basis.name for basis in bases],
        basis_coeffs=b,
        lmax=lmax,
        n_samples=N,
        rms_residual=rms_residual,
    )


def reconstruct_at_times(
    result: BasisFitResult,
    times: np.ndarray,
    reference_time: np.datetime64,
) -> list[HarmonicCoefficients]:
    """
    Reconstruct spherical harmonic coefficients at specified times.

    Args:
        result: BasisFitResult from fit_temporal_basis
        times: Array of datetime64 to reconstruct at
        reference_time: Reference time (t=0) for basis functions

    Returns:
        List of HarmonicCoefficients, one per requested time
    """
    # Get basis functions by their expanded names
    basis_funcs = [_get_basis_func_by_name(name) for name in result.basis_names]

    results = []
    for t in times:
        t_hours = (t - reference_time).astype("timedelta64[s]").astype(np.float64) / 3600.0

        # Evaluate basis functions at this time
        T_k = np.array([func(np.array([t_hours]))[0] for func in basis_funcs])

        # Reconstruct coefficients: a_lm = Σ_k b_lmk × T_k
        coeffs = T_k @ result.basis_coeffs  # (n_coeffs,)

        results.append(
            HarmonicCoefficients(
                time=t,
                lmax=result.lmax,
                coeffs=coeffs,
                n_samples=result.n_samples,
                spatial_coverage=1.0,  # N/A for basis fitting
                rms_residual=result.rms_residual,
            )
        )

    return results


def save_basis_result(result: BasisFitResult, output_path: Path) -> None:
    """Save temporal basis fit result to NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        basis_names=np.array(result.basis_names, dtype=str),
        basis_coeffs=result.basis_coeffs,
        lmax=result.lmax,
        n_samples=result.n_samples,
        rms_residual=result.rms_residual,
    )
    logging.info("Saved basis fit result to %s", output_path)
