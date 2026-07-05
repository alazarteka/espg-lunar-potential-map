"""Poisson noise model for loss-cone fitting.

Reconstructs per-bin electron counts from calibrated ER flux so a variance-
weighted (Poisson-deviance / Cash) objective can be formed, giving the U-width a
genuine confidence-interval basis.

The reconstruction inverts the PDS-PPI LP ER archive flux definition
``F = C / (Δt · G · E)``, i.e. ``C = F · G · E · Δt`` where

* ``G = 0.02 cm²·sr`` (``config.GEOMETRIC_FACTOR``) already folds in the
  per-channel solid angle and the ~25% ΔE/E passband — that is why the count uses
  the center energy ``E`` and no separate ΔE (see ``config`` comment),
* ``Δt = ACCUMULATION_TIME / N_azimuth(band)`` is the per-channel integration
  time (the half-spin accumulation shared among a latitude band's azimuth bins).

Counts are kept **float and unrounded**: they are used as Poisson expectations,
not integer draws. See ``docs`` / the PDS 3D e-flux descriptor.
"""

from __future__ import annotations

import numpy as np

from src import config
from src.utils import thetas as thetas_module


def _mag(x: object) -> float:
    """Return the plain magnitude of a possibly-pint quantity."""
    return float(x.magnitude) if hasattr(x, "magnitude") else float(x)  # type: ignore[attr-defined]


def channel_integration_times(thetas: np.ndarray | None = None) -> np.ndarray:
    """Per-channel accumulation time Δt in seconds, shape ``(CHANNELS,)``.

    ``Δt = ACCUMULATION_TIME / N_azimuth(latitude band)``: the half-spin
    accumulation is shared among a band's azimuth bins (PDS archive).
    """
    if thetas is None:
        thetas = thetas_module.get_thetas()
    acc = _mag(config.ACCUMULATION_TIME)
    return np.array(
        [acc / config.BINS_BY_LATITUDE[float(t)] for t in np.asarray(thetas)],
        dtype=np.float64,
    )


def reconstruct_counts(
    raw_flux: np.ndarray, energies: np.ndarray, dt: np.ndarray | None = None
) -> np.ndarray:
    """Per-bin expected electron counts ``λ = F·G·E·Δt`` (float, unrounded).

    Args:
        raw_flux: (nE, nPitch) directional flux [particle cm⁻² s⁻¹ sr⁻¹ eV⁻¹].
        energies: (nE,) center energies [eV].
        dt: (nPitch,) per-channel integration time [s]; defaults to
            ``channel_integration_times()``.

    Negative flux is clamped to 0 (an unphysical count). This is safe for the
    Poisson deviance, which handles ``C = 0`` natively — unlike a Gaussian weight,
    where a zero count would give infinite weight.
    """
    if dt is None:
        dt = channel_integration_times()
    g = _mag(config.GEOMETRIC_FACTOR)
    raw = np.clip(np.asarray(raw_flux, dtype=np.float64), 0.0, None)
    return (
        raw
        * g
        * np.asarray(energies, dtype=np.float64)[:, None]
        * np.asarray(dt)[None, :]
    )


def count_scale(
    f_scale: np.ndarray, energies: np.ndarray, dt: np.ndarray | None = None
) -> np.ndarray:
    """Per-bin count scale ``κ = F_scale·G·E·Δt`` mapping a normalized model to
    expected counts (``μ = model·κ + background``).

    Args:
        f_scale: (nE,) incident-flux normalization per energy row (paper Eq 1;
            the mean incident-side flux used to normalize that row).
        energies: (nE,) center energies [eV].
        dt: (nPitch,) per-channel integration time [s].
    """
    if dt is None:
        dt = channel_integration_times()
    g = _mag(config.GEOMETRIC_FACTOR)
    return (
        np.asarray(f_scale, dtype=np.float64)[:, None]
        * g
        * np.asarray(energies, dtype=np.float64)[:, None]
        * np.asarray(dt)[None, :]
    )


def chunk_noise(
    raw_flux: np.ndarray,
    norm2d: np.ndarray,
    energies: np.ndarray,
    dt: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-bin (counts λ, count-scale κ, background b) for one sweep.

    Uses the identity ``κ = λ / norm2d`` (since ``norm2d = F_raw / F_scale`` and
    ``λ = F_raw·G·E·Δt``, so ``κ = F_scale·G·E·Δt = λ/norm2d``) — this recovers
    the per-bin count scale, including the per-channel Δt, without re-deriving
    ``F_scale``. Bins where ``norm2d ≤ 0`` (always masked out) get κ = 1.

    Args:
        raw_flux: (nE, nPitch) directional flux.
        norm2d: (nE, nPitch) incident-normalized flux (paper Eq 1).
        energies: (nE,) center energies [eV].
        dt: (nPitch,) per-channel integration time [s].

    Returns:
        (counts, count_scale, background): counts/scale shape (nE, nPitch),
        background shape (nPitch,).
    """
    if dt is None:
        dt = channel_integration_times()
    lam = reconstruct_counts(raw_flux, energies, dt)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = lam / np.asarray(norm2d, dtype=np.float64)
    kappa = np.where(np.isfinite(kappa) & (kappa > 0.0), kappa, 1.0)
    return lam, kappa, background_counts(dt)


def background_counts(dt: np.ndarray | None = None) -> np.ndarray:
    """Per-channel background counts ``b = N_bg · Δt`` [counts], shape ``(nPitch,)``.

    The ~10 counts/s ER background is not subtracted in the archive; it enters the
    Poisson expectation ``μ = model·κ + b``.
    """
    if dt is None:
        dt = channel_integration_times()
    return float(config.LOSS_CONE_BACKGROUND_RATE_CPS) * np.asarray(
        dt, dtype=np.float64
    )
