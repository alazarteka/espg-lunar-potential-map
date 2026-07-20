"""Response-folded loss-cone mean model (Phase 3).

Cell means are formed by integrating the adiabatic endpoint model over the
operational energy response (DeltaE/E ~ 0.5) and optional angular edge broadening.
Point evaluation at bin centres with fixed 15 eV / 7.5° widths is an
approximation that must be disclosed when used.

Masks used here are *calibration* masks only - never the 0.07-0.79 Lillis band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.losscone.model import DEFAULT_BACKGROUND, synth_losscone, synth_losscone_batch

# Operational LP ER effective energy resolution after adjacent-bin combining.
DEFAULT_DE_OVER_E = 0.5


@dataclass(frozen=True)
class ResponseFoldedParams:
    """Physical parameters for the response-folded mean."""

    u_surface: float
    bs_over_bm: float
    beam_amp: float = 0.0
    # Profiled / prior widths - not the legacy fixed 15 eV / 7.5°.
    beam_width_eV: float = 40.0  # prior scale ~ few × intrinsic sigma_ε then folded
    beam_pitch_sigma_deg: float = 15.0
    edge_broadening_deg: float = 0.0
    background: float = DEFAULT_BACKGROUND
    de_over_e: float = DEFAULT_DE_OVER_E
    n_energy_quad: int = 5


def energy_quadrature(
    energy_centers: np.ndarray,
    *,
    de_over_e: float = DEFAULT_DE_OVER_E,
    n_quad: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite-like uniform nodes over ±DeltaE for each centre energy.

    Returns:
        energies: (nE, n_quad) sample energies
        weights: (n_quad,) normalized weights (sum to 1)
    """
    centers = np.asarray(energy_centers, dtype=np.float64)
    if n_quad < 1:
        raise ValueError("n_quad must be >= 1")
    if n_quad == 1:
        return centers[:, None], np.ones(1, dtype=np.float64)

    # Uniform nodes on [E*(1-f/2), E*(1+f/2)] with f = de_over_e.
    half = 0.5 * float(de_over_e)
    nodes = np.linspace(-half, half, n_quad)
    weights = np.ones(n_quad, dtype=np.float64) / n_quad
    energies = centers[:, None] * (1.0 + nodes[None, :])
    energies = np.maximum(energies, 1e-6)
    return energies, weights


def build_calibration_mask(
    flux: np.ndarray,
    *,
    valid_energy_mask: np.ndarray | None = None,
    sunlight_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Documented calibration / geometry validity only (no Lillis band)."""
    flux = np.asarray(flux, dtype=np.float64)
    mask = np.isfinite(flux) & (flux > 0.0)
    if valid_energy_mask is not None:
        mask = mask & np.asarray(valid_energy_mask, dtype=bool)
    if sunlight_mask is not None:
        mask = mask & np.asarray(sunlight_mask, dtype=bool)
    return mask


def response_folded_mean(
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    params: ResponseFoldedParams,
    *,
    u_spacecraft: float | np.ndarray = 0.0,
) -> np.ndarray:
    """Compute response-folded normalized mean flux, shape (nE, nPitch)."""
    energies_q, weights = energy_quadrature(
        energy_centers, de_over_e=params.de_over_e, n_quad=params.n_energy_quad
    )
    nE, nPitch = pitch_grid.shape
    acc = np.zeros((nE, nPitch), dtype=np.float64)

    for q, w in enumerate(weights):
        e_q = energies_q[:, q]
        # Re-broadcast pitches against this energy sample's centres.
        model = synth_losscone(
            energy_grid=e_q,
            pitch_grid=pitch_grid,
            U_surface=params.u_surface,
            U_spacecraft=u_spacecraft,
            bs_over_bm=params.bs_over_bm,
            beam_width_eV=params.beam_width_eV,
            beam_amp=params.beam_amp,
            beam_pitch_sigma_deg=params.beam_pitch_sigma_deg,
            background=params.background,
        )
        if params.edge_broadening_deg > 0:
            model = _smooth_pitch_edge(model, pitch_grid, params.edge_broadening_deg)
        acc += w * model
    return acc


def response_folded_mean_batch(
    energy_centers: np.ndarray,
    pitch_grid: np.ndarray,
    u_surface: np.ndarray,
    bs_over_bm: np.ndarray,
    beam_amp: np.ndarray,
    *,
    u_spacecraft: float | np.ndarray = 0.0,
    beam_width_eV: float = 40.0,
    beam_pitch_sigma_deg: float = 15.0,
    edge_broadening_deg: float = 0.0,
    background: float = DEFAULT_BACKGROUND,
    de_over_e: float = DEFAULT_DE_OVER_E,
    n_energy_quad: int = 5,
) -> np.ndarray:
    """Batch response-folded means, shape (n_params, nE, nPitch)."""
    energies_q, weights = energy_quadrature(
        energy_centers, de_over_e=de_over_e, n_quad=n_energy_quad
    )
    n_params = int(np.asarray(u_surface).size)
    nE, nPitch = pitch_grid.shape
    acc = np.zeros((n_params, nE, nPitch), dtype=np.float64)
    bw = np.full(n_params, float(beam_width_eV), dtype=np.float64)
    bg = np.full(n_params, float(background), dtype=np.float64)

    for q, w in enumerate(weights):
        e_q = energies_q[:, q]
        models = synth_losscone_batch(
            energy_grid=e_q,
            pitch_grid=pitch_grid,
            U_surface=np.asarray(u_surface, dtype=np.float64),
            U_spacecraft=u_spacecraft,
            bs_over_bm=np.asarray(bs_over_bm, dtype=np.float64),
            beam_width_eV=bw,
            beam_amp=np.asarray(beam_amp, dtype=np.float64),
            beam_pitch_sigma_deg=beam_pitch_sigma_deg,
            background=bg,
        )
        if edge_broadening_deg > 0:
            for i in range(n_params):
                models[i] = _smooth_pitch_edge(
                    models[i], pitch_grid, edge_broadening_deg
                )
        acc += w * models
    return acc


def _smooth_pitch_edge(
    model: np.ndarray, pitch_grid: np.ndarray, sigma_deg: float
) -> np.ndarray:
    """Simple pitch-axis Gaussian blur approximating finite angular response."""
    if sigma_deg <= 0:
        return model
    pitches = np.asarray(pitch_grid[0], dtype=np.float64)
    # Assume uniform pitch sampling for the convolution kernel.
    if pitches.size < 3:
        return model
    dp = float(np.median(np.diff(pitches)))
    if dp <= 0:
        return model
    half = max(1, int(np.ceil(3.0 * sigma_deg / dp)))
    x = np.arange(-half, half + 1, dtype=np.float64) * dp
    kern = np.exp(-0.5 * (x / sigma_deg) ** 2)
    kern /= kern.sum()
    out = np.empty_like(model)
    for i in range(model.shape[0]):
        out[i] = np.convolve(model[i], kern, mode="same")
    return out
