"""Canonical float64 forward model for Experiment 006 beam inference.

This module contains no optimizer and no simulation generator. It defines the
scientific residual evaluated by both the CPU reference and GPU candidate
backends. The NumPy implementation is the numerical reference for D1 parity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import erf, expit

N_CONTINUUM_QUANTITIES = 4
N_CONTINUUM_COEFFICIENTS = 12
N_PARAMETERS = 15
SQRT_TWO = np.sqrt(2.0)
SQRT_PI_OVER_TWO = np.sqrt(np.pi / 2.0)


def load_configuration(path: Path | None = None) -> dict[str, Any]:
    """Load the Experiment 006 design configuration."""
    if path is None:
        path = Path(__file__).with_name("configuration-d1.json")
    return json.loads(path.read_text(encoding="ascii"))


def channel_edges_ev(energies_ev: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Infer positive log-spaced channel edges from unique channel centers."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    if energies.ndim != 1 or energies.size < 2:
        raise ValueError("At least two one-dimensional energy centers are required")
    if not np.isfinite(energies).all() or np.any(energies <= 0.0):
        raise ValueError("Energy centers must be finite and positive")
    if np.unique(energies).size != energies.size:
        raise ValueError("Energy centers must be unique")

    order = np.argsort(energies)
    sorted_energy = energies[order]
    internal = np.sqrt(sorted_energy[:-1] * sorted_energy[1:])
    sorted_low = np.empty_like(sorted_energy)
    sorted_high = np.empty_like(sorted_energy)
    sorted_low[1:] = internal
    sorted_high[:-1] = internal
    sorted_low[0] = sorted_energy[0] ** 2 / internal[0]
    sorted_high[-1] = sorted_energy[-1] ** 2 / internal[-1]

    low = np.empty_like(sorted_low)
    high = np.empty_like(sorted_high)
    low[order] = sorted_low
    high[order] = sorted_high
    return low, high


def integrated_energy_response(
    channel_low_ev: np.ndarray,
    channel_high_ev: np.ndarray,
    center_ev: float,
    fwhm_fraction: float,
) -> np.ndarray:
    """Average a linear-energy Gaussian response over each channel interval."""
    low = np.asarray(channel_low_ev, dtype=np.float64)
    high = np.asarray(channel_high_ev, dtype=np.float64)
    if low.shape != high.shape:
        raise ValueError("Channel lower and upper edges must have matching shapes")
    if center_ev <= 0.0 or fwhm_fraction <= 0.0:
        raise ValueError("Beam center and response width must be positive")
    width = high - low
    if np.any(~np.isfinite(width)) or np.any(width <= 0.0):
        raise ValueError("Every channel must have finite positive width")
    sigma = fwhm_fraction * center_ev / 2.355
    upper = (high - center_ev) / (SQRT_TWO * sigma)
    lower = (low - center_ev) / (SQRT_TWO * sigma)
    integral = sigma * SQRT_PI_OVER_TWO * (erf(upper) - erf(lower))
    return np.maximum(integral / width, 0.0)


def energy_basis(
    energies_ev: np.ndarray, energy_domain_ev: tuple[float, float]
) -> np.ndarray:
    """Quadratic basis on a fixed normalized log-energy domain."""
    energies = np.asarray(energies_ev, dtype=np.float64)
    energy_low, energy_high = energy_domain_ev
    if energy_low <= 0.0 or energy_high <= energy_low:
        raise ValueError("Invalid energy domain")
    scaled = (
        2.0
        * (
            (np.log(energies) - np.log(energy_low))
            / (np.log(energy_high) - np.log(energy_low))
        )
        - 1.0
    )
    return np.column_stack((np.ones_like(scaled), scaled, np.square(scaled)))


def continuum_row_parameters(
    energies_ev: np.ndarray,
    coefficients: np.ndarray,
    energy_domain_ev: tuple[float, float],
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
) -> np.ndarray:
    """Map 12 latent coefficients to four bounded row parameters."""
    coefficient_values = np.asarray(coefficients, dtype=np.float64)
    if coefficient_values.shape != (N_CONTINUUM_COEFFICIENTS,):
        raise ValueError("Expected 12 continuum coefficients")
    lower = np.asarray(physical_lower, dtype=np.float64)
    upper = np.asarray(physical_upper, dtype=np.float64)
    if lower.shape != (N_CONTINUUM_QUANTITIES,) or upper.shape != (
        N_CONTINUUM_QUANTITIES,
    ):
        raise ValueError("Continuum physical bounds must contain four values")
    basis = energy_basis(energies_ev, energy_domain_ev)
    coefficient_matrix = coefficient_values.reshape(N_CONTINUUM_QUANTITIES, 3)
    latent = basis @ coefficient_matrix.T
    return lower + (upper - lower) * expit(latent)


def model_values(
    energies_ev: np.ndarray,
    channel_low_ev: np.ndarray,
    channel_high_ev: np.ndarray,
    row_ids: np.ndarray,
    pitches_deg: np.ndarray,
    parameters: np.ndarray,
    *,
    energy_domain_ev: tuple[float, float],
    fwhm_fraction: float,
    physical_lower: np.ndarray,
    physical_upper: np.ndarray,
) -> np.ndarray:
    """Evaluate the channel-integrated smooth-continuum beam model."""
    parameter_values = np.asarray(parameters, dtype=np.float64)
    if parameter_values.shape != (N_PARAMETERS,):
        raise ValueError("Expected 15 beam-model parameters")
    row_index = np.asarray(row_ids, dtype=np.int64)
    pitch = np.asarray(pitches_deg, dtype=np.float64)
    if row_index.shape != pitch.shape:
        raise ValueError("Row identifiers and pitch samples must match")
    if np.any(row_index < 0) or np.any(row_index >= len(energies_ev)):
        raise ValueError("Row identifier outside energy support")

    amplitude, center_ev, pitch_sigma_deg = parameter_values[:3]
    row_parameters = continuum_row_parameters(
        energies_ev,
        parameter_values[3:],
        energy_domain_ev,
        physical_lower,
        physical_upper,
    )
    selected = row_parameters[row_index]
    transition = expit((selected[:, 2] - pitch) / selected[:, 3])
    continuum = selected[:, 0] + selected[:, 1] * transition

    row_energy_response = integrated_energy_response(
        channel_low_ev,
        channel_high_ev,
        float(center_ev),
        fwhm_fraction,
    )
    pitch_response = np.exp(-0.5 * np.square((pitch - 180.0) / float(pitch_sigma_deg)))
    beam = amplitude * row_energy_response[row_index] * pitch_response
    return continuum + beam


def log_residuals(observed: np.ndarray, modeled: np.ndarray) -> np.ndarray:
    """Return the canonical unweighted logarithmic residual vector."""
    values = np.asarray(observed, dtype=np.float64)
    model = np.asarray(modeled, dtype=np.float64)
    if values.shape != model.shape:
        raise ValueError("Observed and modeled arrays must match")
    if np.any(values <= 0.0) or np.any(model <= 0.0):
        raise ValueError("Log-residual inputs must be positive")
    return np.log(model) - np.log(values)


def continuum_physical_bounds(
    configuration: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Extract physical continuum bounds in canonical quantity order."""
    estimator = configuration["model"]
    lower = np.asarray(
        [
            estimator["continuum_cone_level_bounds"][0],
            estimator["continuum_contrast_bounds"][0],
            estimator["continuum_boundary_deg_bounds"][0],
            estimator["continuum_width_deg_bounds"][0],
        ],
        dtype=np.float64,
    )
    upper = np.asarray(
        [
            estimator["continuum_cone_level_bounds"][1],
            estimator["continuum_contrast_bounds"][1],
            estimator["continuum_boundary_deg_bounds"][1],
            estimator["continuum_width_deg_bounds"][1],
        ],
        dtype=np.float64,
    )
    return lower, upper
