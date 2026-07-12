"""Independent sector-level simulator for Experiment 006 C1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.polynomial.legendre import leggauss

if TYPE_CHECKING:
    from support_data import GeometrySupport


@dataclass(frozen=True)
class SimulatedCase:
    """One latent native observation and its two derived normalizations."""

    native_flux: np.ndarray
    native_valid: np.ndarray
    incident_mean: np.ndarray
    conjugate: np.ndarray
    latents: dict[str, Any]


def _uniform(rng: np.random.Generator, bounds: list[float]) -> float:
    return float(rng.uniform(float(bounds[0]), float(bounds[1])))


def _logistic(pitch: np.ndarray, boundary: np.ndarray, width: np.ndarray) -> np.ndarray:
    argument = np.clip((pitch - boundary) / width, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-argument))


def _continuum(
    energy: np.ndarray,
    pitch: np.ndarray,
    family: str,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Evaluate an independent positive continuum on broadcastable arrays."""
    cfg = config["backgrounds"]
    x = np.log(energy / 220.0)
    level = _uniform(rng, cfg["cone_level_bounds"])
    contrast = _uniform(rng, cfg["contrast_bounds"])
    boundary = _uniform(rng, cfg["boundary_deg_bounds"])
    width = _uniform(rng, cfg["transition_width_deg_bounds"])
    slope = _uniform(rng, cfg["log_energy_slope_bounds"])
    curvature = _uniform(rng, cfg["log_energy_curvature_bounds"])
    spectral = np.exp(slope * x + curvature * x * x)
    transition = _logistic(pitch, boundary + 5.0 * slope * x, width)
    reflected = level + contrast * transition
    latents = {
        "level": level,
        "contrast": contrast,
        "boundary_deg": boundary,
        "width_deg": width,
        "energy_slope": slope,
        "energy_curvature": curvature,
    }
    if family == "double_transition_refilling":
        fraction = _uniform(rng, cfg["secondary_transition_fraction_bounds"])
        second = _logistic(pitch, boundary - 18.0, width * 1.4)
        reflected += fraction * contrast * (1.0 - second)
        latents["secondary_fraction"] = fraction
    elif family == "asymmetric_shoulder":
        fraction = _uniform(rng, cfg["shoulder_fraction_bounds"])
        shoulder = np.exp(-0.5 * np.square((pitch - (boundary + 16.0)) / (1.8 * width)))
        reflected += fraction * contrast * shoulder
        latents["shoulder_fraction"] = fraction
    elif family == "correlated_smooth_energy_pitch_field":
        amplitude = _uniform(rng, cfg["correlated_field_log_amplitude_bounds"])
        e_length = _uniform(rng, cfg["correlated_field_log_energy_length_bounds"])
        p_length = _uniform(rng, cfg["correlated_field_pitch_length_deg_bounds"])
        phase = rng.uniform(0.0, 2.0 * np.pi, size=3)
        field = sum(
            np.sin((index + 1) * x / e_length + pitch / p_length + phase[index])
            for index in range(3)
        ) / np.sqrt(3.0)
        reflected *= np.exp(amplitude * field)
        latents.update(
            field_amplitude=amplitude, energy_length=e_length, pitch_length=p_length
        )
    elif family != "independent_smooth_logistic":
        raise ValueError(f"Unknown background family: {family}")
    incident = 1.0 + 0.04 * np.cos(np.deg2rad(pitch)) * np.tanh(x)
    values = spectral * np.where(pitch < 90.0, incident, reflected)
    return np.maximum(values, np.finfo(float).tiny), latents


def _channel_continuum(
    low_ev: np.ndarray,
    high_ev: np.ndarray,
    pitches_deg: np.ndarray,
    family: str,
    rng: np.random.Generator,
    config: dict[str, Any],
    nodes_count: int | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Average the continuum over each native energy-channel width."""
    count = int(nodes_count or config["response"]["channel_quadrature_nodes"])
    nodes, weights = leggauss(count)
    energy = (
        0.5 * (high_ev - low_ev)[:, None] * nodes + 0.5 * (high_ev + low_ev)[:, None]
    )
    values, latents = _continuum(
        energy[:, :, None], pitches_deg[:, None, :], family, rng, config
    )
    return 0.5 * np.sum(values * weights[None, :, None], axis=1), latents


def _beam_response(
    low_ev: np.ndarray,
    high_ev: np.ndarray,
    pitches_deg: np.ndarray,
    row: dict[str, Any],
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    cfg = config["response"]
    family = str(row["response_family"])
    center = float(row["true_center_ev"])
    fwhm = float(cfg["nominal_fwhm_fraction"])
    offset = 0.0
    intrinsic = 0.0
    tail_fraction = 0.0
    tail_scale = 1.0
    if family == "width_stress":
        stratum = int(row["response_direction"])
        fwhm = _uniform(rng, cfg["width_stress_strata"][stratum])
    elif family == "energy_scale_offset":
        offset = _uniform(rng, cfg["energy_scale_fractional_offset_bounds"])
    elif family == "intrinsic_asymmetric_broadening":
        intrinsic = _uniform(rng, cfg["intrinsic_fwhm_fraction_bounds"])
        tail_fraction = _uniform(rng, cfg["asymmetric_tail_fraction_bounds"])
        tail_scale = _uniform(rng, cfg["asymmetric_tail_scale_bounds"])
    elif family not in {"nominal_linear_gaussian", "log_energy_gaussian"}:
        raise ValueError(f"Unknown response family: {family}")
    nodes, weights = leggauss(int(cfg["channel_quadrature_nodes"]))
    energy = (
        0.5 * (high_ev - low_ev)[:, None] * nodes + 0.5 * (high_ev + low_ev)[:, None]
    )
    effective_center = center * (1.0 + offset)
    total_fwhm = np.hypot(fwhm, intrinsic)
    if family == "log_energy_gaussian":
        sigma_log = np.log1p(total_fwhm) / 2.354820045
        profile = np.exp(
            -0.5 * np.square(np.log(energy / effective_center) / sigma_log)
        )
    else:
        sigma = total_fwhm * effective_center / 2.354820045
        profile = np.exp(-0.5 * np.square((energy - effective_center) / sigma))
    if tail_fraction:
        tail_center = effective_center * tail_scale
        sigma = total_fwhm * tail_center / 2.354820045
        profile = (1.0 - tail_fraction) * profile + tail_fraction * np.exp(
            -0.5 * np.square((energy - tail_center) / sigma)
        )
    integrated = 0.5 * np.sum(profile * weights, axis=1)
    pitch = np.exp(
        -0.5
        * np.square(
            (pitches_deg - float(row.get("beam_axis_deg", 180.0)))
            / float(row["pitch_sigma_deg"])
        )
    )
    latents = {
        "response_fwhm_fraction": fwhm,
        "energy_scale_offset": offset,
        "intrinsic_fwhm_fraction": intrinsic,
        "tail_fraction": tail_fraction,
        "tail_scale": tail_scale,
    }
    return float(row["amplitude"]) * integrated[:, None] * pitch, latents


def normalize_native(
    flux: np.ndarray, pitches_deg: np.ndarray, valid: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Derive both frozen normalization branches from one native realization."""
    mean_product = np.full_like(flux, np.nan, dtype=float)
    conjugate_product = np.full_like(flux, np.nan, dtype=float)
    for row in range(flux.shape[0]):
        incident = valid[row] & (pitches_deg[row] < 90.0)
        reflected = valid[row] & (pitches_deg[row] >= 90.0)
        if np.count_nonzero(incident) < 6 or np.count_nonzero(reflected) < 6:
            continue
        denominator = float(np.mean(flux[row, incident]))
        mean_product[row, reflected] = flux[row, reflected] / denominator
        incident_indices = np.flatnonzero(incident)
        for index in np.flatnonzero(reflected):
            target = 180.0 - pitches_deg[row, index]
            nearest = incident_indices[
                np.argmin(np.abs(pitches_deg[row, incident_indices] - target))
            ]
            conjugate_product[row, index] = flux[row, index] / flux[row, nearest]
    return mean_product, conjugate_product


def _log_noise(
    energies_ev: np.ndarray,
    shape: tuple[int, int],
    base_sigma: float,
    rng: np.random.Generator,
    config: dict[str, Any],
) -> tuple[np.ndarray, dict[str, float]]:
    """Draw deterministic heteroskedastic, energy/pitch-correlated log noise."""
    cfg = config["noise"]
    exponent = _uniform(rng, cfg["heteroskedastic_energy_exponent_bounds"])
    correlated_fraction = _uniform(rng, cfg["correlated_fraction_bounds"])
    sigma = base_sigma * np.power(energies_ev / 220.0, exponent)

    independent = rng.standard_normal(shape)
    energy_indices = np.arange(shape[0])
    energy_distance = np.abs(np.subtract.outer(energy_indices, energy_indices))
    pitch_distance = np.abs(np.subtract.outer(np.arange(shape[1]), np.arange(shape[1])))
    energy_covariance = np.exp(-energy_distance / 2.0) + 1e-12 * np.eye(shape[0])
    pitch_covariance = np.exp(-pitch_distance / 2.0) + 1e-12 * np.eye(shape[1])
    energy_cholesky = np.linalg.cholesky(energy_covariance)
    pitch_cholesky = np.linalg.cholesky(pitch_covariance)
    correlated = energy_cholesky @ rng.standard_normal(shape) @ pitch_cholesky.T
    standardized = (
        np.sqrt(1.0 - correlated_fraction) * independent
        + np.sqrt(correlated_fraction) * correlated
    )
    return sigma[:, None] * standardized, {
        "noise_energy_exponent": exponent,
        "noise_correlated_fraction": correlated_fraction,
    }


def simulate_case(
    support: GeometrySupport, row: dict[str, Any], config: dict[str, Any]
) -> SimulatedCase:
    """Generate one deterministic latent sector observation."""
    seeds = {name: int(row[f"seed_{name}"]) for name in config["seeds"]["namespaces"]}
    continuum_rng = np.random.default_rng(seeds["continuum"])
    response_rng = np.random.default_rng(seeds["response"])
    pitches = support.pitches_deg
    continuum, continuum_latents = _channel_continuum(
        support.channel_low_ev,
        support.channel_high_ev,
        pitches,
        str(row["background_family"]),
        continuum_rng,
        config,
    )
    beam, response_latents = _beam_response(
        support.channel_low_ev,
        support.channel_high_ev,
        pitches,
        row,
        response_rng,
        config,
    )
    native = continuum + beam
    temporal_rng = np.random.default_rng(seeds["temporal"])
    temporal_family = str(row["temporal_family"])
    if temporal_family != "none":
        amplitude = _uniform(
            temporal_rng, config["noise"]["temporal_log_amplitude_bounds"]
        )
        coordinate = np.linspace(-1.0, 1.0, native.shape[0])[:, None]
        effect = (
            coordinate
            if temporal_family == "ramp"
            else np.exp(
                -0.5 * np.square((coordinate - temporal_rng.uniform(-0.6, 0.6)) / 0.2)
            )
        )
        native *= np.exp(amplitude * effect)
    noise_rng = np.random.default_rng(seeds["noise"])
    log_noise, noise_latents = _log_noise(
        support.energies_ev,
        native.shape,
        float(row["log_noise_sigma"]),
        noise_rng,
        config,
    )
    native *= np.exp(log_noise)
    mask_rng = np.random.default_rng(seeds["mask"])
    valid = np.isfinite(pitches)
    row_missing = mask_rng.random(native.shape[0]) < float(
        config["noise"]["missing_energy_row_probability"]
    )
    sector_missing = mask_rng.random(native.shape) < float(
        config["noise"]["missing_sector_probability"]
    )
    valid &= ~row_missing[:, None] & ~sector_missing
    native = np.maximum(native, np.finfo(float).tiny)
    incident_mean, conjugate = normalize_native(native, pitches, valid)
    return SimulatedCase(
        native,
        valid,
        incident_mean,
        conjugate,
        {**continuum_latents, **response_latents, **noise_latents},
    )
