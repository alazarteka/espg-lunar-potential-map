"""Synthetic data helpers shared between tests and scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.flux import ERData
from src.physics.kappa import KappaParams, omnidirectional_flux
from src.utils.units import ureg

__all__ = [
    "prepare_flux",
    "prepare_phis",
    "prepare_synthetic_er",
    "prepare_synthetic_er_poisson",
]


def prepare_phis() -> tuple[list[float], np.ndarray]:
    """Prepare mock instrument viewing angles and solid angles.

    Returns
    -------
    list[float], np.ndarray
        Tuple of azimuth angles (degrees) and solid angles (sr) for all
        instrument channels, matching the structure expected by `ERData`.
    """

    phis: list[float] = []
    phis_by_latitude = {
        78.75: ([], 4, 0.119570),
        56.25: ([], 8, 0.170253),
        33.75: ([], 16, 0.127401),
        11.25: ([], 16, 0.150279),
        -11.25: ([], 16, 0.150279),
        -33.75: ([], 16, 0.127401),
        -56.25: ([], 8, 0.170253),
        -78.75: ([], 4, 0.119570),
    }

    thetas: list[float] = []
    for key, (_, count, _solid_angle) in phis_by_latitude.items():
        thetas.extend([key] * count)

    # Sort by absolute latitude to keep a deterministic ordering
    thetas = sorted(np.array(thetas), key=lambda x: abs(x))
    solid_angles = np.array([phis_by_latitude[theta][2] for theta in thetas])

    phi_counter: dict[float, int] = {}
    for theta in thetas:
        if theta not in phi_counter:
            phi_counter[theta] = 0
        n_channels = phis_by_latitude[theta][1]
        phi_value = phi_counter[theta] / n_channels * 360
        phis.append(phi_value)
        phi_counter[theta] += 1

    return phis, solid_angles


def prepare_flux(
    density: float = 1e6,
    kappa: float = 5.0,
    theta: float = 1.1e5,
):
    """Prepare a theoretical omnidirectional particle flux."""

    params = KappaParams(
        density=density * ureg.particle / ureg.meter**3,
        kappa=kappa,
        theta=theta * ureg.meter / ureg.second,
    )

    energy_centers = np.geomspace(2e1, 2e4, config.SWEEP_ROWS) * ureg.electron_volt

    omnidirectional_particle_flux = omnidirectional_flux(
        parameters=params, energy=energy_centers
    )
    return omnidirectional_particle_flux, energy_centers


def prepare_synthetic_er(
    density: float = 1e6,
    kappa: float = 5.0,
    theta: float = 1.1e5,
) -> ERData:
    """Construct a synthetic `ERData` instance for deterministic tests."""

    phis, _solid_angles = prepare_phis()
    omnidirectional_particle_flux, energy_centers = prepare_flux(
        density=density, kappa=kappa, theta=theta
    )

    synthetic_er_data = pd.DataFrame(columns=config.ALL_COLS)
    directional = omnidirectional_particle_flux / (4 * np.pi * ureg.steradian)
    directional = directional.to(
        ureg.particle
        / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)
    )
    synthetic_er_data[config.FLUX_COLS] = np.repeat(
        directional.magnitude[:, None], config.CHANNELS, axis=1
    )

    synthetic_er_data[config.PHI_COLS] = phis
    synthetic_er_data["UTC"] = "2025-07-25T12:30:00"
    synthetic_er_data[config.TIME_COLUMN] = (
        pd.to_datetime(synthetic_er_data["UTC"]).astype("int64") // 10**9
    )
    synthetic_er_data[config.ENERGY_COLUMN] = energy_centers.to(
        ureg.electron_volt
    ).magnitude
    synthetic_er_data[config.SPEC_NO_COLUMN] = 1

    np.random.seed(42)
    synthetic_er_data[config.MAG_COLS] = np.random.rand(3)

    return ERData.from_dataframe(synthetic_er_data, "synthetic")


def prepare_synthetic_er_poisson(
    density: float = 1e10,
    kappa: float = 5.0,
    theta: float = 1e6,
    seed: int | None = 42,
    background_count: float = 0.0,
) -> ERData:
    """
    Construct a synthetic ERData instance with Poisson noise in flux/counts.

    This simulates Poisson counting statistics by sampling counts per channel
    from the expected flux and then reconstructing the flux from those counts.
    """

    base = prepare_synthetic_er(density=density, kappa=kappa, theta=theta)
    df = base.data.copy()

    # Remove count columns so ERData can rebuild with updated flux.
    for col in config.COUNT_COLS:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Compute expected counts per channel from the current flux.
    flux_units = (
        ureg.particle
        / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)
    )
    flux = df[config.FLUX_COLS].to_numpy(dtype=np.float64) * flux_units
    energies = df[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[:, None]
    energies = energies * ureg.electron_volt

    thetas = np.loadtxt(config.DATA_DIR / config.THETA_FILE, dtype=np.float64)
    integration_time = (
        np.array([1 / config.BINS_BY_LATITUDE[x] for x in thetas])
        * config.ACCUMULATION_TIME
    )
    integration_time = integration_time[None, :]

    expected_counts = flux * config.GEOMETRIC_FACTOR * energies * integration_time
    expected_counts_mag = expected_counts.to(ureg.particle).magnitude
    expected_counts_mag = np.clip(expected_counts_mag, 0, None)
    if background_count > 0:
        expected_counts_mag = expected_counts_mag + background_count

    rng = np.random.default_rng(seed)
    noisy_counts = rng.poisson(expected_counts_mag).astype(np.float64)

    # Reconstruct flux from Poisson-sampled counts.
    denom = config.GEOMETRIC_FACTOR * energies * integration_time
    flux_noisy = (noisy_counts * ureg.particle) / denom
    df[config.FLUX_COLS] = flux_noisy.to(flux_units).magnitude

    er = ERData.from_dataframe(df, "synthetic-poisson")

    # Override count columns with Poisson totals to preserve noise statistics.
    count_sum = noisy_counts.sum(axis=1)
    er.data[config.COUNT_COLS[0]] = count_sum
    er.data[config.COUNT_COLS[1]] = np.sqrt(count_sum)

    return er
