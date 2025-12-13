"""Synthetic data helpers shared between tests and scripts."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.flux import ERData
from src.physics.kappa import KappaParams, omnidirectional_flux
from src.utils.units import ureg

__all__ = [
    "prepare_phis",
    "prepare_flux",
    "prepare_synthetic_er",
]


def prepare_phis() -> tuple[list[float], np.ndarray]:
    """Prepare mock instrument viewing angles and solid angles.

    Returns:
        tuple[list[float], np.ndarray]:
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
    """
    Prepare a theoretical omnidirectional particle flux.

    Args:
        density: Density in particles/m^3.
        kappa: Kappa parameter.
        theta: Thermal speed in m/s.

    Returns:
        tuple: (omnidirectional_particle_flux, energy_centers)
    """

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
    """
    Construct a synthetic `ERData` instance for deterministic tests.

    Args:
        density: Density in particles/m^3.
        kappa: Kappa parameter.
        theta: Thermal speed in m/s.

    Returns:
        ERData: A synthetic ERData object.
    """

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
