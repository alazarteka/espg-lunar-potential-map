"""Batch preparation for torch Kappa spacecraft-potential fits."""

from __future__ import annotations

import numbers
from typing import TYPE_CHECKING

import numpy as np

import src.config as config
from src.kappa import Kappa

if TYPE_CHECKING:
    from src.flux import ERData


def _prepare_kappa_batch_data(
    er_data: ERData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    """
    Prepare batched data for torch Kappa fitting.

    Extracts energy grid, flux data, and density estimates for all spectra.

    Args:
        er_data: ERData containing all spectra

    Returns:
        energy: (E,) energy grid [eV]
        flux_data: (N, E) omnidirectional flux per spectrum
        density_estimates: (N,) density estimates [particles/m³]
        weights: (N, E) log-space fit weights (1/σ_log_flux)
        valid_spec_nos: list of valid spectrum numbers
        row_indices: (N,) first row index for each spectrum
    """
    spec_values = er_data.data[config.SPEC_NO_COLUMN].to_numpy()
    unique_specs = np.unique(spec_values)

    energy = None
    flux_list = []
    density_list = []
    weight_list = []
    valid_spec_nos = []
    row_indices = []

    for spec_value in unique_specs:
        if isinstance(spec_value, numbers.Real) and np.isnan(spec_value):
            continue
        try:
            spec_no = int(spec_value)
        except (TypeError, ValueError):
            continue

        try:
            kappa_obj = Kappa(er_data, spec_no)
            if not kappa_obj.is_data_valid:
                continue

            if energy is None:
                energy = kappa_obj.energy_centers_mag

            flux_list.append(kappa_obj.omnidirectional_differential_particle_flux_mag)
            density_list.append(kappa_obj.density_estimate_mag)
            try:
                weight_list.append(kappa_obj.log_flux_weights())
            except Exception:
                weight_list.append(
                    np.ones_like(
                        kappa_obj.omnidirectional_differential_particle_flux_mag
                    )
                )
            valid_spec_nos.append(spec_no)
            # Store first row index for this spectrum
            mask_idx = np.flatnonzero(spec_values == spec_value)
            row_indices.append(mask_idx[0])
        except Exception:
            continue

    if energy is None or len(flux_list) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), [], np.array([])

    return (
        energy,
        np.array(flux_list),
        np.array(density_list),
        np.array(weight_list),
        valid_spec_nos,
        np.array(row_indices),
    )
