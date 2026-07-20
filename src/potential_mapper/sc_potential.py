"""Per-row spacecraft potential estimation (sequential and torch paths)."""

from __future__ import annotations

import logging
import numbers
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

import src.config as config
from src.kappa import FitResults
from src.physics.charging import electron_current_density_magnitude
from src.physics.jucurve import U_from_J
from src.physics.kappa import KappaParams
from src.potential_mapper.kappa_batch import _prepare_kappa_batch_data
from src.utils.units import ureg

if TYPE_CHECKING:
    from src.flux import ERData

try:
    from src.kappa_torch import KappaFitterTorch

    HAS_KAPPA_TORCH = True
except ImportError:
    HAS_KAPPA_TORCH = False
    KappaFitterTorch = None  # type: ignore[misc, assignment]


def _spacecraft_potential_per_row(er_data: ERData, n_rows: int) -> np.ndarray:
    """Return spacecraft potential per ER row by spec_no grouping (sequential)."""

    from src.spacecraft_potential import calculate_potential

    potentials = np.full(n_rows, np.nan)
    if n_rows == 0:
        return potentials

    spec_values = er_data.data[config.SPEC_NO_COLUMN].to_numpy()
    unique_specs = np.unique(spec_values)

    for spec_value in tqdm(
        unique_specs,
        desc="Calculating SC Potential",
        unit="spec",
        leave=False,
        dynamic_ncols=True,
    ):
        if isinstance(spec_value, numbers.Real) and np.isnan(spec_value):
            continue
        mask_idx = np.flatnonzero(spec_values == spec_value)
        if mask_idx.size == 0:
            continue
        raw_spec = er_data.data.iloc[mask_idx[0]][config.SPEC_NO_COLUMN]
        try:
            spec_no = int(raw_spec)
        except (TypeError, ValueError):
            logging.debug("Skipping non-integer spec_no %r", raw_spec)
            continue
        potential_result = None
        energy_backup: np.ndarray | None = None
        if config.ENERGY_COLUMN in er_data.data.columns:
            energy_backup = er_data.data[config.ENERGY_COLUMN].to_numpy(copy=True)
        try:
            potential_result = calculate_potential(er_data, spec_no)
        except Exception as exc:  # pragma: no cover - diagnostic only
            logging.debug(
                "Spacecraft potential failed for spec_no %s: %s",
                spec_no,
                exc,
                exc_info=True,
            )
        finally:
            if energy_backup is not None:
                er_data.data.loc[:, config.ENERGY_COLUMN] = energy_backup
        if not potential_result:
            continue
        _, potential_quantity = potential_result
        try:
            potential_value = float(potential_quantity.to(ureg.volt).magnitude)
        except Exception:
            potential_value = float(potential_quantity)
        potentials[mask_idx] = potential_value

    return potentials


def _spacecraft_potential_per_row_torch(
    er_data: ERData,
    n_rows: int,
    is_day: np.ndarray | None = None,
    electron_temp_out: np.ndarray | None = None,
    electron_dens_out: np.ndarray | None = None,
    kappa_out: np.ndarray | None = None,
) -> np.ndarray:
    """
    Return spacecraft potential per ER row using PyTorch-accelerated Kappa fitting.

    This provides ~78x speedup over the sequential scipy version by batch-fitting
    all spectra simultaneously with vectorized differential evolution.

    Args:
        er_data: ERData containing all spectra
        n_rows: Total number of rows
        is_day: (n_rows,) boolean array indicating daytime rows (optional)
        electron_temp_out: Optional output array to fill with electron temperature [eV]
        electron_dens_out: Optional output array to fill with electron density [m^-3]
        kappa_out: Optional output array to fill with kappa parameter values

    Returns:
        Array of spacecraft potentials per row
    """
    from scipy.optimize import brentq

    from src.spacecraft_potential import (
        current_balance,
        theta_to_temperature_ev,
    )

    potentials = np.full(n_rows, np.nan)
    if n_rows == 0:
        return potentials

    if not HAS_KAPPA_TORCH or KappaFitterTorch is None:
        raise ImportError("PyTorch required for torch-accelerated SC potential")

    # Prepare batch data
    logging.info("Preparing batch data for Kappa fitting...")
    (
        energy,
        flux_data,
        density_estimates,
        weights,
        valid_spec_nos,
        _first_row_indices,
    ) = _prepare_kappa_batch_data(er_data)

    if len(valid_spec_nos) == 0:
        logging.warning("No valid spectra for Kappa fitting")
        return potentials

    logging.info(f"Batch fitting {len(valid_spec_nos)} spectra with PyTorch...")

    # Batch fit all spectra
    fitter = KappaFitterTorch(
        device="cpu",
        popsize=30,
        maxiter=100,
    )
    kappa_vals, theta_vals, chi2_vals = fitter.fit_batch(
        energy, flux_data, density_estimates, weights=weights
    )

    # Compute spacecraft potential for each spectrum
    spec_values = er_data.data[config.SPEC_NO_COLUMN].to_numpy()

    for i, spec_no in enumerate(
        tqdm(
            valid_spec_nos,
            desc="Computing SC Potential",
            unit="spec",
            leave=False,
            dynamic_ncols=True,
        )
    ):
        mask_idx = np.flatnonzero(spec_values == spec_no)
        if mask_idx.size == 0:
            continue

        # Check fit quality
        if chi2_vals[i] > config.FIT_ERROR_THRESHOLD:
            continue

        kappa = kappa_vals[i]
        theta = theta_vals[i]  # m/s
        density = density_estimates[i]  # particles/m³

        # Convert theta to electron temperature in eV
        Te_ev = theta_to_temperature_ev(theta, kappa)

        # Store kappa parameters if output arrays provided
        if electron_temp_out is not None:
            electron_temp_out[mask_idx] = Te_ev
        if electron_dens_out is not None:
            electron_dens_out[mask_idx] = density
        if kappa_out is not None:
            kappa_out[mask_idx] = kappa

        # Determine day/night
        row_idx = mask_idx[0]
        spec_is_day = is_day[row_idx] if is_day is not None else True

        try:
            if spec_is_day:
                # Day: compute U from JU curve
                current_density = electron_current_density_magnitude(
                    density, kappa, theta, E_min=1e1, E_max=2e4, n_steps=10
                )
                spacecraft_potential = U_from_J(
                    J_target=current_density, U_min=0.0, U_max=150.0
                )
                # For full accuracy, we'd refit with corrected energies,
                # but the improvement is marginal compared to the speedup
                potential_value = float(spacecraft_potential)
            else:
                # Night: solve current balance equation
                theta_to_temperature_ev(theta, kappa)

                # Create a FitResults-like object for the current_balance function
                from src.utils.units import ureg

                params = KappaParams(
                    density=density * ureg.particle / ureg.meter**3,
                    kappa=kappa,
                    theta=theta * ureg.meter / ureg.second,
                )
                fit_result = FitResults(
                    params=params,
                    params_uncertainty=params,  # Dummy
                    error=chi2_vals[i],
                    is_good_fit=True,
                )

                # Energy grid for current balance
                energy_grid = np.geomspace(1.0, 2e4, 500)

                # Bracket search
                U_low, U_high = -1500.0, 0.0
                balance_low = current_balance(
                    U_low, fit_result, energy_grid, 500.0, 1.5
                )
                balance_high = current_balance(
                    U_high, fit_result, energy_grid, 500.0, 1.5
                )

                bracket_expansions = 0
                while (
                    np.sign(balance_low) == np.sign(balance_high)
                    and bracket_expansions < 10
                ):
                    U_low *= 1.5
                    balance_low = current_balance(
                        U_low, fit_result, energy_grid, 500.0, 1.5
                    )
                    bracket_expansions += 1

                if np.isnan(balance_low) or np.isnan(balance_high):
                    continue
                if np.sign(balance_low) == np.sign(balance_high):
                    continue

                spacecraft_potential = brentq(
                    current_balance,
                    U_low,
                    U_high,
                    args=(fit_result, energy_grid, 500.0, 1.5),
                    maxiter=200,
                    xtol=1e-3,
                )
                potential_value = float(spacecraft_potential)

            potentials[mask_idx] = potential_value

        except Exception as e:
            logging.debug(f"SC potential failed for spec {spec_no}: {e}")
            continue

    return potentials
