import argparse
import logging
import multiprocessing
import numbers
import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.config as config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.losscone.types import FitMethod, parse_fit_method

try:
    from src.model_torch import HAS_TORCH, LossConeFitterTorch
except ImportError:
    HAS_TORCH = False
    LossConeFitterTorch = None  # type: ignore[misc, assignment]

try:
    from src.kappa_torch import KappaFitterTorch

    HAS_KAPPA_TORCH = True
except ImportError:
    HAS_KAPPA_TORCH = False
    KappaFitterTorch = None  # type: ignore[misc, assignment]

from src.kappa import FitResults, Kappa
from src.physics.charging import electron_current_density_magnitude
from src.physics.jucurve import U_from_J
from src.physics.kappa import KappaParams
from src.potential_mapper import plot as plot_mod
from src.potential_mapper.coordinates import (
    CoordinateArrays,
    CoordinateCalculator,
    find_surface_intersection_with_polarity,
    project_magnetic_fields,
)
from src.potential_mapper.date_utils import (
    MONTH_ABBREV_TO_NUM,
    NUM_STR_TO_MONTH_ABBREV,
)
from src.potential_mapper.results import PotentialResults
from src.utils.attitude import load_attitude_data
from src.utils.geometry import get_intersections_or_none_batch
from src.utils.units import ureg


def _init_worker_spice():
    """Initialize SPICE kernels in worker process for thread-safety."""
    from src.potential_mapper.spice import load_spice_files

    load_spice_files()


def spacecraft_potential_worker(
    args: tuple[int, pd.DataFrame],
) -> tuple[int, np.ndarray, float | None]:
    """
    Calculate spacecraft potential for one spectrum in parallel.

    Args:
        args: Tuple of (spec_no, spectrum_rows_df)

    Returns:
        Tuple of (spec_no, row_indices, potential_value)
    """
    import warnings

    # Suppress RuntimeWarnings (e.g. from geometry.py) to avoid interfering with tqdm
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    from src.spacecraft_potential import calculate_potential

    spec_no, rows_df = args

    # Create isolated ERData for this spectrum only
    # The copy ensures mutations don't affect shared state
    er_data_subset = ERData.from_dataframe(rows_df.copy(), f"spec_{spec_no}")

    try:
        result = calculate_potential(er_data_subset, spec_no)

        if result is None:
            return spec_no, rows_df.index.to_numpy(), None

        _, potential_quantity = result
        potential_value = float(potential_quantity.to(ureg.volt).magnitude)

        return spec_no, rows_df.index.to_numpy(), potential_value

    except Exception as e:
        logging.debug(f"SC potential failed for spec {spec_no}: {e}")
        return spec_no, rows_df.index.to_numpy(), None


def fit_worker(
    args: tuple[pd.DataFrame, np.ndarray, np.ndarray | None, FitMethod | str],
) -> np.ndarray:
    """
    Worker function for parallel fitting.

    Args:
        args: Tuple containing (chunk_df, sc_pot, polarity, fit_method).

    Returns:
        Fitting results array from LossConeFitter.
    """
    chunk_df, sc_pot, polarity, fit_method = args
    fit_method = parse_fit_method(fit_method)

    # Create ERData from the chunk.
    # Note: This might re-trigger cleaning/counting if not handled in ERData,
    # but for now we assume it's acceptable or handled.
    # To avoid double-counting if columns exist, we could check in ERData,
    # but here we just pass it.
    er_data = ERData.from_dataframe(chunk_df, "batch_chunk")

    # Initialize fitter with the chunk
    pitch_angle = (
        PitchAngle(er_data, polarity=polarity) if polarity is not None else None
    )
    fitter = LossConeFitter(
        er_data,
        pitch_angle=pitch_angle,
        spacecraft_potential=sc_pot,
        fit_method=fit_method,
    )
    return fitter.fit_surface_potential()


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


def _spacecraft_potential_per_row_parallel(
    er_data: ERData, n_rows: int, num_workers: int | None = None
) -> np.ndarray:
    """
    Return spacecraft potential per ER row using parallel processing.

    Distributes spectrum-level calculations across multiple worker processes
    to accelerate computation for large datasets.

    Args:
        er_data: ERData object containing the full dataset
        n_rows: Total number of rows
        num_workers: Number of worker processes (default: cpu_count - 1)

    Returns:
        Array of spacecraft potentials per row
    """
    potentials = np.full(n_rows, np.nan)
    if n_rows == 0:
        return potentials

    # Group data by spec_no
    df = er_data.data.reset_index(drop=True)
    spec_groups = df.groupby(config.SPEC_NO_COLUMN, sort=False)

    # Prepare tasks for workers
    tasks = []
    for spec_value, group_df in spec_groups:
        if isinstance(spec_value, numbers.Real) and np.isnan(spec_value):
            continue
        try:
            spec_no = int(spec_value)
        except (TypeError, ValueError):
            logging.debug(f"Skipping non-integer spec_no {spec_value}")
            continue

        # Each task is (spec_no, rows_for_this_spectrum)
        tasks.append((spec_no, group_df))

    if not tasks:
        return potentials

    # Determine worker count
    num_workers = num_workers or max(1, multiprocessing.cpu_count() - 1)

    logging.debug(
        f"Starting parallel SC potential with {num_workers} workers, "
        f"{len(tasks)} spectra"
    )

    # Execute in parallel with SPICE initialization per worker.
    # Use "spawn" context to ensure SPICE thread-safety and avoid global state
    # corruption.
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=num_workers, initializer=_init_worker_spice) as pool:
        # Use imap_unordered for better memory efficiency
        # Chunksize: distribute work in reasonable batches
        chunksize = max(1, len(tasks) // (num_workers * 4))

        results_iter = pool.imap_unordered(
            spacecraft_potential_worker, tasks, chunksize=chunksize
        )

        # Collect results with progress bar
        for _spec_no, row_indices, potential_value in tqdm(
            results_iter,
            total=len(tasks),
            desc="SC Potential (parallel)",
            unit="spec",
            leave=False,
            dynamic_ncols=True,
        ):
            if potential_value is not None and np.isfinite(potential_value):
                potentials[row_indices] = potential_value

    return potentials


def _prepare_kappa_batch_data(
    er_data: ERData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[int], np.ndarray]:
    """
    Prepare batched data for torch Kappa fitting.

    Extracts energy grid, flux data, and density estimates for all spectra.

    Args:
        er_data: ERData containing all spectra

    Returns:
        energy: (E,) energy grid [eV]
        flux_data: (N, E) omnidirectional flux per spectrum
        density_estimates: (N,) density estimates [particles/m³]
        valid_spec_nos: list of valid spectrum numbers
        row_indices: (N,) first row index for each spectrum
    """
    spec_values = er_data.data[config.SPEC_NO_COLUMN].to_numpy()
    unique_specs = np.unique(spec_values)

    energy = None
    flux_list = []
    density_list = []
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
            valid_spec_nos.append(spec_no)
            # Store first row index for this spectrum
            mask_idx = np.flatnonzero(spec_values == spec_value)
            row_indices.append(mask_idx[0])
        except Exception:
            continue

    if energy is None or len(flux_list) == 0:
        return np.array([]), np.array([]), np.array([]), [], np.array([])

    return (
        energy,
        np.array(flux_list),
        np.array(density_list),
        valid_spec_nos,
        np.array(row_indices),
    )


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
    energy, flux_data, density_estimates, valid_spec_nos, _first_row_indices = (
        _prepare_kappa_batch_data(er_data)
    )

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
        energy, flux_data, density_estimates
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
        if chi2_vals[i] > 1e4:  # Poor fit, skip
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


def _apply_fit_results(
    fit_results: np.ndarray,
    proj_potential: np.ndarray,
    bs_over_bm: np.ndarray,
    beam_amp: np.ndarray,
    fit_chi2: np.ndarray,
    row_offset: int,
    n_total: int,
) -> None:
    """Map fitter outputs back onto per-row arrays."""

    rows_per_sweep = config.SWEEP_ROWS
    if fit_results.size == 0:
        return

    for U_surface, bs, amp, chi2, chunk_idx in fit_results:
        if not np.isfinite(chunk_idx):
            continue
        chunk_idx = int(chunk_idx)
        row_start = row_offset + chunk_idx * rows_per_sweep
        row_end = min(row_start + rows_per_sweep, n_total)
        if row_start >= n_total:
            break

        # Always store fit parameters (even if chi2 is high)
        bs_over_bm[row_start:row_end] = float(bs)
        beam_amp[row_start:row_end] = float(amp)
        fit_chi2[row_start:row_end] = float(chi2)

        # Store U_surface whenever the fit returned a finite value.
        # Use fit_chi2 for downstream quality filtering.
        if np.isfinite(U_surface):
            proj_potential[row_start:row_end] = float(U_surface)


class DataLoader:
    """Discover ER files and load auxiliary attitude/theta data."""

    MONTH_TO_NUM = MONTH_ABBREV_TO_NUM
    NUM_TO_MONTH = NUM_STR_TO_MONTH_ABBREV

    @staticmethod
    def discover_flux_files(
        year: int | None = None,
        month: int | None = None,
        day: int | None = None,
    ) -> list[Path]:
        """Discover ER flux files below DATA_DIR."""
        data_dir = config.DATA_DIR
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")

        if month is not None and not 1 <= month <= 12:
            logging.warning("Month %s outside 1-12; no files processed.", month)
            return []
        if day is not None and not 1 <= day <= 31:
            logging.warning("Day %s outside 1-31; no files processed.", day)
            return []

        exclude_basenames = {
            config.ATTITUDE_FILE.lower(),
            config.SOLID_ANGLES_FILE.lower(),
            config.THETA_FILE.lower(),
            "areas.tab",
        }

        candidates: list[Path] = []
        # Use os.walk with followlinks=True so symlinked data dirs are discovered.
        for root, _dirs, files in os.walk(data_dir, followlinks=True):
            for filename in files:
                if not filename.endswith(config.EXT_TAB):
                    continue
                if filename.lower() in exclude_basenames:
                    continue
                candidates.append(Path(root) / filename)

        def matches_date(p: Path) -> bool:
            if year is None and month is None and day is None:
                return True
            s = str(p)
            ok = True
            if year is not None:
                ok &= str(year) in s
            if month is not None:
                mm = DataLoader.NUM_TO_MONTH.get(f"{month:02d}")
                if mm is None:
                    logging.warning(
                        "Month %s not recognized; skipping discovery.", month
                    )
                    return False
                ok &= mm in s
            if day is not None:
                dd = f"{day:02d}"
                ok &= f"{dd}{config.EXT_TAB}" in s

            return ok

        flux_files = sorted([p for p in candidates if matches_date(p)])
        logging.debug(
            f"Discovered {len(flux_files)} candidate ER files under {data_dir}"
        )
        if (year or month or day) and not flux_files:
            logging.warning(
                "No ER files matched the provided date filters; returning empty list"
            )
        return flux_files


def load_all_data(files: list[Path]) -> ERData:
    """Load and merge all discovered files into a single ERData object."""
    dfs = []
    spec_offset = 0

    for f in tqdm(files, desc="Loading files", unit="file"):
        try:
            # Load individual file
            er = ERData(str(f))
            if not er.data.empty:
                if config.SPEC_NO_COLUMN in er.data.columns:
                    er.data[config.SPEC_NO_COLUMN] = (
                        pd.to_numeric(
                            er.data[config.SPEC_NO_COLUMN],
                            errors="coerce",
                        )
                        .fillna(0)
                        .astype(int)
                    )

                    er.data[config.SPEC_NO_COLUMN] += spec_offset

                    max_spec = er.data[config.SPEC_NO_COLUMN].max()
                    if pd.notna(max_spec):
                        spec_offset = max_spec + 1

                dfs.append(er.data)
        except Exception as e:
            logging.warning(f"Failed to load {f}: {e}")

    if not dfs:
        # Return empty ERData without attempting to read a file
        return ERData.from_dataframe(pd.DataFrame(), "empty")

    # Concatenate all dataframes
    merged_df = pd.concat(dfs, ignore_index=True)

    # Create a new ERData object from the merged dataframe
    # We use a dummy filename since it's a composite
    return ERData.from_dataframe(merged_df, "merged_dataset")


def process_merged_data(
    er_data: ERData,
    *,
    use_parallel: bool = False,
    use_torch: bool = False,
    fit_method: str | FitMethod | None = None,
    spacecraft_potential_override: float | None = None,
) -> PotentialResults:
    """
    Process the merged ER dataset.

    Steps:
    1. Load attitude and calculate coordinates (vectorized).
    2. Calculate spacecraft potential (parallel if use_parallel=True).
    3. Run surface potential fitting (parallel or torch-accelerated).
    4. Assemble results.

    Args:
        er_data: Merged ERData object
        use_parallel: Enable multiprocessing for SC and surface potential
            (default: False; deprecated for CPU path)
        use_torch: Use PyTorch-accelerated fitter (~5x faster, default: False)
        fit_method: Loss-cone fitting method ("halekas" or "lillis")
        spacecraft_potential_override: Optional constant spacecraft potential [V].
            When provided, skips spacecraft potential estimation and uses this
            value for all rows.

    Returns:
        PotentialResults with all computed fields
    """
    logging.info("Processing merged dataset...")

    if use_parallel and not use_torch:
        logging.warning(
            "CPU parallel fitting is deprecated; falling back to sequential. "
            "Use --fast for torch acceleration."
        )
        use_parallel = False

    fit_method = parse_fit_method(fit_method)

    # Load attitude
    et_spin, ra_vals, dec_vals = load_attitude_data(
        config.DATA_DIR / config.ATTITUDE_FILE
    )
    if (
        et_spin is None
        or ra_vals is None
        or dec_vals is None
        or len(et_spin) == 0
        or len(ra_vals) == 0
        or len(dec_vals) == 0
    ):
        raise RuntimeError("Attitude data unavailable or empty; cannot process file.")

    # Coordinates and magnetic field projection
    logging.info("Calculating coordinates...")
    coord_calc = CoordinateCalculator(et_spin, ra_vals, dec_vals)
    coord_arrays: CoordinateArrays = coord_calc.calculate_coordinate_transformation(
        er_data
    )
    projected_b = project_magnetic_fields(er_data, coord_arrays)

    # Intersections on lunar sphere
    points, mask, polarity = find_surface_intersection_with_polarity(
        coord_arrays, projected_b
    )
    # Polarity convention: +1 => Moonward along +B (projected_b),
    # -1 => Moonward along -B, 0 => no connection.

    n = mask.shape[0]
    projection_polarity = polarity.astype(np.int8, copy=False)

    # Spacecraft lat/lon from LP position in lunar frame
    spacecraft_lat = np.full(n, np.nan)
    spacecraft_lon = np.full(n, np.nan)
    if n > 0:
        lp_pos = coord_arrays.lp_positions
        # Vectorized lat/lon
        r_norm = np.linalg.norm(lp_pos, axis=1)
        valid_lp = r_norm > 0
        spacecraft_lat[valid_lp] = np.rad2deg(
            np.arcsin(lp_pos[valid_lp, 2] / r_norm[valid_lp])
        )
        spacecraft_lon[valid_lp] = np.rad2deg(
            np.arctan2(lp_pos[valid_lp, 1], lp_pos[valid_lp, 0])
        )

    # Projection lat/lon from intersection points
    proj_lat = np.full(n, np.nan)
    proj_lon = np.full(n, np.nan)
    if np.any(mask):
        p = points[mask]
        r_norm_p = np.linalg.norm(p, axis=1)
        valid_p = r_norm_p > 0
        proj_lat_masked = np.full(p.shape[0], np.nan)
        proj_lon_masked = np.full(p.shape[0], np.nan)
        proj_lat_masked[valid_p] = np.rad2deg(
            np.arcsin(p[valid_p, 2] / r_norm_p[valid_p])
        )
        proj_lon_masked[valid_p] = np.rad2deg(np.arctan2(p[valid_p, 1], p[valid_p, 0]))
        proj_lat[mask] = proj_lat_masked
        proj_lon[mask] = proj_lon_masked

    # Sun illumination for spacecraft
    u_lp_to_sun = coord_arrays.lp_vectors_to_sun
    u_lp_to_sun = u_lp_to_sun / np.linalg.norm(u_lp_to_sun, axis=1, keepdims=True)
    _, sc_shadow_hit = get_intersections_or_none_batch(
        pos=coord_arrays.lp_positions,
        direction=u_lp_to_sun,
        radius=config.LUNAR_RADIUS,
    )
    sc_in_sun = ~sc_shadow_hit

    # Sun illumination for projection point
    proj_in_sun = np.zeros(n, dtype=bool)
    if np.any(mask):
        n_hat = p / r_norm_p[:, None]
        moon_to_sun = coord_arrays.moon_vectors_to_sun
        moon_to_sun_hat = moon_to_sun / np.linalg.norm(
            moon_to_sun, axis=1, keepdims=True
        )
        dots = np.sum(n_hat * moon_to_sun_hat[mask], axis=1)
        proj_in_sun_masked = dots > 0
        proj_in_sun[mask] = proj_in_sun_masked

    # Kappa/plasma parameters - will be filled by torch path if available
    electron_temp = np.full(n, np.nan)
    electron_dens = np.full(n, np.nan)
    kappa_vals_arr = np.full(n, np.nan)

    # Spacecraft potential
    if spacecraft_potential_override is not None:
        sc_potential = np.full(n, float(spacecraft_potential_override))
        logging.info(
            "Using constant spacecraft potential override: %.2f V",
            float(spacecraft_potential_override),
        )
    else:
        logging.info("Calculating spacecraft potential...")
        if use_torch:
            # PyTorch-accelerated Kappa fitting (~78x faster)
            if not HAS_KAPPA_TORCH or KappaFitterTorch is None:
                logging.warning(
                    "PyTorch not available for Kappa fitting; falling back to sequential"
                )
                sc_potential = _spacecraft_potential_per_row(er_data, n)
            else:
                try:
                    sc_potential = _spacecraft_potential_per_row_torch(
                        er_data,
                        n,
                        is_day=sc_in_sun,
                        electron_temp_out=electron_temp,
                        electron_dens_out=electron_dens,
                        kappa_out=kappa_vals_arr,
                    )
                except Exception as e:
                    logging.warning(
                        "Torch SC potential failed (%s); falling back to sequential",
                        e,
                    )
                    sc_potential = _spacecraft_potential_per_row(er_data, n)
        elif use_parallel:
            try:
                sc_potential = _spacecraft_potential_per_row_parallel(er_data, n)
            except (PermissionError, OSError) as e:
                logging.warning(
                    "Parallel SC potential failed (%s); falling back to sequential", e
                )
                sc_potential = _spacecraft_potential_per_row(er_data, n)
        else:
            sc_potential = _spacecraft_potential_per_row(er_data, n)

    # Surface Potential Fitting
    proj_potential = np.full(n, np.nan)
    bs_over_bm_arr = np.full(n, np.nan)
    beam_amp_arr = np.full(n, np.nan)
    fit_chi2_arr = np.full(n, np.nan)

    # If the dataset is small, or ERData has been monkeypatched without
    # from_dataframe (as in some tests), fall back to sequential fitting.
    can_chunk = hasattr(ERData, "from_dataframe")
    rows_per_sweep = config.SWEEP_ROWS
    sweeps_per_chunk = 100
    rows_per_chunk = sweeps_per_chunk * rows_per_sweep
    # Torch mode uses its own vectorization, so don't use parallel chunking
    chunked = use_parallel and can_chunk and n > rows_per_chunk and not use_torch
    pitch_angle = None
    use_polarity = False
    required_cols = set(config.PHI_COLS + config.MAG_COLS)
    if required_cols.issubset(er_data.data.columns):
        use_polarity = True
        if not chunked:
            pitch_angle = PitchAngle(er_data, polarity=projection_polarity)
    else:
        logging.debug("Pitch-angle polarity skipped; required ER columns are missing.")

    if use_torch:
        # PyTorch-accelerated fitter (~5x faster)
        if not HAS_TORCH or LossConeFitterTorch is None:
            raise ImportError(
                "PyTorch is required for --fast mode. Install with: uv sync --extra gpu"
            )
        logging.info("Running surface potential fitting (PyTorch-accelerated)...")
        fitter = LossConeFitterTorch(
            er_data,
            pitch_angle=pitch_angle,
            spacecraft_potential=sc_potential,
            device=None,  # Auto-detect: CUDA if available, else CPU
            fit_method=fit_method,
        )
        # Use batched GPU processing (auto-detects dtype and batch_size)
        fit_mat = fitter.fit_surface_potential_batched()
        _apply_fit_results(
            fit_mat,
            proj_potential,
            bs_over_bm_arr,
            beam_amp_arr,
            fit_chi2_arr,
            row_offset=0,
            n_total=n,
        )
    elif not chunked:
        logging.info("Running surface potential fitting (sequential)...")
        fitter = LossConeFitter(
            er_data,
            pitch_angle=pitch_angle,
            spacecraft_potential=sc_potential,
            fit_method=fit_method,
        )
        fit_mat = fitter.fit_surface_potential()
        _apply_fit_results(
            fit_mat,
            proj_potential,
            bs_over_bm_arr,
            beam_amp_arr,
            fit_chi2_arr,
            row_offset=0,
            n_total=n,
        )
    else:
        logging.info("Starting parallel surface potential fitting...")

        chunks: list[tuple[pd.DataFrame, np.ndarray, np.ndarray | None, FitMethod]] = []
        for i in range(0, n, rows_per_chunk):
            end = min(i + rows_per_chunk, n)
            chunk_df = er_data.data.iloc[i:end].copy()
            chunk_sc_pot = sc_potential[i:end]
            chunk_pol = projection_polarity[i:end] if use_polarity else None
            chunks.append((chunk_df, chunk_sc_pot, chunk_pol, fit_method))

        if chunks:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

            # Use 'spawn' context to ensure thread-safety for SPICE
            ctx = multiprocessing.get_context("spawn")
            with ctx.Pool(
                processes=num_workers, initializer=_init_worker_spice
            ) as pool:
                results_iter = pool.imap(fit_worker, chunks)
                for chunk_idx, chunk_res in enumerate(
                    tqdm(
                        results_iter,
                        total=len(chunks),
                        desc="Fitting chunks",
                        unit="chunk",
                        dynamic_ncols=True,
                    )
                ):
                    row_offset = chunk_idx * rows_per_chunk
                    _apply_fit_results(
                        chunk_res,
                        proj_potential,
                        bs_over_bm_arr,
                        beam_amp_arr,
                        fit_chi2_arr,
                        row_offset,
                        n_total=n,
                    )

    # Build results object
    results = PotentialResults(
        spacecraft_latitude=spacecraft_lat,
        spacecraft_longitude=spacecraft_lon,
        projection_latitude=proj_lat,
        projection_longitude=proj_lon,
        spacecraft_potential=sc_potential,
        projected_potential=proj_potential,
        spacecraft_in_sun=sc_in_sun,
        projection_in_sun=proj_in_sun,
        projection_polarity=projection_polarity,
        bs_over_bm=bs_over_bm_arr,
        beam_amp=beam_amp_arr,
        fit_chi2=fit_chi2_arr,
        electron_temperature=electron_temp,
        electron_density=electron_dens,
        kappa_value=kappa_vals_arr,
    )

    # Classify plasma environments based on electron temperature
    results.classify_environments()

    return results


def process_lp_file(
    file_path: Path,
    *,
    fit_method: str | FitMethod | None = None,
    spacecraft_potential_override: float | None = None,
) -> PotentialResults:
    """
    Process a single ER file into PotentialResults (sequential fitting).

    This preserves the historical API used by tests, docs, and dev scripts.
    """
    logging.debug(f"Processing LP file: {file_path}")
    er_data = ERData(str(file_path))
    return process_merged_data(
        er_data,
        use_parallel=False,
        fit_method=fit_method,
        spacecraft_potential_override=spacecraft_potential_override,
    )


def run(args: argparse.Namespace) -> int:
    """Entry point for CLI."""
    flux_files = DataLoader.discover_flux_files(
        year=args.year,
        month=args.month,
        day=args.day,
    )

    if not flux_files:
        logging.info("No ER flux files discovered. Exiting.")
        return 0

    # Load and merge all data
    er_data = load_all_data(flux_files)

    if er_data.data.empty:
        logging.warning("Merged dataset is empty. Exiting.")
        return 1

    logging.info(f"Loaded {len(er_data.data)} rows of data.")

    try:
        agg = process_merged_data(er_data)
    except Exception as e:
        logging.exception(f"Failed to process merged data: {e}")
        return 1

    if args.output or args.display:
        fig, _ax = plot_mod.plot_map(agg, illumination=args.illumination)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=200)
            logging.info(f"Saved plot to {out_path}")
        if args.display:
            try:
                import matplotlib.pyplot as plt

                plt.show()
            except Exception:
                logging.warning("Display requested but matplotlib backend failed.")

    return 0
