import argparse
import logging
import multiprocessing
import numbers
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.config as config
from src.flux import ERData, LossConeFitter
from src.potential_mapper import plot as plot_mod
from src.potential_mapper.coordinates import (
    CoordinateArrays,
    CoordinateCalculator,
    find_surface_intersection,
    project_magnetic_fields,
)
from src.potential_mapper.results import PotentialResults
from src.utils.attitude import load_attitude_data
from src.utils.geometry import get_intersections_or_none_batch
from src.utils.units import ureg, VoltageType


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


def fit_worker(args: tuple[pd.DataFrame, str, np.ndarray]) -> np.ndarray:
    """
    Worker function for parallel fitting.

    Args:
        args: Tuple containing (chunk_df, theta_path, sc_pot).

    Returns:
        Fitting results array from LossConeFitter.
    """
    chunk_df, theta_path, sc_pot = args

    # Create ERData from the chunk.
    # Note: This might re-trigger cleaning/counting if not handled in ERData,
    # but for now we assume it's acceptable or handled.
    # To avoid double-counting if columns exist, we could check in ERData,
    # but here we just pass it.
    er_data = ERData.from_dataframe(chunk_df, "batch_chunk")

    # Initialize fitter with the chunk
    fitter = LossConeFitter(
        er_data,
        theta_path,
        spacecraft_potential=sc_pot,
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
        unique_specs, desc="Calculating SC Potential", unit="spec", leave=False
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

    # Execute in parallel with SPICE initialization per worker
    with multiprocessing.Pool(
        processes=num_workers, initializer=_init_worker_spice
    ) as pool:
        # Use imap_unordered for better memory efficiency
        # Chunksize: distribute work in reasonable batches
        chunksize = max(1, len(tasks) // (num_workers * 4))

        results_iter = pool.imap_unordered(
            spacecraft_potential_worker, tasks, chunksize=chunksize
        )

        # Collect results with progress bar
        for spec_no, row_indices, potential_value in tqdm(
            results_iter,
            total=len(tasks),
            desc="SC Potential (parallel)",
            unit="spec",
            leave=False,
        ):
            if potential_value is not None and np.isfinite(potential_value):
                potentials[row_indices] = potential_value

    return potentials


def _apply_fit_results(
    fit_results: np.ndarray, proj_potential: np.ndarray, row_offset: int, n_total: int
) -> None:
    """Map fitter outputs back onto per-row projected potential array."""

    rows_per_sweep = config.SWEEP_ROWS
    if fit_results.size == 0:
        return

    for U_surface, _bs, _beam_amp, chi2, chunk_idx in fit_results:
        chunk_idx = int(chunk_idx)
        if not np.isfinite(U_surface) or not np.isfinite(chi2):
            continue
        if chi2 > config.FIT_ERROR_THRESHOLD:
            continue

        row_start = row_offset + chunk_idx * rows_per_sweep
        row_end = min(row_start + rows_per_sweep, n_total)
        if row_start >= n_total:
            break
        proj_potential[row_start:row_end] = float(U_surface)


class DataLoader:
    """Discover ER files and load auxiliary attitude/theta data."""

    MONTH_TO_NUM = {
        "JAN": "01",
        "FEB": "02",
        "MAR": "03",
        "APR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AUG": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12",
    }

    NUM_TO_MONTH = {v: k for k, v in MONTH_TO_NUM.items()}

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

        candidates = [
            p
            for p in data_dir.rglob(f"*{config.EXT_TAB}")
            if p.is_file() and p.name.lower() not in exclude_basenames
        ]

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
    er_data: ERData, *, use_parallel: bool = False
) -> PotentialResults:
    """
    Process the merged ER dataset.

    Steps:
    1. Load attitude and calculate coordinates (vectorized).
    2. Calculate spacecraft potential (parallel if use_parallel=True).
    3. Run surface potential fitting (parallel if use_parallel=True).
    4. Assemble results.

    Args:
        er_data: Merged ERData object
        use_parallel: Enable multiprocessing for SC and surface potential (default: False)

    Returns:
        PotentialResults with all computed fields
    """
    logging.info("Processing merged dataset...")

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
    points, mask = find_surface_intersection(coord_arrays, projected_b)

    n = mask.shape[0]

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

    # Spacecraft potential
    logging.info("Calculating spacecraft potential...")
    if use_parallel:
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
    theta_path = str(config.DATA_DIR / config.THETA_FILE)
    proj_potential = np.full(n, np.nan)

    # If the dataset is small, or ERData has been monkeypatched without
    # from_dataframe (as in some tests), fall back to sequential fitting.
    can_chunk = hasattr(ERData, "from_dataframe")
    rows_per_sweep = config.SWEEP_ROWS
    sweeps_per_chunk = 100
    rows_per_chunk = sweeps_per_chunk * rows_per_sweep
    chunked = use_parallel and can_chunk and n > rows_per_chunk

    if not chunked:
        logging.info("Running surface potential fitting (sequential)...")
        fitter = LossConeFitter(
            er_data,
            theta_path,
            spacecraft_potential=sc_potential,
        )
        fit_mat = fitter.fit_surface_potential()
        _apply_fit_results(fit_mat, proj_potential, row_offset=0, n_total=n)
    else:
        logging.info("Starting parallel surface potential fitting...")

        chunks: list[tuple[pd.DataFrame, str, np.ndarray]] = []
        for i in range(0, n, rows_per_chunk):
            end = min(i + rows_per_chunk, n)
            chunk_df = er_data.data.iloc[i:end].copy()
            chunk_sc_pot = sc_potential[i:end]
            chunks.append((chunk_df, theta_path, chunk_sc_pot))

        if chunks:
            num_workers = max(1, multiprocessing.cpu_count() - 1)

            with multiprocessing.Pool(processes=num_workers) as pool:
                results_iter = pool.imap(fit_worker, chunks)
                for chunk_idx, chunk_res in enumerate(
                    tqdm(
                        results_iter,
                        total=len(chunks),
                        desc="Fitting chunks",
                        unit="chunk",
                    )
                ):
                    row_offset = chunk_idx * rows_per_chunk
                    _apply_fit_results(chunk_res, proj_potential, row_offset, n_total=n)

    return PotentialResults(
        spacecraft_latitude=spacecraft_lat,
        spacecraft_longitude=spacecraft_lon,
        projection_latitude=proj_lat,
        projection_longitude=proj_lon,
        spacecraft_potential=sc_potential,
        projected_potential=proj_potential,
        spacecraft_in_sun=sc_in_sun,
        projection_in_sun=proj_in_sun,
    )


def process_lp_file(file_path: Path) -> PotentialResults:
    """
    Process a single ER file into PotentialResults (sequential fitting).

    This preserves the historical API used by tests, docs, and dev scripts.
    """
    logging.debug(f"Processing LP file: {file_path}")
    er_data = ERData(str(file_path))
    return process_merged_data(er_data, use_parallel=False)


def _concat_results(results: list[PotentialResults]) -> PotentialResults:
    """Concatenate fields from multiple PotentialResults objects (row-wise)."""

    def cat(attr: str):
        return np.concatenate([getattr(r, attr) for r in results])

    return PotentialResults(
        spacecraft_latitude=cat("spacecraft_latitude"),
        spacecraft_longitude=cat("spacecraft_longitude"),
        projection_latitude=cat("projection_latitude"),
        projection_longitude=cat("projection_longitude"),
        spacecraft_potential=cat("spacecraft_potential"),
        projected_potential=cat("projected_potential"),
        spacecraft_in_sun=cat("spacecraft_in_sun"),
        projection_in_sun=cat("projection_in_sun"),
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
        fig, ax = plot_mod.plot_map(agg, illumination=args.illumination)
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
