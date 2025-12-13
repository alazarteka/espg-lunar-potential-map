import argparse
import logging
import numbers
from pathlib import Path

import numpy as np

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
from src.utils.units import ureg


def _spacecraft_potential_per_row(er_data: ERData, n_rows: int) -> np.ndarray:
    """
    Return spacecraft potential per ER row by spec_no grouping.

    Args:
        er_data: The ERData object.
        n_rows: Total number of rows.

    Returns:
        np.ndarray: Array of spacecraft potentials per row.
    """

    from src.spacecraft_potential import calculate_potential

    potentials = np.full(n_rows, np.nan)
    if n_rows == 0:
        return potentials

    spec_values = er_data.data[config.SPEC_NO_COLUMN].to_numpy()
    unique_specs = np.unique(spec_values)
    for spec_value in unique_specs:
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


class DataLoader:
    """Discover ER files and load auxiliary attitude/theta data.

    File discovery is layout-agnostic by default (recursive glob of *EXT_TAB)
    and excludes known support tables. Optional year/month/day filters are
    applied via best-effort token matching in paths.
    """

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
        """
        Discover ER flux files below DATA_DIR.

        Strategy:
        - Recursively glob for `*{EXT_TAB}` under `DATA_DIR`.
        - Exclude known support tables (attitude/solid_angles/theta/areas).
        - Optionally filter by year/month/day via token matching if present.
        """
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


def process_lp_file(file_path: Path) -> PotentialResults:
    """
    Process a single ER file into PotentialResults.

    Steps: load ER + attitude; build coordinate arrays; project B; find surface
    intersections; compute lat/lon and day/night flags; fit U_surface per 15-row chunk
    and map to rows.
    """
    logging.debug(f"Processing LP file: {file_path}")

    # Load ER data and attitude
    er_data = ERData(str(file_path))
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

    # Sun illumination for spacecraft: ray to sun intersects the Moon -> shadow
    u_lp_to_sun = coord_arrays.lp_vectors_to_sun
    u_lp_to_sun = u_lp_to_sun / np.linalg.norm(u_lp_to_sun, axis=1, keepdims=True)
    _, sc_shadow_hit = get_intersections_or_none_batch(
        pos=coord_arrays.lp_positions, direction=u_lp_to_sun, radius=config.LUNAR_RADIUS
    )
    sc_in_sun = ~sc_shadow_hit

    # Sun illumination for projection point: dot(n_hat, moon_to_sun_hat) > 0
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

    # Spacecraft potential per spectrum (spec_no) – reused when combining U_surface.
    sc_potential = _spacecraft_potential_per_row(er_data, n)

    # Fit surface potential U_surface per 15-row chunk and map to rows
    try:
        # Pass spacecraft potential so fitter can de-bias energies per spectrum.
        fitter = LossConeFitter(
            er_data,
            str(config.DATA_DIR / config.THETA_FILE),
            spacecraft_potential=sc_potential,
        )
        fit_mat = fitter.fit_surface_potential()  # shape (n_chunks, 5)
        # Columns: [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
        proj_potential = np.full(n, np.nan)
        if fit_mat.size > 0:
            for U_surface, _bs, _beam_amp, chi2, chunk_idx in fit_mat:
                i = int(chunk_idx)
                if not np.isfinite(U_surface) or not np.isfinite(chi2):
                    continue
                if chi2 > config.FIT_ERROR_THRESHOLD:
                    continue
                s = i * config.SWEEP_ROWS
                e = min((i + 1) * config.SWEEP_ROWS, n)
                if s >= n:
                    break
                delta_u_value = float(U_surface)
                proj_chunk = proj_potential[s:e]
                proj_chunk[:] = delta_u_value
    except Exception as e:
        logging.exception(f"LossCone fitting failed for {file_path}: {e}")
        proj_potential = np.full(n, np.nan)

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


def _concat_results(results: list[PotentialResults]) -> PotentialResults:
    """
    Concatenate fields from multiple PotentialResults objects (row-wise).

    Args:
        results: List of PotentialResults objects.

    Returns:
        PotentialResults: Concatenated results.
    """

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
    """Entry point for CLI to orchestrate processing and optional plotting."""
    flux_files = DataLoader.discover_flux_files(
        year=args.year,
        month=args.month,
        day=args.day,
    )

    if not flux_files:
        logging.info("No ER flux files discovered. Exiting.")
        return 0

    results: list[PotentialResults] = []
    for fp in flux_files:
        try:
            logging.debug(f"Processing {fp}")
            res = process_lp_file(fp)
            results.append(res)
        except Exception as e:
            logging.exception(f"Failed to process {fp}: {e}")

    if not results:
        logging.warning("All files failed to process; nothing to plot or save.")
        return 1

    # Aggregate results across files
    agg = _concat_results(results) if len(results) > 1 else results[0]

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
