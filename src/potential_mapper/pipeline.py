"""End-to-end potential mapping pipeline: per-ER-file kappa fits, spacecraft
potential, loss-cone surface potential fits, and surface footprint projection."""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import src.config as config
from src.flux import ERData, LossConeFitter, PitchAngle
from src.losscone.types import FitMethod, parse_fit_method

try:
    from src.losscone_torch import HAS_TORCH, LossConeFitterTorch
except ImportError:
    HAS_TORCH = False
    LossConeFitterTorch = None  # type: ignore[misc, assignment]

try:
    from src.kappa_torch import KappaFitterTorch

    HAS_KAPPA_TORCH = True
except ImportError:
    HAS_KAPPA_TORCH = False
    KappaFitterTorch = None  # type: ignore[misc, assignment]

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
    discover_flux_files,
)
from src.potential_mapper.kappa_batch import _prepare_kappa_batch_data
from src.potential_mapper.results import PotentialResults
from src.potential_mapper.sc_potential import (
    _spacecraft_potential_per_row,
    _spacecraft_potential_per_row_torch,
)

# Re-export for tests and callers that import privates from pipeline.
__all__ = [
    "DataLoader",
    "_apply_fit_results",
    "_prepare_kappa_batch_data",
    "_spacecraft_potential_per_row",
    "_spacecraft_potential_per_row_torch",
    "load_all_data",
    "process_lp_file",
    "process_merged_data",
]
from src.utils.attitude import load_attitude_data
from src.utils.geometry import get_intersections_or_none_batch


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
        return discover_flux_files(year=year, month=month, day=day)


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
    use_torch: bool = False,
    fit_method: str | FitMethod | None = None,
    spacecraft_potential_override: float | None = None,
    emit_u_width_qc: bool = False,
    u_width_identifiable_max_v: float = 200.0,
) -> PotentialResults:
    """
    Process the merged ER dataset.

    Steps:
    1. Load attitude and calculate coordinates (vectorized).
    2. Calculate spacecraft potential.
    3. Run surface potential fitting (sequential or torch-accelerated).
    4. Assemble results.

    Args:
        er_data: Merged ERData object
        use_torch: Use PyTorch-accelerated fitter (~5x faster, default: False)
        fit_method: Loss-cone fitting method ("halekas" or "lillis")
        spacecraft_potential_override: Optional constant spacecraft potential [V].
            When provided, skips spacecraft potential estimation and uses this
            value for all rows.

    Returns:
        PotentialResults with all computed fields
    """
    logging.info("Processing merged dataset...")

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
                    "PyTorch not available for Kappa fitting; "
                    "falling back to sequential"
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
        else:
            sc_potential = _spacecraft_potential_per_row(er_data, n)

    # Surface Potential Fitting
    proj_potential = np.full(n, np.nan)
    bs_over_bm_arr = np.full(n, np.nan)
    beam_amp_arr = np.full(n, np.nan)
    fit_chi2_arr = np.full(n, np.nan)
    u_width_lhs_dchi2red_0p001_arr = np.full(n, np.nan)
    u_is_identifiable_lhs_dchi2red_0p001_arr = np.zeros(n, dtype=bool)

    rows_per_sweep = config.SWEEP_ROWS
    pitch_angle = None
    required_cols = set(config.PHI_COLS + config.MAG_COLS)
    if required_cols.issubset(er_data.data.columns):
        pitch_angle = PitchAngle(er_data, polarity=projection_polarity)
    else:
        logging.debug("Pitch-angle polarity skipped; required ER columns are missing.")

    emit_u_width_qc = bool(emit_u_width_qc) and fit_method == FitMethod.LILLIS

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
        # Torch path defaults to batched processing (auto-detects dtype and batch_size)
        if emit_u_width_qc:
            fit_mat, u_width_chunks, u_is_identifiable_chunks = (
                fitter.fit_surface_potential_with_u_width_qc(
                    delta_reduced=0.001,
                    identifiable_width_max_v=float(u_width_identifiable_max_v),
                )
            )
        else:
            fit_mat = fitter.fit_surface_potential()
            u_width_chunks = None
            u_is_identifiable_chunks = None
        _apply_fit_results(
            fit_mat,
            proj_potential,
            bs_over_bm_arr,
            beam_amp_arr,
            fit_chi2_arr,
            0,
            n,
        )
    else:
        logging.info("Running surface potential fitting (sequential)...")
        fitter = LossConeFitter(
            er_data,
            pitch_angle=pitch_angle,
            spacecraft_potential=sc_potential,
            fit_method=fit_method,
        )
        if emit_u_width_qc:
            fit_mat, u_width_chunks = fitter.fit_surface_potential_with_u_width_qc(
                delta_reduced=0.001
            )
            u_is_identifiable_chunks = np.isfinite(u_width_chunks) & (
                u_width_chunks <= float(u_width_identifiable_max_v)
            )
        else:
            fit_mat = fitter.fit_surface_potential()
            u_width_chunks = None
            u_is_identifiable_chunks = None
        _apply_fit_results(
            fit_mat,
            proj_potential,
            bs_over_bm_arr,
            beam_amp_arr,
            fit_chi2_arr,
            0,
            n,
        )

    if (
        emit_u_width_qc
        and u_width_chunks is not None
        and u_is_identifiable_chunks is not None
    ):
        n_chunks = len(er_data.data) // rows_per_sweep
        for chunk_idx in range(n_chunks):
            row_start = chunk_idx * rows_per_sweep
            row_end = min(row_start + rows_per_sweep, n)
            u_width_lhs_dchi2red_0p001_arr[row_start:row_end] = float(
                u_width_chunks[chunk_idx]
            )
            u_is_identifiable_lhs_dchi2red_0p001_arr[row_start:row_end] = bool(
                u_is_identifiable_chunks[chunk_idx]
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
        u_width_lhs_dchi2red_0p001=u_width_lhs_dchi2red_0p001_arr,
        u_is_identifiable_lhs_dchi2red_0p001=u_is_identifiable_lhs_dchi2red_0p001_arr,
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
