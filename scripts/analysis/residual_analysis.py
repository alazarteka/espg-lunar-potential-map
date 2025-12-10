#!/usr/bin/env python
"""
Analyze model residuals (U_obs - U_model).

Computes residuals between observed surface potential and a temporal basis model,
then analyzes them by:
1. Spatial distribution (Map)
2. Solar Zenith Angle (SZA)
3. Plasma Environment (Solar Wind, Wake, Lobes, Plasma Sheet)

Usage:
    uv run python scripts/analysis/residual_analysis.py \
        --start 1998-07-01 --end 1998-07-31 \
        --lmax 5 \
        --temporal-basis constant,synodic
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice
from scipy.stats import binned_statistic, binned_statistic_2d
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.potential_mapper.spice import load_spice_files
from src.temporal.basis import _get_basis_func_by_name, fit_temporal_basis
from src.temporal.coefficients import (
    DEFAULT_CACHE_DIR,
    _build_harmonic_design,
    _discover_npz,
    _harmonic_coefficient_count,
    _load_all_data,
)
from src.utils.spice_ops import get_sun_vector_wrt_moon_batch
from src.visualization import style


def latlon_to_vector(lat, lon):
    """Convert lat/lon (degrees) to unit vector on sphere."""
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack((x, y, z))


def compute_sza(et, lat, lon):
    """Compute Solar Zenith Angle for each point."""
    sun_vecs = get_sun_vector_wrt_moon_batch(et)
    surf_vecs = latlon_to_vector(lat, lon)

    # Normalize sun vecs
    sun_norms = np.linalg.norm(sun_vecs, axis=1, keepdims=True)
    # Handle potentially zero norms (though sun shouldn't be at 0 distance)
    sun_dirs = np.zeros_like(sun_vecs)
    valid = sun_norms.flatten() > 0
    sun_dirs[valid] = sun_vecs[valid] / sun_norms[valid]

    dot = np.sum(sun_dirs * surf_vecs, axis=1)
    sza_deg = np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
    return sza_deg


def classify_environment(et: np.ndarray, sza_deg: np.ndarray) -> np.ndarray:
    """
    Classify plasma environment based on Moon position and SZA.

    Categories:
    - "Solar Wind": Sun-Earth-Moon angle < 135 deg, SZA < 90
    - "Wake": Sun-Earth-Moon angle < 135 deg, SZA >= 90
    - "Plasma Sheet": In Tail (> 135 deg) and |Z_Eclip| < 5 RE
    - "Lobes": In Tail (> 135 deg) and |Z_Eclip| >= 5 RE
    """
    n = len(et)
    env = np.full(n, "Unknown", dtype=object)

    logging.info("Classifying environment for %d points...", n)

    # Batch process in chunks to avoid memory issues if n is huge?
    # n ~ 15000 per month, so it's fine.

    # 1. Determine Tail vs SW (Sun-Earth-Moon Angle)
    pos_sun = np.zeros((n, 3))
    pos_moon = np.zeros((n, 3))

    # This loop might be slow for very large datasets, but acceptable for analysis scripts
    for i in tqdm(range(n), desc="Environment Classification", leave=False):
        t = et[i]
        try:
            p_sun, _ = spice.spkpos("SUN", t, "J2000", "NONE", "EARTH")
            p_moon, _ = spice.spkpos("MOON", t, "J2000", "NONE", "EARTH")
            pos_sun[i] = p_sun
            pos_moon[i] = p_moon
        except Exception:
            pass

    norm_sun = np.linalg.norm(pos_sun, axis=1)
    norm_moon = np.linalg.norm(pos_moon, axis=1)

    # Avoid div/0
    norm_sun[norm_sun == 0] = 1.0
    norm_moon[norm_moon == 0] = 1.0

    u_sun = pos_sun / norm_sun[:, None]
    u_moon = pos_moon / norm_moon[:, None]

    cos_theta = np.sum(u_sun * u_moon, axis=1)
    theta_deg = np.rad2deg(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    in_tail = theta_deg > 135.0
    mask_sw = ~in_tail

    # 2. Assign SW / Wake
    env[mask_sw & (sza_deg < 90)] = "Solar Wind"
    env[mask_sw & (sza_deg >= 90)] = "Wake"

    # 3. Assign Magnetotail regions
    if np.any(in_tail):
        tail_indices = np.where(in_tail)[0]
        pos_moon_eclip = np.zeros((len(tail_indices), 3))

        for idx, i in enumerate(tail_indices):
            t = et[i]
            try:
                # Transform Moon position (already in J2000) to ECLIPJ2000
                mat = spice.pxform("J2000", "ECLIPJ2000", t)
                pos_moon_eclip[idx] = spice.mxv(mat, pos_moon[i])
            except Exception:
                pass

        z_dist_km = np.abs(pos_moon_eclip[:, 2])
        RE_km = 6371.0

        is_ps = z_dist_km < (5 * RE_km)

        env[tail_indices[is_ps]] = "Plasma Sheet"
        env[tail_indices[~is_ps]] = "Lobes"

    return env


def predict_with_result(
    result,
    utc: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    reference_time: np.datetime64,
) -> np.ndarray:
    """Predict potential using fitted basis coefficients."""
    t_hours = (utc - reference_time).astype("timedelta64[s]").astype(np.float64) / 3600.0

    K = len(result.basis_names)
    basis_funcs = [_get_basis_func_by_name(name) for name in result.basis_names]
    T = np.column_stack([func(t_hours) for func in basis_funcs])

    Y = _build_harmonic_design(lat, lon, result.lmax)

    n_coeffs = _harmonic_coefficient_count(result.lmax)
    design = np.empty((len(utc), K * n_coeffs), dtype=np.complex128)
    for k in range(K):
        design[:, k * n_coeffs : (k + 1) * n_coeffs] = Y * T[:, k : k + 1]

    b_flat = result.basis_coeffs.flatten()
    return np.real(design @ b_flat)


def analyze_residuals(
    utc, lat, lon, potential, predicted, et, args
):
    """Perform analysis and generate plots."""
    residual = potential - predicted

    logging.info("Computing SZA...")
    sza = compute_sza(et, lat, lon)

    logging.info("Classifying Environment...")
    env = classify_environment(et, sza)

    # Create output directory
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global Residual Map
    logging.info("Generating Global Residual Map...")
    fig1, ax1 = plt.subplots(figsize=(10, 5))

    # Binning
    lat_bins = np.linspace(-90, 90, 37)  # 5 deg
    lon_bins = np.linspace(-180, 180, 73) # 5 deg

    stat_mean, x_edge, y_edge, _ = binned_statistic_2d(
        lon, lat, residual, statistic='mean', bins=[lon_bins, lat_bins]
    )

    # Plot using imshow (transpose because statistic is bin_x, bin_y)
    im = ax1.imshow(
        stat_mean.T,
        extent=[x_edge[0], x_edge[-1], y_edge[0], y_edge[-1]],
        origin='lower',
        cmap='RdBu_r',
        vmin=-np.nanpercentile(np.abs(residual), 95),
        vmax=np.nanpercentile(np.abs(residual), 95)
    )
    cbar1 = plt.colorbar(im, ax=ax1, label='Mean Residual (V)')
    cbar1.ax.tick_params(labelsize=style.FONT_SIZE_TEXT)
    ax1.set_xlabel('Longitude (deg)', fontsize=style.FONT_SIZE_LABEL)
    ax1.set_ylabel('Latitude (deg)', fontsize=style.FONT_SIZE_LABEL)
    style.apply_paper_style(ax1)
    if args.title_map:
        ax1.set_title(args.title_map, fontsize=style.FONT_SIZE_TITLE)
    else:
        ax1.set_title(f'Mean Residual Map (Obs - Model)\n{args.start} to {args.end}', fontsize=style.FONT_SIZE_TITLE)
    fig1.savefig(out_dir / "residual_map_mean.png", dpi=150)
    plt.close(fig1)

    # 2. Residual vs SZA
    logging.info("Generating Residual vs SZA plot...")
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    sza_bins = np.linspace(0, 180, 37) # 5 deg
    sza_centers = 0.5 * (sza_bins[1:] + sza_bins[:-1])

    mean_res, _, _ = binned_statistic(sza, residual, statistic='mean', bins=sza_bins)
    std_res, _, _ = binned_statistic(sza, residual, statistic='std', bins=sza_bins)

    ax2.plot(sza_centers, mean_res, 'b-', label='Mean Residual')
    ax2.fill_between(sza_centers, mean_res - std_res, mean_res + std_res, color='b', alpha=0.2, label='±1 Std Dev')

    ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
    ax2.axvline(90, color='r', linestyle='--', label='Terminator')

    ax2.set_xlabel('Solar Zenith Angle (deg)', fontsize=style.FONT_SIZE_LABEL)
    ax2.set_ylabel('Residual (V)', fontsize=style.FONT_SIZE_LABEL)
    if args.title_sza:
        ax2.set_title(args.title_sza, fontsize=style.FONT_SIZE_TITLE)
    else:
        ax2.set_title('Residual vs SZA', fontsize=style.FONT_SIZE_TITLE)
    ax2.legend()
    style.apply_paper_style(ax2)
    fig2.savefig(out_dir / "residual_vs_sza.png", dpi=150)
    plt.close(fig2)

    # 3. Residual vs Environment
    logging.info("Generating Residual by Environment plot...")
    fig3, ax3 = plt.subplots(figsize=(10, 6))

    # Collect data for boxplot
    unique_envs = ["Solar Wind", "Wake", "Plasma Sheet", "Lobes"]
    data_groups = []
    labels = []

    for e in unique_envs:
        mask = env == e
        vals = residual[mask]
        if len(vals) > 0:
            data_groups.append(vals)
            labels.append(f"{e}\n(n={len(vals)})")

    if data_groups:
        ax3.boxplot(data_groups, labels=labels, showfliers=False)
        ax3.set_ylabel('Residual (V)', fontsize=style.FONT_SIZE_LABEL)
        if args.title_env:
            ax3.set_title(args.title_env, fontsize=style.FONT_SIZE_TITLE)
        else:
            ax3.set_title('Residual Distribution by Plasma Environment', fontsize=style.FONT_SIZE_TITLE)
        style.apply_paper_style(ax3, grid=False)
        ax3.grid(True, axis='y', **style.GRID_STYLE)
        ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    else:
        ax3.text(0.5, 0.5, "No environment data classified", ha='center')

    fig3.savefig(out_dir / "residual_by_environment.png", dpi=150)
    plt.close(fig3)

    # Summary text
    summary_path = out_dir / "residual_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Residual Analysis Summary\n")
        f.write("=========================\n")
        f.write(f"Date Range: {args.start} to {args.end}\n")
        f.write(f"Model: lmax={args.lmax}, basis={args.temporal_basis}\n")
        f.write(f"Total Points: {len(residual)}\n")
        f.write(f"Global RMS Residual: {np.sqrt(np.mean(residual**2)):.2f} V\n\n")

        f.write("By Environment:\n")
        for e in unique_envs:
            mask = env == e
            vals = residual[mask]
            if len(vals) > 0:
                f.write(f"  {e}: Mean={np.mean(vals):.2f}, Std={np.std(vals):.2f}, N={len(vals)}\n")
            else:
                f.write(f"  {e}: N=0\n")

    logging.info(f"Analysis complete. Results saved to {out_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze model residuals")
    parser.add_argument("--start", type=np.datetime64, required=True, help="Start date")
    parser.add_argument("--end", type=np.datetime64, required=True, help="End date")
    parser.add_argument("--lmax", type=int, default=5, help="Spherical harmonic degree")
    parser.add_argument("--temporal-basis", default="constant,synodic", help="Basis spec")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--l2-penalty", type=float, default=100.0)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/analysis/residuals"), help="Output directory for plots")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--title-map", type=str, default=None, help="Override title for residual map plot")
    parser.add_argument("--title-sza", type=str, default=None, help="Override title for SZA plot")
    parser.add_argument("--title-env", type=str, default=None, help="Override title for environment plot")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    # 1. Load Data
    logging.info("Discovering data...")
    files = _discover_npz(args.cache_dir)
    if not files:
        logging.error(f"No cache files found in {args.cache_dir}")
        return 1

    start_ts = args.start.astype("datetime64[s]")
    end_ts = (args.end + np.timedelta64(1, "D")).astype("datetime64[s]")

    utc, lat, lon, potential = _load_all_data(files, start_ts, end_ts)

    if len(utc) == 0:
        logging.error("No data found in specified date range.")
        return 1

    logging.info(f"Loaded {len(utc)} points.")

    # 2. Fit Model
    logging.info("Fitting model...")
    result = fit_temporal_basis(
        utc, lat, lon, potential,
        lmax=args.lmax,
        basis_spec=args.temporal_basis,
        l2_penalty=args.l2_penalty
    )

    # 3. Predict
    logging.info("Predicting model values...")
    predicted = predict_with_result(result, utc, lat, lon, utc.min())

    # 4. Compute Geometry & Analyze
    logging.info("Loading SPICE...")
    try:
        load_spice_files()
    except Exception as e:
        logging.error(f"Failed to load SPICE kernels: {e}")
        return 1

    # Convert utc to ET for SPICE
    logging.info("Converting timestamps to ET...")
    utc_str = np.datetime_as_string(utc, unit='s')
    et = np.zeros(len(utc))
    for i, t_str in enumerate(tqdm(utc_str, desc="Converting UTC to ET", leave=False)):
        try:
            et[i] = spice.str2et(t_str)
        except:
            et[i] = np.nan

    # Filter invalid ET
    mask = np.isfinite(et)
    if not np.all(mask):
        logging.warning(f"Dropping {np.sum(~mask)} points with invalid ET")
        utc = utc[mask]
        lat = lat[mask]
        lon = lon[mask]
        potential = potential[mask]
        predicted = predicted[mask]
        et = et[mask]

    # 5. Analyze
    analyze_residuals(utc, lat, lon, potential, predicted, et, args)

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
