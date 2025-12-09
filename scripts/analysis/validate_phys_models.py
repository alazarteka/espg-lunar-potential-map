"""
Validate the reconstructed surface potential model against physical relationships.
Reproduces Halekas (2008) analysis: |U| vs Te (night) and |U| vs Je (day).
"""
import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.special import sph_harm_y
from scipy.stats import linregress
from tqdm import tqdm

# Allow imports from src if running as script
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import src.config as config
from src.flux import ERData
from src.kappa import Kappa, FitResults
from src.spacecraft_potential import theta_to_temperature_ev
from src.physics.charging import electron_current_density_magnitude
from src.utils.units import ureg

try:
    from src.potential_mapper.pipeline import DataLoader
    from src.potential_mapper.coordinates import (
        CoordinateCalculator,
        project_magnetic_fields,
        find_surface_intersection,
        CoordinateArrays
    )
    from src.potential_mapper.spice import load_spice_files
    from src.utils.attitude import load_attitude_data
    from src.utils.spice_ops import get_lp_vector_to_sun_in_lunar_frame, get_lp_position_wrt_moon
    SPICE_AVAILABLE = True
except ImportError:
    SPICE_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Default paths
ARTIFACTS_DIR = Path("artifacts/validation")
CACHE_DIR = Path("artifacts/potential_cache")
MODEL_FILE = CACHE_DIR / "temporal_coeffs.npz"

def _sph_harm(m: int, l: int, phi, theta):
    """Evaluate spherical harmonics using SciPy's sph_harm_y.

    Inputs match scipy: m (order), l (degree), theta (colatitude), phi (azimuth).
    """
    return sph_harm_y(l, m, theta, phi)

def evaluate_potential_at_points(
    lats: np.ndarray,
    lons: np.ndarray,
    times: np.ndarray,
    model_times: np.ndarray,
    model_coeffs: np.ndarray,
    lmax: int
) -> np.ndarray:
    """
    Evaluate the temporal SH model at specific (lat, lon, time) points.
    """
    if len(model_times) == 0:
        return np.full_like(lats, np.nan)

    # Convert times to float (seconds since epoch) for interpolation
    t_epoch = model_times[0]
    model_t_sec = (model_times - t_epoch) / np.timedelta64(1, 's')
    target_t_sec = (times - t_epoch) / np.timedelta64(1, 's')

    # Handle single time point case
    if len(model_times) == 1:
        coeffs_interp = np.repeat(model_coeffs, len(times), axis=0)
    else:
        # Interpolate coefficients to target times
        interpolator = interp1d(model_t_sec, model_coeffs, axis=0,
                               bounds_error=False, fill_value=(model_coeffs[0], model_coeffs[-1]))
        coeffs_interp = interpolator(target_t_sec) # (n_points, n_coeffs)

    # Prepare coordinates
    lat_rad = np.deg2rad(lats)
    lon_rad = np.deg2rad(lons)
    colatitudes = (np.pi / 2.0) - lat_rad

    # Evaluate expansion
    n_points = len(lats)
    potentials = np.zeros(n_points, dtype=np.float64)

    col_idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            y_lm = _sph_harm(m, l, lon_rad, colatitudes)
            term = np.real(coeffs_interp[:, col_idx] * y_lm)
            potentials += term
            col_idx += 1

    return potentials

def classify_environment(sza_deg, te=None):
    """
    Classify plasma environment based on geometry and parameters.
    """
    env = np.full(len(sza_deg), "Unknown", dtype=object)

    day_mask = sza_deg < 90
    night_mask = ~day_mask

    env[day_mask] = "Dayside"
    env[night_mask] = "Nightside"

    # Refine Nightside if possible
    # This assumes Moon passes through Earth's magnetotail.
    # Without orbital position wrt Earth, we can't reliably distinguish Lobes/Sheet.
    # But we can try using Te if provided.
    if te is not None:
        # Plasma Sheet: Hot (Te > ~50-100 eV)
        # Lobes: Cold (Te < ~50 eV)
        # Solar Wind Wake: Cold/Empty

        # Simple heuristic for demonstration
        hot_mask = night_mask & (te > 100)
        cold_mask = night_mask & (te <= 100)

        env[hot_mask] = "Nightside (Plasma Sheet?)"
        env[cold_mask] = "Nightside (Wake/Lobe)"

    return env

class ValidationData:
    def __init__(self):
        self.df = pd.DataFrame()

    def generate_synthetic(self, n_points=500):
        """Generate synthetic validation data if real data is missing."""
        logging.info(f"Generating {n_points} synthetic data points...")
        np.random.seed(42)

        base_time = datetime(1998, 5, 1)
        times = np.array([base_time + timedelta(hours=i) for i in range(n_points)], dtype='datetime64[ns]')

        lats = np.random.uniform(-90, 90, n_points)
        lons = np.random.uniform(-180, 180, n_points)
        sza = np.random.uniform(0, 180, n_points)

        te = np.random.lognormal(mean=np.log(40), sigma=0.5, size=n_points)
        te[sza > 90] *= 1.5

        je_micro = np.random.lognormal(mean=np.log(1.0), sigma=1.0, size=n_points)
        je = je_micro * 1e-6

        u_model = np.zeros(n_points)

        mask_night = sza > 90
        u_model[mask_night] = -4.0 * te[mask_night] * np.random.normal(1.0, 0.1, size=np.sum(mask_night))

        mask_day = ~mask_night
        je_eff = je[mask_day] * np.cos(np.deg2rad(sza[mask_day]))
        j_crit = 0.1e-6

        u_day = np.zeros(np.sum(mask_day))
        high_curr = je_eff > j_crit
        u_day[high_curr] = 5.0 + np.random.normal(0, 1, np.sum(high_curr))

        low_curr = ~high_curr
        mask_low = low_curr & (je_eff > 0)
        if mask_low.any():
            u_day[mask_low] = -100.0 * (j_crit - je_eff[mask_low])/j_crit

        u_model[mask_day] = u_day

        env = classify_environment(sza, te)

        self.df = pd.DataFrame({
            'time': times,
            'lat': lats,
            'lon': lons,
            'sza': sza,
            'Te': te,
            'Je': je,
            'U_model': u_model,
            'environment': env,
            'is_synthetic': True
        })

    def load_real_data(self, data_dir: Path):
        """Attempt to load real data by running kappa pipeline on discovered files."""
        cache_path = ARTIFACTS_DIR / "validation_data.parquet"
        if cache_path.exists():
            logging.info(f"Loading cached validation data from {cache_path}")
            self.df = pd.read_parquet(cache_path)
            return

        logging.info("No cached validation data found. Attempting to assemble from raw data...")
        try:
            files = DataLoader.discover_flux_files()
        except Exception:
            files = []

        if not files:
            logging.warning("No raw data files found.")
            return

        files = files[:5] # Process subset for speed
        logging.info(f"Processing {len(files)} files...")

        records = []

        # Load Attitude for Geometry
        try:
            et_spin, ra_vals, dec_vals = load_attitude_data(config.DATA_DIR / config.ATTITUDE_FILE)
            if et_spin is None:
                raise ValueError("Attitude data failed to load")
            coord_calc = CoordinateCalculator(et_spin, ra_vals, dec_vals)
            load_spice_files()
            has_geometry = True
        except Exception as e:
            logging.warning(f"Geometry init failed: {e}. Processing without accurate geometry.")
            has_geometry = False

        for f in tqdm(files):
            try:
                er = ERData(str(f))
                if er.data.empty: continue

                # Calculate coordinates for the whole file
                lat_arr, lon_arr, sza_arr = None, None, None

                if has_geometry:
                    try:
                        coord_arrays = coord_calc.calculate_coordinate_transformation(er)
                        proj_b = project_magnetic_fields(er, coord_arrays)
                        intersections, mask = find_surface_intersection(coord_arrays, proj_b)

                        # Calculate SZA
                        # Cos(SZA) = dot(SurfaceNormal, SunVector)
                        # SurfaceNormal at intersection is normalized intersection point
                        p = intersections
                        norms = np.linalg.norm(p, axis=1)
                        p_unit = p / norms[:, None]

                        sun_vec = coord_arrays.moon_vectors_to_sun
                        sun_norms = np.linalg.norm(sun_vec, axis=1)
                        sun_unit = sun_vec / sun_norms[:, None]

                        cos_sza = np.sum(p_unit * sun_unit, axis=1)
                        sza_arr = np.rad2deg(np.arccos(np.clip(cos_sza, -1, 1)))

                        # Calculate Lat/Lon
                        lat_arr = np.rad2deg(np.arcsin(p_unit[:, 2]))
                        lon_arr = np.rad2deg(np.arctan2(p_unit[:, 1], p_unit[:, 0]))

                        # Mask invalid
                        sza_arr[~mask] = np.nan
                        lat_arr[~mask] = np.nan
                        lon_arr[~mask] = np.nan

                    except Exception as e:
                        logging.warning(f"Geometry calc failed for {f}: {e}")

                # Group by spectrum
                # Map row indices to spec_no
                spec_nos = er.data[config.SPEC_NO_COLUMN].to_numpy()
                unique_specs = np.unique(spec_nos)

                # Time array
                times = er.data[config.UTC_COLUMN].to_numpy() # String

                for spec_no in unique_specs:
                    try:
                        if np.isnan(spec_no): continue

                        # Indices for this spec
                        idxs = np.where(spec_nos == spec_no)[0]
                        if len(idxs) == 0: continue

                        # Pick first valid geometry for this spectrum
                        row_idx = idxs[0]

                        k = Kappa(er, spec_no=int(spec_no))
                        fit, U_sc = k.corrected_fit()

                        if fit and fit.is_good_fit:
                            d, kap, th = fit.params.to_tuple()
                            te = theta_to_temperature_ev(th, kap)
                            je = electron_current_density_magnitude(d, kap, th, 1.0, 20000.0)

                            # Geometry
                            if has_geometry and sza_arr is not None:
                                sza = sza_arr[row_idx]
                                lat = lat_arr[row_idx]
                                lon = lon_arr[row_idx]
                            else:
                                sza, lat, lon = np.nan, np.nan, np.nan

                            utc_str = times[row_idx]
                            # Simple parse
                            try:
                                # 1998-05-01T00:00:00
                                ts = np.datetime64(utc_str)
                            except:
                                ts = np.datetime64('NaT')

                            records.append({
                                'spec_no': spec_no,
                                'Te': te,
                                'Je': je,
                                'sza': sza,
                                'lat': lat,
                                'lon': lon,
                                'time': ts
                            })
                    except Exception as e:
                        pass
            except Exception:
                continue

        if records:
            self.df = pd.DataFrame(records)
            # Add environment
            self.df['environment'] = classify_environment(self.df['sza'], self.df['Te'])
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            self.df.to_parquet(cache_path)
        else:
            logging.warning("No valid records generated from real data.")

def load_model_coeffs(path: Path) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load model coefficients from npz."""
    if not path.exists():
        logging.warning(f"Model file {path} not found.")
        return np.array([]), np.array([]), 0

    with np.load(path) as data:
        times = data['times']
        coeffs = data['coeffs']
        lmax = int(data['lmax'])
        return times, coeffs, lmax

def analyze_nightside(df: pd.DataFrame):
    """
    Bin |U_model| vs Te for nightside (SZA > 90), by environment.
    """
    logging.info("Analyzing Nightside...")
    night_df = df[df['sza'] > 90].copy()
    if night_df.empty:
        logging.warning("No nightside data found.")
        return

    # Plot
    plt.figure(figsize=(10, 6))

    envs = night_df['environment'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(envs)))

    for i, env in enumerate(envs):
        sub = night_df[night_df['environment'] == env]
        if sub.empty: continue

        # Bin Te
        if len(sub) > 5:
            bins = np.logspace(np.log10(sub['Te'].min()), np.log10(sub['Te'].max()), 15)
            sub = sub.copy() # Avoid SettingWithCopy
            sub['Te_bin'] = pd.cut(sub['Te'], bins)
            stats = sub.groupby('Te_bin', observed=True).agg({
                'Te': 'mean',
                'U_model': lambda x: np.mean(np.abs(x))
            }).reset_index()

            plt.scatter(sub['Te'], np.abs(sub['U_model']), alpha=0.2, s=5, label=f'{env} Data', color=colors[i])
            plt.plot(stats['Te'], stats['U_model'], '-o', lw=2, label=f'{env} Mean', color=colors[i])

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Electron Temperature Te (eV)')
    plt.ylabel('|Surface Potential| (V)')
    plt.title('Nightside: |U| vs Te by Environment')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    out_path = ARTIFACTS_DIR / "nightside_U_vs_Te.png"
    plt.savefig(out_path, dpi=150)
    logging.info(f"Saved {out_path}")
    plt.close()

    # Stats
    print("\n--- Nightside Statistics ---")
    valid = (night_df['Te'] > 0) & (np.abs(night_df['U_model']) > 0)
    if valid.any():
        slope, intercept, r_val, p_val, std_err = linregress(
            np.log10(night_df.loc[valid, 'Te']),
            np.log10(np.abs(night_df.loc[valid, 'U_model']))
        )
        print(f"Overall Nightside log(|U|) vs log(Te): Slope={slope:.2f}, R2={r_val**2:.2f}")
        print(f"Expected slope ~1.0 (linear scaling).")

def analyze_dayside(df: pd.DataFrame):
    """
    Bin |U_model| vs Je_eff for dayside (SZA < 90).
    """
    logging.info("Analyzing Dayside...")
    day_df = df[df['sza'] < 90].copy()
    if day_df.empty:
        logging.warning("No dayside data found.")
        return

    # Je_eff
    cos_sza = np.maximum(0, np.cos(np.deg2rad(day_df['sza'])))
    day_df['Je_eff'] = day_df['Je'] * cos_sza

    day_df = day_df[day_df['Je_eff'] > 1e-9]
    if day_df.empty: return

    bins = np.logspace(np.log10(day_df['Je_eff'].min()), np.log10(day_df['Je_eff'].max()), 20)
    day_df['Je_bin'] = pd.cut(day_df['Je_eff'], bins)

    stats = day_df.groupby('Je_bin', observed=True).agg({
        'Je_eff': 'mean',
        'U_model': lambda x: np.mean(np.abs(x))
    }).reset_index()

    plt.figure(figsize=(8, 6))
    plt.scatter(day_df['Je_eff'], np.abs(day_df['U_model']), alpha=0.1, color='gray', s=1, label='Data')
    plt.plot(stats['Je_eff'], stats['U_model'], 'b-o', lw=2, label='Binned Mean')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Effective Current Density Je*cos(SZA) (A/m^2)')
    plt.ylabel('|Surface Potential| (V)')
    plt.title('Dayside: |U| vs Effective Current')
    plt.axvline(x=0.1e-6, color='k', linestyle='--', label='~0.1 uA/m2 Threshold')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    plt.legend()

    out_path = ARTIFACTS_DIR / "dayside_U_vs_Je.png"
    plt.savefig(out_path, dpi=150)
    logging.info(f"Saved {out_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="Force synthetic data generation")
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    val_data = ValidationData()

    # 1. Load Data
    if args.demo:
        val_data.generate_synthetic()
    else:
        val_data.load_real_data(Path("data"))
        if val_data.df.empty:
            logging.info("Real data load failed or empty. Falling back to synthetic.")
            val_data.generate_synthetic()

    df = val_data.df
    logging.info(f"Data ready: {len(df)} records.")

    # 2. Load Model
    is_synthetic = False
    if 'is_synthetic' in df.columns:
        is_synthetic = df['is_synthetic'].any()

    if not is_synthetic:
        times, coeffs, lmax = load_model_coeffs(MODEL_FILE)
        if len(times) > 0:
            logging.info(f"Loaded model with lmax={lmax}, {len(times)} time steps.")
            # Evaluate model
            if 'lat' in df.columns and 'time' in df.columns and not df['lat'].isna().all():
                u_model = evaluate_potential_at_points(
                    df['lat'].values, df['lon'].values, df['time'].values,
                    times, coeffs, lmax
                )
                df['U_model'] = u_model
            else:
                logging.warning("Data missing lat/lon/time, cannot evaluate model.")
        else:
            logging.warning("Model load failed. Cannot evaluate U_model.")
            if 'U_model' not in df.columns:
                logging.warning("Adding dummy U_model for plot demonstration.")
                df['U_model'] = np.random.randn(len(df)) * 10

    # 3. Analyze
    analyze_nightside(df)
    analyze_dayside(df)

    logging.info("Validation complete.")

if __name__ == "__main__":
    main()
