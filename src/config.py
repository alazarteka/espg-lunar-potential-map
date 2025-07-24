"""
Central configuration constants for the Lunar Prospector Plasma Analysis pipeline.
"""

from pathlib import Path

import scipy.constants

from .utils.units import (ureg, Length, Mass, Charge)

# ========== Data and chunk settings ==========
SWEEP_ROWS = 15  # rows per spacecraft sweep of energy spectrum
CHANNELS = 88  # number of ER electron flux channels

# ========== Physical parameters ==========
LUNAR_RADIUS_KM = 1737.400  # average radius of the Moon in kilometers
LUNAR_RADIUS: Length = 1737.400 * ureg.kilometer  # average radius of the Moon
ELECTRON_MASS: Mass = scipy.constants.electron_mass * ureg.kilogram  # electron
ELECTRON_CHARGE: Charge = scipy.constants.e * ureg.coulomb  # elementary charge in Coulombs

# ========== File extensions ==========
EXT_TAB = ".TAB"  # ER data file extension
EXT_BSP = ".bsp"  # SPICE ephemeris kernel
EXT_TLS = ".tls"  # SPICE leap seconds kernel
EXT_TPC = ".tpc"  # SPICE planetary constants kernel

# ========== Download manager settings ==========
MAX_DOWNLOAD_WORKERS = 10  # default threads for parallel downloads
CHUNK_SIZE_BYTES = 4 * 1024 * 1024  # chunk size for streaming downloads
REQUESTS_PER_SECOND = 5  # rate-limit threshold

# ========== Numerical tolerances ==========
EPS = 1e-6  # small epsilon to avoid division by zero

# ========== Data Column names ==========
MISC_COLS = ["UTC", "time", "energy", "spec_no", "mag_x", "mag_y", "mag_z"]
FLUX_COLS = [f"ele_flux_{i}" for i in range(88)]
PHI_COLS = [f"dist_phi_{i}" for i in range(88)]
ALL_COLS = MISC_COLS + FLUX_COLS + PHI_COLS
MAG_COLS = ["mag_x", "mag_y", "mag_z"]

# ========== Directory paths ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root directory
DATA_DIR = PROJECT_ROOT / "data"  # main data directory
KERNELS_DIR = PROJECT_ROOT / "data" / "spice_kernels"

# ========== File names ==========
THETA_FILE = "theta.tab"
SOLID_ANGLES_FILE = "solid_angles.tab"
ATTITUDE_FILE = "attitude.tab"
MOON_MAP_FILE = "moon_map.tif"
