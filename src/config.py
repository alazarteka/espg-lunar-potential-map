"""
Central configuration constants for the Lunar Prospector Plasma Analysis pipeline.
"""

from pathlib import Path

import scipy.constants

from .utils.units import ChargeType, LengthType, MassType, ureg

# ========== Data and chunk settings ==========
SWEEP_ROWS = 15  # rows per spacecraft sweep of energy spectrum
CHANNELS = 88  # number of ER electron flux channels

ACCUMULATION_TIME = 2.5 * ureg.second # accumulation time for electron flux data
GEOMETRIC_FACTOR = 0.02 * ureg.centimeter**2 * ureg.steradian # geometric factor for electron flux
BINS_BY_LATITUDE = {
        78.75: 4,
        56.25: 8,
        33.75: 16,
        11.25: 16,
        -11.25: 16,
        -33.75: 16,
        -56.25: 8,
        -78.75: 4,
    }


# ========== Physical parameters ==========
LUNAR_RADIUS_KM = 1737.400  # average radius of the Moon in kilometers
LUNAR_RADIUS: LengthType = 1737.400 * ureg.kilometer  # average radius of the Moon
ELECTRON_MASS: MassType = scipy.constants.electron_mass * ureg.kilogram  # electron
ELECTRON_CHARGE: ChargeType = (
    scipy.constants.e * ureg.coulomb
)  # elementary charge in Coulombs

# ========== File extensions ==========
EXT_TAB = ".TAB"  # ER data file extension
EXT_BSP = ".bsp"  # SPICE ephemeris kernel
EXT_TLS = ".tls"  # SPICE leap seconds kernel
EXT_TPC = ".tpc"  # SPICE planetary constants kernel

# ========== Download manager settings ==========
MAX_DOWNLOAD_WORKERS = 20  # threads for parallel downloads
CHUNK_SIZE_BYTES = 16 * 1024 * 1024  # chunk size for streaming downloads
REQUESTS_PER_SECOND = 10  # rate-limit threshold
CONNECTION_POOL_SIZE = 50  # connection pool size for reuse

# ========== Numerical tolerances ==========
EPS = 1e-6  # small epsilon to avoid division by zero
E_GAIN = 0.05  # typical gain uncertainty
E_G = 0.10  # typical geometric factor uncertainty
N_BG = 0.0  # background noise


# ========== Data Column names ==========
MISC_COLS = ["UTC", "time", "energy", "spec_no"]
FLUX_COLS = [f"ele_flux_{i}" for i in range(88)]
PHI_COLS = [f"dist_phi_{i}" for i in range(88)]
COUNT_COLS = ["count", "count_err"]
MAG_COLS = ["mag_x", "mag_y", "mag_z"]
ALL_COLS = MISC_COLS + FLUX_COLS + PHI_COLS + MAG_COLS + COUNT_COLS

# ========== Directory paths ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root directory
DATA_DIR = PROJECT_ROOT / "data"  # main data directory
KERNELS_DIR = PROJECT_ROOT / "data" / "spice_kernels"

# ========== File names ==========
THETA_FILE = "theta.tab"
SOLID_ANGLES_FILE = "solid_angles.tab"
ATTITUDE_FILE = "attitude.tab"
MOON_MAP_FILE = "moon_map.tif"
