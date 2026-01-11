"""
Central configuration constants for the Lunar Prospector Plasma Analysis pipeline.
"""

from pathlib import Path

import scipy.constants

from .utils.units import ChargeType, LengthType, MassType, ureg

# ========== Data and chunk settings ==========
SWEEP_ROWS = 15  # rows per spacecraft sweep of energy spectrum
CHANNELS = 88  # number of ER electron flux channels

ACCUMULATION_TIME = 2.5 * ureg.second  # accumulation time for electron flux data
GEOMETRIC_FACTOR = (
    0.02 * ureg.centimeter**2 * ureg.steradian
)  # geometric factor for electron flux
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
LUNAR_RADIUS_KM: float = 1737.400  # average radius of the Moon in kilometers
LUNAR_RADIUS: LengthType = 1737.400 * ureg.kilometer  # average radius of the Moon
ELECTRON_MASS: MassType = scipy.constants.electron_mass * ureg.kilogram  # electron
ELECTRON_MASS_MAGNITUDE: float = scipy.constants.electron_mass  # electron mass in kg
ELECTRON_MASS_EV_S2_M2: float = (
    5.685630e-12  # Electron mass in eV*s^2/m^2 units (for energy-velocity conversion)
)
ELECTRON_CHARGE: ChargeType = (
    scipy.constants.e * ureg.coulomb
)  # elementary charge in Coulombs
ELECTRON_CHARGE_MAGNITUDE: float = (
    scipy.constants.e
)  # Charge of an electron in Coulombs
BOLTZMANN_CONSTANT_MAGNITUDE: float = scipy.constants.Boltzmann  # Unit J/K
PROTON_MASS_MAGNITUDE: float = scipy.constants.proton_mass  # Unit kg
CM2_TO_M2: float = 1e-4  # cm^2 to m^2 conversion factor

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
ENERGY_WINDOW_WIDTH_RELATIVE = 0.5

# ========== Fitting parameters ==========
# TODO: See docs/analysis/fitter_error_analysis.md for threshold decision discussion
# Current value (2.15e10) is very permissive (~99th+ percentile, accepts almost all fits)
# Alternative: 215_000 (95th percentile) or 657_000 (99th percentile) for stricter quality control
FIT_ERROR_THRESHOLD = 657000  # chi-squared threshold for a good fit (99th percentile)
LOSS_CONE_LHS_SEED = 42  # ensures deterministic Latin hypercube sampling

# Surface potential bounds (see Halekas 2008 Section 5.2)
# Electron reflectometry cannot reliably measure positive potentials because the
# photoelectron sheath (~meters) is much thinner than magnetic field scales.
# Detection threshold is ~20V; values within ±20V are "below threshold".
LOSS_CONE_U_SURFACE_MIN = -2000.0  # lower bound in volts (extreme plasma sheet)
LOSS_CONE_U_SURFACE_MAX = 20.0  # upper bound in volts (detection threshold)
LOSS_CONE_DETECTION_THRESHOLD = 20.0  # values within ±threshold are unreliable

# B_s/B_m bounds: avoid unrealistically low ratios that can cause degenerate fits.
LOSS_CONE_BS_OVER_BM_MIN = 0.3
LOSS_CONE_BS_OVER_BM_MAX = 1.1

# Beam parameters
# Beam width: fixed at ~15 eV (LP energy resolution), NOT scaling with |U_surface|
# The previous scaling (0.5 * |U|) caused runaway at extreme potentials.
LOSS_CONE_BEAM_WIDTH_EV = 15.0  # fixed beam width in eV (instrument resolution)
LOSS_CONE_BEAM_WIDTH_FACTOR = 0.5  # DEPRECATED: kept for backward compatibility
LOSS_CONE_BEAM_AMP_MIN = 0.0  # lower bound for normalized beam amplitude
LOSS_CONE_BEAM_AMP_MAX = 5.0  # upper bound (reduced from 100; data is normalized 0-1)
# Pitch sigma: reasonable angular spread for secondary electron beam
LOSS_CONE_BEAM_PITCH_SIGMA_DEG = 7.5  # spread toward 180° (upward beam)
LOSS_CONE_BACKGROUND = (
    0.05  # baseline model value outside loss cone (for log stability)
)


# ========== Data Column names ==========
UTC_COLUMN = "UTC"  # column name for UTC time in dataframes
TIME_COLUMN = "time"  # column name for time in dataframes
ENERGY_COLUMN = "energy"  # column name for energy in dataframes
SPEC_NO_COLUMN = "spec_no"  # column name for spectrum number in dataframes

MISC_COLS = [UTC_COLUMN, TIME_COLUMN, ENERGY_COLUMN, SPEC_NO_COLUMN]
FLUX_COLS = [f"ele_flux_{i}" for i in range(88)]
PHI_COLS = [f"dist_phi_{i}" for i in range(88)]
MAG_COLS = ["mag_x", "mag_y", "mag_z"]
ALL_COLS = MISC_COLS + MAG_COLS + FLUX_COLS + PHI_COLS  # Do not modify

COUNT_COLS = ["count", "count_err"]  # This is count data inferred from the flux data

# ========== Directory paths ==========
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # project root directory
DATA_DIR = PROJECT_ROOT / "data"  # main data directory
SPICE_KERNELS_DIR = PROJECT_ROOT / "data" / "spice_kernels"

# ========== File names ==========
THETA_FILE = "theta.tab"
SOLID_ANGLES_FILE = "solid_angles.tab"
ATTITUDE_FILE = "attitude.tab"
MOON_MAP_FILE = "moon_map.tif"
