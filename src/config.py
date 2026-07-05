"""
Central configuration constants for the Lunar Prospector Plasma Analysis pipeline.
"""

from pathlib import Path

import scipy.constants

from .utils.units import ChargeType, LengthType, MassType, ureg

# ========== Data and chunk settings ==========
SWEEP_ROWS = 15  # rows per spacecraft sweep of energy spectrum
CHANNELS = 88  # number of ER electron flux channels

ACCUMULATION_TIME = 2.5 * ureg.second  # accumulation time (half spin) for ER flux
# Geometric factor per the PDS-PPI LP ER archive: F = C / (Δt · G · E), so counts
# are C = F · G · E · Δt. This single number ALREADY folds in the per-channel
# solid angle AND the ~25% ΔE/E energy passband (that is why counts use center
# energy E, not ΔE). Do NOT reintroduce ΔΩ (solid_angles.tab) or a ΔE/E factor.
GEOMETRIC_FACTOR = (
    0.02 * ureg.centimeter**2 * ureg.steradian
)  # geometric factor (folds in ΔΩ and ΔE/E≈0.25)
# ER background count rate (~10 counts/s per the PDS descriptor); NOT subtracted
# in the archive, so it enters the Poisson expectation μ = model·κ + N_bg·Δt.
LOSS_CONE_BACKGROUND_RATE_CPS = 10.0
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
PROTON_MASS_MAGNITUDE: float = scipy.constants.proton_mass  # Unit kg
CM2_TO_M2: float = 1e-4  # cm^2 to m^2 conversion factor

# ========== File extensions ==========
EXT_TAB = ".TAB"  # ER data file extension
EXT_BSP = ".bsp"  # SPICE ephemeris kernel
EXT_TLS = ".tls"  # SPICE leap seconds kernel
EXT_TPC = ".tpc"  # SPICE planetary constants kernel

# ========== Download manager settings ==========
MAX_DOWNLOAD_WORKERS = 20  # threads for parallel downloads (override via --max-workers)
# Streaming write granularity. Also the resume granularity: on an interrupted
# download, bytes since the last full chunk are re-fetched. 1 MB keeps that
# waste small and peak memory low (~workers x chunk) without costing throughput,
# which is TCP/OS-bound well below this size. Data files are ~20-25 MB, kernels
# up to ~100 MB, so a large chunk would coarsen resume with no upside.
CHUNK_SIZE_BYTES = 1 * 1024 * 1024  # chunk size for streaming downloads
CONNECTION_POOL_SIZE = 50  # connection pool size for reuse
DOWNLOAD_MAX_RETRIES = 5  # transient-failure retries per request
# Exponential backoff between retries: sleep = backoff * (2 ** (retry - 1)).
# With 1.0s this yields 0s, 1s, 2s, 4s, 8s so we back off politely on throttling.
DOWNLOAD_BACKOFF_FACTOR = 1.0

# ========== Numerical tolerances ==========
EPS = 1e-6  # small epsilon to avoid division by zero
E_GAIN = 0.05  # typical gain uncertainty
E_G = 0.10  # typical geometric factor uncertainty
N_BG = 0.0  # background noise
ENERGY_WINDOW_WIDTH_RELATIVE = 0.5

# ========== Fitting parameters ==========
# See docs/archive/analysis/fitter_error_analysis.md for the threshold-decision discussion.
# Options considered: 215_000 (95th pct) or 657_000 (99th pct, current) for QC strictness.
FIT_ERROR_THRESHOLD = 657000  # chi-squared threshold for a good fit (99th percentile)
LOSS_CONE_LHS_SEED = 42  # ensures deterministic Latin hypercube sampling
LOSS_CONE_LHS_SAMPLES = 400  # phase-1 samples per chunk (CPU + torch)

# Differential Evolution settings (CPU + torch parity)
LOSS_CONE_DE_SEED = 42
LOSS_CONE_DE_POPSIZE = 50  # population size (torch); CPU converts to SciPy multiplier
LOSS_CONE_DE_MAXITER = 500
LOSS_CONE_DE_MUTATION = 0.5
LOSS_CONE_DE_CROSSOVER = 0.9
LOSS_CONE_DE_ATOL = 1e-3
LOSS_CONE_DE_TOL = 1e-3  # SciPy-only relative tolerance
LOSS_CONE_DE_POLISH = False  # SciPy-only local polishing step

# Loss-cone fit method ("halekas" log-chi2 or "lillis" masked linear chi2)
LOSS_CONE_FIT_METHOD = "halekas"

# Surface potential bounds (see Halekas 2008 Section 5.2)
# Electron reflectometry cannot reliably measure positive potentials because the
# photoelectron sheath (~meters) is much thinner than magnetic field scales.
# Detection threshold is ~20V; values within ±20V are "below threshold".
LOSS_CONE_U_SURFACE_MIN = -2000.0  # lower bound in volts (extreme plasma sheet)
LOSS_CONE_U_SURFACE_MAX = 20.0  # upper bound in volts (detection threshold)

# B_s/B_m bounds: avoid unrealistically low ratios that can cause degenerate fits.
LOSS_CONE_BS_OVER_BM_MIN = 0.3
LOSS_CONE_BS_OVER_BM_MAX = 1.1

# Beam parameters
# Beam width: fixed at ~15 eV (LP energy resolution), NOT scaling with |U_surface|
# The previous scaling (0.5 * |U|) caused runaway at extreme potentials.
LOSS_CONE_BEAM_WIDTH_EV = 15.0  # fixed beam width in eV (instrument resolution)
LOSS_CONE_BEAM_AMP_MIN = 0.0  # lower bound for normalized beam amplitude
LOSS_CONE_BEAM_AMP_MAX = 5.0  # upper bound (reduced from 100; data is normalized 0-1)
# Pitch sigma: reasonable angular spread for secondary electron beam
LOSS_CONE_BEAM_PITCH_SIGMA_DEG = 7.5  # spread toward 180° (upward beam)
LOSS_CONE_BACKGROUND = (
    0.05  # baseline model value outside loss cone (for log stability)
)
# Loss-cone boundary smoothing (degrees). Replaces the hard pitch-angle step at
# the cone edge with a sigmoid of scale tau, making the objective differentiable
# and the global fit robust (removes the step-vs-bin aliasing that destabilizes
# the U-width). Physically ~ the ER effective pitch-angle resolution: detector
# polar sectors span 22.5 deg and the beam angular width is 7.5 deg, so a few
# degrees is appropriate. tau = 0 recovers the exact hard step (kept for a
# hard-vs-smooth comparison).
LOSS_CONE_BOUNDARY_TAU_DEG = 5.0

# U-width identifiability, profile-likelihood estimator.
# For each sweep the U axis is scanned on a grid; at each U the objective is
# minimized over the nuisance parameters (bs/bm, beam_amp) on a sub-grid, giving
# a profiled chi2(U). The U-width is the extent of U where
# chi2_profile(U) - chi2_min <= LOSS_CONE_U_WIDTH_DELTA_CHI2. With Lillis' proper
# (masked linear) chi2, delta_chi2 = 1.0 is the standard 1-parameter 68% (1-sigma)
# interval. This replaces the earlier LHS max-min proxy (which depended on sample
# density and used an arbitrary 0.001 band); that proxy is still emitted as a
# side-by-side diagnostic.
LOSS_CONE_U_WIDTH_DELTA_CHI2 = 1.0  # 68% (1-sigma) for a 1-parameter profile
LOSS_CONE_U_WIDTH_N_U = 80  # U-grid points spanning the surface-potential bounds
LOSS_CONE_U_WIDTH_N_BS = 17  # bs/bm sub-grid points (nuisance minimization)
LOSS_CONE_U_WIDTH_N_BEAM = 9  # beam-amp sub-grid points (nuisance minimization)

# Lillis 2008 masked-fit thresholds (relative flux-based)
LILLIS_RELATIVE_FLUX_MIN = 0.07
LILLIS_RELATIVE_FLUX_MAX = 0.79
LILLIS_MIN_VALID_BINS = 5
LILLIS_CHI2_REDUCED_MAX = 2.0

# Cash / Poisson-deviance acceptance: reduced deviance (D / max(n_valid-3,1))
# above this rejects the fit. Expected ~1 for a good fit; a few is generous.
LOSS_CONE_DEVIANCE_REDUCED_MAX = 3.0


# ========== Data Column names ==========
UTC_COLUMN = "UTC"  # column name for UTC time in dataframes - unique per row
TIME_COLUMN = "time"  # column name for time in dataframes - seconds since epoch - unique per spec_no
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
