'''
Central configuration constants for the Lunar Prospector Plasma Analysis pipeline.
'''  

import scipy.constants

# ========== Data and chunk settings ==========  
SWEEP_ROWS = 15                 # rows per spacecraft sweep of energy spectrum  
CHANNELS = 88                  # number of ER electron flux channels  

# ========== Physical parameters ==========  
LUNAR_RADIUS_KM = 1737.4       # mean lunar radius in kilometers  
LUNAR_RADIUS_M = LUNAR_RADIUS_KM * 1e3  # mean lunar radius in meters  

ELECTRON_MASS_KG = scipy.constants.electron_mass  # electron
ELECTRON_CHARGE_C = scipy.constants.e  # elementary charge in Coulombs

# ========== File extensions ==========  
EXT_TAB = '.TAB'               # ER data file extension  
EXT_BSP = '.bsp'               # SPICE ephemeris kernel  
EXT_TLS = '.tls'               # SPICE leap seconds kernel  
EXT_TPC = '.tpc'               # SPICE planetary constants kernel  

# ========== Download manager settings ==========  
MAX_DOWNLOAD_WORKERS = 10      # default threads for parallel downloads  
CHUNK_SIZE_BYTES = 4 * 1024 * 1024  # chunk size for streaming downloads  
REQUESTS_PER_SECOND = 5        # rate-limit threshold  

# ========== Numerical tolerances ==========  
EPS = 1e-6                     # small epsilon to avoid division by zero

# ========== Data Column names ==========
MISC_COLS = ["UTC", "time", "energy", "spec_no", "mag_x", "mag_y", "mag_z"]
FLUX_COLS = [f"ele_flux_{i}" for i in range(88)]
PHI_COLS = [f"dist_phi_{i}" for i in range(88)]
ALL_COLS = MISC_COLS + FLUX_COLS + PHI_COLS
MAG_COLS = ["mag_x", "mag_y", "mag_z"]

# ========== Directory paths ==========
DATA_DIR = '../data'              # main data directory
KERNELS_DIR = '../spice_kernels'  # SPICE kernels directory

# ========== File names ==========
THETA_FILE = 'theta.tab'
ATTITUDE_FILE = 'attitude.tab'
MOON_MAP_FILE = 'moon_map.tif'