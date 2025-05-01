'''
Central configuration constants for the Lunar Prospector Plasma Analysis pipeline.
'''  

# ========== Data and chunk settings ==========  
SWEEP_ROWS = 15                 # rows per spacecraft sweep of energy spectrum  
CHANNELS = 88                  # number of ER electron flux channels  

# ========== Physical parameters ==========  
LUNAR_RADIUS_KM = 1737.4       # mean lunar radius in kilometers  
LUNAR_RADIUS_M = LUNAR_RADIUS_KM * 1e3  # mean lunar radius in meters  

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
