"""
Utils package for Lunar Prospector analysis.

This package provides utilities for:
- File and directory operations
- SPICE ephemeris operations
- Coordinate transformations
- Attitude data handling
- Geometric calculations

For backward compatibility, commonly used functions are imported at package level.
"""

# Import commonly used functions for backward compatibility
from .attitude import get_current_ra_dec, get_time_range, load_attitude_data
from .coordinates import (
    build_scd_to_j2000,
    cartesian_to_lat_lon,
    lat_lon_to_cartesian,
    ra_dec_to_unit,
)
from .energy import make_relative_energy_bounds
from .file_ops import list_files, list_folder_files
from .flux_files import select_flux_day_file
from .geometry import (
    get_connection_and_polarity,
    get_connections_and_polarity_batch,
    get_intersection_or_none,
)
from .spice_ops import (
    get_j2000_iau_moon_transform_matrix,
    get_lp_position_wrt_moon,
    get_lp_vector_to_sun_in_lunar_frame,
    get_sun_vector_wrt_moon,
)
from .synthetic import (
    prepare_flux,
    prepare_phis,
    prepare_synthetic_er,
)

# Re-export for backward compatibility
__all__ = [
    "build_scd_to_j2000",
    "cartesian_to_lat_lon",
    "get_current_ra_dec",
    # Geometry
    "get_connection_and_polarity",
    "get_connections_and_polarity_batch",
    "get_intersection_or_none",
    "get_j2000_iau_moon_transform_matrix",
    # SPICE operations
    "get_lp_position_wrt_moon",
    "get_lp_vector_to_sun_in_lunar_frame",
    "get_sun_vector_wrt_moon",
    "get_time_range",
    "lat_lon_to_cartesian",
    # File operations
    "list_files",
    "list_folder_files",
    # Attitude operations
    "load_attitude_data",
    # Energy helpers
    "make_relative_energy_bounds",
    "prepare_flux",
    # Synthetic data helpers
    "prepare_phis",
    "prepare_synthetic_er",
    # Coordinate transformations
    "ra_dec_to_unit",
    # Flux file helpers
    "select_flux_day_file",
]
