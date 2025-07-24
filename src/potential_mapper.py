import argparse
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import spiceypy as spice

from . import config
from .flux import FluxData
from .utils import *

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


@dataclass
class CoordinateArrays:
    """Container for coordinate transformation arrays."""

    lp_positions: np.ndarray
    lp_vectors_to_sun: np.ndarray
    ra_dec_cartesian: np.ndarray
    moon_vectors_to_sun: np.ndarray
    j2000_to_iau_moon_mats: np.ndarray
    scd_to_iau_moon_mats: np.ndarray


@dataclass
class PotentialResults:
    """Container for potential calculation results."""

    latitudes: np.ndarray
    longitudes: np.ndarray
    potentials: np.ndarray
    facing_sun: np.ndarray
    intersection_positions: np.ndarray
    lp_positions: np.ndarray


class SpiceManager:
    """Manages SPICE kernel loading and cleanup."""

    @staticmethod
    def load_spice_files(spice_directory: Union[str, Path]) -> None:
        """Load SPICE files from the specified directory."""
        spice_dir = Path(spice_directory)
        if not spice_dir.exists():
            raise FileNotFoundError(f"SPICE directory {spice_dir} not found")

        patterns = ["*.bsp", "*.tpc", "*.tls"]
        spice_files = []
        for pattern in patterns:
            spice_files.extend(spice_dir.glob(pattern))

        if not spice_files:
            raise FileNotFoundError(f"No SPICE files found in {spice_dir}")

        for spice_file in spice_files:
            try:
                logging.info(f"Loading SPICE file: {spice_file}")
                spice.furnsh(str(spice_file))
            except Exception as e:
                logging.error(f"Error loading SPICE file {spice_file}: {e}")
                raise


class DataLoader:
    """Handles data file discovery and loading."""

    MONTH_TO_NUM = {
        "JAN": "01",
        "FEB": "02",
        "MAR": "03",
        "APR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AUG": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12",
    }

    NUM_TO_MONTH = {v: k for k, v in MONTH_TO_NUM.items()}

    @staticmethod
    def discover_flux_files(
        data_directory: Union[str, Path],
        years: Optional[List[str]] = None,
        months: Optional[List[str]] = None,
    ) -> List[Path]:
        """Discover flux files in the data directory structure."""
        data_dir = Path(data_directory)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory {data_dir} not found")

        flux_files = []

        # Get year directories
        year_dirs = [d for d in data_dir.iterdir() if d.is_dir() and d.name.isdigit()]
        if years:
            year_dirs = [d for d in year_dirs if d.name in years]

        for year_dir in sorted(year_dirs):
            logging.info(f"Scanning year: {year_dir.name}")

            # Get month directories
            month_dirs = [d for d in year_dir.iterdir() if d.is_dir()]
            if months:
                months = [DataLoader.NUM_TO_MONTH[m] for m in months]
                month_dirs = [d for d in month_dirs if d.name[-3:] in months]

            for month_dir in sorted(month_dirs):
                logging.info(f"Scanning month: {month_dir.name}")
                tab_files = list(month_dir.glob("*.TAB"))
                flux_files.extend(tab_files)

        return sorted(flux_files)

    @staticmethod
    def load_attitude_files(
        data_directory: Union[str, Path],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load attitude data from the data directory."""
        data_dir = Path(data_directory)
        attitude_file = data_dir / config.ATTITUDE_FILE

        if not attitude_file.exists():
            raise FileNotFoundError(f"Attitude file {attitude_file} not found")

        logging.info(f"Loading attitude data from {attitude_file}")
        return load_attitude_data(str(attitude_file))

    @staticmethod
    def get_theta_file(data_directory: Union[str, Path]) -> str:
        """Get the theta file path."""
        data_dir = Path(data_directory)
        theta_file = data_dir / config.THETA_FILE

        if not theta_file.exists():
            raise FileNotFoundError(f"Theta file {theta_file} not found")

        return str(theta_file)


class CoordinateCalculator:
    """Handles coordinate transformations and calculations."""

    def __init__(self, et_spin: np.ndarray, ra_vals: np.ndarray, dec_vals: np.ndarray):
        self.et_spin = et_spin
        self.ra_vals = ra_vals
        self.dec_vals = dec_vals

    def calculate_coordinate_arrays(self, flux_data: FluxData) -> CoordinateArrays:
        """Calculate all coordinate transformation arrays for the flux data."""
        n_points = len(flux_data.data)

        # Initialize arrays
        lp_positions = np.zeros((n_points, 3))
        lp_vectors_to_sun = np.zeros((n_points, 3))
        ra_dec_cartesian = np.zeros((n_points, 3))
        moon_vectors_to_sun = np.zeros((n_points, 3))
        j2000_to_iau_moon_mats = np.zeros((n_points, 3, 3))

        logging.info("Calculating positions and vectors...")
        for i, utc_time in enumerate(flux_data.data["UTC"]):
            time = spice.str2et(utc_time)

            # Calculate positions and vectors
            lp_position = get_lp_position_wrt_moon(time)
            lp_vector_to_sun = get_lp_vector_to_sun_in_lunar_frame(time)
            ra, dec = get_current_ra_dec(
                time, self.et_spin, self.ra_vals, self.dec_vals
            )

            # Validate results
            if any(x is None for x in [ra, dec, lp_position, lp_vector_to_sun]):
                logging.warning(f"Invalid data at time {utc_time}, skipping...")
                continue

            # Store results
            lp_positions[i] = lp_position
            lp_vectors_to_sun[i] = lp_vector_to_sun
            ra_dec_cartesian[i] = ra_dec_to_unit(ra, dec)
            moon_vectors_to_sun[i] = get_sun_vector_wrt_moon(time)
            j2000_to_iau_moon_mats[i] = get_j2000_iau_moon_transform_matrix(time)

        # Calculate transformation matrices
        logging.info("Calculating transformation matrices...")
        unit_vectors_to_sun = lp_vectors_to_sun / np.linalg.norm(
            lp_vectors_to_sun, axis=1, keepdims=True
        )
        scd_to_j2000_mats = build_scd_to_j2000(ra_dec_cartesian, unit_vectors_to_sun)
        scd_to_iau_moon_mats = np.einsum(
            "nij,njk->nik", j2000_to_iau_moon_mats, scd_to_j2000_mats
        )

        return CoordinateArrays(
            lp_positions=lp_positions,
            lp_vectors_to_sun=lp_vectors_to_sun,
            ra_dec_cartesian=ra_dec_cartesian,
            moon_vectors_to_sun=moon_vectors_to_sun,
            j2000_to_iau_moon_mats=j2000_to_iau_moon_mats,
            scd_to_iau_moon_mats=scd_to_iau_moon_mats,
        )


class MagneticFieldProcessor:
    """Handles magnetic field processing and projection."""

    @staticmethod
    def project_magnetic_field(
        flux_data: FluxData, coord_arrays: CoordinateArrays
    ) -> np.ndarray:
        """Project magnetic field vectors to lunar coordinates."""
        logging.info("Projecting magnetic field...")

        magnetic_field = flux_data.data[config.MAG_COLS].to_numpy(dtype=np.float64)
        unit_magnetic_field = magnetic_field / np.linalg.norm(
            magnetic_field, axis=1, keepdims=True
        )
        projected_magnetic_field = np.einsum(
            "nij,nj->ni", coord_arrays.scd_to_iau_moon_mats, unit_magnetic_field
        )

        return projected_magnetic_field


class SurfaceIntersectionFinder:
    """Finds surface intersections along magnetic field lines."""

    @staticmethod
    def find_intersections(
        coord_arrays: CoordinateArrays,
        projected_magnetic_field: np.ndarray,
        potentials: np.ndarray,
    ) -> PotentialResults:
        """Find surface intersections and compile results."""
        logging.info("Finding surface intersections...")

        latitudes = []
        longitudes = []
        potential_values = []
        facing_sun = []
        intersection_positions = []

        for i, (position, mag_field, moon_vector_to_sun) in enumerate(
            zip(
                coord_arrays.lp_positions,
                projected_magnetic_field,
                coord_arrays.moon_vectors_to_sun,
                strict=False,
            )
        ):
            intersection = get_intersection_or_none(position, mag_field)

            if intersection is not None:
                chunk_index = i // config.SWEEP_ROWS
                if chunk_index < len(potentials):
                    latitude, longitude = cartesian_to_lat_lon(intersection)
                    latitudes.append(latitude)
                    longitudes.append(longitude)
                    facing_sun.append(moon_vector_to_sun)
                    potential_values.append(potentials[chunk_index][0])  # delta_U
                    intersection_positions.append(intersection)
            else:
                logging.debug(f"No intersection found for index {i}")

        return PotentialResults(
            latitudes=np.array(latitudes),
            longitudes=np.array(longitudes),
            potentials=np.array(potential_values),
            facing_sun=np.array(facing_sun),
            intersection_positions=np.array(intersection_positions),
            lp_positions=coord_arrays.lp_positions,
        )


class PotentialMapper:
    """Main class for potential mapping operations."""

    def __init__(
        self, data_directory: Union[str, Path], spice_directory: Union[str, Path]
    ):
        self.data_dir = Path(data_directory)
        self.spice_dir = Path(spice_directory)
        self.theta_file = None
        self.attitude_data = None

    def initialize(self):
        """Initialize SPICE kernels and load auxiliary data."""
        # Load SPICE files
        SpiceManager.load_spice_files(self.spice_dir)

        # Load attitude and theta files
        self.attitude_data = DataLoader.load_attitude_files(self.data_dir)
        self.theta_file = DataLoader.get_theta_file(self.data_dir)

        logging.info("Initialization complete")

    def process_flux_file(self, file_path: Union[str, Path]) -> PotentialResults:
        """Process a single flux file and return potential results."""
        logging.info(f"Processing file: {file_path}")

        # Load flux data
        flux_data = FluxData(str(file_path), self.theta_file)

        # Calculate coordinate transformations
        et_spin, ra_vals, dec_vals = self.attitude_data
        coord_calculator = CoordinateCalculator(et_spin, ra_vals, dec_vals)
        coord_arrays = coord_calculator.calculate_coordinate_arrays(flux_data)

        # Project magnetic field
        projected_magnetic_field = MagneticFieldProcessor.project_magnetic_field(
            flux_data, coord_arrays
        )

        # Calculate surface potentials
        logging.info("Calculating surface potential...")
        potentials = flux_data.fit_surface_potential()

        logging.info(f"Computed {len(potentials)} potential values")
        for potential_result in potentials:
            chunk_index = int(potential_result[-1])
            corresponding_time = flux_data.data["UTC"].iloc[
                chunk_index * config.SWEEP_ROWS
            ]
            logging.info(
                f"Potential {potential_result[0]:.5f} V at time {corresponding_time}"
            )

        # Compute kappa parameters
        from kappa_fitter import KappaFitter  # Import here to avoid circular dependency

        logging.info("Calculating kappa parameters...")
        er_data = flux_data.er_data
        for spec_no in er_data.data["spec_no"].unique():
            fitter = KappaFitter(er_data, spec_no)
            kappa_params = fitter.fit()
            logging.info(f"Kappa parameters for spec {spec_no}: {kappa_params}")

        # Find surface intersections
        results = SurfaceIntersectionFinder.find_intersections(
            coord_arrays, projected_magnetic_field, potentials
        )

        return results

    def create_potential_map(
        self, results_list: List[PotentialResults], output_path: Optional[str] = None
    ) -> None:
        """Create and display/save the potential map visualization."""
        if not results_list:
            logging.warning("No results to plot")
            return

        # Combine all results
        all_latitudes = np.concatenate([r.latitudes for r in results_list])
        all_longitudes = np.concatenate([r.longitudes for r in results_list])
        all_potentials = np.concatenate([r.potentials for r in results_list])
        all_facing_sun = np.concatenate([r.facing_sun for r in results_list])
        all_intersections = np.concatenate(
            [r.intersection_positions for r in results_list]
        )
        all_lp_positions = np.concatenate([r.lp_positions for r in results_list])

        if len(all_intersections) == 0:
            logging.warning("No valid surface intersections found; skipping plot")
            return

        # Normalize vectors
        all_facing_sun = all_facing_sun / np.linalg.norm(
            all_facing_sun, axis=1, keepdims=True
        )
        all_intersections = all_intersections / np.linalg.norm(
            all_intersections, axis=1, keepdims=True
        )

        # Filter for dayside intersections with finite potentials
        is_day = np.einsum("ij,ij->i", all_facing_sun, all_intersections) > 0
        finite_mask = np.isfinite(all_potentials) & is_day

        latitudes_plot = all_latitudes[finite_mask]
        longitudes_plot = all_longitudes[finite_mask]
        potentials_plot = all_potentials[finite_mask]

        if len(potentials_plot) == 0:
            logging.warning("No valid dayside potentials to plot")
            return

        # Create the plot
        plt.figure(figsize=(12, 6))
        norm = plt.Normalize(vmin=potentials_plot.min(), vmax=potentials_plot.max())
        sc = plt.scatter(
            longitudes_plot,
            latitudes_plot,
            c=potentials_plot,
            cmap="viridis",
            marker="o",
            norm=norm,
            s=1,
        )

        # Add LP trajectory
        # lp_lat = np.rad2deg(np.arcsin(all_lp_positions[:, 2] / np.linalg.norm(all_lp_positions, axis=1)))
        # lp_long = np.rad2deg(np.arctan2(all_lp_positions[:, 1], all_lp_positions[:, 0]))
        # step = 15
        # plt.plot(lp_long[::step], lp_lat[::step], 'r-', label='LP Path', linewidth=0.5)

        # Add moon map background
        moon_map_path = self.data_dir / config.MOON_MAP_FILE
        if moon_map_path.exists():
            img = plt.imread(str(moon_map_path))
            plt.imshow(img, extent=(-180, 180, -90, 90), aspect="equal", zorder=-1)

        # Formatting
        plt.xlabel("Longitude (degrees)")
        plt.ylabel("Latitude (degrees)")
        plt.title("Lunar Surface Potential Map")
        plt.colorbar(sc, label="Potential (V)")
        plt.legend()

        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches="tight")
            logging.info(f"Plot saved to {output_path}")
        else:
            plt.show()


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Lunar Prospector Surface Potential Mapper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in data directory
  python potential_mapper.py

  # Process specific files
  python potential_mapper.py --files data/1998/01/file1.TAB data/1998/01/file2.TAB

  # Process specific year and month
  python potential_mapper.py --years 1998 --months 01 02

  # Save output to file
  python potential_mapper.py --output potential_map.png
        """,
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=config.DATA_DIR,
        help="Path to data directory (default: ../data)",
    )

    parser.add_argument(
        "--spice-dir",
        type=str,
        default=config.KERNELS_DIR,
        help="Path to SPICE kernels directory (default: ../spice_kernels)",
    )

    parser.add_argument(
        "--files", nargs="+", type=str, help="Specific flux files to process"
    )

    parser.add_argument(
        "--years", nargs="+", type=str, help="Years to process (e.g., 1998 1999)"
    )

    parser.add_argument(
        "--months", nargs="+", type=str, help="Months to process (e.g., 01 02 03)"
    )

    parser.add_argument(
        "--output",
        type=str,
        help="Output file path for the plot (if not specified, plot is displayed)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()

    # Set logging level
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # Initialize mapper
    mapper = PotentialMapper(args.data_dir, args.spice_dir)
    mapper.initialize()

    # Determine which files to process
    if args.files:
        # Process specific files
        flux_files = [Path(f) for f in args.files]
        for file_path in flux_files:
            if not file_path.exists():
                raise FileNotFoundError(f"File {file_path} not found")
    else:
        # Discover files based on year/month filters
        flux_files = DataLoader.discover_flux_files(
            mapper.data_dir, years=args.years, months=args.months
        )

    if not flux_files:
        logging.error("No flux files found to process")
        return

    logging.info(f"Found {len(flux_files)} files to process")

    # Process all files
    all_results = []
    for file_path in flux_files:
        try:
            results = mapper.process_flux_file(file_path)
            all_results.append(results)
        except Exception as e:
            logging.error(f"Error processing {file_path}: {e}")
            continue

    if not all_results:
        logging.error("No files were successfully processed")
        return

    # Create the potential map
    mapper.create_potential_map(all_results, args.output)

    logging.info("Processing complete")


if __name__ == "__main__":
    main()
