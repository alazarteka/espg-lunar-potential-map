from flux import FluxData
from utils import *
from typing import Tuple

import numpy as np
import pandas as pd
import os # consider using pathlib instead
import spiceypy as spice
import matplotlib.pyplot as plt
import glob
import logging
import config


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_spice_files(spice_directory: str) -> None:
    """
    Load SPICE files from the specified directory.
    """
    spice_files =   glob.glob(os.path.join(spice_directory, '*.bsp')) + \
                    glob.glob(os.path.join(spice_directory, '*.tpc')) + \
                    glob.glob(os.path.join(spice_directory, '*.tls'))

    if len(spice_files) == 0:
        logging.error(f"No SPICE files found in {spice_directory}")
        return

    for spice_file in spice_files:
        try:
            logging.info(f"Loading SPICE file: {spice_file}")
            spice.furnsh(spice_file)
        except Exception as e:
            logging.error(f"Error loading SPICE file {spice_file}: {e}")


def load_attitude(data_directory: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    attitude_file = os.path.join(data_directory, 'attitude.tab')

    if not os.path.exists(attitude_file):
        logging.error(f"Attitude file {attitude_file} not found.")
        return

    logging.info(f"Loading attitude data from {attitude_file}")
    return load_attitude_data(attitude_file)

def load_theta_file(data_directory: str) -> str:
    theta_file = os.path.join(data_directory, 'theta.tab')

    if not os.path.exists(theta_file):
        logging.error(f"Theta file {theta_file} not found.")
        return

    return theta_file

def main():
    logging.info("Starting the potential mapper...")
    current_directory = os.getcwd()

    # Load SPICE files
    spice_directory = os.path.join(current_directory, 'spice_kernels')
    load_spice_files(spice_directory)

    # Load ER files
    data_directory = os.path.join(current_directory, 'data')

    attitude = load_attitude(data_directory)
    theta_file = load_theta_file(data_directory)

    if attitude is None or theta_file is None:
        logging.error("Failed to load attitude or theta file.")
        return

    et_spin, ra_vals, dec_vals = attitude

    years = sorted(list_folder_files(data_directory))
    if len(years) == 0:
        logging.error("No year directories found in the data directory.")
        return

    if not all([year.isdigit() for year in years]):
        logging.error("Year directories should only contain digits.")
        return

    # Loop through each year and month
    for year in years:
        logging.info(f"Processing year: {year}")
        year_directory = os.path.join(data_directory, year)
        months = sorted(list_folder_files(year_directory))


        latitudes = []
        longitudes = []
        potentials = []
        facing_sun = []
        intersection_cartesian_pos = []

        monthly_flux_data = []

        for month in months:
            logging.info(f"Processing month: {month}, {year}")
            month_directory = os.path.join(year_directory, month)
            files = sorted(glob.glob(os.path.join(month_directory, '*.TAB')))

            monthly_flux_data.extend(files)

        for selected_file_path in monthly_flux_data[74:75]: # 103:104
            logging.info(f"Selected file: {selected_file_path}")

            flux = FluxData(selected_file_path, theta_file)
            logging.info("Processing flux data...")

            lp_position_array = np.zeros((len(flux.data), 3))
            lp_vector_to_sun_array = np.zeros((len(flux.data), 3))
            ra_dec_cartesian_array = np.zeros((len(flux.data), 3))
            moon_vector_to_sun_array = np.zeros((len(flux.data), 3))
            j2000_to_iau_moon_mats = np.zeros((len(flux.data), 3, 3))

            logging.info("Calculating positions and vectors...")
            for i, UTC_time in enumerate(flux.data['UTC']):
                time = spice.str2et(UTC_time)

                lp_position = get_lp_position_wrt_moon(time)
                lp_vector_to_sun = get_lp_vector_to_sun_in_lunar_frame(time)
                ra, dec = get_current_ra_dec(time, et_spin, ra_vals, dec_vals)


                if ra is None or dec is None:
                    logging.error("RA/Dec could not be determined.")
                    continue

                if lp_position is None:
                    logging.error("LP position could not be determined.")
                    continue

                if lp_vector_to_sun is None:
                    logging.error("Vector to Sun could not be determined.")
                    continue

                lp_position_array[i] = lp_position # IAU Moon frame
                lp_vector_to_sun_array[i] = lp_vector_to_sun # IAU Moon frame
                ra_dec_cartesian_array[i] = ra_dec_to_unit(ra, dec) # J2000 frame
                moon_vector_to_sun_array[i] = get_sun_vector_wrt_moon(time) # IAU Moon frame
                j2000_to_iau_moon_mats[i] = get_j2000_iau_moon_transform_matrix(time) # J2000 to IAU Moon frame

            unit_vectors_to_sun = lp_vector_to_sun_array / np.linalg.norm(lp_vector_to_sun_array, axis=1, keepdims=True)

            logging.info("Calculating transformation matrices...")
            scd_to_j2000_mats = build_scd_to_j2000(ra_dec_cartesian_array, unit_vectors_to_sun)
            scd_to_iau_moon_mats = np.einsum("nij,njk->nik",
                                             j2000_to_iau_moon_mats,
                                            scd_to_j2000_mats
            )

            logging.info("Projecting magnetic field...")
            magnetic_field = flux.data[config.MAG_COLS].to_numpy(dtype=np.float64)
            unit_magnetic_field = magnetic_field / np.linalg.norm(magnetic_field, axis=1, keepdims=True)
            projected_magnetic_field = np.einsum("nij,nj->ni", scd_to_iau_moon_mats, unit_magnetic_field)

            logging.info("Calculating surface potential...")
            potential = flux.fit_surface_potential()

            logging.info(f"Potentials computed: {len(potential)}")
            for V in potential:
                corresponding_time = flux.data["UTC"][V[-1] * config.SWEEP_ROWS]
                logging.info(f"Potential {V[0]:.5f} V at time {corresponding_time}.") # potential to 5 s.d.

            for i, (position, projected_magnetic_field, moon_vector_to_sun) in enumerate(zip(lp_position_array, projected_magnetic_field, moon_vector_to_sun_array)):
                intersection = get_intersection_or_none(position, projected_magnetic_field)

                chunk_index = i // 15
                if intersection is not None:
                    latitude, longitude = cartesian_to_lat_lon(intersection)
                    latitudes.append(latitude)
                    longitudes.append(longitude)
                    facing_sun.append(moon_vector_to_sun)
                    potentials.append(potential[chunk_index][0])
                    intersection_cartesian_pos.append(intersection)
                else:
                    logging.debug(f"Intersection not found for index {i}.")


        potentials = np.array(potentials)
        latitudes = np.array(latitudes)
        longitudes = np.array(longitudes)

        facing_sun = np.array(facing_sun)
        facing_sun = facing_sun / np.linalg.norm(facing_sun, axis=1, keepdims=True)


        intersection_cartesian_pos = np.array(intersection_cartesian_pos)
        if len(intersection_cartesian_pos) == 0:
            logging.warning("No valid surface intersections this cycle; skipping plot.")
            continue
        intersection_cartesian_pos = intersection_cartesian_pos / np.linalg.norm(intersection_cartesian_pos, axis=1, keepdims=True)

        is_day = np.einsum("ij,ij->i", facing_sun, intersection_cartesian_pos) > 0

        finite_mask = np.isfinite(potentials) & is_day
        latitudes_finite = latitudes[finite_mask]
        longitudes_finite = longitudes[finite_mask]
        potentials_finite = potentials[finite_mask]
        norm = plt.Normalize(vmin=potentials_finite.min(), vmax=potentials_finite.max())
        plt.figure(figsize=(12, 6))  # Adjusted for a realistic map aspect ratio
        sc = plt.scatter(longitudes_finite, latitudes_finite, c=potentials_finite, cmap='viridis', marker='o', norm=norm, s=1)


        lp_lat = np.rad2deg(np.arcsin(lp_position_array[:, 2] / np.linalg.norm(lp_position_array, axis=1)))
        lp_long = np.rad2deg(np.arctan2(lp_position_array[:, 1], lp_position_array[:, 0]))
        step = 15
        lp_path_plot = plt.plot(lp_long[::step], lp_lat[::step], 'r-', label='LP Path', linewidth=0.5)


        moon_map_path = os.path.join(current_directory, 'data/moon_map.tif')
        img = plt.imread(moon_map_path)

        plt.imshow(img, extent=(-180, 180, -90, 90), aspect='equal', zorder=-1)  # Set aspect to 'equal'
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Surface Potential Map')
        plt.colorbar(sc, label='Potential')
        plt.show()






if __name__ == "__main__":
    main()
