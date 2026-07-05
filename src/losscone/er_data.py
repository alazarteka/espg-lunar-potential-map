"""ER data container: loading, sweep cleaning, and count reconstruction."""

import logging

import numpy as np
import pandas as pd

from src import config
from src.utils.units import ureg

__all__ = ["ERData"]

logger = logging.getLogger(__name__)


class ERData:
    def __init__(self, er_data_file: str):
        """
        Initialize the ERData class with the path to the ER data file.
        """
        self.er_data_file = er_data_file
        self.data: pd.DataFrame = pd.DataFrame()

        self._load_data()
        self._add_count_columns()

    @classmethod
    def from_dataframe(cls, data: pd.DataFrame, er_data_file: str):
        """
        Create an ERData instance from existing DataFrame data.

        Args:
            data: The pandas DataFrame containing the ER data
            er_data_file: The original file path for reference

        Returns:
            ERData instance with the provided data
        """
        instance = cls.__new__(cls)
        instance.er_data_file = er_data_file
        instance.data = data
        instance._clean_sweep_data()
        instance._add_count_columns()
        return instance

    def _load_data(self) -> None:
        """
        Load the ER data from the specified file.

        Reads the specified file into a pandas DataFrame, using the column names
        defined in ALL_COLS. If the file is not found, or if there is an error
        parsing the file, the data attribute is set to None.
        """
        try:
            self.data = pd.read_csv(
                self.er_data_file,
                sep=" ",
                engine="c",
                skipinitialspace=True,
                header=None,
                names=config.ALL_COLS,
            )
            self._clean_sweep_data()
        except FileNotFoundError:
            logger.error(f"Error: The file {self.er_data_file} was not found.")
            self.data = pd.DataFrame()
        except pd.errors.ParserError:
            logger.error(
                f"Error: The file {self.er_data_file} could not be parsed. "
                "Please check the file format."
            )
            self.data = pd.DataFrame()
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            self.data = pd.DataFrame()

    def _clean_sweep_data(self) -> None:
        """
        Remove entire sweeps that contain any invalid rows.

        Identifies sweeps with invalid timestamps or magnetic field data,
        then removes all rows belonging to those spec_no values.
        """
        if self.data.empty:
            return

        original_rows = len(self.data)

        # Identify invalid rows
        magnetic_field = self.data[config.MAG_COLS].to_numpy(dtype=np.float64)
        magnetic_field_magnitude = np.linalg.norm(magnetic_field, axis=1)

        invalid_mag_mask = (magnetic_field_magnitude <= 1e-9) | (
            magnetic_field_magnitude >= 1e3
        )
        invalid_time_mask = self.data[config.TIME_COLUMN] == "1970-01-01T00:00:00"
        invalid_rows_mask = invalid_mag_mask | invalid_time_mask

        # Get spec_no values for invalid rows
        invalid_spec_nos = set(
            self.data[config.SPEC_NO_COLUMN][invalid_rows_mask].unique()
        )

        if invalid_spec_nos:
            logger.debug(f"Removing {len(invalid_spec_nos)} sweeps with invalid data")

            # Remove all rows belonging to invalid spec_nos
            valid_mask = ~self.data[config.SPEC_NO_COLUMN].isin(list(invalid_spec_nos))
            self.data = self.data[valid_mask].reset_index(drop=True)

            removed_rows = original_rows - len(self.data)
            logger.debug(
                ("Removed %d rows (%.1f%%) from %d invalid sweeps"),
                removed_rows,
                (removed_rows / original_rows * 100.0),
                len(invalid_spec_nos),
            )

    def _add_count_columns(self) -> None:
        """
        Reconstruct integer electron counts from the flux columns.

        Adds two new DataFrame blocks:
            - `count`: Integer electron counts for each energy bin.
            - `count_err`: Estimated error in the electron counts.
        """
        theta_path = config.DATA_DIR / config.THETA_FILE
        try:
            thetas = np.loadtxt(theta_path, dtype=np.float64)
        except OSError as exc:
            logger.warning(
                "Theta table %s unavailable (%s); skipping count reconstruction.",
                theta_path,
                exc,
            )
            return

        if self.data.empty:
            return

        F = self.data[config.FLUX_COLS].to_numpy(dtype=np.float64) * (
            ureg.particle
            / (ureg.centimeter**2 * ureg.second * ureg.steradian * ureg.electron_volt)
        )

        negative_flux_mask = F.magnitude < 0
        if np.any(negative_flux_mask):
            n_negative = np.sum(negative_flux_mask)
            total_values = negative_flux_mask.size
            logger.debug(
                ("Found %d negative flux values (%.2f%%) - clamping to zero"),
                n_negative,
                (n_negative / total_values * 100.0),
            )

            F = np.maximum(F, 0 * F.units)

        energies = self.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
        energies = energies[:, None] * ureg.electron_volt  # Reshape for broadcasting
        integration_time = (
            np.array([1 / config.BINS_BY_LATITUDE[x] for x in thetas])
            * config.ACCUMULATION_TIME
        )
        integration_time = integration_time[None, :]  # Reshape for broadcasting
        count_estimate = F * config.GEOMETRIC_FACTOR * energies * integration_time
        count_estimate = np.rint(count_estimate.to(ureg.particle).magnitude).astype(int)

        count_estimate_sum = count_estimate.sum(axis=1)
        if np.any(count_estimate_sum < 0):
            logger.debug(
                "Negative count sums encountered; clamping to zero before sqrt"
            )
        count_estimate_sum = np.clip(count_estimate_sum, 0, None)
        count_err = np.sqrt(count_estimate_sum.astype(np.float64, copy=False))

        count_df = pd.DataFrame(
            {config.COUNT_COLS[0]: count_estimate_sum, config.COUNT_COLS[1]: count_err}
        )

        self.data = pd.concat([self.data, count_df], axis=1)
