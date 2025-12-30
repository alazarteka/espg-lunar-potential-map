import logging

import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale
from tqdm import tqdm

from src import config
from src.model import synth_losscone, synth_losscone_batch
from src.utils.units import ureg

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


class PitchAngle:
    """
    Initialize the PitchAngle class with the ER data and theta values.

    Data rows with invalid B-field are retained; all such rows are flagged via
    valid_mask and their derived quantities are NaN. Down-stream algorithms
    must honor this mask.

    Attributes:
        er_data: The ER data object.
        thetas: The theta values in degrees.
        cartesian_coords: The Cartesian coordinates of the data points.
        pitch_angles: The pitch angles in degrees.
        unit_magnetic_field: The unit magnetic field vectors.
        valid_mask: A mask indicating valid data points.
    """

    def __init__(self, er_data: ERData, thetas: str):
        """
        Initialize the PitchAngle class with the ER data and theta values.

        Args:
            er_data (ERData): The ER data object.
            thetas (str): The path to the theta values file.
        """
        self.er_data = er_data
        self.thetas = np.loadtxt(
            thetas, dtype=np.float64
        )  # Expects theta values in degrees
        self.cartesian_coords = np.array([])
        self.pitch_angles = np.array([])
        self.unit_magnetic_field = np.array([])
        self.valid_mask = np.array([])

        self._process_data()

    def _get_cartesian_coords(self, phis: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """
        Convert spherical coordinates (phi, theta) to Cartesian coordinates (X, Y, Z).

        Args:
            phis (np.ndarray): The phi values in radians.
            thetas (np.ndarray): The theta values in radians.

        Returns:
            np.ndarray: The Cartesian coordinates (X, Y, Z).
        """
        X = np.cos(phis) * np.cos(thetas)
        Y = np.sin(phis) * np.cos(thetas)
        z_base = np.sin(thetas)
        Z = np.broadcast_to(z_base, X.shape)
        return np.stack((X, Y, Z), axis=-1)

    def _process_data(self) -> None:
        """
        Process the ER data to calculate the Cartesian coordinates and prepare
        the unit magnetic field vectors for pitch angle calculation.

        This function performs data validation and transformation from spherical
        to Cartesian coordinates. It also normalizes the magnetic field vectors
        and stores indices of valid and invalid data points.
        """

        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )
        assert len(self.thetas) == config.CHANNELS, (
            f"Theta values must match the number of channels {config.CHANNELS}."
        )

        # Convert spherical coordinates (phi, theta) to Cartesian coordinates (X, Y, Z)
        phis = np.deg2rad(self.er_data.data[config.PHI_COLS].to_numpy(dtype=np.float64))
        thetas = np.deg2rad(self.thetas)
        self.cartesian_coords = self._get_cartesian_coords(phis, thetas)

        magnetic_field = self.er_data.data[config.MAG_COLS].to_numpy(dtype=np.float64)
        magnetic_field_magnitude = np.linalg.norm(magnetic_field, axis=1, keepdims=True)
        # ER convention points +B roughly sunward; loss-cone tracing expects the
        # opposite orientation (toward the Moon), so flip the direction here.
        unit_magnetic_field = -magnetic_field / magnetic_field_magnitude
        unit_magnetic_field = np.tile(
            unit_magnetic_field[:, None, :], (1, config.CHANNELS, 1)
        )
        self.unit_magnetic_field = unit_magnetic_field

        self.calculate_pitch_angles()

    def calculate_pitch_angles(self) -> None:
        """
        Calculate the pitch angles based on the loaded ER data and theta values.

        The pitch angle is the angle between the magnetic field line and the
        radial direction. It is calculated as the arccosine of the dot product
        between the unit magnetic field vector and the radial direction vector.
        """
        # Check if data is loaded
        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )

        dot_product = np.einsum(
            "ijk,ijk->ij", self.unit_magnetic_field, self.cartesian_coords
        ).clip(-1, 1)
        pitch_angles = np.arccos(dot_product)
        pitch_angles = np.rad2deg(pitch_angles)

        self.pitch_angles = pitch_angles


class LossConeFitter:
    def __init__(
        self,
        er_data: ERData,
        thetas: str,
        pitch_angle: PitchAngle | None = None,
        spacecraft_potential: np.ndarray | None = None,
        normalization_mode: str = "ratio",
        beam_amp_fixed: float | None = None,
        incident_flux_stat: str = "mean",
        loss_cone_background: float | None = None,
    ):
        """
        Initialize the LossConeFitter class with the ER data and theta values.

        Args:
            er_data (ERData): The ER data object.
            thetas (str): The path to the theta values file.
            pitch_angle (PitchAngle, optional): Pre-computed pitch angle
                object. If None, creates a new one.
            spacecraft_potential (np.ndarray | None): Optional per-row spacecraft
                potential [V] aligned with `er_data.data`; used in synthetic model.
            normalization_mode (str): How to normalize flux for fitting.
                - "global": divide entire 2D array by max incident flux
                - "ratio": per-energy ratio of reflected/incident flux (default)
                - "ratio2": pairwise normalization (incident→1, reflected→reflected/incident)
                - "ratio_rescaled": per-energy ratio, then rescale to [0, 1]
            beam_amp_fixed (float | None): If set, fix the Gaussian beam amplitude
                to this value instead of fitting it.
            incident_flux_stat (str): Statistic for incident flux normalization
                ("mean" or "max").
            loss_cone_background (float | None): Baseline model value outside the
                loss cone to stabilise log-space χ² (defaults to config value).
        """
        self.er_data = er_data
        self.thetas = np.loadtxt(thetas, dtype=np.float64)
        self.pitch_angle = (
            pitch_angle if pitch_angle is not None else PitchAngle(er_data, thetas)
        )
        self.spacecraft_potential = spacecraft_potential

        self.beam_width_factor = config.LOSS_CONE_BEAM_WIDTH_FACTOR
        self.beam_amp_min = config.LOSS_CONE_BEAM_AMP_MIN
        self.beam_amp_max = config.LOSS_CONE_BEAM_AMP_MAX
        if beam_amp_fixed is not None:
            self.beam_amp_min = beam_amp_fixed
            self.beam_amp_max = beam_amp_fixed
        self.beam_pitch_sigma_deg = config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG

        if normalization_mode not in {"global", "ratio", "ratio2", "ratio_rescaled"}:
            raise ValueError(f"Unknown normalization_mode: {normalization_mode}")
        self.normalization_mode = normalization_mode
        if incident_flux_stat not in {"mean", "max"}:
            raise ValueError(f"Unknown incident_flux_stat: {incident_flux_stat}")
        self.incident_flux_stat = incident_flux_stat
        if loss_cone_background is None:
            loss_cone_background = config.LOSS_CONE_BACKGROUND
        if loss_cone_background <= 0:
            raise ValueError("loss_cone_background must be positive")
        self.background = float(loss_cone_background)

        self.lhs = self._generate_latin_hypercube()

    def _generate_latin_hypercube(self) -> np.ndarray:
        """
        Generate a Latin Hypercube sample.

        Returns:
            np.ndarray: The Latin Hypercube sample.
        """
        # Generate a Latin Hypercube sample across U_surface, B_s/B_m, and beam amplitude
        lower_bounds = np.array([-1000.0, 0.1, self.beam_amp_min], dtype=float)
        upper_bounds = np.array([1000.0, 1.1, self.beam_amp_max], dtype=float)
        if upper_bounds[2] <= lower_bounds[2]:
            upper_bounds[2] = lower_bounds[2] + 1e-12
        sampler = LatinHypercube(
            d=len(lower_bounds), scramble=False, seed=config.LOSS_CONE_LHS_SEED
        )
        lhs = sampler.random(n=400)  # 400 points

        scaled = scale(lhs, lower_bounds, upper_bounds)
        if self.beam_amp_max <= self.beam_amp_min:
            scaled[:, 2] = self.beam_amp_min
        return scaled

    def _get_normalized_flux(
        self, energy_bin: int, measurement_chunk: int
    ) -> np.ndarray:
        """
        Get the normalized flux for a specific energy bin and measurement chunk.

        Args:
            energy_bin (int): The index of the energy bin.
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            np.ndarray: The normalized flux for the specified energy bin and
                measurement chunk.
        """
        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )

        index = measurement_chunk * config.SWEEP_ROWS + energy_bin

        if index >= len(self.er_data.data):
            return np.full(config.CHANNELS, np.nan)

        electron_flux = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[
            index
        ]
        if len(self.pitch_angle.pitch_angles) == 0:
            return np.full(config.CHANNELS, np.nan)
        angles = self.pitch_angle.pitch_angles[index]
        incident_mask = angles < 90
        # TODO: Check reconsider the reflected mask
        # reflected_mask = ~incident_mask

        # Check if the electron flux is valid
        if not incident_mask.any():
            return np.full_like(electron_flux, np.nan)

        incident_vals = electron_flux[incident_mask]
        incident_vals = incident_vals[np.isfinite(incident_vals)]
        incident_vals = incident_vals[incident_vals > 0]
        if len(incident_vals) == 0:
            return np.full_like(electron_flux, np.nan)

        if self.incident_flux_stat == "mean":
            incident_flux = float(np.mean(incident_vals))
        else:
            incident_flux = float(np.max(incident_vals))

        incident_flux = max(config.EPS, incident_flux)
        return electron_flux / incident_flux

    def build_norm2d(self, measurement_chunk: int) -> np.ndarray:
        """
        Build a 2D normalized flux distribution for a specific measurement chunk.

        Normalization modes:
        - 'global': divide entire 2D array by max incident flux
        - 'ratio': divide each energy by its own mean incident flux
        - 'ratio2': pairwise normalization (incident→1.0, reflected→reflected/incident)
        - 'ratio_rescaled': per-energy ratio, then rescale to [0, 1]

        Args:
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            np.ndarray: The 2D normalized flux distribution.
        """
        assert not self.er_data.data.empty, (
            "Data not loaded. Please load the data first."
        )

        if self.normalization_mode == "ratio":
            # Per-energy normalization: divide each energy by its own incident flux
            norm2d = np.vstack(
                [
                    self._get_normalized_flux(energy_bin, measurement_chunk)
                    for energy_bin in range(config.SWEEP_ROWS)
                ]
            )
        elif self.normalization_mode == "ratio2":
            # Pairwise normalization: mirror incident/reflected angles around 90°
            # Each reflected angle normalized by its closest mirrored incident angle
            s = measurement_chunk * config.SWEEP_ROWS
            e = min((measurement_chunk + 1) * config.SWEEP_ROWS, len(self.er_data.data))

            flux_2d = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[
                s:e
            ]
            pitches_2d = self.pitch_angle.pitch_angles[s:e]

            nE, nPitch = flux_2d.shape
            norm2d = np.full((nE, nPitch), np.nan, dtype=np.float64)

            for row in range(nE):
                pitch_row = pitches_2d[row]
                flux_row = flux_2d[row]

                incident_mask = pitch_row < 90.0
                reflected_mask = ~incident_mask

                incident_idx = np.nonzero(incident_mask)[0]
                reflected_idx = np.nonzero(reflected_mask)[0]

                if len(incident_idx) == 0 or len(reflected_idx) == 0:
                    continue

                incident_flux = flux_row[incident_idx]
                valid_incident = (incident_flux > 0) & np.isfinite(incident_flux)
                if not valid_incident.any():
                    continue

                valid_inc_indices = incident_idx[valid_incident]
                norm2d[row, valid_inc_indices] = 1.0

                for i_ref in reflected_idx:
                    ref_flux = flux_row[i_ref]
                    if ref_flux <= 0 or not np.isfinite(ref_flux):
                        continue

                    target_angle = 180.0 - pitch_row[i_ref]
                    mirror_idx = valid_inc_indices[
                        np.argmin(np.abs(pitch_row[valid_inc_indices] - target_angle))
                    ]
                    denom = flux_row[mirror_idx]
                    if denom <= 0 or not np.isfinite(denom):
                        continue

                    norm2d[row, i_ref] = ref_flux / denom

                # Ensure the bin closest to 90° is defined
                mid = int(np.argmin(np.abs(pitch_row - 90.0)))
                norm2d[row, mid] = 1.0

        elif self.normalization_mode == "ratio_rescaled":
            # Two-step hybrid: per-energy ratio, then global rescale to [0, 1]
            # Step 1: Per-energy normalization (same as "ratio")
            norm2d = np.vstack(
                [
                    self._get_normalized_flux(energy_bin, measurement_chunk)
                    for energy_bin in range(config.SWEEP_ROWS)
                ]
            )

            # Step 2: Global rescale to [0, 1]
            # Find max across all finite values
            finite_vals = norm2d[np.isfinite(norm2d)]
            if len(finite_vals) > 0:
                global_max = np.max(finite_vals)
                if global_max > 0:
                    norm2d = norm2d / global_max
            # Now norm2d is in [0, 1] while preserving ratio structure

        else:  # "global"
            # Global normalization: divide entire 2D array by maximum incident flux
            # First build the 2D flux array
            s = measurement_chunk * config.SWEEP_ROWS
            e = min((measurement_chunk + 1) * config.SWEEP_ROWS, len(self.er_data.data))

            flux_2d = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[
                s:e
            ]
            pitches_2d = self.pitch_angle.pitch_angles[s:e]

            # Find maximum incident flux across all energies
            incident_mask = pitches_2d < 90
            incident_flux_vals = flux_2d[incident_mask]
            incident_flux_vals = incident_flux_vals[
                incident_flux_vals > 0
            ]  # Remove zeros/negatives

            if len(incident_flux_vals) == 0:
                return np.full((config.SWEEP_ROWS, config.CHANNELS), np.nan)

            global_incident_flux = float(np.max(incident_flux_vals))
            norm2d = flux_2d / max(global_incident_flux, config.EPS)

        return norm2d

    def build_norm2d_batch(self, chunk_indices: list[int]) -> np.ndarray:
        """
        Build normalized 2D flux distributions for multiple chunks at once.

        Vectorized implementation for significant speedup over calling
        build_norm2d() in a loop.

        Args:
            chunk_indices: List of measurement chunk indices to process

        Returns:
            np.ndarray: Shape (n_chunks, SWEEP_ROWS, CHANNELS) normalized flux.
                        Invalid chunks are filled with NaN.
        """
        if not chunk_indices:
            return np.zeros((0, config.SWEEP_ROWS, config.CHANNELS), dtype=np.float64)

        n_chunks = len(chunk_indices)
        n_rows = len(self.er_data.data)
        nE = config.SWEEP_ROWS
        nP = config.CHANNELS

        # Load all flux and pitch data once
        flux_all = self.er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)
        pitches_all = self.pitch_angle.pitch_angles

        # Build index arrays for all chunks
        chunk_indices_arr = np.array(chunk_indices, dtype=np.int64)
        start_indices = chunk_indices_arr * nE
        valid_chunks = start_indices < n_rows

        # Pre-allocate output
        result = np.full((n_chunks, nE, nP), np.nan, dtype=np.float64)

        if not valid_chunks.any():
            return result

        # Get valid chunk data
        valid_chunk_idx = np.where(valid_chunks)[0]

        if self.normalization_mode == "ratio":
            # Fully vectorized per-energy normalization
            for i in valid_chunk_idx:
                chunk_idx = chunk_indices[i]
                s = chunk_idx * nE
                e = min(s + nE, n_rows)
                actual_rows = e - s

                flux_chunk = flux_all[s:e]  # (actual_rows, nP)
                pitch_chunk = pitches_all[s:e]  # (actual_rows, nP)

                # Incident mask per row
                incident_mask = pitch_chunk < 90.0  # (actual_rows, nP)

                # Valid flux mask
                valid_flux = np.isfinite(flux_chunk) & (flux_chunk > 0)

                # Combined mask for valid incident flux
                valid_incident = incident_mask & valid_flux  # (actual_rows, nP)

                # Compute normalization factor per row using masked operations
                # Replace non-incident values with NaN for aggregation
                flux_for_norm = np.where(valid_incident, flux_chunk, np.nan)

                if self.incident_flux_stat == "mean":
                    # nanmean per row
                    norm_factors = np.nanmean(flux_for_norm, axis=1)  # (actual_rows,)
                else:
                    # nanmax per row
                    norm_factors = np.nanmax(flux_for_norm, axis=1)  # (actual_rows,)

                # Handle rows with no valid incident flux
                norm_factors = np.where(
                    np.isfinite(norm_factors) & (norm_factors > 0), norm_factors, np.nan
                )
                norm_factors = np.maximum(norm_factors, config.EPS)

                # Normalize: flux / norm_factor (broadcast over columns)
                result[i, :actual_rows, :] = flux_chunk / norm_factors[:, np.newaxis]

        elif self.normalization_mode == "global":
            # Vectorized global normalization
            for i in valid_chunk_idx:
                chunk_idx = chunk_indices[i]
                s = chunk_idx * nE
                e = min(s + nE, n_rows)
                actual_rows = e - s

                flux_chunk = flux_all[s:e]
                pitch_chunk = pitches_all[s:e]

                # Incident mask
                incident_mask = pitch_chunk < 90.0
                valid_flux = np.isfinite(flux_chunk) & (flux_chunk > 0)
                valid_incident = incident_mask & valid_flux

                # Get max incident flux
                incident_vals = flux_chunk[valid_incident]
                if len(incident_vals) == 0:
                    continue

                global_norm = np.max(incident_vals)
                result[i, :actual_rows, :] = flux_chunk / max(global_norm, config.EPS)

        elif self.normalization_mode == "ratio_rescaled":
            # Per-energy ratio then global rescale
            for i in valid_chunk_idx:
                chunk_idx = chunk_indices[i]
                s = chunk_idx * nE
                e = min(s + nE, n_rows)
                actual_rows = e - s

                flux_chunk = flux_all[s:e]
                pitch_chunk = pitches_all[s:e]

                incident_mask = pitch_chunk < 90.0
                valid_flux = np.isfinite(flux_chunk) & (flux_chunk > 0)
                valid_incident = incident_mask & valid_flux

                flux_for_norm = np.where(valid_incident, flux_chunk, np.nan)

                if self.incident_flux_stat == "mean":
                    norm_factors = np.nanmean(flux_for_norm, axis=1)
                else:
                    norm_factors = np.nanmax(flux_for_norm, axis=1)

                norm_factors = np.where(
                    np.isfinite(norm_factors) & (norm_factors > 0), norm_factors, np.nan
                )
                norm_factors = np.maximum(norm_factors, config.EPS)

                chunk_result = flux_chunk / norm_factors[:, np.newaxis]

                # Global rescale
                finite_vals = chunk_result[np.isfinite(chunk_result)]
                if len(finite_vals) > 0:
                    global_max = np.max(finite_vals)
                    if global_max > 0:
                        chunk_result = chunk_result / global_max

                result[i, :actual_rows, :] = chunk_result

        else:  # ratio2 - complex pairwise normalization
            # Fall back to per-chunk for ratio2
            for i in valid_chunk_idx:
                result[i] = self.build_norm2d(chunk_indices[i])

        return result

    def _fit_surface_potential(
        self, measurement_chunk: int
    ) -> tuple[float, float, float]:
        """
        Fit surface potential (U_surface) and B_s/B_m for one 15-row measurement chunk
        using χ² minimisation with scipy.optimize.minimize.

        Args:
            measurement_chunk (int): The index of the measurement chunk.

        Returns:
            tuple[float, float, float]:
                - U_surface: best-fit surface potential in volts
                - bs_over_bm: best-fit B_s/B_m ratio
                - beam_amp: best-fit Gaussian beam amplitude
                - chi2: final χ² value
        """
        assert not self.er_data.data.empty, "Data not loaded."

        eps = 1e-6
        norm2d = self.build_norm2d(measurement_chunk)

        if np.isnan(norm2d).all():
            return np.nan, np.nan, np.nan, np.nan

        s = measurement_chunk * config.SWEEP_ROWS
        e = (measurement_chunk + 1) * config.SWEEP_ROWS

        max_rows = len(self.er_data.data)
        if s >= max_rows:
            return np.nan, np.nan, np.nan, np.nan
        e = min(e, max_rows)

        energies = self.er_data.data[config.ENERGY_COLUMN].to_numpy(dtype=np.float64)[
            s:e
        ]

        if self.pitch_angle.pitch_angles is None or s >= len(
            self.pitch_angle.pitch_angles
        ):
            return np.nan, np.nan, np.nan, np.nan
        pitches = self.pitch_angle.pitch_angles[s:e]
        spacecraft_slice = (
            self.spacecraft_potential[s:e]
            if self.spacecraft_potential is not None
            else 0.0
        )

        actual_rows = e - s
        if norm2d.shape[0] > actual_rows:
            norm2d = norm2d[:actual_rows]

        data_mask = np.isfinite(norm2d) & (norm2d > 0)
        if not data_mask.any():
            return np.nan, np.nan, np.nan, np.nan
        log_data = np.zeros_like(norm2d, dtype=float)
        log_data[data_mask] = np.log(norm2d[data_mask] + eps)
        data_mask_3d = data_mask[None, :, :]

        # self.lhs is (N_samples, 3) -> [U_surface, bs_over_bm, beam_amp]
        lhs_U_surface = self.lhs[:, 0]
        lhs_bs_over_bm = self.lhs[:, 1]
        lhs_beam_amp = self.lhs[:, 2]

        # Calculate beam widths for all samples
        # beam_width = max(abs(U_surface) * factor, EPS)
        lhs_beam_width = np.maximum(
            np.abs(lhs_U_surface) * self.beam_width_factor, config.EPS
        )

        # Evaluate models in batch: (N_samples, nE, nPitch)
        models = synth_losscone_batch(
            energy_grid=energies,
            pitch_grid=pitches,
            U_surface=lhs_U_surface,
            U_spacecraft=spacecraft_slice,
            bs_over_bm=lhs_bs_over_bm,
            beam_width_eV=lhs_beam_width,
            beam_amp=lhs_beam_amp,
            beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
            background=self.background,
        )

        log_models = np.log(models + eps)

        diff = (log_data[None, :, :] - log_models) * data_mask_3d

        # Sum over energy and pitch axes (1, 2)
        chi2_vals = np.sum(diff**2, axis=(1, 2))

        bad_mask = ~np.isfinite(chi2_vals)
        chi2_vals[bad_mask] = 1e30

        best_idx = int(np.argmin(chi2_vals))
        best_lhs_chi2 = chi2_vals[best_idx]
        x0 = self.lhs[best_idx]

        # 2) Global optimization with differential_evolution
        # Objective for optimizer (scalar)
        def chi2_scalar(params):
            U_surface, bs_over_bm, beam_amp = params
            beam_amp = float(np.clip(beam_amp, self.beam_amp_min, self.beam_amp_max))
            beam_width = max(abs(U_surface) * self.beam_width_factor, config.EPS)
            model = synth_losscone(
                energy_grid=energies,
                pitch_grid=pitches,
                U_surface=U_surface,
                U_spacecraft=spacecraft_slice,
                bs_over_bm=bs_over_bm,
                beam_width_eV=beam_width,
                beam_amp=beam_amp,
                beam_pitch_sigma_deg=self.beam_pitch_sigma_deg,
                background=self.background,
            )

            if not np.all(np.isfinite(model)) or (model <= 0).all():
                return 1e30  # big penalty

            log_model = np.log(model + eps)
            diff = (log_data - log_model) * data_mask
            chi2 = np.sum(diff * diff)

            if not np.isfinite(chi2):
                return 1e30
            return chi2

        # Use differential_evolution (global optimizer with bounds)
        # More robust than LHS + Nelder-Mead for avoiding local minima
        from scipy.optimize import differential_evolution

        bounds = [
            (-2000.0, 2000.0),  # U_surface
            (0.1, 1.1),  # bs_over_bm
            (self.beam_amp_min, self.beam_amp_max),  # beam_amp
        ]

        result = differential_evolution(
            chi2_scalar,
            bounds,
            seed=42,
            maxiter=1000,
            atol=1e-3,
            tol=1e-3,
            workers=1,  # Single-threaded for reproducibility
            updating="deferred",  # Faster convergence
        )

        if not result.success:
            # Fallback to best LHS if optimization fails
            return float(x0[0]), float(x0[1]), float(x0[2]), float(best_lhs_chi2)

        U_surface, bs_over_bm, beam_amp = result.x

        # Clip to ensure exact bounds (DE should respect them, but be safe)
        bs_over_bm = float(np.clip(bs_over_bm, 0.1, 1.1))
        beam_amp = float(np.clip(beam_amp, self.beam_amp_min, self.beam_amp_max))

        return float(U_surface), bs_over_bm, beam_amp, float(result.fun)

    def fit_surface_potential(self) -> np.ndarray:
        """
        Fit surface potential (U_surface) and B_s/B_m for all 15-row measurement chunks
        using χ² minimisation with scipy.optimize.minimize.

        Returns:
            np.ndarray: Array with columns [U_surface, bs_over_bm, beam_amp, chi2, chunk_index]
                - U_surface: best-fit surface potential in volts
                - bs_over_bm: best-fit B_s/B_m ratio
                - beam_amp: best-fit Gaussian beam amplitude
                - chi2: final χ² value
                - chunk_index: measurement chunk index
        """
        assert not self.er_data.data.empty, "Data not loaded."

        # Fit for each chunk independently (no warm-starting)
        n_chunks = len(self.er_data.data) // config.SWEEP_ROWS
        results = np.zeros((n_chunks, 5))

        for i in tqdm(
            range(n_chunks), desc="Fitting chunks", unit="chunk", dynamic_ncols=True
        ):
            U_surface, bs_over_bm, beam_amp, chi2 = self._fit_surface_potential(i)
            results[i] = [U_surface, bs_over_bm, beam_amp, chi2, i]

        return results


class FluxData:
    def __init__(self, er_data_file: str, thetas: str):
        """
        Initialize the FluxData class as an orchestrator using the new class structure.

        Args:
            er_data_file (str): Path to the ER data file
            thetas (str): Path to the theta values file
        """
        # Use the new class structure
        self.er_data = ERData(er_data_file)
        self.pitch_angle = PitchAngle(self.er_data, thetas)
        self.loss_cone_fitter = LossConeFitter(self.er_data, thetas, self.pitch_angle)

        # Expose data for backward compatibility
        self.data = self.er_data.data

    def get_normalized_flux(
        self, energy_bin: int, measurement_chunk: int
    ) -> np.ndarray:
        """
        Get the normalized flux for a specific energy bin and measurement chunk.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter._get_normalized_flux(energy_bin, measurement_chunk)

    def build_norm2d(self, measurement_chunk: int):
        """
        Build a 2D normalized flux distribution for a specific measurement chunk.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter.build_norm2d(measurement_chunk)

    def _fit_surface_potential(self, measurement_chunk: int):
        """
        Fit surface potential for one measurement chunk.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter._fit_surface_potential(measurement_chunk)

    def fit_surface_potential(self):
        """
        Fit surface potential for all measurement chunks.
        Delegates to LossConeFitter.
        """
        return self.loss_cone_fitter.fit_surface_potential()
