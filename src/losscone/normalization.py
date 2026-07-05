"""Flux normalization helpers for loss-cone fitting.

Free functions extracted from ``LossConeFitter`` so both the CPU and torch
fitters can share a single normalization implementation.
"""

import numpy as np

from src import config
from src.losscone.er_data import ERData
from src.losscone.pitch_angle import PitchAngle
from src.losscone.types import NormalizationMode

__all__ = [
    "build_norm2d",
    "build_norm2d_batch",
    "get_normalized_flux",
]


def get_normalized_flux(
    er_data: ERData,
    pitch_angle: PitchAngle,
    incident_flux_stat: str,
    energy_bin: int,
    measurement_chunk: int,
) -> np.ndarray:
    """
    Get the normalized flux for a specific energy bin and measurement chunk.

    Args:
        er_data: ER data container.
        pitch_angle: Pre-computed pitch-angle geometry.
        incident_flux_stat: Statistic for incident flux ("mean" or "max").
        energy_bin (int): The index of the energy bin.
        measurement_chunk (int): The index of the measurement chunk.

    Returns:
        np.ndarray: The normalized flux for the specified energy bin and
            measurement chunk.
    """
    assert not er_data.data.empty, "Data not loaded. Please load the data first."

    index = measurement_chunk * config.SWEEP_ROWS + energy_bin

    if index >= len(er_data.data):
        return np.full(config.CHANNELS, np.nan)

    electron_flux = er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[index]
    if len(pitch_angle.pitch_angles) == 0:
        return np.full(config.CHANNELS, np.nan)
    angles = pitch_angle.pitch_angles[index]
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

    if incident_flux_stat == "mean":
        incident_flux = float(np.mean(incident_vals))
    else:
        incident_flux = float(np.max(incident_vals))

    incident_flux = max(config.EPS, incident_flux)
    return electron_flux / incident_flux


def build_norm2d(
    er_data: ERData,
    pitch_angle: PitchAngle,
    normalization_mode: NormalizationMode,
    incident_flux_stat: str,
    measurement_chunk: int,
) -> np.ndarray:
    """
    Build a 2D normalized flux distribution for a specific measurement chunk.

    Normalization modes:
    - 'ratio': per-energy normalization by incident flux
    - 'ratio2': pairwise normalization (incident→1.0, reflected→reflected/incident)

    Args:
        er_data: ER data container.
        pitch_angle: Pre-computed pitch-angle geometry.
        normalization_mode: Flux normalization strategy.
        incident_flux_stat: Statistic for incident flux ("mean" or "max").
        measurement_chunk (int): The index of the measurement chunk.

    Returns:
        np.ndarray: The 2D normalized flux distribution.
    """
    assert not er_data.data.empty, "Data not loaded. Please load the data first."

    if normalization_mode == NormalizationMode.RATIO:
        # Per-energy normalization: divide each energy by its own incident flux
        norm2d = np.vstack(
            [
                get_normalized_flux(
                    er_data,
                    pitch_angle,
                    incident_flux_stat,
                    energy_bin,
                    measurement_chunk,
                )
                for energy_bin in range(config.SWEEP_ROWS)
            ]
        )
    elif normalization_mode == NormalizationMode.RATIO2:
        # Pairwise normalization: mirror incident/reflected angles around 90°
        # Each reflected angle normalized by its closest mirrored incident angle
        s = measurement_chunk * config.SWEEP_ROWS
        e = min((measurement_chunk + 1) * config.SWEEP_ROWS, len(er_data.data))

        flux_2d = er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)[s:e]
        pitches_2d = pitch_angle.pitch_angles[s:e]

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

    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")

    return norm2d


def build_norm2d_batch(
    er_data: ERData,
    pitch_angle: PitchAngle,
    normalization_mode: NormalizationMode,
    incident_flux_stat: str,
    chunk_indices: list[int],
) -> np.ndarray:
    """
    Build normalized 2D flux distributions for multiple chunks at once.

    Vectorized implementation for significant speedup over calling
    build_norm2d() in a loop.

    Args:
        er_data: ER data container.
        pitch_angle: Pre-computed pitch-angle geometry.
        normalization_mode: Flux normalization strategy.
        incident_flux_stat: Statistic for incident flux ("mean" or "max").
        chunk_indices: List of measurement chunk indices to process

    Returns:
        np.ndarray: Shape (n_chunks, SWEEP_ROWS, CHANNELS) normalized flux.
                    Invalid chunks are filled with NaN.
    """
    if not chunk_indices:
        return np.zeros((0, config.SWEEP_ROWS, config.CHANNELS), dtype=np.float64)

    n_chunks = len(chunk_indices)
    n_rows = len(er_data.data)
    nE = config.SWEEP_ROWS
    nP = config.CHANNELS

    # Load all flux and pitch data once
    flux_all = er_data.data[config.FLUX_COLS].to_numpy(dtype=np.float64)
    pitches_all = pitch_angle.pitch_angles

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

    if normalization_mode == NormalizationMode.RATIO:
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

            if incident_flux_stat == "mean":
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

    elif normalization_mode == NormalizationMode.RATIO2:
        # ratio2 - complex pairwise normalization; fall back to per-chunk
        for i in valid_chunk_idx:
            result[i] = build_norm2d(
                er_data,
                pitch_angle,
                normalization_mode,
                incident_flux_stat,
                chunk_indices[i],
            )

    else:
        raise ValueError(f"Unsupported normalization_mode: {normalization_mode}")

    return result
