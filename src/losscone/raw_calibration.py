"""Forensic association of Level-0 ER codes with calibrated 3-D sweeps.

This module ranks time-near raw/code and calibrated/flux candidates using only
response-independent angular-shape evidence. It does not reconstruct detector
counts or calibrated flux and must not be used as a likelihood input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src import config
from src.losscone.level0 import Level0Er3DSweep, decompress_er_counter

if TYPE_CHECKING:
    from datetime import datetime
    from pathlib import Path

CALIBRATED_MISSING_FLUX = 9.999e-99


@dataclass(frozen=True)
class CalibratedEr3DSweep:
    """One unmodified 15-row calibrated ER sweep, including its time span."""

    spec_no: int
    start_time: datetime
    end_time: datetime
    energies_ev: np.ndarray
    flux: np.ndarray


@dataclass(frozen=True)
class RawCalibratedCandidate:
    """A coarse raw/calibrated association candidate and scale-free score."""

    raw_first_record_index: int
    raw_earth_receive_time: datetime
    calibrated_spec_no: int
    calibrated_start_time: datetime
    calibrated_end_time: datetime
    receive_minus_end_seconds: float
    angular_log_correlation: float
    positive_cell_fraction: float


def read_calibrated_er3d_sweeps(path: Path) -> list[CalibratedEr3DSweep]:
    """Read complete PDS 3-D sweeps without cleaning or pseudo-count creation."""
    data = pd.read_csv(
        path,
        sep=" ",
        engine="c",
        skipinitialspace=True,
        header=None,
        names=config.ALL_COLS,
    )
    sweeps: list[CalibratedEr3DSweep] = []
    for spec_no, rows in data.groupby(config.SPEC_NO_COLUMN, sort=True):
        if len(rows) != config.SWEEP_ROWS:
            continue
        timestamps = pd.to_datetime(rows[config.UTC_COLUMN], utc=True)
        sweeps.append(
            CalibratedEr3DSweep(
                spec_no=int(spec_no),
                start_time=timestamps.iloc[0].to_pydatetime(),
                end_time=timestamps.iloc[-1].to_pydatetime(),
                energies_ev=rows[config.ENERGY_COLUMN].to_numpy(dtype=np.float64),
                flux=rows[config.FLUX_COLS].to_numpy(dtype=np.float64),
            )
        )
    return sweeps


def angular_log_correlation(
    compressed_codes: np.ndarray,
    calibrated_flux: np.ndarray,
) -> tuple[float, float]:
    """Return a response-independent per-energy angular-shape agreement score.

    Each energy row is separately centred and scaled before concatenation, so
    the score cannot determine an absolute response or an energy-dependent
    exposure. Values are suitable only for ranking time-near associations.
    """
    decoded = decompress_er_counter(compressed_codes).astype(np.float64)
    flux = np.asarray(calibrated_flux, dtype=np.float64)
    if decoded.shape != flux.shape:
        raise ValueError(
            "Raw code and calibrated flux shapes differ: "
            f"{decoded.shape} versus {flux.shape}"
        )

    raw_terms: list[np.ndarray] = []
    flux_terms: list[np.ndarray] = []
    n_positive = 0
    n_possible = 0
    for raw_row, flux_row in zip(decoded, flux, strict=True):
        # The PDS missing sentinel must not become a very small valid flux.
        valid = np.isfinite(flux_row) & (flux_row > CALIBRATED_MISSING_FLUX * 10)
        valid &= raw_row > 0
        if valid.sum() < 3:
            continue
        raw_log = np.log(raw_row[valid])
        flux_log = np.log(flux_row[valid])
        raw_std = raw_log.std()
        flux_std = flux_log.std()
        if raw_std == 0 or flux_std == 0:
            continue
        raw_terms.append((raw_log - raw_log.mean()) / raw_std)
        flux_terms.append((flux_log - flux_log.mean()) / flux_std)
        n_positive += int(valid.sum())
        n_possible += int(np.isfinite(flux_row).sum())

    if not raw_terms:
        return float("nan"), 0.0
    raw_vector = np.concatenate(raw_terms)
    flux_vector = np.concatenate(flux_terms)
    return float(np.corrcoef(raw_vector, flux_vector)[0, 1]), n_positive / n_possible


def rank_time_near_raw_calibrated_candidates(
    raw_sweeps: list[Level0Er3DSweep],
    calibrated_sweeps: list[CalibratedEr3DSweep],
    *,
    max_receive_minus_end_seconds: float = 600.0,
) -> list[RawCalibratedCandidate]:
    """Score every time-near pair without asserting event-time correspondence.

    The retained time difference is between raw Earth-receive time and the end
    of the calibrated energy sweep. It is only a coarse search coordinate;
    event time still needs the documented timing/control-state reconstruction.
    """
    if (
        not np.isfinite(max_receive_minus_end_seconds)
        or max_receive_minus_end_seconds <= 0
    ):
        raise ValueError("max_receive_minus_end_seconds must be finite and positive")

    candidates: list[RawCalibratedCandidate] = []
    for raw_sweep in raw_sweeps:
        if raw_sweep.earth_receive_time is None:
            raise ValueError("Raw sweeps need Earth-receive timestamps for ranking")
        for calibrated_sweep in calibrated_sweeps:
            receive_minus_end_seconds = (
                raw_sweep.earth_receive_time - calibrated_sweep.end_time
            ).total_seconds()
            if abs(receive_minus_end_seconds) > max_receive_minus_end_seconds:
                continue
            correlation, positive_fraction = angular_log_correlation(
                raw_sweep.compressed_counts,
                calibrated_sweep.flux,
            )
            candidates.append(
                RawCalibratedCandidate(
                    raw_first_record_index=raw_sweep.first_record_index,
                    raw_earth_receive_time=raw_sweep.earth_receive_time,
                    calibrated_spec_no=calibrated_sweep.spec_no,
                    calibrated_start_time=calibrated_sweep.start_time,
                    calibrated_end_time=calibrated_sweep.end_time,
                    receive_minus_end_seconds=receive_minus_end_seconds,
                    angular_log_correlation=correlation,
                    positive_cell_fraction=positive_fraction,
                )
            )

    return sorted(
        candidates,
        key=lambda candidate: (
            np.nan_to_num(candidate.angular_log_correlation, nan=-np.inf),
            -abs(candidate.receive_minus_end_seconds),
        ),
        reverse=True,
    )
