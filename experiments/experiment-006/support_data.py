"""Geometry-only loading for frozen Experiment 006 support templates."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import beam_model
import numpy as np
import pandas as pd
import spiceypy as spice

from src import config
from src.losscone.pitch_angle import PitchAngle
from src.potential_mapper.coordinates import (
    CoordinateCalculator,
    find_surface_intersection_with_polarity,
    project_magnetic_fields,
)
from src.potential_mapper.spice import load_spice_files
from src.utils.attitude import load_attitude_data
from src.utils.spice_lock import verify_kernels_lock

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class GeometrySupport:
    """One polarity-oriented native LP support without observed flux."""

    spec_no: int
    energies_ev: np.ndarray
    channel_low_ev: np.ndarray
    channel_high_ev: np.ndarray
    pitches_deg: np.ndarray
    polarity: np.ndarray


def _manifest_specs(manifest_path: Path, split: str) -> list[int]:
    # The manifest is an exclusion boundary: only the geometry key and split are
    # inspected.  In particular, confirmation identifiers never enter D1.
    manifest = pd.read_csv(manifest_path, usecols=["spec_no", "split"])
    selected = manifest.loc[manifest["split"] == split, "spec_no"]
    if selected.empty:
        raise ValueError(f"No support specs found for split {split!r}")
    return sorted(int(value) for value in selected)


def load_geometry_supports(
    data_path: Path,
    manifest_path: Path,
    *,
    split: str = "calibration",
) -> list[GeometrySupport]:
    """Load selected geometry while never parsing ER flux columns."""
    if split != "calibration":
        raise ValueError("D1 geometry loading is restricted to calibration supports")
    specs = _manifest_specs(manifest_path, split)
    geometry_columns = [
        *config.MISC_COLS,
        *config.MAG_COLS,
        *config.PHI_COLS,
    ]
    data = pd.read_csv(
        data_path,
        sep=" ",
        engine="c",
        skipinitialspace=True,
        header=None,
        names=config.ALL_COLS,
        usecols=geometry_columns,
    )
    data = data[data[config.SPEC_NO_COLUMN].isin(specs)].copy()
    data.sort_values([config.SPEC_NO_COLUMN, config.UTC_COLUMN], inplace=True)
    data.reset_index(drop=True, inplace=True)
    counts = data.groupby(config.SPEC_NO_COLUMN).size()
    missing = sorted(set(specs) - {int(value) for value in counts.index})
    incomplete = sorted(int(value) for value, count in counts.items() if count != 15)
    if missing or incomplete:
        raise RuntimeError(
            f"Geometry supports missing={missing}, incomplete={incomplete}"
        )

    lock_path = config.SPICE_KERNELS_DIR / "kernels.lock"
    verification = verify_kernels_lock(
        config.SPICE_KERNELS_DIR, lock_path, verify_hashes=True
    )
    if verification.missing or verification.extra or verification.mismatched:
        raise RuntimeError(f"SPICE kernel lock verification failed: {verification}")
    spice.kclear()
    load_spice_files()
    et_spin, ra_vals, dec_vals = load_attitude_data(
        config.DATA_DIR / config.ATTITUDE_FILE
    )
    if et_spin is None or ra_vals is None or dec_vals is None:
        raise RuntimeError("Attitude data unavailable")

    geometry_data = SimpleNamespace(data=data)
    coordinate_arrays = CoordinateCalculator(
        et_spin, ra_vals, dec_vals
    ).calculate_coordinate_transformation(geometry_data)
    projected_b = project_magnetic_fields(geometry_data, coordinate_arrays)
    _points, connected, polarity = find_surface_intersection_with_polarity(
        coordinate_arrays, projected_b
    )
    if not np.all(connected):
        raise RuntimeError("Frozen calibration support became disconnected")
    pitches = PitchAngle(geometry_data, polarity=polarity).pitch_angles

    spec_values = data[config.SPEC_NO_COLUMN].to_numpy(dtype=np.int64)
    supports: list[GeometrySupport] = []
    for spec_no in specs:
        selected = spec_values == spec_no
        energies = data.loc[selected, config.ENERGY_COLUMN].to_numpy(dtype=np.float64)
        low, high = beam_model.channel_edges_ev(energies)
        supports.append(
            GeometrySupport(
                spec_no=spec_no,
                energies_ev=energies,
                channel_low_ev=low,
                channel_high_ev=high,
                pitches_deg=pitches[selected],
                polarity=polarity[selected].copy(),
            )
        )
    spice.kclear()
    return supports
