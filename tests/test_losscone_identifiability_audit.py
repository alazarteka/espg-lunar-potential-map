from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.losscone.identifiability_audit import (
    _GEOMETRY_COLUMNS,
    _build_design,
    _load_polarity_shard,
    _read_geometry_table,
    deterministic_stratified_sample,
    parse_args,
    validate_time_limit,
    validate_workers,
)


def _census_frame() -> pd.DataFrame:
    records = [
        {
            "source_path": f"data/{month}/3D{month_index}{index:03d}.TAB",
            "geometry_shard": f"artifacts/{month_index}-{index:03d}.npz",
            "source_stem": (
                "3D980116" if month == "1998-01" and index == 0 else f"s{index}"
            ),
            "spec_no": 1 if index == 0 else index + 1,
            "utc_start": f"{month}-{index % 27 + 1:02d}T00:00:00+00:00",
            "month": month,
            "structural_valid": index != 39,
            "energy_signature": f"grid-{index % 2}",
            "energy_min_ev": 20.0 + index,
            "energy_max_ev": 1000.0 + index,
            "energy_leverage_sc_m100": 0.01 + index / 10000.0,
            "energy_leverage_sc_0": 0.02 + index / 10000.0,
            "energy_leverage_sc_10": 0.03 + index / 10000.0,
            "threshold_gap_median": 0.005 + index / 10000.0,
            "row_max_gap_median": 0.01 + index / 1000.0,
            "row_max_gap_max": 0.02 + index / 1000.0,
            "safe_rows_sc_150p0": 10 + index % 6,
            "reflected_cells": 500 + index,
            "reflected_rows": 15,
        }
        for month_index, month in enumerate(("1998-01", "1998-02"))
        for index in range(40)
    ]
    return pd.DataFrame.from_records(records)


def test_worker_and_time_limits_are_hard_caps() -> None:
    assert validate_workers(8) == 8
    with pytest.raises(ValueError, match="between 1 and 8"):
        validate_workers(9)
    with pytest.raises(ValueError, match="time limit"):
        validate_time_limit(56)


def test_cli_parses_all_four_subcommands() -> None:
    for command in ("census", "sample", "audit", "summarize"):
        args = parse_args([command])
        assert args.command == command


def test_geometry_loader_does_not_request_flux_columns() -> None:
    assert set(_GEOMETRY_COLUMNS).isdisjoint(config.FLUX_COLS)


def test_deterministic_sample_is_reproducible_and_forces_reference() -> None:
    census = _census_frame()
    first = deterministic_stratified_sample(census, per_month=8)
    second = deterministic_stratified_sample(census, per_month=8)
    pd.testing.assert_frame_equal(
        first.reset_index(drop=True), second.reset_index(drop=True)
    )
    reference = first[(first["source_stem"] == "3D980116") & (first["spec_no"] == 1)]
    assert len(reference) == 1
    assert bool(reference.iloc[0]["forced_reference"])
    assert set(first["month"]) == {"1998-01", "1998-02"}
    assert first.groupby("month").size().to_dict() == {"1998-01": 8, "1998-02": 8}
    valid = census[census["structural_valid"]]
    for column in (
        "energy_min_ev",
        "energy_max_ev",
        "energy_leverage_sc_0",
        "threshold_gap_median",
        "row_max_gap_median",
        "row_max_gap_max",
        "reflected_cells",
        "reflected_rows",
    ):
        assert first[column].min() == pytest.approx(valid[column].min())
        assert first[column].max() == pytest.approx(valid[column].max())


def test_reference_sweep_geometry_when_local_archive_is_available() -> None:
    project_root = Path(__file__).resolve().parents[1]
    data_path = project_root / "data/1998/001_031JAN/3D980116.TAB"
    shard_path = (
        project_root / "artifacts/experiment-008/geometry-shards/0000-3D980116.npz"
    )
    theta_path = project_root / "data/theta.tab"
    if not data_path.exists() or not shard_path.exists() or not theta_path.exists():
        pytest.skip("local LP archive or corrected-polarity shard is unavailable")

    table = _read_geometry_table(data_path)
    group = table[table[config.SPEC_NO_COLUMN] == 1].reset_index(drop=True)
    lookup, arrays = _load_polarity_shard(shard_path)
    design = _build_design(
        group,
        shard_index=lookup[1],
        shard_arrays=arrays,
        thetas=np.loadtxt(theta_path),
    )
    assert design.energies_ev.shape == (15,)
    assert design.pitches_deg.shape == (15, 88)
    assert np.all(design.valid_rows)
    assert np.ptp(design.row_times_s) > 0
