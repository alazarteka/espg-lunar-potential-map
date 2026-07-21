"""Verification for the documented calibrated ER reference-sweep anchor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src import config

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = PROJECT_ROOT / "tests/fixtures/er3d_19980116_spec_0001.json"


def test_reference_sweep_manifest_matches_local_calibrated_product() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text())
    data_path = PROJECT_ROOT / manifest["relative_path"]
    if not data_path.exists():
        pytest.skip("PDS ER calibrated data are not installed locally")

    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == manifest["sha256"]

    data = pd.read_csv(
        data_path,
        sep=" ",
        engine="c",
        skipinitialspace=True,
        header=None,
    )
    assert data.shape[1] == manifest["expected_fields_per_row"]

    spec_no_column = config.ALL_COLS.index(config.SPEC_NO_COLUMN)
    energy_column = config.ALL_COLS.index(config.ENERGY_COLUMN)
    sweep = data.loc[data.iloc[:, spec_no_column] == manifest["spec_no"]]
    assert len(sweep) == manifest["expected_rows"]
    assert sweep.iloc[:, energy_column].min() == pytest.approx(
        manifest["expected_energy_min_ev"]
    )
    assert sweep.iloc[:, energy_column].max() == pytest.approx(
        manifest["expected_energy_max_ev"]
    )
