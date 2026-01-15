from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest  # noqa: TC002

import src.config as config
from scripts.dev import potential_mapper_batch as batch
from src.potential_mapper.results import PotentialResults


class DummyERData:
    def __init__(self, df: pd.DataFrame):
        self.data = df


def _make_results(n: int, projected: np.ndarray | None = None) -> PotentialResults:
    proj = projected if projected is not None else np.full(n, np.nan)
    return PotentialResults(
        spacecraft_latitude=np.zeros(n),
        spacecraft_longitude=np.zeros(n),
        projection_latitude=np.zeros(n),
        projection_longitude=np.zeros(n),
        spacecraft_potential=np.zeros(n),
        projected_potential=proj,
        spacecraft_in_sun=np.zeros(n, dtype=bool),
        projection_in_sun=np.zeros(n, dtype=bool),
    )


def test_prepare_payload_spec_summary():
    df = pd.DataFrame(
        {
            config.SPEC_NO_COLUMN: [10, 10, 11],
            config.TIME_COLUMN: [
                "2000-01-01T00:00:00",
                "2000-01-01T00:05:00",
                "2000-01-01T00:10:00",
            ],
            config.UTC_COLUMN: [
                "2000-01-01T00:00:00",
                "2000-01-01T00:05:00",
                "2000-01-01T00:10:00",
            ],
        }
    )
    results = _make_results(3, projected=np.array([np.nan, -5.0, np.nan]))
    payload = batch._prepare_payload(DummyERData(df), results)

    assert payload["rows_spec_no"].tolist() == [10, 10, 11]
    assert payload["spec_spec_no"].tolist() == [10, 11]
    assert payload["spec_time_start"].tolist()[0] == "2000-01-01T00:00:00"
    assert payload["spec_time_end"].tolist()[0] == "2000-01-01T00:05:00"
    assert payload["spec_has_fit"].tolist() == [True, False]


def test_process_file_writes_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    df = pd.DataFrame(
        {
            config.SPEC_NO_COLUMN: [42, 42],
            config.TIME_COLUMN: ["2000-01-01T00:00:00", "2000-01-01T00:05:00"],
            config.UTC_COLUMN: ["2000-01-01T00:00:00", "2000-01-01T00:05:00"],
        }
    )
    results = _make_results(2, projected=np.array([-1.0, -2.0]))

    def fake_process_lp_file(_path: Path) -> PotentialResults:
        return results

    def fake_er_data(_path: str) -> DummyERData:
        return DummyERData(df)

    monkeypatch.setattr(batch, "process_lp_file", fake_process_lp_file)
    monkeypatch.setattr(batch, "ERData", fake_er_data)
    monkeypatch.setattr(batch.config, "DATA_DIR", tmp_path)
    monkeypatch.setattr(batch, "load_spice_files", lambda: None)
    batch._SPICE_LOADED = False

    er_file = tmp_path / "example.tab"
    er_file.write_text("dummy")
    out_dir = tmp_path / "out"

    _, status = batch._process_file(er_file, out_dir, overwrite=False)
    assert status == "written"

    npz_files = list(out_dir.rglob("*.npz"))
    assert len(npz_files) == 1

    data = np.load(npz_files[0])
    assert np.allclose(data["rows_projected_potential"], [-1.0, -2.0])

    # Second call should skip without rewriting
    _, status2 = batch._process_file(er_file, out_dir, overwrite=False)
    assert status2 == "skipped"
