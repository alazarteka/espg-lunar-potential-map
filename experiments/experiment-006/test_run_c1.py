"""Focused mocked tests for the deterministic Experiment 006 C1 runner."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load(filename: str, name: str):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).with_name(filename)
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_load("beam_model.py", "beam_model")
_load("fit_reference.py", "fit_reference")
_load("start_parameters.py", "start_parameters")
_load("support_data.py", "support_data")
_load("simulator.py", "simulator")
_load("c1_estimator.py", "c1_estimator")
runner = _load("run_c1.py", "experiment_006_run_c1")


def _manifest(path: Path, count: int = 3) -> None:
    fields = [
        "case_id",
        "split",
        "support_id",
        *runner.TRUTH_FIELDS,
        *runner.FACTOR_FIELDS,
    ]
    with path.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for case_id in range(count):
            writer.writerow(
                {
                    "case_id": case_id,
                    "split": "bias",
                    "support_id": 10,
                    "anchor_ev": 100,
                    "true_center_ev": 100,
                    "beam_amplitude": 8,
                    "pitch_sigma_deg": 20,
                    "beam_axis_deg": 180,
                    "background_family": "background",
                    "noise_level": 0.2,
                    "response_family": "response",
                    "amplitude_quartile": 0,
                    "pitch_sigma_quartile": 0,
                    "temporal_family": "none",
                    "response_direction": 0,
                }
            )


def test_geometry_loader_is_explicitly_calibration_only(monkeypatch, tmp_path) -> None:
    seen = {}

    def fake_loader(data_path, manifest_path, *, split):
        seen.update(data=data_path, manifest=manifest_path, split=split)
        return [SimpleNamespace(spec_no=10)]

    monkeypatch.setattr(runner.support_data, "load_geometry_supports", fake_loader)
    result = runner.load_calibration_geometry(
        tmp_path / "geometry", tmp_path / "supports"
    )
    assert set(result) == {10}
    assert seen["split"] == "calibration"


def test_run_shards_resumes_and_checkpoints_each_pending_case(
    monkeypatch, tmp_path
) -> None:
    manifest_path = tmp_path / "manifest.csv"
    output = tmp_path / "results.csv"
    config_path = tmp_path / "config.json"
    _manifest(manifest_path)
    config_path.write_text("{}", encoding="ascii")
    output.write_text("case_id,value\n1,old\n", encoding="ascii")
    support = SimpleNamespace(spec_no=10)
    monkeypatch.setattr(
        runner, "load_calibration_geometry", lambda *args: {10: support}
    )
    calls = []

    def fake_run_case(row, supplied_support, config):
        calls.append(int(row["case_id"]))
        assert supplied_support is support
        return {"case_id": int(row["case_id"]), "value": "new"}

    monkeypatch.setattr(runner, "run_case", fake_run_case)
    runner.run(
        start=1,
        stop=3,
        output=output,
        manifest_path=manifest_path,
        config_path=config_path,
    )
    assert calls == [2]
    with output.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["case_id"]) for row in rows] == [1, 2]
    assert b"\r\n" not in output.read_bytes()
    provenance = json.loads(
        output.with_name(f"{output.name}.provenance.json").read_text(encoding="ascii")
    )
    assert provenance["range"] == {"start": 1, "stop": 3}
    assert provenance["manifest_sha256"] == runner.sha256_file(manifest_path)
    assert provenance["configuration_sha256"] == runner.sha256_file(config_path)
    assert provenance["output_sha256"] == runner.sha256_file(output)
    assert set(provenance["source_sha256"]) == {
        "run_c1.py",
        "simulator.py",
        "c1_estimator.py",
        "support_data.py",
    }
    assert provenance["confirmation_flux_opened"] is False


def test_run_case_writes_truth_centers_se_qc_reasons_and_factors(monkeypatch) -> None:
    row = {
        "case_id": "7",
        "split": "interval",
        "support_id": "10",
        **dict.fromkeys(runner.TRUTH_FIELDS, "1"),
        **dict.fromkeys(runner.FACTOR_FIELDS, "factor"),
    }
    support = SimpleNamespace(
        energies_ev=np.array([100.0]),
        channel_low_ev=np.array([80.0]),
        channel_high_ev=np.array([120.0]),
        pitches_deg=np.array([[180.0]]),
    )
    simulated = SimpleNamespace(
        native_flux=np.ones((1, 1)),
        native_valid=np.ones((1, 1), dtype=bool),
        incident_mean=np.ones((1, 1)),
        conjugate=np.ones((1, 1)),
        latents={"tail_fraction": 0.2},
    )
    fit = SimpleNamespace(
        parameters=np.array([8.0, 101.5]),
        standard_errors=np.array([1.0, 2.5]),
    )
    primary = SimpleNamespace(name="incident_mean", fit=fit, reason_codes=())
    secondary = SimpleNamespace(
        name="conjugate", fit=None, reason_codes=("fit_failed",)
    )
    estimate = SimpleNamespace(
        incident_mean=primary,
        conjugate=secondary,
        base_valid=False,
        reason_codes=("fit_failed",),
    )
    monkeypatch.setattr(runner.simulator, "simulate_case", lambda *args: simulated)
    monkeypatch.setattr(runner.c1_estimator, "fit_c1", lambda *args, **kwargs: estimate)
    result = runner.run_case(row, support, {})
    assert result["true_center_ev"] == "1"
    assert result["incident_mean_center_ev"] == 101.5
    assert result["incident_mean_center_se_ev"] == 2.5
    assert result["conjugate_reason_codes"] == "fit_failed"
    assert result["background_family"] == "factor"
    assert result["simulation_factors_json"] == '{"tail_fraction":0.2}'
