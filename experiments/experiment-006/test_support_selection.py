"""Focused tests for the outcome-blind Experiment 006 support selector."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _load_module():
    path = Path(__file__).with_name("support_selection.py")
    spec = importlib.util.spec_from_file_location(
        "experiment_006_support_selection", path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


selection = _load_module()


def metric(spec_no: int, stratum: int, offset: float):
    return selection.SupportMetric(
        spec_no=spec_no,
        utc_start="1999-04-29T13:00:00Z",
        utc_end="1999-04-29T13:01:00Z",
        utc_midpoint="1999-04-29T13:00:30Z",
        temporal_stratum=stratum,
        structural_rows=15,
        connected_rows=15,
        polarity=1 if spec_no % 2 else -1,
        guard_energy_rows=8,
        usable_guard_rows=8,
        axis_support_rows=4 + int(offset),
        minimum_axis_sector_count=2.0,
        median_axis_sector_count=2.0 + offset,
        median_incident_sector_count=30.0 - offset,
        median_reflected_sector_count=30.0 + offset,
        median_pitch_span_deg=160.0 - offset,
        field_elevation_deg=35.0 + 10.0 * offset,
        spacecraft_altitude_km=90.0 + offset,
        footprint_lat_deg=-30.0 + offset,
        footprint_lon_deg=20.0 + offset,
        footprint_lon_sin=float(np.sin(np.deg2rad(20.0 + offset))),
        footprint_lon_cos=float(np.cos(np.deg2rad(20.0 + offset))),
        magnetic_variation_deg=offset,
        mean_anchor_log_distance=0.1 + 0.01 * offset,
        eligible=True,
        exclusion_reason="",
    )


def test_temporal_strata_are_half_open_and_clipped() -> None:
    start = pd.Timestamp("1999-04-29T12:00:00Z")
    end = pd.Timestamp("1999-04-29T13:00:00Z")
    assert selection.temporal_stratum(start, start, end, 12) == 0
    second = selection.temporal_stratum(start + pd.Timedelta(minutes=5), start, end, 12)
    assert second == 1
    assert selection.temporal_stratum(end, start, end, 12) == 11


def test_maximin_selector_chooses_feature_extremes() -> None:
    metrics = [
        metric(10, 0, 0.0),
        metric(11, 0, 1.0),
        metric(12, 0, 2.0),
        metric(13, 0, 3.0),
        metric(14, 0, 6.0),
    ]
    groups, distances = selection.select_groups(metrics, 1, 4)
    assert {10, 14}.issubset({value.spec_no for value in groups[0]})
    assert all(distances[value.spec_no] > 0.0 for value in groups[0])


def test_assignment_is_disjoint_balanced_and_deterministic() -> None:
    groups = [
        [metric(10 * index + rank, index, float(rank)) for rank in range(1, 5)]
        for index in range(4)
    ]
    first = selection.assign_groups(groups)
    second = selection.assign_groups(groups)
    calibration, confirmation, ranks, imbalance = first
    assert calibration == second[0]
    assert confirmation == second[1]
    assert ranks == second[2]
    assert imbalance == second[3]
    assert calibration.isdisjoint(confirmation)
    assert len(calibration) == len(confirmation) == 8
    assert len(calibration | confirmation) == 16
    assert set(ranks.values()) == {1, 2, 3, 4}
    assert np.isfinite(imbalance)


def test_selection_features_exclude_flux_morphology() -> None:
    forbidden = {"flux", "beam", "continuum", "residual", "amplitude"}
    for feature in selection.SELECTION_FEATURES:
        assert not any(word in feature for word in forbidden)


def test_provenance_name_is_repository_relative() -> None:
    path = Path(__file__).with_name("support_selection.py")
    assert selection.provenance_name(path) == (
        "experiments/experiment-006/support_selection.py"
    )


def test_frozen_manifest_invariants() -> None:
    experiment_dir = Path(__file__).resolve().parent
    manifest = pd.read_csv(experiment_dir / "support_manifest.csv")
    provenance = json.loads(
        (experiment_dir / "support_provenance.json").read_text(encoding="ascii")
    )

    calibration = set(manifest.loc[manifest.split == "calibration", "spec_no"])
    confirmation = set(manifest.loc[manifest.split == "confirmation", "spec_no"])
    assert calibration == set(provenance["calibration_specs"])
    assert confirmation == set(provenance["confirmation_specs"])
    assert calibration.isdisjoint(confirmation)
    assert len(calibration) == len(confirmation) == 12
    assert not ({572, 652, 653, 654, 772, 969} & (calibration | confirmation))
    assert manifest.eligible.all()
    assert (manifest.connected_rows == 15).all()
    assert (manifest.field_elevation_deg >= 30.0).all()
    assert set(manifest.groupby(["temporal_stratum", "split"]).size()) == {2}
