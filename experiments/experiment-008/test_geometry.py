"""Focused tests for Experiment 008 geometry extraction."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("geometry.py")
SPEC = importlib.util.spec_from_file_location("experiment_008_geometry", MODULE_PATH)
assert SPEC is not None
assert SPEC.loader is not None
geometry = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def test_contiguous_group_slices_preserve_runs() -> None:
    values = np.array([1, 1, 2, 2, 2, 1])
    groups = geometry.contiguous_group_slices(values)
    assert [(group.start, group.stop) for group in groups] == [(0, 2), (2, 5), (5, 6)]


def test_legacy_lookup_selects_next_sample() -> None:
    attitude_et = np.array([0.0, 10.0, 20.0])
    ra = np.array([1.0, 2.0, 3.0])
    dec = np.array([4.0, 5.0, 6.0])
    query = np.array([0.0, 5.0, 10.0, 20.0, 21.0])
    actual_ra, actual_dec = geometry.legacy_next_attitude(query, attitude_et, ra, dec)
    np.testing.assert_allclose(actual_ra[:3], [2.0, 2.0, 3.0])
    np.testing.assert_allclose(actual_dec[:3], [5.0, 5.0, 6.0])
    assert np.isnan(actual_ra[3:]).all()
    assert np.isnan(actual_dec[3:]).all()


def test_points_to_lat_lon() -> None:
    points = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [np.nan, 0.0, 0.0]])
    lat, lon = geometry.points_to_lat_lon(points, np.array([True, True, False]))
    np.testing.assert_allclose(lat[:2], [0.0, 45.0])
    np.testing.assert_allclose(lon[:2], [0.0, 90.0])
    assert np.isnan(lat[2])
    assert np.isnan(lon[2])


def test_manifest_reader_is_limited_to_raw_inputs(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "kind,path,size_bytes,mtime_ns,sha256\n"
        "raw_er_mag,data/a.TAB,1,1,a\n"
        "attitude,data/attitude.tab,1,1,b\n"
        "spice,data/a.bsp,1,1,c\n",
        encoding="ascii",
    )
    assert geometry.raw_paths_from_manifest(manifest, tmp_path) == [
        tmp_path / "data/a.TAB"
    ]


def test_empty_group_list() -> None:
    assert geometry.contiguous_group_slices(np.array([], dtype=np.int64)) == []


def test_display_path_handles_external_output(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    external = tmp_path / "external" / "result.npz"
    internal = project_root / "artifacts" / "result.npz"
    assert geometry.display_path(internal, project_root) == "artifacts/result.npz"
    assert geometry.display_path(external, project_root) == external.as_posix()


def test_robust_spkpos_isolates_uncovered_epoch(monkeypatch) -> None:
    def fake_spkpos(target, et, reference_frame, correction, observer):
        del target, reference_frame, correction, observer
        epochs = np.atleast_1d(et)
        if np.any(epochs == 2.0):
            raise RuntimeError("uncovered")
        return np.column_stack([epochs, epochs + 1.0, epochs + 2.0]), np.zeros(
            len(epochs)
        )

    monkeypatch.setattr(geometry.spice, "spkpos", fake_spkpos)
    actual = geometry.robust_spkpos(
        "target", np.array([1.0, 2.0, 3.0]), "frame", "observer"
    )
    np.testing.assert_allclose(actual[[0, 2]], [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]])
    assert np.isnan(actual[1]).all()


def test_corrected_projection_uses_eclipj2000_for_both_defining_vectors(
    monkeypatch,
) -> None:
    frames: list[tuple[str, str]] = []
    monkeypatch.setattr(geometry, "_ATTITUDE_ET", np.array([0.0]))
    monkeypatch.setattr(geometry, "_ATTITUDE_RA", np.array([0.0]))
    monkeypatch.setattr(geometry, "_ATTITUDE_DEC", np.array([90.0]))
    monkeypatch.setattr(
        geometry,
        "get_current_ra_dec_batch",
        lambda *_args: (np.array([0.0]), np.array([90.0])),
    )

    def fake_spkpos(target, et, reference_frame, observer):
        frames.append((target, reference_frame))
        if target == "-25":
            return np.array([[2_000.0, 0.0, 0.0]])
        return np.array([[1.0, 0.0, 0.0]])

    def fake_pxform(from_frame, to_frame, et):
        del et
        frames.append((from_frame, to_frame))
        return np.eye(3)

    monkeypatch.setattr(geometry, "robust_spkpos", fake_spkpos)
    monkeypatch.setattr(geometry.spice, "pxform", fake_pxform)
    monkeypatch.setattr(
        geometry,
        "get_connections_and_polarity_batch",
        lambda positions, field, radius: (
            field,
            np.array([True]),
            np.array([1], dtype=np.int8),
        ),
    )

    latitude, longitude, polarity = geometry._project_rows(
        np.array([1.0]),
        np.array([[1.0, 0.0, 0.0]]),
        legacy_attitude=False,
        attitude_frame="ECLIPJ2000",
        frame_mixed_sun=False,
    )

    assert ("10", "ECLIPJ2000") in frames
    assert ("ECLIPJ2000", "IAU_MOON") in frames
    np.testing.assert_allclose(latitude, [0.0], atol=1e-12)
    np.testing.assert_allclose(longitude, [0.0], atol=1e-12)
    np.testing.assert_array_equal(polarity, [1])
