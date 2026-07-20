"""CI-safe integration seams: batch NPZ → temporal harmonics → engineering stats."""

from __future__ import annotations

import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config
from src.engineering.analysis import compute_global_stats, extract_site_stats
from src.engineering.sites import Site
from src.potential_mapper import batch as batch_mod
from src.potential_mapper.batch import _prepare_payload, run_batch
from src.potential_mapper.cli_args import validate_date_filters
from src.potential_mapper.npz_io import write_npz_atomic
from src.potential_mapper.results import PotentialResults
from src.temporal.coefficients import (
    compute_temporal_harmonics,
    save_temporal_coefficients,
)
from src.temporal.dataset import TemporalDataset, load_temporal_coefficients

REQUIRED_ROW_KEYS = {
    "rows_spec_no",
    "rows_utc",
    "rows_time",
    "rows_spacecraft_latitude",
    "rows_spacecraft_longitude",
    "rows_projection_latitude",
    "rows_projection_longitude",
    "rows_spacecraft_potential",
    "rows_projected_potential",
    "rows_spacecraft_in_sun",
    "rows_projection_in_sun",
    "rows_projection_polarity",
    "rows_bs_over_bm",
    "rows_beam_amp",
    "rows_fit_chi2",
    "rows_electron_temperature",
    "rows_electron_density",
    "rows_kappa_value",
    "rows_environment_class",
}

REQUIRED_SPEC_KEYS = {
    "spec_spec_no",
    "spec_time_start",
    "spec_time_end",
    "spec_has_fit",
    "spec_row_count",
    "spec_u_surface",
    "spec_bs_over_bm",
    "spec_fit_chi2",
    "spec_electron_temperature",
    "spec_electron_density",
    "spec_environment_class",
}


def _make_er(
    spec_no: list[int],
    utc_prefix: str = "1998-01-01",
) -> types.SimpleNamespace:
    n = len(spec_no)
    df = pd.DataFrame(
        {
            config.SPEC_NO_COLUMN: np.asarray(spec_no, dtype=np.int64),
            config.UTC_COLUMN: [f"{utc_prefix}T00:00:{i:02d}" for i in range(n)],
            config.TIME_COLUMN: np.arange(n, dtype=float),
        }
    )
    return types.SimpleNamespace(data=df)


def _make_results(
    n: int,
    *,
    projected_potential: np.ndarray | None = None,
    projection_lat: np.ndarray | None = None,
    projection_lon: np.ndarray | None = None,
) -> PotentialResults:
    z = np.zeros(n, dtype=float)
    if projected_potential is None:
        projected_potential = np.full(n, -42.0, dtype=float)
    if projection_lat is None:
        projection_lat = z.copy()
    if projection_lon is None:
        projection_lon = z.copy()
    return PotentialResults(
        spacecraft_latitude=z.copy(),
        spacecraft_longitude=z.copy(),
        projection_latitude=np.asarray(projection_lat, dtype=float),
        projection_longitude=np.asarray(projection_lon, dtype=float),
        spacecraft_potential=z.copy(),
        projected_potential=np.asarray(projected_potential, dtype=float),
        spacecraft_in_sun=np.zeros(n, dtype=bool),
        projection_in_sun=np.zeros(n, dtype=bool),
    )


def _write_fake_day_cache(
    path: Path,
    *,
    day: str,
    n: int = 24,
    seed: int = 0,
) -> None:
    rng = np.random.default_rng(seed)
    utc = np.array(
        [f"{day}T{hour:02d}:00:00" for hour in range(n)],
        dtype="U32",
    )
    lat = rng.uniform(-60.0, 60.0, n)
    lon = rng.uniform(-180.0, 180.0, n)
    pot = 50.0 + 5.0 * np.sin(np.deg2rad(lat))
    np.savez_compressed(
        path,
        rows_utc=utc,
        rows_projection_latitude=lat,
        rows_projection_longitude=lon,
        rows_projected_potential=pot,
    )


def test_validate_date_filters_rejects_invalid_calendar_dates() -> None:
    assert validate_date_filters(1999, 2, 30) is not None
    assert validate_date_filters(1999, 13, 1) is not None
    assert validate_date_filters(1999, 4, 0) is not None
    assert validate_date_filters(1999, 4, None) is None
    assert validate_date_filters(1999, 4, 29) is None


def test_prepare_payload_npz_keys_and_contiguous_spec(tmp_path: Path) -> None:
    sweep = config.SWEEP_ROWS
    spec_no = [10] * sweep + [11] * sweep
    u_vals = np.concatenate([np.full(sweep, -100.0), np.full(sweep, -50.0)])
    er = _make_er(spec_no)
    results = _make_results(len(spec_no), projected_potential=u_vals)
    payload = _prepare_payload(er, results)

    assert REQUIRED_ROW_KEYS.issubset(payload)
    assert REQUIRED_SPEC_KEYS.issubset(payload)

    assert np.array_equal(payload["rows_spec_no"], np.asarray(spec_no, dtype=np.int64))
    assert list(payload["spec_spec_no"]) == [10, 11]
    assert list(payload["spec_row_count"]) == [sweep, sweep]
    assert np.allclose(payload["spec_u_surface"], [-100.0, -50.0])
    assert np.all(np.isfinite(payload["spec_u_surface"]))
    assert np.all(payload["spec_has_fit"])

    out = tmp_path / "potential_batch_1998_01_01.npz"
    write_npz_atomic(out, payload)
    with np.load(out) as data:
        assert set(REQUIRED_ROW_KEYS | REQUIRED_SPEC_KEYS).issubset(data.files)
        assert np.all(np.isfinite(data["spec_u_surface"]))


def test_run_batch_writes_npz_with_monkeypatched_processor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sweep = config.SWEEP_ROWS
    spec_no = [1] * sweep
    er = _make_er(spec_no, utc_prefix="1999-04-29")
    results = _make_results(
        sweep,
        projected_potential=np.full(sweep, -75.0),
    )

    monkeypatch.setattr(batch_mod, "load_spice_files", lambda: None)
    monkeypatch.setattr(
        batch_mod.DataLoader,
        "discover_flux_files",
        staticmethod(lambda **_kwargs: [tmp_path / "fake.TAB"]),
    )
    monkeypatch.setattr(batch_mod, "load_all_data", lambda _files: er)
    monkeypatch.setattr(
        batch_mod,
        "process_merged_data",
        lambda *_args, **_kwargs: results,
    )

    code = run_batch(
        output_dir=tmp_path,
        year=1999,
        month=4,
        day=29,
        overwrite=True,
    )
    assert code == 0

    out = tmp_path / "potential_batch_1999_04_29.npz"
    assert out.is_file()
    with np.load(out) as data:
        assert np.array_equal(data["rows_spec_no"], np.asarray(spec_no, dtype=np.int64))
        assert np.allclose(data["spec_u_surface"], [-75.0])
        assert bool(data["spec_has_fit"][0])


@pytest.mark.parametrize(
    ("argv_extra",),
    [
        (["--year", "1999", "--month", "13", "--day", "1"],),
        (["--year", "1999", "--month", "2", "--day", "30"],),
    ],
)
def test_batch_cli_invalid_dates_exit_1(
    monkeypatch: pytest.MonkeyPatch,
    argv_extra: list[str],
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        ["batch", "--output-dir", "/tmp", *argv_extra],
    )
    assert batch_mod.main() == 1


def test_fake_cache_temporal_to_engineering(tmp_path: Path) -> None:
    _write_fake_day_cache(tmp_path / "day1.npz", day="1998-01-01", seed=1)
    _write_fake_day_cache(tmp_path / "day2.npz", day="1998-01-02", seed=2)

    results = compute_temporal_harmonics(
        cache_dir=tmp_path,
        start_date=np.datetime64("1998-01-01"),
        end_date=np.datetime64("1998-01-02"),
        lmax=1,
        window_hours=24.0,
        min_samples=10,
        min_coverage=0.0,
    )
    assert len(results) >= 1
    for result in results:
        assert result.coeffs.shape == (4,)
        assert result.n_samples >= 10
        assert np.all(np.isfinite(result.coeffs))

    coeff_path = tmp_path / "temporal_coeffs.npz"
    save_temporal_coefficients(results, coeff_path)
    dataset = load_temporal_coefficients(coeff_path)
    assert isinstance(dataset, TemporalDataset)
    assert dataset.lmax == 1
    assert dataset.coeffs.ndim == 2
    assert dataset.coeffs.shape[1] == 4

    global_stats = compute_global_stats(dataset, lat_steps=8, lon_steps=16)
    assert global_stats.mean_potential.shape == (8, 16)
    assert np.all(np.isfinite(global_stats.mean_potential))

    site = Site(name="Equator", lat=0.0, lon=0.0, description="integration")
    site_stats = extract_site_stats(dataset, site)
    assert np.isfinite(site_stats.mean_potential)
