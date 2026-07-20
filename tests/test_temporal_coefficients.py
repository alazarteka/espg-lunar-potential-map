"""Validation tests for temporal harmonic windowing."""

from pathlib import Path

import numpy as np
import pytest

from src.temporal import cli
from src.temporal.coefficients import (
    _partition_into_windows,
    compute_temporal_harmonics,
)


@pytest.mark.parametrize("window_hours", [0.0, -1.0, np.nan, np.inf])
def test_compute_temporal_harmonics_rejects_invalid_window_hours(
    window_hours: float,
) -> None:
    with pytest.raises(ValueError, match="window_hours must be finite and positive"):
        compute_temporal_harmonics(
            cache_dir=Path("does-not-matter"),
            start_date=np.datetime64("1998-01-01"),
            end_date=np.datetime64("1998-01-01"),
            lmax=0,
            window_hours=window_hours,
        )


@pytest.mark.parametrize("stride_hours", [0.0, -1.0, np.nan, np.inf])
def test_compute_temporal_harmonics_rejects_invalid_explicit_stride(
    stride_hours: float,
) -> None:
    with pytest.raises(
        ValueError,
        match="stride_hours must be finite and positive when provided",
    ):
        compute_temporal_harmonics(
            cache_dir=Path("does-not-matter"),
            start_date=np.datetime64("1998-01-01"),
            end_date=np.datetime64("1998-01-01"),
            lmax=0,
            stride_hours=stride_hours,
        )


@pytest.mark.parametrize("parameter", ["window_hours", "stride_hours"])
def test_compute_temporal_harmonics_rejects_subsecond_duration(
    parameter: str,
) -> None:
    with pytest.raises(ValueError, match=f"{parameter} must be at least one second"):
        compute_temporal_harmonics(
            cache_dir=Path("does-not-matter"),
            start_date=np.datetime64("1998-01-01"),
            end_date=np.datetime64("1998-01-01"),
            lmax=0,
            **{parameter: 0.0001},
        )


def test_partition_rejects_zero_stride_before_looping() -> None:
    utc = np.array(["1998-01-01T00:00:00"], dtype="datetime64[s]")
    with pytest.raises(
        ValueError,
        match="stride_hours must be finite and positive when provided",
    ):
        list(
            _partition_into_windows(
                utc,
                np.array([0.0]),
                np.array([0.0]),
                np.array([1.0]),
                window_hours=1.0,
                stride_hours=0.0,
            )
        )


@pytest.mark.parametrize(
    ("fit_mode", "window_option", "window_value"),
    [
        ("basis", "--window-hours", "nan"),
        ("window", "--window-stride", "0"),
    ],
)
def test_cli_validates_window_parameters_for_both_fit_modes(
    monkeypatch: pytest.MonkeyPatch,
    fit_mode: str,
    window_option: str,
    window_value: str,
) -> None:
    monkeypatch.setattr(
        "sys.argv",
        [
            "temporal",
            "--start",
            "1998-01-01",
            "--end",
            "1998-01-01",
            "--output",
            "unused.npz",
            "--fit-mode",
            fit_mode,
            window_option,
            window_value,
        ],
    )

    assert cli.main() == 1
