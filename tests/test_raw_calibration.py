from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from scripts.diagnostics.level0_calibrated_match import _candidate_to_json
from src.losscone.level0 import Level0Er3DSweep, earth_receive_time_from_level0_record
from src.losscone.raw_calibration import (
    CalibratedEr3DSweep,
    RawCalibratedCandidate,
    angular_log_correlation,
    rank_time_near_raw_calibrated_candidates,
)


def test_earth_receive_time_uses_documented_merged_record_offsets() -> None:
    record = bytearray(472)
    record[456:458] = (16).to_bytes(2, byteorder="big")
    record[458:462] = (12_345).to_bytes(4, byteorder="big")

    assert earth_receive_time_from_level0_record(bytes(record), year=1998) == datetime(
        1998,
        1,
        16,
        0,
        0,
        12,
        345_000,
        tzinfo=UTC,
    )


def test_angular_log_correlation_is_scale_free_within_each_energy() -> None:
    codes = np.array(
        [[16, 17, 18, 19], [32, 34, 36, 38]],
        dtype=np.uint8,
    )
    flux = np.array(
        [[2.0, 4.0, 8.0, 16.0], [10.0, 20.0, 40.0, 80.0]],
        dtype=np.float64,
    )

    correlation, positive_fraction = angular_log_correlation(codes, flux)

    assert correlation > 0.99
    assert positive_fraction == 1.0


def test_ranking_prefers_matching_angular_pattern_over_time_near_decoy() -> None:
    start = datetime(1998, 1, 16, tzinfo=UTC)
    expected_flux = np.array(
        [[2.0, 4.0, 8.0, 16.0], [10.0, 20.0, 40.0, 80.0]],
        dtype=np.float64,
    )
    raw = Level0Er3DSweep(
        first_record_index=5,
        compressed_counts=np.array(
            [[16, 17, 18, 19], [32, 34, 36, 38]], dtype=np.uint8
        ),
        earth_receive_time=start + timedelta(seconds=100),
    )
    matching = CalibratedEr3DSweep(
        spec_no=1,
        start_time=start,
        end_time=start + timedelta(seconds=95),
        energies_ev=np.array([100.0, 50.0]),
        flux=expected_flux,
    )
    decoy = CalibratedEr3DSweep(
        spec_no=2,
        start_time=start,
        end_time=start + timedelta(seconds=99),
        energies_ev=np.array([100.0, 50.0]),
        flux=np.fliplr(expected_flux),
    )

    candidates = rank_time_near_raw_calibrated_candidates(
        [raw],
        [matching, decoy],
        max_receive_minus_end_seconds=10.0,
    )

    assert [candidate.calibrated_spec_no for candidate in candidates] == [1, 2]


@pytest.mark.parametrize("window", [0.0, -1.0, np.nan, np.inf, -np.inf])
def test_ranking_rejects_non_finite_or_non_positive_time_window(window: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        rank_time_near_raw_calibrated_candidates(
            [],
            [],
            max_receive_minus_end_seconds=window,
        )


def test_unscorable_candidate_serializes_as_strict_json_null() -> None:
    timestamp = datetime(1998, 1, 16, tzinfo=UTC)
    candidate = RawCalibratedCandidate(
        raw_first_record_index=5,
        raw_earth_receive_time=timestamp,
        calibrated_spec_no=1,
        calibrated_start_time=timestamp,
        calibrated_end_time=timestamp,
        receive_minus_end_seconds=0.0,
        angular_log_correlation=float("nan"),
        positive_cell_fraction=0.0,
    )

    encoded = json.dumps(
        _candidate_to_json(candidate),
        allow_nan=False,
    )

    assert json.loads(encoded)["angular_log_correlation"] is None
