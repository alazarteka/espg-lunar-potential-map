from __future__ import annotations

import numpy as np

from src.potential_mapper.u_profile_qc import augment_batch_arrays_with_u_width


def test_augment_batch_arrays_with_u_width_aligns_and_broadcasts():
    batch_arrays = {
        "spec_spec_no": np.array([10, 11, 12], dtype=np.int64),
        "rows_spec_no": np.array([10, 10, 12], dtype=np.int64),
    }
    profile_arrays = {
        # Deliberately out of order to exercise alignment-by-spec_no.
        "spec_spec_no": np.array([12, 10], dtype=np.int64),
        "u_width_dchi2red_0p001": np.array([50.0, 250.0], dtype=np.float64),
    }

    out = augment_batch_arrays_with_u_width(
        batch_arrays=batch_arrays,
        profile_arrays=profile_arrays,
        delta_reduced=0.001,
        identifiable_width_max_v=200.0,
        include_rows=True,
    )

    assert "spec_u_width_dchi2red_0p001" in out
    assert "spec_u_is_identifiable_dchi2red_0p001" in out
    assert "rows_u_width_dchi2red_0p001" in out
    assert "rows_u_is_identifiable_dchi2red_0p001" in out

    # spec 10 width is 250 (not identifiable); spec 11 missing (NaN); spec 12 width 50.
    assert np.allclose(
        out["spec_u_width_dchi2red_0p001"],
        np.array([250.0, np.nan, 50.0], dtype=np.float64),
        equal_nan=True,
    )
    assert out["spec_u_is_identifiable_dchi2red_0p001"].tolist() == [False, False, True]

    # Row-level arrays should follow rows_spec_no.
    assert np.allclose(
        out["rows_u_width_dchi2red_0p001"],
        np.array([250.0, 250.0, 50.0], dtype=np.float64),
        equal_nan=True,
    )
    assert out["rows_u_is_identifiable_dchi2red_0p001"].tolist() == [False, False, True]


def test_augment_batch_arrays_with_u_width_can_skip_rows():
    batch_arrays = {
        "spec_spec_no": np.array([10], dtype=np.int64),
        "rows_spec_no": np.array([10], dtype=np.int64),
    }
    profile_arrays = {
        "spec_spec_no": np.array([10], dtype=np.int64),
        "u_width_dchi2red_0p001": np.array([10.0], dtype=np.float64),
    }

    out = augment_batch_arrays_with_u_width(
        batch_arrays=batch_arrays,
        profile_arrays=profile_arrays,
        delta_reduced=0.001,
        identifiable_width_max_v=200.0,
        include_rows=False,
    )

    assert "spec_u_width_dchi2red_0p001" in out
    assert "spec_u_is_identifiable_dchi2red_0p001" in out
    assert "rows_u_width_dchi2red_0p001" not in out
    assert "rows_u_is_identifiable_dchi2red_0p001" not in out
