"""Tests for src.potential_mapper.batch._prepare_payload spec aggregation.

Guards the Wave-0 fixes: the spec_no contiguity guard and the has_fit/u_surface
consistency (both driven by the first finite value per spectrum).
"""

import types

import numpy as np
import pandas as pd
import pytest

from src import config
from src.potential_mapper.batch import _prepare_payload
from src.potential_mapper.results import PotentialResults


def _make_er(spec_no: list[int]) -> types.SimpleNamespace:
    n = len(spec_no)
    df = pd.DataFrame(
        {
            config.SPEC_NO_COLUMN: np.asarray(spec_no, dtype=np.int64),
            config.UTC_COLUMN: [f"1998-09-16T00:00:{i:02d}" for i in range(n)],
            config.TIME_COLUMN: np.arange(n, dtype=float),
        }
    )
    return types.SimpleNamespace(data=df)


def _make_results(n: int, projected_potential: list[float]) -> PotentialResults:
    z = np.zeros(n)
    return PotentialResults(
        spacecraft_latitude=z.copy(),
        spacecraft_longitude=z.copy(),
        projection_latitude=z.copy(),
        projection_longitude=z.copy(),
        spacecraft_potential=z.copy(),
        projected_potential=np.asarray(projected_potential, dtype=float),
        spacecraft_in_sun=np.zeros(n, dtype=bool),
        projection_in_sun=np.zeros(n, dtype=bool),
    )


def test_noncontiguous_spec_no_is_rejected() -> None:
    # spec 1 appears in two separate blocks -> the (start, count) slicing would
    # silently mix spectra; the guard must reject it.
    er = _make_er([1, 2, 1])
    res = _make_results(3, [1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="not contiguous"):
        _prepare_payload(er, res)


def test_has_fit_and_u_surface_are_consistent() -> None:
    # spec 1: leading NaN then a finite value -> valid, u_surface = first finite.
    # spec 2: all NaN -> not valid, u_surface NaN.
    er = _make_er([1, 1, 2, 2])
    res = _make_results(4, [np.nan, 5.0, np.nan, np.nan])
    payload = _prepare_payload(er, res)

    assert list(payload["spec_spec_no"]) == [1, 2]
    assert list(payload["spec_has_fit"]) == [True, False]

    u = payload["spec_u_surface"]
    assert u[0] == 5.0  # first finite value, not the leading NaN
    assert np.isnan(u[1])

    # has_fit must agree with u_surface finiteness for every spectrum.
    for has_fit, u_val in zip(payload["spec_has_fit"], u, strict=True):
        assert bool(has_fit) == bool(np.isfinite(u_val))


def test_cash_qc_emits_profile_fields() -> None:
    # A Cash fit must emit the profile-likelihood 68% CI fields (not the LHS proxy).
    er = _make_er([1, 1, 2, 2])
    res = _make_results(4, [5.0, 5.0, 7.0, 7.0])
    res.u_width_profile_dev68 = np.array([120.0, 120.0, 500.0, 500.0])
    res.u_is_identifiable_profile = np.array([True, True, False, False])

    payload = _prepare_payload(
        er, res, u_width_identifiable_max_v=200.0, fit_method="cash"
    )

    assert payload["rows_u_width_profile_dev68"].shape == (4,)
    assert payload["rows_u_is_identifiable_profile"].dtype == bool
    assert payload["spec_u_width_profile_dev68"].tolist() == [120.0, 500.0]
    assert payload["spec_u_is_identifiable_profile"].tolist() == [True, False]
    assert float(payload["u_width_profile_identifiable_max_v"]) == 200.0
    assert float(payload["u_width_profile_delta_chi2"]) == float(
        config.LOSS_CONE_U_WIDTH_DELTA_CHI2
    )
    assert str(payload["u_width_fit_method"]) == "cash"
    assert int(payload["u_width_schema_version"]) == 2
    # Legacy LHS fields must NOT be emitted for a Cash run.
    assert "rows_u_width_lhs_dchi2red_0p001" not in payload
    assert "spec_u_width_lhs_dchi2red_0p001" not in payload


def test_lillis_qc_emits_legacy_lhs_fields() -> None:
    er = _make_er([1, 1])
    res = _make_results(2, [5.0, 5.0])
    res.u_width_lhs_dchi2red_0p001 = np.array([50.0, 50.0])
    res.u_is_identifiable_lhs_dchi2red_0p001 = np.array([True, True])

    payload = _prepare_payload(
        er, res, u_width_identifiable_max_v=200.0, fit_method="lillis"
    )

    assert payload["rows_u_width_lhs_dchi2red_0p001"].tolist() == [50.0, 50.0]
    assert payload["spec_u_is_identifiable_lhs_dchi2red_0p001"].tolist() == [True]
    assert str(payload["u_width_fit_method"]) == "lillis"
    # Profile fields must NOT be emitted for a Lillis run.
    assert "rows_u_width_profile_dev68" not in payload


# The exact NPZ key set emitted by `main` at commit 0de3ec5, before the
# losscone-weighted-width merge, with the U-width QC block enabled. Cached
# artifacts and every downstream reader (temporal, engineering, visualization,
# the plotting scripts) were written against these names. The merge is only
# allowed to ADD keys for the default and Lillis paths, never to rename or drop
# one, or the 560-day production cache stops being readable by existing tools.
_LEGACY_QC_PAYLOAD_KEYS = frozenset(
    {
        "rows_beam_amp",
        "rows_bs_over_bm",
        "rows_electron_density",
        "rows_electron_temperature",
        "rows_environment_class",
        "rows_fit_chi2",
        "rows_kappa_value",
        "rows_projected_potential",
        "rows_projection_in_sun",
        "rows_projection_latitude",
        "rows_projection_longitude",
        "rows_projection_polarity",
        "rows_spacecraft_in_sun",
        "rows_spacecraft_latitude",
        "rows_spacecraft_longitude",
        "rows_spacecraft_potential",
        "rows_spec_no",
        "rows_time",
        "rows_u_is_identifiable_lhs_dchi2red_0p001",
        "rows_u_width_lhs_dchi2red_0p001",
        "rows_utc",
        "spec_bs_over_bm",
        "spec_electron_density",
        "spec_electron_temperature",
        "spec_environment_class",
        "spec_fit_chi2",
        "spec_has_fit",
        "spec_row_count",
        "spec_spec_no",
        "spec_time_end",
        "spec_time_start",
        "spec_u_is_identifiable_lhs_dchi2red_0p001",
        "spec_u_surface",
        "spec_u_width_lhs_dchi2red_0p001",
        "u_width_lhs_delta_reduced",
        "u_width_lhs_identifiable_max_v",
    }
)

# Same, without the QC block (the QC keys are opt-in via --emit-u-width-qc).
_LEGACY_BASE_PAYLOAD_KEYS = frozenset(
    key
    for key in _LEGACY_QC_PAYLOAD_KEYS
    if "u_width" not in key and "u_is_identifiable" not in key
)


def _payload_for_schema_check(**kwargs) -> dict:
    er = _make_er([1, 1, 2, 2])
    res = _make_results(4, [5.0, 5.0, 7.0, 7.0])
    return _prepare_payload(er, res, **kwargs)


@pytest.mark.parametrize("fit_method", [None, "lillis", "halekas"])
def test_npz_schema_is_backward_readable_without_qc(fit_method) -> None:
    """No legacy NPZ key may disappear from the default/Lillis/Halekas payload."""
    kwargs = {} if fit_method is None else {"fit_method": fit_method}
    keys = set(_payload_for_schema_check(**kwargs))
    assert keys >= _LEGACY_BASE_PAYLOAD_KEYS, sorted(_LEGACY_BASE_PAYLOAD_KEYS - keys)


@pytest.mark.parametrize("fit_method", [None, "lillis"])
def test_npz_schema_is_additive_only_with_qc(fit_method) -> None:
    """With QC enabled the payload is main's key set plus additive keys only."""
    kwargs = {"u_width_identifiable_max_v": 200.0}
    if fit_method is not None:
        kwargs["fit_method"] = fit_method
    keys = set(_payload_for_schema_check(**kwargs))

    missing = _LEGACY_QC_PAYLOAD_KEYS - keys
    assert not missing, (
        f"NPZ keys removed, breaking existing readers: {sorted(missing)}"
    )

    # Only provenance keys are new; anything else here is an unreviewed schema change.
    assert keys - _LEGACY_QC_PAYLOAD_KEYS == {
        "u_width_schema_version",
        "u_width_fit_method",
    }
