"""Tests for temporal identifiability-key auto-resolution.

The batch pipeline can emit either the Cash profile-CI flag
(``rows_u_is_identifiable_profile``) or the legacy Lillis LHS proxy
(``rows_u_is_identifiable_lhs_dchi2red_0p001``). The temporal loader must prefer
the profile key when present, fall back to the legacy key otherwise, and honor an
explicit override.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.temporal.coefficients import (
    LEGACY_U_IDENTIFIABLE_KEY,
    PROFILE_U_IDENTIFIABLE_KEY,
    resolve_u_identifiable_key,
)


def _write_npz(path: Path, keys: list[str]) -> Path:
    np.savez(path, **{k: np.zeros(2, dtype=bool) for k in keys})
    return path


def test_resolve_prefers_profile(tmp_path: Path) -> None:
    p = _write_npz(
        tmp_path / "a.npz",
        [PROFILE_U_IDENTIFIABLE_KEY, LEGACY_U_IDENTIFIABLE_KEY],
    )
    assert resolve_u_identifiable_key([p]) == PROFILE_U_IDENTIFIABLE_KEY


def test_resolve_profile_across_files(tmp_path: Path) -> None:
    # First file has only legacy; a later file has profile -> profile wins.
    legacy = _write_npz(tmp_path / "a.npz", [LEGACY_U_IDENTIFIABLE_KEY])
    profile = _write_npz(tmp_path / "b.npz", [PROFILE_U_IDENTIFIABLE_KEY])
    assert resolve_u_identifiable_key([legacy, profile]) == PROFILE_U_IDENTIFIABLE_KEY


def test_resolve_falls_back_to_legacy(tmp_path: Path) -> None:
    p = _write_npz(tmp_path / "a.npz", [LEGACY_U_IDENTIFIABLE_KEY])
    assert resolve_u_identifiable_key([p]) == LEGACY_U_IDENTIFIABLE_KEY


def test_resolve_override_wins(tmp_path: Path) -> None:
    p = _write_npz(tmp_path / "a.npz", [PROFILE_U_IDENTIFIABLE_KEY])
    assert resolve_u_identifiable_key([p], override="custom_key") == "custom_key"


def test_resolve_default_when_absent(tmp_path: Path) -> None:
    p = _write_npz(tmp_path / "a.npz", ["rows_projected_potential"])
    assert resolve_u_identifiable_key([p]) == LEGACY_U_IDENTIFIABLE_KEY
