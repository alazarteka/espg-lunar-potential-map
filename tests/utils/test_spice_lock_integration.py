from __future__ import annotations

import os

import pytest

from src import config
from src.utils.spice_lock import verify_kernels_lock


@pytest.mark.skip_ci
def test_local_spice_kernels_match_lock_if_present() -> None:
    kernels_dir = config.SPICE_KERNELS_DIR
    lock = kernels_dir / "kernels.lock"

    if not kernels_dir.exists() or not lock.exists():
        pytest.skip("Local SPICE kernels or lock file not present")

    verify_hashes = os.getenv("SPICE_LOCK_FULL_HASH") == "1"
    res = verify_kernels_lock(kernels_dir, lock_path=lock, verify_hashes=verify_hashes)

    if res.missing:
        pytest.skip(f"Missing {len(res.missing)} kernels, skipping integration test")

    # At minimum, structure should match locally
    assert res.missing == []
    assert res.extra == []
    if verify_hashes:
        assert res.mismatched == []
