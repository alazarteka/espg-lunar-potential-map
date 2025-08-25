from __future__ import annotations

from pathlib import Path
import hashlib

from src.utils.spice_lock import read_lock, verify_kernels_lock


def _sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def test_verify_kernels_lock_with_temp_files(tmp_path: Path) -> None:
    # Create a fake kernels dir with small files
    kernels_dir = tmp_path
    a = kernels_dir / "a.bin"
    b = kernels_dir / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")

    lock = kernels_dir / "kernels.lock"
    lock.write_text(
        f"{_sha1_bytes(b'hello')}  {a.name}\n"
        f"{_sha1_bytes(b'world')}  {b.name}\n",
        encoding="utf-8",
    )

    # All good initially
    res = verify_kernels_lock(kernels_dir, lock_path=lock, verify_hashes=True)
    assert res.missing == []
    assert res.extra == []
    assert res.mismatched == []

    # Change one file to trigger mismatch
    a.write_bytes(b"HELLO")
    res = verify_kernels_lock(kernels_dir, lock_path=lock, verify_hashes=True)
    assert any(t[0].name == "a.bin" for t in res.mismatched)

    # Remove one file to trigger missing and add an extra file
    b.unlink()
    c = kernels_dir / "c.bin"
    c.write_bytes(b"extra")
    res = verify_kernels_lock(kernels_dir, lock_path=lock, verify_hashes=False)
    # b is missing
    assert any(p.name == "b.bin" for p in res.missing)
    # c is extra
    assert any(p.name == "c.bin" for p in res.extra)
