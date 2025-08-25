from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class LockEntry:
    """One line from kernels.lock: '<sha1>  spice_kernels/<name>'"""

    sha1: str
    relpath: str


@dataclass(frozen=True)
class VerificationResult:
    """Result of verifying files against kernels.lock."""

    missing: List[Path]
    extra: List[Path]
    mismatched: List[Tuple[Path, str, str]]  # (path, expected_sha1, actual_sha1)


def _iter_lock_lines(text: str) -> Iterable[str]:
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        yield line


def read_lock(lock_path: Path) -> List[LockEntry]:
    """Parse kernels.lock into entries.

    Expects lines like: '<sha1>  spice_kernels/<filename>' (two spaces between).
    """
    entries: List[LockEntry] = []
    text = lock_path.read_text(encoding="utf-8")
    for line in _iter_lock_lines(text):
        parts = line.split()
        if len(parts) < 2:
            # Skip malformed lines silently to keep function robust
            continue
        sha1, relpath = parts[0], parts[-1]
        # Normalize legacy entries like 'spice_kernels/<name>' to just '<name>'
        name_only = Path(relpath).name
        entries.append(LockEntry(sha1=sha1, relpath=name_only))
    return entries


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-1 of a file in chunks (1 MiB default)."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_kernels_lock(
    kernels_dir: Path,
    lock_path: Path,
    *,
    verify_hashes: bool = True,
) -> VerificationResult:
    """Verify that files in `kernels_dir` match entries in `kernels.lock`.

    - Flags files present in lock but missing on disk (missing).
    - Flags files on disk but not present in lock (extra).
    - Optionally verifies SHA-1s (mismatched) when `verify_hashes` is True.
    """
    entries = read_lock(lock_path)
    # Map expected filenames (strict) to expected entry
    expected_by_name = {e.relpath: e for e in entries}
    # Actual files (exclude the lock itself)
    actual_files = {
        p.name: p
        for p in kernels_dir.iterdir()
        if p.is_file() and p.name != lock_path.name
    }

    missing: List[Path] = [kernels_dir / name for name in expected_by_name if name not in actual_files]
    extra: List[Path] = [path for name, path in actual_files.items() if name not in expected_by_name]

    mismatched: List[Tuple[Path, str, str]] = []
    if verify_hashes:
        for name, path in actual_files.items():
            entry = expected_by_name.get(name)
            if entry is None:
                continue
            actual_sha1 = sha1_file(path)
            if actual_sha1 != entry.sha1:
                mismatched.append((path, entry.sha1, actual_sha1))

    return VerificationResult(missing=missing, extra=extra, mismatched=mismatched)
