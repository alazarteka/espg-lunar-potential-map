#!/usr/bin/env python3
"""Freeze the raw geometry-only input inventory for Experiment 008.

Run from the repository root:

    uv run python experiments/experiment-008/freeze_inputs.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = Path(__file__).with_name("input-manifest.csv")
HASH_CHUNK_BYTES = 8 * 1024 * 1024


def discover_inputs(project_root: Path) -> list[tuple[str, Path]]:
    """Return the frozen raw, attitude, and loaded-kernel input files."""
    data_dir = project_root / "data"
    inputs: list[tuple[str, Path]] = []
    for year in ("1998", "1999"):
        inputs.extend(
            ("raw_er_mag", path)
            for path in sorted((data_dir / year).glob("**/3D*.TAB"))
        )
    inputs.append(("attitude", data_dir / "attitude.tab"))
    spice_dir = data_dir / "spice_kernels"
    for suffix in ("*.bsp", "*.tpc", "*.tls"):
        inputs.extend(("spice", path) for path in sorted(spice_dir.glob(suffix)))
    missing = [path for _, path in inputs if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing frozen inputs: {missing}")
    return inputs


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_rows(
    project_root: Path, inputs: Iterable[tuple[str, Path]]
) -> Iterable[dict[str, str | int]]:
    """Yield deterministic manifest rows."""
    for kind, path in inputs:
        stat = path.stat()
        yield {
            "kind": kind,
            "path": path.relative_to(project_root).as_posix(),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": sha256_file(path),
        }


def write_manifest(project_root: Path, output: Path) -> int:
    """Write the complete manifest and return its row count."""
    rows = list(manifest_rows(project_root, discover_inputs(project_root)))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("kind", "path", "size_bytes", "mtime_ns", "sha256"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = write_manifest(args.project_root.resolve(), args.output.resolve())
    print(f"Wrote {count} frozen inputs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
