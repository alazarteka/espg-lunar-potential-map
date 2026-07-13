#!/usr/bin/env python3
"""Write the final tracked/ignored output manifest for Experiment 008."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment-008"
OUTPUT_PATH = EXPERIMENT_DIR / "result-manifest.csv"


def sha256_file(path: Path) -> str:
    """Hash one output in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def result_paths() -> list[tuple[str, Path]]:
    """Return compact tracked results and large ignored replay artifacts."""
    paths = [
        ("tracked_result", path)
        for path in sorted(RESULTS_DIR.rglob("*"))
        if path.is_file()
    ]
    paths.extend(
        ("tracked_result", EXPERIMENT_DIR / relative)
        for relative in (
            "RESULTS.md",
            "phase-1-results.md",
            "supplement/README.md",
            "supplement/main.tex",
            "supplement/main.pdf",
        )
    )
    artifact_names = (
        "geometry.npz",
        "canonical-sweep-basis-l10.npy",
        "canonical-sweep-basis-l10.json",
        "canonical-sweep-temporal-synodic-l10.npy",
        "canonical-sweep-temporal-synodic-l10.json",
        "injection-cells.csv",
        "injection-errors.csv",
    )
    paths.extend(
        ("ignored_replay_artifact", ARTIFACT_DIR / name) for name in artifact_names
    )
    paths.extend(
        ("ignored_replay_artifact", path)
        for path in sorted(ARTIFACT_DIR.glob("sensitivity-basis-*.npy"))
    )
    missing = [path for _, path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing final outputs: {missing}")
    return paths


def main() -> int:
    """Write deterministic hashes for all final outputs."""
    records = []
    for kind, path in result_paths():
        records.append(
            {
                "kind": kind,
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    with OUTPUT_PATH.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=records[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)
    print(f"Wrote {len(records)} output hashes to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
