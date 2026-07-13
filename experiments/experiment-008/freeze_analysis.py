#!/usr/bin/env python3
"""Freeze Experiment 008 analysis code, caches, and deterministic cell seeds."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment-008"
OUTPUT_PATH = EXPERIMENT_DIR / "analysis-freeze.json"
CELL_PATH = ARTIFACT_DIR / "injection-cells.csv"
CONFIGURATION_PATH = EXPERIMENT_DIR / "configuration.json"


def sha256_file(path: Path) -> str:
    """Hash one file in bounded memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def stable_seed_hex(namespace: str, *parts: object) -> str:
    """Return the frozen 128-bit root seed for a labeled analysis cell."""
    payload = "\x1f".join((namespace, *(str(part) for part in parts))).encode("ascii")
    return hashlib.sha256(payload).digest()[:16].hex()


def injection_cells() -> Iterable[dict[str, object]]:
    """Yield every prospectively declared window/control recovery cell."""
    sys.path.insert(0, str(EXPERIMENT_DIR))
    import observation_operator as operator
    import run_analysis

    with CONFIGURATION_PATH.open(encoding="ascii") as stream:
        configuration = json.load(stream)
    namespace = str(configuration["injections"]["seed_namespace"])
    geometry = operator.load_sweep_geometry(run_analysis.GEOMETRY_PATH)
    for duration in configuration["basis"]["window_hours"]:
        windows, _ = run_analysis.iter_windows(
            geometry.midpoint_utc_ns, float(duration)
        )
        for lmax in configuration["basis"]["window_lmax"]:
            for window_id, _ in windows:
                labels = ("window", float(duration), int(lmax), int(window_id))
                yield {
                    "class": labels[0],
                    "duration_hours": labels[1],
                    "lmax": labels[2],
                    "model": "independent",
                    "window_or_split": labels[3],
                    "root_seed_hex": stable_seed_hex(namespace, *labels),
                }
    for lmax in configuration["basis"]["static_lmax"]:
        labels = ("static", int(lmax), "mission")
        yield {
            "class": labels[0],
            "duration_hours": "",
            "lmax": labels[1],
            "model": "constant",
            "window_or_split": labels[2],
            "root_seed_hex": stable_seed_hex(namespace, *labels),
        }
    for model in configuration["basis"]["temporal_models"]:
        for lmax in configuration["basis"]["temporal_lmax"]:
            labels = ("temporal", str(model), int(lmax), "mission")
            yield {
                "class": labels[0],
                "duration_hours": "",
                "lmax": labels[2],
                "model": labels[1],
                "window_or_split": labels[3],
                "root_seed_hex": stable_seed_hex(namespace, *labels),
            }
    for model in configuration["basis"]["temporal_models"]:
        for split in (
            "mission_tail",
            *(f"longitude_{sector:02d}" for sector in range(12)),
            *(f"seven_day_fold_{fold}" for fold in range(4)),
        ):
            labels = ("blocked", str(model), split)
            yield {
                "class": labels[0],
                "duration_hours": "",
                "lmax": 5,
                "model": labels[1],
                "window_or_split": labels[2],
                "root_seed_hex": stable_seed_hex(namespace, *labels),
            }


def write_cells() -> int:
    """Write the deterministic ignored cell manifest."""
    rows = list(injection_cells())
    CELL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CELL_PATH.open("w", encoding="ascii", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=rows[0].keys(),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def git_head() -> str:
    """Return the committed analysis-code revision."""
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()


def main() -> int:
    """Write the final pre-outcome analysis freeze."""
    cell_count = write_cells()
    relative_paths = (
        "experiments/experiment-008/protocol.md",
        "experiments/experiment-008/configuration.json",
        "experiments/experiment-008/protocol-amendment-01.md",
        "experiments/experiment-008/configuration-amendment-01.json",
        "experiments/experiment-008/protocol-amendment-02.md",
        "experiments/experiment-008/configuration-amendment-02.json",
        "experiments/experiment-008/protocol-amendment-03.md",
        "experiments/experiment-008/configuration-amendment-03.json",
        "experiments/experiment-008/input-manifest.csv",
        "experiments/experiment-008/geometry-census.csv",
        "experiments/experiment-008/geometry.py",
        "experiments/experiment-008/observation_operator.py",
        "experiments/experiment-008/run_analysis.py",
        "experiments/experiment-008/test_geometry.py",
        "experiments/experiment-008/test_operator.py",
        "experiments/experiment-008/test_run_analysis.py",
        "artifacts/experiment-008/geometry.npz",
        "artifacts/experiment-008/canonical-sweep-basis-l10.npy",
        "artifacts/experiment-008/canonical-sweep-basis-l10.json",
        "artifacts/experiment-008/injection-cells.csv",
    )
    records = []
    for relative in relative_paths:
        path = PROJECT_ROOT / relative
        records.append(
            {
                "path": relative,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    freeze = {
        "analysis_git_head": git_head(),
        "injection_cell_count": cell_count,
        "records": records,
        "schema_version": 1,
        "status": "frozen_before_scientific_operator_outcomes",
    }
    with OUTPUT_PATH.open("w", encoding="ascii") as stream:
        json.dump(freeze, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(f"Wrote {OUTPUT_PATH} with {len(records)} hashes and {cell_count} cells")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
