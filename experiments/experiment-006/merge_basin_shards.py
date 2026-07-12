#!/usr/bin/env python3
"""Validate and merge frozen Experiment 006 D1 shard artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import basin_certification


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shards(paths: list[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load shards after enforcing commit, configuration, and partition identity."""
    provenances = [
        json.loads((path / "d1-certification-provenance.json").read_text())
        for path in paths
    ]
    counts = {item["shard_count"] for item in provenances}
    indices = [item["shard_index"] for item in provenances]
    commits = {item["repository_commit"] for item in provenances}
    configurations = {item["configuration_sha256"] for item in provenances}
    if counts != {len(paths)} or sorted(indices) != list(range(len(paths))):
        raise ValueError("Shard set is not one complete indexed partition")
    if len(commits) != 1 or len(configurations) != 1:
        raise ValueError("Shard commits or configuration hashes differ")
    rows: list[dict[str, Any]] = []
    for path, provenance in zip(paths, provenances, strict=True):
        results_path = path / "d1-certification-results.csv"
        if sha256_file(results_path) != provenance["results_sha256"]:
            raise ValueError(f"Results hash mismatch in {path}")
        with results_path.open(encoding="ascii", newline="") as handle:
            rows.extend(csv.DictReader(handle))
    ids = [int(row["case_id"]) for row in rows]
    expected = list(range(512))
    if len(ids) != len(set(ids)) or sorted(ids) != expected:
        raise ValueError("Merged case IDs are duplicated, missing, or outside 0..511")
    rows.sort(key=lambda row: int(row["case_id"]))
    return rows, provenances[0]


def coerce_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restore scalar CSV fields used by the summary implementation."""
    bool_fields = {
        "reference_success",
        "used_fallback",
        "final_contacts_bound",
        *(f"recall_k{k}" for k in range(1, 13)),
    }
    integer_fields = {
        "gpu_finite_candidates",
        "gpu_converged_candidates",
        "final_rank",
    }
    float_fields = {
        "final_objective_relative_difference",
        "final_log_model_rms",
        "final_center_difference_response_fwhm",
        "starts_seconds",
        "reference_seconds",
        "gpu_seconds",
        "all_gpu_polish_seconds",
    }
    for row in rows:
        for field in bool_fields:
            row[field] = row[field] == "True"
        for field in integer_fields:
            row[field] = int(row[field])
        for field in float_fields:
            row[field] = float(row[field])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    experiment_dir = Path(__file__).resolve().parent
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, default=experiment_dir)
    args = parser.parse_args()
    rows, provenance = load_shards(args.shards)
    configuration_path = experiment_dir / "configuration-d1-certification.json"
    if sha256_file(configuration_path) != provenance["configuration_sha256"]:
        raise ValueError("Current frozen configuration does not match shards")
    summary = basin_certification.summarize(
        coerce_rows(rows), basin_certification.load_configuration(configuration_path)
    )
    summary["repository_commit"] = provenance["repository_commit"]
    summary["execution"] = "merged_shards"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.output_dir / "d1-certification-results.csv"
    basin_certification.write_csv(results_path, rows)
    summary_path = args.output_dir / "d1-certification-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    merged_provenance = {
        "repository_commit": provenance["repository_commit"],
        "configuration_sha256": provenance["configuration_sha256"],
        "results_sha256": sha256_file(results_path),
        "summary_sha256": sha256_file(summary_path),
        "confirmation_flux_opened": False,
        "shard_count": len(args.shards),
    }
    (args.output_dir / "d1-certification-provenance.json").write_text(
        json.dumps(merged_provenance, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(main())
