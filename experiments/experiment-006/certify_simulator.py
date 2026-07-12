#!/usr/bin/env python3
"""Run pre-C1 structural certification of the frozen sector simulator."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import simulator
import support_data


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def certify(
    rows: list[dict[str, str]],
    supports: dict[int, support_data.GeometrySupport],
    config: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate deterministic, positivity, masking, and branch-separation gates."""
    failures: Counter[str] = Counter()
    branch_differences: list[float] = []
    usable_rows: list[int] = []
    family_counts: Counter[str] = Counter()
    for row in rows:
        support = supports[int(row["support_id"])]
        case = simulator.simulate_case(support, row, config)
        replay = simulator.simulate_case(support, row, config)
        if not np.array_equal(case.native_flux, replay.native_flux):
            failures["nondeterministic_native_flux"] += 1
        if not np.array_equal(case.native_valid, replay.native_valid):
            failures["nondeterministic_mask"] += 1
        if not np.isfinite(case.native_flux).all() or not np.all(case.native_flux > 0):
            failures["nonpositive_or_nonfinite_native_flux"] += 1
        common = np.isfinite(case.incident_mean) & np.isfinite(case.conjugate)
        rows_with_common = int(np.count_nonzero(np.any(common, axis=1)))
        usable_rows.append(rows_with_common)
        if rows_with_common < 2:
            failures["fewer_than_two_common_normalized_rows"] += 1
        if np.any(common):
            branch_differences.append(
                float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                np.log(case.incident_mean[common])
                                - np.log(case.conjugate[common])
                            )
                        )
                    )
                )
            )
        else:
            failures["no_common_normalized_samples"] += 1
        family_counts[str(row["background_family"])] += 1
    difference = np.asarray(branch_differences)
    return {
        "cases": len(rows),
        "failures": dict(sorted(failures.items())),
        "minimum_common_normalized_rows": min(usable_rows),
        "branch_log_rms_median": float(np.median(difference)),
        "branch_log_rms_minimum": float(np.min(difference)),
        "background_counts": dict(sorted(family_counts.items())),
        "determinism_pass": not any(
            key.startswith("nondeterministic") for key in failures
        ),
        "positivity_pass": failures["nonpositive_or_nonfinite_native_flux"] == 0,
        "normalization_support_pass": failures["fewer_than_two_common_normalized_rows"]
        == 0,
        "nontrivial_branch_pass": bool(np.min(difference) > 1e-10),
        "full_gate_pass": not failures and bool(np.min(difference) > 1e-10),
    }


def main() -> None:
    directory = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--output", type=Path, default=directory / "c1-simulator-certification.json"
    )
    args = parser.parse_args()
    config_path = directory / "configuration-c1.json"
    manifest_path = directory / "c1-case-manifest.csv"
    config = json.loads(config_path.read_text(encoding="ascii"))
    with manifest_path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit is not None:
        rows = rows[: args.limit]
    geometry = support_data.load_geometry_supports(
        Path("data/1999/091_120APR/3D990429.TAB"),
        directory / "support_manifest.csv",
    )
    summary = certify(rows, {support.spec_no: support for support in geometry}, config)
    summary["configuration_sha256"] = sha256(config_path)
    summary["manifest_sha256"] = sha256(manifest_path)
    summary["simulator_sha256"] = sha256(directory / "simulator.py")
    summary["confirmation_flux_opened"] = False
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
