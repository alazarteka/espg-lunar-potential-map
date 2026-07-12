#!/usr/bin/env python3
"""Run deterministic, resumable shards of the frozen Experiment 006 C1 corpus.

Usage:
    python experiments/experiment-006/run_c1.py --start 0 --stop 300 \
        --output artifacts/experiment-006/c1-0000-0299.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

import c1_estimator
import simulator
import support_data

if TYPE_CHECKING:
    from collections.abc import Iterable

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "c1-case-manifest.csv"
DEFAULT_CONFIG = ROOT / "configuration-c1.json"
DEFAULT_SUPPORT_MANIFEST = ROOT / "support_manifest.csv"
DEFAULT_DATA = Path("data/1999/091_120APR/3D990429.TAB")
DEFAULT_OUTPUT = ROOT / "c1-results.csv"

FACTOR_FIELDS = (
    "background_family",
    "noise_level",
    "response_family",
    "amplitude_quartile",
    "pitch_sigma_quartile",
    "temporal_family",
    "response_direction",
)
TRUTH_FIELDS = (
    "anchor_ev",
    "true_center_ev",
    "beam_amplitude",
    "pitch_sigma_deg",
    "beam_axis_deg",
)


def load_manifest(path: Path = DEFAULT_MANIFEST) -> list[dict[str, str]]:
    """Load the frozen manifest and require its canonical contiguous case IDs."""
    with path.open(encoding="ascii", newline="") as handle:
        rows = list(csv.DictReader(handle))
    ids = [int(row["case_id"]) for row in rows]
    if ids != list(range(len(rows))):
        raise ValueError("C1 manifest case IDs must be contiguous and ordered")
    return rows


def load_configuration(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the frozen C1 configuration without consulting confirmation data."""
    return json.loads(path.read_text(encoding="ascii"))


def load_calibration_geometry(
    data_path: Path = DEFAULT_DATA,
    support_manifest_path: Path = DEFAULT_SUPPORT_MANIFEST,
) -> dict[int, support_data.GeometrySupport]:
    """Load only calibration geometry through the geometry-only support reader."""
    supports = support_data.load_geometry_supports(
        data_path, support_manifest_path, split="calibration"
    )
    return {support.spec_no: support for support in supports}


def _fit_fields(assessment: c1_estimator.BranchAssessment) -> dict[str, Any]:
    fit = assessment.fit
    prefix = assessment.name
    return {
        f"{prefix}_fit_success": fit is not None,
        f"{prefix}_center_ev": "" if fit is None else float(fit.parameters[1]),
        f"{prefix}_center_se_ev": "" if fit is None else float(fit.standard_errors[1]),
        f"{prefix}_qc_valid": fit is not None and not assessment.reason_codes,
        f"{prefix}_reason_codes": ";".join(assessment.reason_codes),
    }


def run_case(
    manifest_row: dict[str, str],
    support: support_data.GeometrySupport,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    """Simulate and fit one manifest row, returning one flat CSV record."""
    simulated = simulator.simulate_case(support, manifest_row, configuration)
    estimate = c1_estimator.fit_c1(
        support.energies_ev,
        support.channel_low_ev,
        support.channel_high_ev,
        support.pitches_deg,
        simulated.native_flux,
        simulated.native_valid,
        simulated.incident_mean,
        simulated.conjugate,
        configuration=configuration,
    )
    result: dict[str, Any] = {
        "case_id": int(manifest_row["case_id"]),
        "split": manifest_row["split"],
        "support_id": int(manifest_row["support_id"]),
        **{field: manifest_row[field] for field in TRUTH_FIELDS},
        **{field: manifest_row[field] for field in FACTOR_FIELDS},
        "base_valid": estimate.base_valid,
        "reason_codes": ";".join(estimate.reason_codes),
        **_fit_fields(estimate.incident_mean),
        **_fit_fields(estimate.conjugate),
        "simulation_factors_json": json.dumps(
            simulated.latents,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ),
    }
    return result


def _read_checkpoint(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="ascii", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    """Atomically replace a checkpoint with rows sorted by case ID."""
    ordered = sorted(rows, key=lambda row: int(row["case_id"]))
    if not ordered:
        return
    fieldnames = list(ordered[0])
    if any(list(row) != fieldnames for row in ordered):
        raise ValueError("C1 result rows have inconsistent CSV fields")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(ordered)
    os.replace(temporary, path)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one frozen input or generated artifact."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_provenance(
    output: Path,
    *,
    manifest_path: Path,
    config_path: Path,
    start: int,
    stop: int,
) -> None:
    """Write the resumable shard's frozen-input and source provenance."""
    source_paths = {
        name: ROOT / name
        for name in ("run_c1.py", "simulator.py", "c1_estimator.py", "support_data.py")
    }
    provenance = {
        "configuration_sha256": sha256_file(config_path),
        "confirmation_flux_opened": False,
        "manifest_sha256": sha256_file(manifest_path),
        "output_sha256": sha256_file(output),
        "range": {"start": start, "stop": stop},
        "source_sha256": {
            name: sha256_file(path) for name, path in source_paths.items()
        },
    }
    sidecar = output.with_name(f"{output.name}.provenance.json")
    temporary = sidecar.with_name(f".{sidecar.name}.tmp")
    temporary.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    os.replace(temporary, sidecar)


def run(
    *,
    start: int,
    stop: int | None,
    output: Path,
    manifest_path: Path = DEFAULT_MANIFEST,
    config_path: Path = DEFAULT_CONFIG,
    data_path: Path = DEFAULT_DATA,
    support_manifest_path: Path = DEFAULT_SUPPORT_MANIFEST,
) -> list[dict[str, Any]]:
    """Run ``[start, stop)`` and resume completed IDs from ``output``."""
    manifest = load_manifest(manifest_path)
    end = len(manifest) if stop is None else stop
    if start < 0 or end < start or end > len(manifest):
        raise ValueError(f"range must satisfy 0 <= start <= stop <= {len(manifest)}")
    prior = _read_checkpoint(output)
    completed: dict[int, dict[str, Any]] = {}
    for row in prior:
        case_id = int(row["case_id"])
        if case_id in completed or not 0 <= case_id < len(manifest):
            raise ValueError(f"invalid or duplicate checkpoint case ID {case_id}")
        completed[case_id] = row

    pending = [
        row for row in manifest[start:end] if int(row["case_id"]) not in completed
    ]
    if not pending:
        if output.exists():
            write_provenance(
                output,
                manifest_path=manifest_path,
                config_path=config_path,
                start=start,
                stop=end,
            )
        return list(completed.values())
    configuration = load_configuration(config_path)
    geometry = load_calibration_geometry(data_path, support_manifest_path)
    for manifest_row in pending:
        support_id = int(manifest_row["support_id"])
        if support_id not in geometry:
            raise RuntimeError(f"calibration geometry missing support {support_id}")
        result = run_case(manifest_row, geometry[support_id], configuration)
        completed[int(manifest_row["case_id"])] = result
        write_csv(output, completed.values())
        write_provenance(
            output,
            manifest_path=manifest_path,
            config_path=config_path,
            start=start,
            stop=end,
        )
    return list(completed.values())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(start=args.start, stop=args.stop, output=args.output)


if __name__ == "__main__":
    main()
