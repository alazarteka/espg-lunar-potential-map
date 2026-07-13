#!/usr/bin/env python3
"""Generate the prospective 192-case Experiment 006 G2 certification corpus."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
CONFIG = ROOT / "configuration-g2.json"
SUPPORTS = ROOT / "support_manifest.csv"
OUTPUT = ROOT / "g2-certification-corpus.csv"

ARM_COUNTS = {
    "matched": 48,
    "discrepancy": 72,
    "calibration_geometry": 48,
    "control": 24,
}
BACKGROUNDS = (
    "independent_smooth_logistic",
    "double_transition_refilling",
    "asymmetric_shoulder",
    "correlated_smooth_energy_pitch_field",
)
RESPONSES = (
    "nominal_linear_gaussian",
    "width_stress",
    "energy_scale_offset",
    "log_energy_gaussian",
    "intrinsic_asymmetric_broadening",
)
TEMPORAL = ("none", "ramp", "pulse")
NOISE = (0.2, 0.25, 0.35, 0.45)
ANCHORS = (100.0, 140.0, 220.0, 340.0, 500.0)
NAMESPACES = (
    "allocation",
    "continuum",
    "beam_morphology",
    "response",
    "noise",
    "temporal",
    "mask",
    "diagnostic_resampling",
)


def canonical_json(value: Any) -> str:
    """Return canonical compact ASCII JSON."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def seed(root: str, case_id: int, namespace: str) -> int:
    """Derive one deterministic unsigned BLAKE2b-128 seed."""
    payload = canonical_json([root, int(case_id), namespace]).encode("ascii")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=16).digest(), "little")


def _supports(path: Path) -> list[int]:
    with path.open(encoding="ascii", newline="") as handle:
        values = [
            int(row["spec_no"])
            for row in csv.DictReader(handle)
            if row["split"] == "calibration" and row["eligible"] == "True"
        ]
    if len(values) != 12 or len(set(values)) != 12:
        raise ValueError("G2 requires exactly 12 unique calibration supports")
    return sorted(values)


def generate(
    config_path: Path = CONFIG, support_path: Path = SUPPORTS
) -> list[dict[str, Any]]:
    """Generate all rows without opening geometry, flux, or fit outcomes."""
    config = json.loads(config_path.read_text(encoding="ascii"))
    root = str(config["corpus"]["seed_root"])
    supports = _supports(support_path)
    rows: list[dict[str, Any]] = []
    case_id = 0
    for arm, count in ARM_COUNTS.items():
        for local in range(count):
            rng = np.random.default_rng(seed(root, case_id, "allocation"))
            center = ANCHORS[local % len(ANCHORS)]
            amplitude = float(np.exp(rng.uniform(math.log(4.0), math.log(16.0))))
            if arm == "discrepancy":
                center = float(np.exp(rng.uniform(math.log(80.0), math.log(700.0))))
            elif arm == "control":
                amplitude = (0.0, 0.5, 1.0, 2.0)[local % 4]
                center = (60.0, 100.0, 500.0, 750.0)[local % 4]
            coefficients = np.random.default_rng(
                seed(root, case_id, "continuum")
            ).uniform(-1.5, 1.5, size=12)
            row: dict[str, Any] = {
                "case_id": case_id,
                "arm": arm,
                "generator": (
                    "canonical_matched" if arm == "matched" else "independent_sector"
                ),
                "support_id": supports[local % len(supports)],
                "normalization": "incident_mean" if local % 2 == 0 else "conjugate",
                "true_center_ev": center,
                "amplitude": amplitude,
                "beam_amplitude": amplitude,
                "pitch_sigma_deg": float(rng.uniform(12.0, 42.0)),
                "beam_axis_deg": 180.0,
                "background_family": BACKGROUNDS[local % len(BACKGROUNDS)],
                "response_family": RESPONSES[local % len(RESPONSES)],
                "response_direction": local % 2,
                "temporal_family": TEMPORAL[local % len(TEMPORAL)],
                "log_noise_sigma": NOISE[local % len(NOISE)],
                **{
                    f"coefficient_{index}": float(value)
                    for index, value in enumerate(coefficients)
                },
                **{
                    f"seed_{namespace}": seed(root, case_id, namespace)
                    for namespace in NAMESPACES
                },
            }
            rows.append(row)
            case_id += 1
    validate(rows, config, supports)
    return rows


def validate(
    rows: list[dict[str, Any]], config: dict[str, Any], supports: list[int]
) -> None:
    """Reject corpus drift before any fitting occurs."""
    if len(rows) != int(config["corpus"]["total_cases"]):
        raise ValueError("G2 corpus row count differs from configuration")
    if [int(row["case_id"]) for row in rows] != list(range(len(rows))):
        raise ValueError("G2 case IDs must be contiguous and ordered")
    for arm, expected in ARM_COUNTS.items():
        selected = [row for row in rows if row["arm"] == arm]
        if len(selected) != expected:
            raise ValueError(f"G2 arm {arm} has the wrong size")
    if {int(row["support_id"]) for row in rows} != set(supports):
        raise ValueError("G2 corpus does not span all calibration supports")
    if any(row["normalization"] not in {"incident_mean", "conjugate"} for row in rows):
        raise ValueError("Unknown normalization branch")


def write(rows: list[dict[str, Any]], output: Path = OUTPUT) -> dict[str, Any]:
    """Write the frozen corpus and source-bearing provenance."""
    with output.open("w", encoding="ascii", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    provenance = {
        "rows": len(rows),
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "configuration_sha256": hashlib.sha256(CONFIG.read_bytes()).hexdigest(),
        "support_manifest_sha256": hashlib.sha256(SUPPORTS.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "partial_c1_scientific_outcomes_opened": False,
        "confirmation_flux_opened": False,
    }
    output.with_suffix(output.suffix + ".provenance.json").write_text(
        canonical_json(provenance) + "\n", encoding="ascii"
    )
    return provenance


def main() -> None:
    """Generate and print frozen corpus provenance."""
    print(json.dumps(write(generate()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
