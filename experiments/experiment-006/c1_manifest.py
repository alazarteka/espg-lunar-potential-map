"""Generate the frozen deterministic Experiment 006 C1 case manifest.

Usage:
    python experiments/experiment-006/c1_manifest.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "configuration-c1.json"
DEFAULT_SUPPORTS = ROOT / "support_manifest.csv"
DEFAULT_OUTPUT = ROOT / "c1-case-manifest.csv"

FACTOR_VALUES: dict[str, tuple[str, ...]] = {
    "background_family": (
        "independent_smooth_logistic",
        "double_transition_refilling",
        "asymmetric_shoulder",
        "correlated_smooth_energy_pitch_field",
    ),
    "noise_level": ("0.2", "0.25", "0.35", "0.45"),
    "response_family": (
        "nominal_linear_gaussian",
        "width_stress",
        "energy_scale_offset",
        "log_energy_gaussian",
        "intrinsic_asymmetric_broadening",
    ),
    "amplitude_quartile": ("0", "1", "2", "3"),
    "pitch_sigma_quartile": ("0", "1", "2", "3"),
    "temporal_family": ("none", "ramp", "pulse"),
}


def canonical_json(value: Any) -> str:
    """Return the ASCII compact canonical JSON representation used for seeds."""
    return json.dumps(
        value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
    )


def namespace_seed(
    root: str,
    namespace: str,
    anchor_ev: float,
    split: str,
    case_id: int,
    support_id: int,
) -> int:
    """Derive one BLAKE2b-128 little-endian namespace seed."""
    payload = canonical_json(
        [root, namespace, float(anchor_ev), split, int(case_id), int(support_id)]
    ).encode("ascii")
    digest = hashlib.blake2b(payload, digest_size=16).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def _load_calibration_supports(path: Path) -> list[int]:
    with path.open(newline="", encoding="ascii") as handle:
        supports = [
            int(row["spec_no"])
            for row in csv.DictReader(handle)
            if row["split"] == "calibration" and row["eligible"] == "True"
        ]
    if len(supports) != 12 or len(set(supports)) != 12:
        raise ValueError(
            f"expected 12 unique eligible calibration supports, got {supports}"
        )
    return sorted(supports)


def _expanded(values: tuple[str, ...], counts: tuple[int, ...]) -> list[str]:
    return [
        value for value, count in zip(values, counts, strict=True) for _ in range(count)
    ]


def _pair_penalty(left: list[str], right: list[str]) -> int:
    left_counts = Counter(left)
    right_counts = Counter(right)
    cells = Counter(zip(left, right, strict=True))
    n = len(left)
    penalty = 0
    for a, count_a in left_counts.items():
        for b, count_b in right_counts.items():
            expectation = count_a * count_b / n
            low, high = math.floor(expectation), math.ceil(expectation)
            observed = cells[a, b]
            if observed < low:
                penalty += low - observed
            elif observed > high:
                penalty += observed - high
    return penalty


def _allocation_penalty(columns: dict[str, list[str]]) -> int:
    return sum(
        _pair_penalty(columns[a], columns[b])
        for a, b in itertools.combinations(columns, 2)
    )


def _balanced_block(
    seed: int, supports: list[int], max_attempts: int
) -> dict[str, list[str]]:
    """Construct a 300-row allocation satisfying every pairwise balance bound."""
    rng = random.Random(seed)
    specifications = {
        "support_id": (tuple(map(str, supports)), (25,) * 12),
        "background_family": (FACTOR_VALUES["background_family"], (75,) * 4),
        "noise_level": (FACTOR_VALUES["noise_level"], (75,) * 4),
        "response_family": (FACTOR_VALUES["response_family"], (60,) * 5),
        "amplitude_quartile": (FACTOR_VALUES["amplitude_quartile"], (75,) * 4),
        "pitch_sigma_quartile": (
            FACTOR_VALUES["pitch_sigma_quartile"],
            (75,) * 4,
        ),
        "temporal_family": (FACTOR_VALUES["temporal_family"], (150, 75, 75)),
    }
    columns: dict[str, list[str]] = {}
    for name, (values, counts) in specifications.items():
        if not columns:
            column = _expanded(values, counts)
            rng.shuffle(column)
            columns[name] = column
            continue
        row_count = 300
        level_count = len(values)
        constraints: list[tuple[list[int], float, float]] = [
            ([row * level_count + level for level in range(level_count)], 1, 1)
            for row in range(row_count)
        ]
        for level, count in enumerate(counts):
            constraints.append(
                ([row * level_count + level for row in range(row_count)], count, count)
            )
        for previous in columns.values():
            previous_counts = Counter(previous)
            for previous_value, previous_count in previous_counts.items():
                selected_rows = [
                    row for row, value in enumerate(previous) if value == previous_value
                ]
                for level, count in enumerate(counts):
                    expectation = previous_count * count / row_count
                    indices = [row * level_count + level for row in selected_rows]
                    constraints.append(
                        (indices, math.floor(expectation), math.ceil(expectation))
                    )
        matrix = lil_matrix((len(constraints), row_count * level_count), dtype=float)
        lower = np.empty(len(constraints))
        upper = np.empty(len(constraints))
        for constraint_index, (indices, low, high) in enumerate(constraints):
            matrix[constraint_index, indices] = 1.0
            lower[constraint_index] = low
            upper[constraint_index] = high
        result = milp(
            np.zeros(row_count * level_count),
            integrality=np.ones(row_count * level_count),
            bounds=Bounds(0.0, 1.0),
            constraints=LinearConstraint(matrix.tocsr(), lower, upper),
            options={"time_limit": float(max_attempts) / 1000.0},
        )
        if not result.success or result.x is None:
            raise RuntimeError(f"exact allocation failed for {name}: {result.message}")
        assignment = np.argmax(result.x.reshape(row_count, level_count), axis=1)
        columns[name] = [values[level] for level in assignment]
    if _allocation_penalty(columns):
        raise RuntimeError("allocation failed independent pairwise validation")
    return columns


def _stratified_value(
    seed: int, quartile: int, low: float, high: float, log: bool
) -> float:
    fraction = (quartile + random.Random(seed).random()) / 4.0
    if log:
        return math.exp(math.log(low) + fraction * math.log(high / low))
    return low + fraction * (high - low)


def generate_manifest(config_path: Path, support_path: Path) -> list[dict[str, Any]]:
    """Generate and validate all 3,000 deterministic C1 rows."""
    config = json.loads(config_path.read_text(encoding="ascii"))
    supports = _load_calibration_supports(support_path)
    root = config["seeds"]["root"]
    namespaces = config["seeds"]["namespaces"]
    max_attempts = int(config["allocation"]["allocation_search_max_attempts"])
    rows: list[dict[str, Any]] = []
    case_id = 0
    for anchor in config["allocation"]["anchors_ev"]:
        for split, ordinal_start in (("bias", 0), ("interval", 300)):
            allocation_seed = namespace_seed(
                root, "allocation", anchor, split, case_id, 0
            )
            columns = _balanced_block(allocation_seed, supports, max_attempts)
            response_occurrences: Counter[str] = Counter()
            for local in range(300):
                support_id = int(columns["support_id"][local])
                seeds = {
                    f"seed_{namespace}": namespace_seed(
                        root, namespace, anchor, split, case_id, support_id
                    )
                    for namespace in namespaces
                }
                amplitude_q = int(columns["amplitude_quartile"][local])
                pitch_q = int(columns["pitch_sigma_quartile"][local])
                row: dict[str, Any] = {
                    "case_id": case_id,
                    "anchor_ev": float(anchor),
                    "true_center_ev": float(anchor),
                    "anchor_ordinal": ordinal_start + local,
                    "split": split,
                    "support_id": support_id,
                    **{name: columns[name][local] for name in FACTOR_VALUES},
                    "beam_amplitude": _stratified_value(
                        seeds["seed_beam_morphology"], amplitude_q, 4.0, 16.0, True
                    ),
                    "pitch_sigma_deg": _stratified_value(
                        seeds["seed_continuum"], pitch_q, 12.0, 42.0, False
                    ),
                    "native_log_channel_phase": random.Random(
                        seeds["seed_response"]
                    ).uniform(-0.5, 0.5),
                    "beam_axis_deg": 180.0,
                    "response_direction": response_occurrences[
                        columns["response_family"][local]
                    ]
                    % 2,
                    **seeds,
                }
                row["amplitude"] = row["beam_amplitude"]
                row["log_noise_sigma"] = float(row["noise_level"])
                response_occurrences[columns["response_family"][local]] += 1
                rows.append(row)
                case_id += 1
    validate_manifest(rows, config, supports)
    return rows


def validate_manifest(
    rows: list[dict[str, Any]], config: dict[str, Any], supports: list[int]
) -> None:
    """Reject any manifest that violates identifiers, marginals, or pairwise balance."""
    if len(rows) != 3000 or [row["case_id"] for row in rows] != list(range(3000)):
        raise ValueError("manifest must contain case IDs 0 through 2999 exactly once")
    for anchor in config["allocation"]["anchors_ev"]:
        anchor_rows = [row for row in rows if row["anchor_ev"] == anchor]
        if len(anchor_rows) != 600:
            raise ValueError(f"anchor {anchor} does not contain 600 cases")
        for split in ("bias", "interval"):
            block = [row for row in anchor_rows if row["split"] == split]
            if len(block) != 300:
                raise ValueError(
                    f"anchor {anchor} split {split} does not contain 300 cases"
                )
            expected = {
                "support_id": Counter(dict.fromkeys(supports, 25)),
                "background_family": Counter(
                    dict.fromkeys(FACTOR_VALUES["background_family"], 75)
                ),
                "noise_level": Counter(dict.fromkeys(FACTOR_VALUES["noise_level"], 75)),
                "response_family": Counter(
                    dict.fromkeys(FACTOR_VALUES["response_family"], 60)
                ),
                "amplitude_quartile": Counter(
                    dict.fromkeys(FACTOR_VALUES["amplitude_quartile"], 75)
                ),
                "pitch_sigma_quartile": Counter(
                    dict.fromkeys(FACTOR_VALUES["pitch_sigma_quartile"], 75)
                ),
                "temporal_family": Counter({"none": 150, "ramp": 75, "pulse": 75}),
            }
            for name, counts in expected.items():
                if Counter(row[name] for row in block) != counts:
                    raise ValueError(f"invalid {name} marginal at {anchor} {split}")
            width_directions = Counter(
                int(row["response_direction"])
                for row in block
                if row["response_family"] == "width_stress"
            )
            if width_directions != Counter({0: 30, 1: 30}):
                raise ValueError(f"invalid width-stress directions at {anchor} {split}")
            columns = {name: [str(row[name]) for row in block] for name in expected}
            if _allocation_penalty(columns):
                raise ValueError(f"invalid pairwise balance at {anchor} {split}")
            for row in block:
                q_amp, q_pitch = (
                    int(row["amplitude_quartile"]),
                    int(row["pitch_sigma_quartile"]),
                )
                amp_fraction = math.log(row["beam_amplitude"] / 4.0) / math.log(4.0)
                pitch_fraction = (row["pitch_sigma_deg"] - 12.0) / 30.0
                if not q_amp / 4 <= amp_fraction < (q_amp + 1) / 4:
                    raise ValueError("amplitude lies outside its quartile")
                if not q_pitch / 4 <= pitch_fraction < (q_pitch + 1) / 4:
                    raise ValueError("pitch sigma lies outside its quartile")
                if not -0.5 <= row["native_log_channel_phase"] <= 0.5:
                    raise ValueError("native channel phase is out of bounds")


def write_manifest(
    rows: list[dict[str, Any]], output: Path, config: Path, supports: Path
) -> dict[str, Any]:
    """Write CSV plus a checksum-bearing provenance sidecar."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="ascii") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    provenance = {
        "rows": len(rows),
        "sha256": digest,
        "manifest": output.name,
        "configuration_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "support_manifest_sha256": hashlib.sha256(supports.read_bytes()).hexdigest(),
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "confirmation_flux_opened": False,
    }
    sidecar = output.with_suffix(output.suffix + ".provenance.json")
    sidecar.write_text(canonical_json(provenance) + "\n", encoding="ascii")
    return provenance


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--supports", type=Path, default=DEFAULT_SUPPORTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    rows = generate_manifest(args.config, args.supports)
    if not args.validate_only:
        provenance = write_manifest(rows, args.output, args.config, args.supports)
        print(canonical_json(provenance))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
