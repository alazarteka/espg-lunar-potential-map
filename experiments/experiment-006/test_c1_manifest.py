from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("c1_manifest", ROOT / "c1_manifest.py")
assert SPEC is not None
assert SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture(scope="module")
def manifest():
    return MODULE.generate_manifest(
        ROOT / "configuration-c1.json", ROOT / "support_manifest.csv"
    )


def test_seed_uses_canonical_blake2b_little_endian() -> None:
    fields = ["experiment-006-c1-v1", "noise", 140.0, "interval", 840, 596]
    digest = hashlib.blake2b(
        json.dumps(fields, separators=(",", ":")).encode("ascii"), digest_size=16
    ).digest()
    assert MODULE.namespace_seed(*fields) == int.from_bytes(digest, "little")


def test_generation_is_deterministic_and_valid(manifest) -> None:
    repeated = MODULE.generate_manifest(
        ROOT / "configuration-c1.json", ROOT / "support_manifest.csv"
    )
    assert repeated == manifest
    assert len(manifest) == 3000
    assert len({row["seed_noise"] for row in manifest}) == 3000


def test_each_block_has_exact_marginals_and_pairwise_balance(manifest) -> None:
    config = json.loads((ROOT / "configuration-c1.json").read_text())
    supports = MODULE._load_calibration_supports(ROOT / "support_manifest.csv")
    MODULE.validate_manifest(manifest, config, supports)


def test_continuous_values_are_stratified_not_point_masses(manifest) -> None:
    for field in ("beam_amplitude", "pitch_sigma_deg", "native_log_channel_phase"):
        assert len({row[field] for row in manifest}) > 2900


def test_writer_emits_reproducible_checksum(manifest, tmp_path: Path) -> None:
    output = tmp_path / "manifest.csv"
    provenance = MODULE.write_manifest(
        manifest,
        output,
        ROOT / "configuration-c1.json",
        ROOT / "support_manifest.csv",
    )
    assert provenance["rows"] == 3000
    assert provenance["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with output.open(newline="", encoding="ascii") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == 3000
    sidecar = json.loads(output.with_suffix(".csv.provenance.json").read_text())
    assert sidecar == provenance
