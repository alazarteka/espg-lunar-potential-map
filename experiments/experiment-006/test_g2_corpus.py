"""Tests for the prospectively frozen G2 certification corpus."""

from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path


def _load():
    path = Path(__file__).with_name("g2_corpus.py")
    spec = importlib.util.spec_from_file_location("experiment_006_g2_corpus", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


corpus = _load()


def test_generation_is_deterministic_and_has_exact_arms() -> None:
    first = corpus.generate()
    second = corpus.generate()
    assert first == second
    assert Counter(row["arm"] for row in first) == Counter(corpus.ARM_COUNTS)


def test_validation_and_support_coverage() -> None:
    rows = corpus.generate()
    config = json.loads(corpus.CONFIG.read_text(encoding="ascii"))
    supports = corpus._supports(corpus.SUPPORTS)
    corpus.validate(rows, config, supports)
    assert {row["support_id"] for row in rows} == set(supports)


def test_writer_is_reproducible_and_records_information_barrier(tmp_path) -> None:
    output = tmp_path / "corpus.csv"
    first = corpus.write(corpus.generate(), output)
    first_bytes = output.read_bytes()
    second = corpus.write(corpus.generate(), output)
    assert output.read_bytes() == first_bytes
    assert first["sha256"] == second["sha256"]
    assert not first["partial_c1_scientific_outcomes_opened"]
    assert not first["confirmation_flux_opened"]
