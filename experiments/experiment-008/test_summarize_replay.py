"""Focused tests for deterministic Experiment 008 replay synthesis."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd

EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(EXPERIMENT_DIR))
SPEC = importlib.util.spec_from_file_location(
    "experiment008_summarize_replay", EXPERIMENT_DIR / "summarize_replay.py"
)
assert SPEC is not None
assert SPEC.loader is not None
summary = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summary
SPEC.loader.exec_module(summary)


def test_recoverable_window_classes_preserve_frozen_duration_labels() -> None:
    grid = pd.DataFrame(
        {
            "class_pass": [True, False, True],
            "duration_hours": [708.734112, 24.0, 72.0],
            "lmax": [10, 5, 1],
        }
    )

    assert summary.recoverable_window_classes(grid) == [
        "72h_lmax1",
        "708.734112h_lmax10",
    ]
