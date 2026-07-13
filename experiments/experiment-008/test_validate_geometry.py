"""Focused tests for Experiment 008 geometry validation helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

MODULE_PATH = Path(__file__).with_name("validate_geometry.py")
SPEC = importlib.util.spec_from_file_location(
    "experiment_008_validate_geometry", MODULE_PATH
)
assert SPEC is not None
assert SPEC.loader is not None
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


def test_angular_separation_degrees() -> None:
    separation = validation.angular_separation_degrees(
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 90.0]),
        np.array([90.0, 0.0]),
    )
    np.testing.assert_allclose(separation, [90.0, 90.0])
