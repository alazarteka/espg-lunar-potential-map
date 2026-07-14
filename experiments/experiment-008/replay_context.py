"""Resolve isolated output paths for an Experiment 008 replay."""

from __future__ import annotations

import os
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).resolve().parent
RUN_ID_ENV = "EXPERIMENT_008_RUN_ID"
RUN_ID = os.environ.get(RUN_ID_ENV, "")

if RUN_ID and re.fullmatch(r"[a-z0-9][a-z0-9-]*", RUN_ID) is None:
    raise ValueError(f"{RUN_ID_ENV} must be a lowercase ASCII slug")

if RUN_ID:
    OUTPUT_DIR = EXPERIMENT_DIR / "replays" / RUN_ID
    ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment-008" / RUN_ID
else:
    OUTPUT_DIR = EXPERIMENT_DIR
    ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "experiment-008"

RESULTS_DIR = OUTPUT_DIR / "results"
CENSUS_PATH = OUTPUT_DIR / "geometry-census.csv"
RESULT_MANIFEST_PATH = OUTPUT_DIR / "result-manifest.csv"
