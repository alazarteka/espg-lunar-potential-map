#!/usr/bin/env python3
"""Run the geometry-only hard-bin loss-cone identification audit.

The workflow never reads calibrated flux values into the identification model.
It uses the archived energy, time, magnetic-field, and look-direction geometry
to study zero-beam hard-boundary plateaus under declared nuisance envelopes.

Examples
--------
Run the mission geometry census with at most eight single-threaded workers::

    uv run python scripts/diagnostics/losscone_hardbin_identifiability.py census \\
      --workers 8

Select the deterministic monthly sample, audit it, and summarize results::

    uv run python scripts/diagnostics/losscone_hardbin_identifiability.py sample
    uv run python scripts/diagnostics/losscone_hardbin_identifiability.py audit \\
      --workers 8
    uv run python scripts/diagnostics/losscone_hardbin_identifiability.py summarize
"""

from __future__ import annotations

import os

# Keep every worker single-threaded before importing NumPy/SciPy. The CLI also
# rejects more than eight worker processes and does not import torch.
for variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "HIGHS_THREADS",
):
    os.environ[variable] = "1"

from src.losscone.identifiability_audit import main  # noqa: E402, I001


if __name__ == "__main__":
    raise SystemExit(main())
