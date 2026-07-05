"""Import-hygiene guards against latent circular imports.

These run in a fresh subprocess so that a project module being the very first
import in a clean interpreter cannot silently re-introduce an import cycle.
"""

from __future__ import annotations

import subprocess
import sys


def _first_import_ok(statement: str) -> None:
    result = subprocess.run(
        [sys.executable, "-c", statement],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"first import failed for {statement!r}:\n{result.stderr}"
    )


def test_losscone_is_safe_first_import() -> None:
    _first_import_ok("from src.losscone import LossConeFitter")


def test_chi2_is_safe_first_import() -> None:
    _first_import_ok("import src.losscone.chi2")


def test_flux_and_utils_first_import() -> None:
    _first_import_ok("import src.flux, src.utils, src.utils.synthetic")
