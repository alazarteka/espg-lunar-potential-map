"""Shared atomic NPZ writing for the batch and QC output paths."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np


def write_npz_atomic(out_path: Path, payload: dict[str, np.ndarray]) -> None:
    """Write a compressed NPZ atomically via a temp file + fsync + rename.

    The temp file is created in the destination directory so ``os.replace`` is a
    same-filesystem atomic rename; readers never observe a partially written NPZ.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=out_path.parent, suffix=".tmp", delete=False
    ) as tmp:
        np.savez_compressed(tmp, **payload)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = Path(tmp.name)
    os.replace(tmp_path, out_path)
