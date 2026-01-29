"""Backward-compatible shim for torch loss-cone APIs.

Prefer importing from `src.losscone_torch`.
"""

from src.losscone_torch import *  # noqa: F403
from src.losscone_torch import __all__  # noqa: F401
