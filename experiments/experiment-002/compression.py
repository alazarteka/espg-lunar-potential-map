"""LP ER experiment-local counter compression semantics.

The MAG/ER telemetry SIS Appendix G, section 2.6, defines an 8-bit code with
a 4-bit exponent and 4-bit mantissa.  Low-order bits are truncated before
transmission, so the decoded value is the lower endpoint of an interval rather
than a rounded observation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CountInterval:
    """Pre-compression integer counts represented by one telemetry byte."""

    lower: int
    upper: int | None

    def contains(self, count: float) -> bool:
        """Return whether ``count`` lies in this code's represented interval."""
        if count < self.lower:
            return False
        return self.upper is None or count <= self.upper


def decode_lower(code: int) -> int:
    """Decode a telemetry byte to its documented lower-endpoint count."""
    if not 0 <= code <= 0xFF:
        raise ValueError(f"Counter code must be in [0, 255], got {code}")
    if code < 0x20:
        return code
    exponent, mantissa = divmod(code, 0x10)
    return (0x10 + mantissa) << (exponent - 1)


def count_interval(code: int) -> CountInterval:
    """Return the count interval represented by ``code``.

    The archive documentation does not establish whether ``0xff`` has a finite
    upper endpoint or includes counter saturation, so its upper bound remains
    open rather than being inferred from an untransmitted ninth-bit code.
    """
    lower = decode_lower(code)
    upper = None if code == 0xFF else decode_lower(code + 1) - 1
    return CountInterval(lower=lower, upper=upper)


def _self_check() -> None:
    expected = {
        0x00: (0, 0),
        0x1F: (31, 31),
        0x20: (32, 33),
        0x2F: (62, 63),
        0x30: (64, 67),
        0xFE: (491_520, 507_903),
        0xFF: (507_904, None),
    }
    for code, bounds in expected.items():
        interval = count_interval(code)
        assert (interval.lower, interval.upper) == bounds


if __name__ == "__main__":
    _self_check()
    print("Compression table checks passed.")
