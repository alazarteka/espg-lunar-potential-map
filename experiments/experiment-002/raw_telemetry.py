"""Reconstruct LP ER 3-D counter-code blocks from Level-0 merged telemetry.

This parser implements only the fixed record and Format-1 byte layout needed by
Experiment 002.  It does not assign a Level-1 spectrum number: that crosswalk
requires the documented accumulation-time calculation and is audited
separately.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

RECORD_BYTES = 472
MAGER_START = 286
MAGER_BYTES = 168
ER_START_IN_MAGER = 85
ER_BYTES = 83

# Inclusive byte ranges in the 1,320-byte Format-1 3-D array.  Non-3-D payload
# precedes these bytes in each 83-byte ER frame, so each fragment is at its tail.
FORMAT1_RANGES: dict[int, tuple[int, int] | None] = {
    0: None,
    1: (0, 1),
    2: (2, 4),
    3: (5, 23),
    4: (24, 74),
    5: (75, 125),
    6: (126, 176),
    7: (177, 227),
    8: (228, 278),
    9: (279, 281),
    10: (282, 300),
    11: (301, 351),
    12: (352, 402),
    13: (403, 453),
    14: (454, 504),
    15: (505, 555),
    16: (556, 606),
    17: (607, 657),
    18: (658, 660),
    19: (661, 663),
    20: (664, 666),
    21: (667, 685),
    22: (686, 736),
    23: (737, 787),
    24: (788, 838),
    25: (839, 889),
    26: (890, 940),
    27: (941, 943),
    28: (944, 962),
    29: (963, 1013),
    30: (1014, 1064),
    31: (1065, 1115),
    32: (1116, 1166),
    33: (1167, 1217),
    34: (1218, 1268),
    35: (1269, 1319),
}


@dataclass(frozen=True)
class TelemetryFrame:
    record_index: int
    vcdu_count: int
    frame_code: int
    frame_type: int
    er_payload: bytes


@dataclass(frozen=True)
class ThreeDBlock:
    first_record: int
    last_record: int
    first_vcdu: int
    last_vcdu: int
    codes: bytes


def read_frames(path: Path) -> list[TelemetryFrame]:
    """Read fixed-size Level-0 records and extract their MAG/ER packets."""
    raw = path.read_bytes()
    if len(raw) % RECORD_BYTES:
        raise ValueError(
            f"{path} has {len(raw)} bytes, not a multiple of {RECORD_BYTES}"
        )

    frames: list[TelemetryFrame] = []
    for record_index in range(len(raw) // RECORD_BYTES):
        record = raw[record_index * RECORD_BYTES : (record_index + 1) * RECORD_BYTES]
        mager = record[MAGER_START : MAGER_START + MAGER_BYTES]
        code = mager[0]
        frames.append(
            TelemetryFrame(
                record_index=record_index,
                vcdu_count=int.from_bytes(record[6:9], "big"),
                frame_code=code,
                frame_type=code & 0x3F,
                er_payload=mager[ER_START_IN_MAGER : ER_START_IN_MAGER + ER_BYTES],
            )
        )
    return frames


def _fragment(frame: TelemetryFrame) -> bytes:
    byte_range = FORMAT1_RANGES[frame.frame_type]
    if byte_range is None:
        return b""
    start, end = byte_range
    length = end - start + 1
    return frame.er_payload[-length:]


def reconstruct_blocks(frames: list[TelemetryFrame]) -> list[ThreeDBlock]:
    """Collect uninterrupted ordered frame types 0..35, skipping other modes."""
    blocks: list[ThreeDBlock] = []
    pending: list[TelemetryFrame] = []
    expected = 0

    for frame in frames:
        if frame.frame_type > 35:
            continue
        if frame.frame_type == expected:
            pending.append(frame)
            expected += 1
            if expected != 36:
                continue

            codes = b"".join(_fragment(item) for item in pending)
            if len(codes) != 1320:
                raise AssertionError(f"Format-1 block has {len(codes)} bytes")
            blocks.append(
                ThreeDBlock(
                    first_record=pending[0].record_index,
                    last_record=pending[-1].record_index,
                    first_vcdu=pending[0].vcdu_count,
                    last_vcdu=pending[-1].vcdu_count,
                    codes=codes,
                )
            )
            pending = []
            expected = 0
        elif frame.frame_type == 0:
            pending = [frame]
            expected = 1
        else:
            pending = []
            expected = 0

    return blocks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry", type=Path)
    args = parser.parse_args()

    frames = read_frames(args.telemetry)
    blocks = reconstruct_blocks(frames)
    print(f"records={len(frames)} complete_3d_blocks={len(blocks)}")
    if blocks:
        first, last = blocks[0], blocks[-1]
        print(
            f"first=records:{first.first_record}-{first.last_record} "
            f"vcdu:{first.first_vcdu}-{first.last_vcdu}"
        )
        print(
            f"last=records:{last.first_record}-{last.last_record} "
            f"vcdu:{last.first_vcdu}-{last.last_vcdu}"
        )


if __name__ == "__main__":
    main()
