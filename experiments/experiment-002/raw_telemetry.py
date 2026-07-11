"""Reconstruct LP ER 3-D counter-code blocks from Level-0 merged telemetry.

This parser implements the fixed record layout plus the documented Format-1 and
Format-3 ER frame layouts. April 1999 uses Format 3. It does not assign a
Level-1 spectrum number: that crosswalk is audited separately.
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

# Frame type -> (inclusive 3-D array start, inclusive array end, payload offset).
# Format 3 frame 33 is the important exception to tail packing: magnetic-vector
# bytes follow its final 3-D fragment.
FORMAT_LAYOUTS: dict[int, dict[int, tuple[int, int, int] | None]] = {
    1: {
        0: None,
        1: (0, 1, 81),
        2: (2, 4, 80),
        3: (5, 23, 64),
        4: (24, 74, 32),
        5: (75, 125, 32),
        6: (126, 176, 32),
        7: (177, 227, 32),
        8: (228, 278, 32),
        9: (279, 281, 80),
        10: (282, 300, 64),
        11: (301, 351, 32),
        12: (352, 402, 32),
        13: (403, 453, 32),
        14: (454, 504, 32),
        15: (505, 555, 32),
        16: (556, 606, 32),
        17: (607, 657, 32),
        18: (658, 660, 80),
        19: (661, 663, 80),
        20: (664, 666, 80),
        21: (667, 685, 64),
        22: (686, 736, 32),
        23: (737, 787, 32),
        24: (788, 838, 32),
        25: (839, 889, 32),
        26: (890, 940, 32),
        27: (941, 943, 80),
        28: (944, 962, 64),
        29: (963, 1013, 32),
        30: (1014, 1064, 32),
        31: (1065, 1115, 32),
        32: (1116, 1166, 32),
        33: (1167, 1217, 32),
        34: (1218, 1268, 32),
        35: (1269, 1319, 32),
    },
    3: {
        0: None,
        1: (0, 1, 81),
        2: (2, 4, 80),
        3: (5, 23, 64),
        4: (24, 74, 32),
        5: (75, 125, 32),
        6: (126, 176, 32),
        7: (177, 227, 32),
        8: (228, 278, 32),
        9: (279, 313, 48),
        10: (314, 364, 32),
        11: (365, 415, 32),
        12: (416, 466, 32),
        13: (467, 517, 32),
        14: (518, 568, 32),
        15: (569, 619, 32),
        16: (620, 670, 32),
        17: (671, 721, 32),
        18: (722, 724, 80),
        19: (725, 727, 80),
        20: (728, 730, 80),
        21: (731, 749, 64),
        22: (750, 800, 32),
        23: (801, 851, 32),
        24: (852, 902, 32),
        25: (903, 953, 32),
        26: (954, 1004, 32),
        27: (1005, 1039, 48),
        28: (1040, 1090, 32),
        29: (1091, 1141, 32),
        30: (1142, 1192, 32),
        31: (1193, 1243, 32),
        32: (1244, 1294, 32),
        33: (1295, 1319, 32),
        34: None,
        35: None,
    },
}


def _check_layouts() -> None:
    for telemetry_format, layout in FORMAT_LAYOUTS.items():
        array_ranges = []
        for frame_type in range(36):
            fragment = layout[frame_type]
            if fragment is None:
                continue
            start, end, payload_offset = fragment
            length = end - start + 1
            if payload_offset < 0 or payload_offset + length > ER_BYTES:
                raise AssertionError(
                    f"Format {telemetry_format} frame {frame_type} exceeds payload"
                )
            array_ranges.extend(range(start, end + 1))
        if array_ranges != list(range(1320)):
            raise AssertionError(
                f"Format {telemetry_format} does not cover 3-D bytes 0..1319"
            )


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


def _fragment(
    frame: TelemetryFrame, layout: dict[int, tuple[int, int, int] | None]
) -> bytes:
    fragment_layout = layout[frame.frame_type]
    if fragment_layout is None:
        return b""
    start, end, payload_offset = fragment_layout
    length = end - start + 1
    return frame.er_payload[payload_offset : payload_offset + length]


def reconstruct_blocks(
    frames: list[TelemetryFrame], telemetry_format: int = 3
) -> list[ThreeDBlock]:
    """Collect uninterrupted ordered frame types 0..35, skipping other modes."""
    try:
        layout = FORMAT_LAYOUTS[telemetry_format]
    except KeyError as error:
        raise ValueError(f"Unsupported telemetry format {telemetry_format}") from error
    _check_layouts()
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

            codes = b"".join(_fragment(item, layout) for item in pending)
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
    parser.add_argument("--format", type=int, choices=FORMAT_LAYOUTS, default=3)
    args = parser.parse_args()

    frames = read_frames(args.telemetry)
    blocks = reconstruct_blocks(frames, args.format)
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
