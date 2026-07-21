"""Narrow Level-0 ER 3-D packet probe.

This module assembles the 1,320 compressed 3-D ER values from real-time MAG/ER
telemetry frames. It deliberately does not infer calibrated flux, event time,
exposure, dead-time correction, or quality flags; those require additional
telemetry/control-state decoding and validation against the calibrated product.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator

ER_ENERGIES = 15
ER_ANGLES = 88
ER_3D_BYTES = ER_ENERGIES * ER_ANGLES
LEVEL0_RECORD_BYTES = 472
MAG_ER_PACKET_OFFSET = 286
MAG_ER_PACKET_BYTES = 168
ER_DATA_OFFSET = 85
ER_DATA_BYTES = 83
MAG_STATUS_OFFSET = 3
DIGITAL_SUBCOM_FIRST_BYTE_OFFSET = 1
DIGITAL_SUBCOM_VERSION_FRAME = 3
SUPPORTED_3D_TELEMETRY_VERSIONS = frozenset((1, 2))

# Telemetry formats 1 and 2 share this 3-D-byte layout. Format 3 moves part of
# the payload to make room for magnetic-angle data and is intentionally refused
# by this early feasibility probe.
_FORMAT_1_2_3D_LAYOUT: tuple[tuple[int, int], ...] = (
    (0, 0),
    (0, 2),
    (2, 3),
    (5, 19),
    (24, 51),
    (75, 51),
    (126, 51),
    (177, 51),
    (228, 51),
    (279, 3),
    (282, 19),
    (301, 51),
    (352, 51),
    (403, 51),
    (454, 51),
    (505, 51),
    (556, 51),
    (607, 51),
    (658, 3),
    (661, 3),
    (664, 3),
    (667, 19),
    (686, 51),
    (737, 51),
    (788, 51),
    (839, 51),
    (890, 51),
    (941, 3),
    (944, 19),
    (963, 51),
    (1014, 51),
    (1065, 51),
    (1116, 51),
    (1167, 51),
    (1218, 51),
    (1269, 51),
)


@dataclass(frozen=True)
class Level0Er3DSweep:
    """One complete, compressed 15×88 real-time ER 3-D telemetry sweep."""

    first_record_index: int
    compressed_counts: np.ndarray
    earth_receive_time: datetime | None = None
    telemetry_format_version: int | None = None


def infer_level0_year(path: Path) -> int | None:
    """Infer the four-digit year from a standard ``Myyddd*.B`` Level-0 name."""
    match = re.match(r"M(?P<year>\d{2})\d{3}", Path(path).name)
    if match is None:
        return None
    return 1900 + int(match.group("year"))


def earth_receive_time_from_level0_record(record: bytes, *, year: int) -> datetime:
    """Decode the documented ERT field in one 472-byte merged Level-0 record.

    This is the ground-receive timestamp, not the ER observation time. The
    caller must still account for light time and instrument/downlink latency
    before using it to align to a calibrated product.
    """
    if len(record) != LEVEL0_RECORD_BYTES:
        raise ValueError(
            "Level-0 record must contain "
            f"{LEVEL0_RECORD_BYTES} bytes, got {len(record)}"
        )

    # PDS Level-0 SIS, Table 2.5.1-2: words 457-458 are day of year and
    # words 459-462 are milliseconds of day, both stored most-significant byte
    # first. Convert its one-based word references to zero-based byte offsets.
    day_of_year = int.from_bytes(record[456:458], byteorder="big")
    milliseconds = int.from_bytes(record[458:462], byteorder="big")
    if not 1 <= day_of_year <= 366:
        raise ValueError(f"Invalid Level-0 Earth-receive day of year: {day_of_year}")
    if not 0 <= milliseconds < 86_400_000:
        raise ValueError(
            f"Invalid Level-0 Earth-receive millisecond of day: {milliseconds}"
        )
    return datetime(year, 1, 1, tzinfo=UTC) + timedelta(
        days=day_of_year - 1,
        milliseconds=milliseconds,
    )


def decompress_er_counter(codes: np.ndarray) -> np.ndarray:
    """Decode the documented 4-bit-exponent/4-bit-mantissa ER counter code.

    The result is the lower endpoint of the count interval represented by each
    truncated telemetry code, not an exact raw count.
    """
    code_values = np.asarray(codes, dtype=np.uint8)
    exponent = code_values >> 4
    mantissa = code_values & 0x0F
    decoded = mantissa.astype(np.int64)
    compressed = exponent > 0
    decoded[compressed] = (16 + mantissa[compressed].astype(np.int64)) << (
        exponent[compressed].astype(np.int64) - 1
    )
    return decoded


def digital_subcom_software_version(packet: bytes) -> int | None:
    """Return the Digital Subcom ER software version when present in a packet.

    Digital Subcom is decommutated by the MAG-status frame number. Its software
    version occupies the high nibble of its first byte on frame 3. ``None``
    means that this packet carries another Digital Subcom word.
    """
    if len(packet) != MAG_ER_PACKET_BYTES:
        raise ValueError(
            f"MAG/ER packet must contain {MAG_ER_PACKET_BYTES} bytes, got {len(packet)}"
        )
    mag_frame_number = packet[MAG_STATUS_OFFSET] >> 4
    if mag_frame_number != DIGITAL_SUBCOM_VERSION_FRAME:
        return None
    return packet[DIGITAL_SUBCOM_FIRST_BYTE_OFFSET] >> 4


def iter_level0_er3d_sweeps(
    path: Path,
    *,
    year: int | None = None,
) -> Iterator[Level0Er3DSweep]:
    """Yield complete format-1/2 real-time ER 3-D code arrays from a raw file.

    Burst, memory-dump, incomplete, duplicate, and unsupported-format frame
    sequences are excluded. A sequence must carry one consistent Digital
    Subcom software version of 1 or 2; version 3 uses a different 3-D layout.
    The caller must separately establish timing and calibrated-product
    correspondence before treating yielded sweeps as observations.
    """
    raw_path = Path(path)
    raw = raw_path.read_bytes()
    if len(raw) % LEVEL0_RECORD_BYTES:
        raise ValueError(
            f"Level-0 file length {len(raw)} is not divisible by "
            f"{LEVEL0_RECORD_BYTES} bytes"
        )

    current_codes: np.ndarray | None = None
    current_index = -1
    next_frame_type: int | None = None
    current_versions: set[int] = set()
    resolved_year = year if year is not None else infer_level0_year(raw_path)

    for record_index in range(len(raw) // LEVEL0_RECORD_BYTES):
        record_start = record_index * LEVEL0_RECORD_BYTES
        packet_start = record_start + MAG_ER_PACKET_OFFSET
        packet = raw[packet_start : packet_start + MAG_ER_PACKET_BYTES]
        frame_type = packet[0] & 0x3F
        if frame_type > 35:
            continue
        software_version = digital_subcom_software_version(packet)

        if frame_type == 0:
            current_codes = np.zeros(ER_3D_BYTES, dtype=np.uint8)
            current_index = record_index
            next_frame_type = 1
            current_versions = (
                {software_version} if software_version is not None else set()
            )
            continue

        if current_codes is None or next_frame_type is None:
            continue
        if frame_type != next_frame_type:
            current_codes = None
            next_frame_type = None
            current_versions = set()
            continue

        if software_version is not None:
            current_versions.add(software_version)

        destination_start, n_bytes = _FORMAT_1_2_3D_LAYOUT[frame_type]
        er_data = packet[ER_DATA_OFFSET : ER_DATA_OFFSET + ER_DATA_BYTES]
        if n_bytes:
            current_codes[destination_start : destination_start + n_bytes] = (
                np.frombuffer(er_data[-n_bytes:], dtype=np.uint8)
            )
        next_frame_type += 1

        if next_frame_type == 36:
            if (
                len(current_versions) != 1
                or not current_versions <= SUPPORTED_3D_TELEMETRY_VERSIONS
            ):
                current_codes = None
                next_frame_type = None
                current_versions = set()
                continue

            first_record = raw[
                current_index * LEVEL0_RECORD_BYTES : (current_index + 1)
                * LEVEL0_RECORD_BYTES
            ]
            earth_receive_time = (
                earth_receive_time_from_level0_record(first_record, year=resolved_year)
                if resolved_year is not None
                else None
            )
            yield Level0Er3DSweep(
                first_record_index=current_index,
                compressed_counts=current_codes.reshape(ER_ENERGIES, ER_ANGLES),
                earth_receive_time=earth_receive_time,
                telemetry_format_version=current_versions.pop(),
            )
            current_codes = None
            next_frame_type = None
            current_versions = set()
