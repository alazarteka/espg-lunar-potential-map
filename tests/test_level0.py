from __future__ import annotations

import numpy as np

from src.losscone.level0 import (
    ER_3D_BYTES,
    LEVEL0_RECORD_BYTES,
    MAG_ER_PACKET_OFFSET,
    SUPPORTED_3D_TELEMETRY_VERSIONS,
    decompress_er_counter,
    digital_subcom_software_version,
    iter_level0_er3d_sweeps,
)

FORMAT_1_2_VERSION = min(SUPPORTED_3D_TELEMETRY_VERSIONS)


def _format_1_2_raw(
    *,
    frame_types: list[int] | None = None,
    software_versions: dict[int, int] | None = None,
) -> tuple[bytearray, np.ndarray]:
    ordered_frame_types = list(range(36)) if frame_types is None else frame_types
    raw = bytearray(len(ordered_frame_types) * LEVEL0_RECORD_BYTES)
    expected = np.arange(ER_3D_BYTES, dtype=np.uint8)
    layout = (
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
    for record_index, frame_type in enumerate(ordered_frame_types):
        frame_start = record_index * LEVEL0_RECORD_BYTES + MAG_ER_PACKET_OFFSET
        raw[frame_start] = frame_type
        if frame_type >= len(layout):
            continue
        start, n_bytes = layout[frame_type]
        if n_bytes:
            raw[frame_start + 168 - n_bytes : frame_start + 168] = expected[
                start : start + n_bytes
            ].tobytes()

    versions = (
        {4: FORMAT_1_2_VERSION} if software_versions is None else software_versions
    )
    for record_index, version in versions.items():
        packet_start = record_index * LEVEL0_RECORD_BYTES + MAG_ER_PACKET_OFFSET
        raw[packet_start + 1] = version << 4
        raw[packet_start + 3] = 3 << 4
    return raw, expected


def test_decompress_er_counter_returns_interval_lower_endpoints() -> None:
    codes = np.array([0x00, 0x01, 0x0F, 0x10, 0x1F, 0x20, 0xA0, 0xFF])

    assert decompress_er_counter(codes).tolist() == [
        0,
        1,
        15,
        16,
        31,
        32,
        8192,
        507904,
    ]


def test_iter_level0_er3d_sweeps_assembles_format_1_2_layout(tmp_path) -> None:
    raw, expected = _format_1_2_raw()

    path = tmp_path / "level0.bin"
    path.write_bytes(raw)

    sweeps = list(iter_level0_er3d_sweeps(path))

    assert len(sweeps) == 1
    assert sweeps[0].first_record_index == 0
    assert sweeps[0].telemetry_format_version == FORMAT_1_2_VERSION
    assert np.array_equal(sweeps[0].compressed_counts.ravel(), expected)


def test_iter_level0_er3d_sweeps_allows_non_realtime_interruptions(tmp_path) -> None:
    frame_types = [*range(11), 40, 63, *range(11, 36)]
    raw, expected = _format_1_2_raw(frame_types=frame_types)
    path = tmp_path / "interrupted_level0.bin"
    path.write_bytes(raw)

    sweeps = list(iter_level0_er3d_sweeps(path))

    assert len(sweeps) == 1
    assert np.array_equal(sweeps[0].compressed_counts.ravel(), expected)


def test_iter_level0_er3d_sweeps_rejects_duplicate_frame_type(tmp_path) -> None:
    frame_types = [*range(11), 10, *range(11, 36)]
    raw, _ = _format_1_2_raw(frame_types=frame_types)
    path = tmp_path / "duplicate_frame_level0.bin"
    path.write_bytes(raw)

    assert list(iter_level0_er3d_sweeps(path)) == []


def test_iter_level0_er3d_sweeps_rejects_out_of_order_frame_type(tmp_path) -> None:
    frame_types = [0, 1, 3, 2, *range(4, 36)]
    raw, _ = _format_1_2_raw(frame_types=frame_types)
    path = tmp_path / "out_of_order_level0.bin"
    path.write_bytes(raw)

    assert list(iter_level0_er3d_sweeps(path)) == []


def test_iter_level0_er3d_sweeps_rejects_hybrid_sequence(tmp_path) -> None:
    frame_types = [*range(11), *range(1, 36)]
    raw, _ = _format_1_2_raw(frame_types=frame_types)
    path = tmp_path / "hybrid_level0.bin"
    path.write_bytes(raw)

    assert list(iter_level0_er3d_sweeps(path)) == []


def test_iter_level0_er3d_sweeps_rejects_format_3_layout(tmp_path) -> None:
    raw, _ = _format_1_2_raw(software_versions={4: 3})
    path = tmp_path / "format_3_level0.bin"
    path.write_bytes(raw)

    assert list(iter_level0_er3d_sweeps(path)) == []


def test_iter_level0_er3d_sweeps_rejects_mixed_software_versions(tmp_path) -> None:
    raw, _ = _format_1_2_raw(software_versions={4: 1, 20: 2})
    path = tmp_path / "mixed_version_level0.bin"
    path.write_bytes(raw)

    assert list(iter_level0_er3d_sweeps(path)) == []


def test_iter_level0_er3d_sweeps_rejects_missing_software_version(tmp_path) -> None:
    raw, _ = _format_1_2_raw(software_versions={})
    path = tmp_path / "missing_version_level0.bin"
    path.write_bytes(raw)

    assert list(iter_level0_er3d_sweeps(path)) == []


def test_digital_subcom_software_version_uses_mag_status_frame_number() -> None:
    packet = bytearray(168)
    packet[1] = 0x2A
    packet[3] = 0x30

    assert digital_subcom_software_version(packet) == 2
