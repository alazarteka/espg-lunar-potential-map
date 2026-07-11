"""Empirically verify 3-D fragment offsets inside each 83-byte ER frame.

For every legal byte offset, correlate candidate raw bytes through time with the
matching Level-1 energy/direction bins.  This is a layout diagnostic only: it
does not alter counts or fit a physical calibration.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from compression import decode_lower
from infer_response import load_spectra
from raw_telemetry import FORMAT_LAYOUTS, read_frames, reconstruct_blocks


def median_temporal_correlation(raw: np.ndarray, flux: np.ndarray) -> float:
    correlations = []
    for byte_index in range(raw.shape[1]):
        x = np.log1p(raw[:, byte_index])
        y = np.log1p(flux[:, byte_index])
        if np.std(x) > 0.0 and np.std(y) > 0.0:
            correlations.append(float(np.corrcoef(x, y)[0, 1]))
    return float(np.median(correlations)) if correlations else np.nan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--first-block-spec", type=int, default=630)
    parser.add_argument("--first-spec", type=int, default=638)
    parser.add_argument("--last-spec", type=int, default=668)
    parser.add_argument("--format", type=int, choices=FORMAT_LAYOUTS, default=3)
    args = parser.parse_args()

    spec_numbers = list(range(args.first_spec, args.last_spec + 1))
    spectra = load_spectra(args.data, set(spec_numbers))
    frames = read_frames(args.telemetry)
    blocks = reconstruct_blocks(frames, args.format)
    layout = FORMAT_LAYOUTS[args.format]

    frame_lookup = {}
    for block_index, block in enumerate(blocks):
        for frame in frames[block.first_record : block.last_record + 1]:
            if frame.frame_type <= 35:
                frame_lookup[block_index, frame.frame_type] = frame

    print(
        "frame_type,documented_offset,best_offset,documented_score,best_score,"
        "best_lag,best_lag_score"
    )
    for frame_type in range(1, 36):
        fragment_layout = layout[frame_type]
        if fragment_layout is None:
            continue
        first_byte, last_byte, documented_offset = fragment_layout
        length = last_byte - first_byte + 1
        flux = np.array(
            [
                spectra[spec_no][1].ravel()[first_byte : last_byte + 1]
                for spec_no in spec_numbers
            ]
        )

        scores = []
        for offset in range(84 - length):
            raw = []
            for spec_no in spec_numbers:
                block_index = spec_no - args.first_block_spec
                payload = frame_lookup[block_index, frame_type].er_payload
                codes = payload[offset : offset + length]
                raw.append([decode_lower(code) for code in codes])
            scores.append(median_temporal_correlation(np.array(raw), flux))

        best_offset = int(np.nanargmax(scores))

        lag_scores = []
        for lag in range(-2, 3):
            common_specs = [
                spec_no
                for spec_no in spec_numbers
                if args.first_spec <= spec_no + lag <= args.last_spec
            ]
            raw = []
            lagged_flux = []
            for spec_no in common_specs:
                block_index = spec_no - args.first_block_spec
                payload = frame_lookup[block_index, frame_type].er_payload
                codes = payload[documented_offset : documented_offset + length]
                raw.append([decode_lower(code) for code in codes])
                lagged_flux.append(
                    spectra[spec_no + lag][1].ravel()[first_byte : last_byte + 1]
                )
            lag_scores.append(
                median_temporal_correlation(np.array(raw), np.array(lagged_flux))
            )
        best_lag_index = int(np.nanargmax(lag_scores))
        best_lag = best_lag_index - 2
        print(
            f"{frame_type},{documented_offset},{best_offset},"
            f"{scores[documented_offset]:.6f},{scores[best_offset]:.6f},"
            f"{best_lag},{lag_scores[best_lag_index]:.6f}"
        )


if __name__ == "__main__":
    main()
