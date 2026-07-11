"""Infer and hold out-test the archived raw-count response per 3-D bin.

The inferred quantity has units of seconds under the PDS formula, but is called
an effective response because undocumented exposure and response calibration
cannot yet be separated.  Spectrum 653 is excluded from estimation and used
only for validation against its raw compression intervals.
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
from compression import count_interval, decode_lower
from raw_telemetry import read_frames, reconstruct_blocks

GEOMETRIC_FACTOR = 0.02
DEAD_TIME_SECONDS = 0.3e-6
SHAPE = (15, 88)


def load_spectra(
    path: Path, wanted: set[int]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    rows: dict[int, list[list[str]]] = {spec_no: [] for spec_no in wanted}
    with path.open(encoding="ascii") as source:
        for line in source:
            fields = line.split()
            spec_no = int(fields[3])
            if spec_no in rows:
                rows[spec_no].append(fields)

    spectra: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for spec_no, spec_rows in rows.items():
        if len(spec_rows) != SHAPE[0]:
            raise ValueError(
                f"Expected {SHAPE[0]} rows for spectrum {spec_no}, got {len(spec_rows)}"
            )
        energies = np.array([float(row[2]) for row in spec_rows])
        flux = np.array([[float(value) for value in row[7:95]] for row in spec_rows])
        spectra[spec_no] = energies, flux
    return spectra


def measured_rate(
    flux: np.ndarray, energies: np.ndarray, dead_time_seconds: float
) -> np.ndarray:
    corrected = flux * GEOMETRIC_FACTOR * energies[:, None]
    return corrected / (1.0 + dead_time_seconds * corrected)


def lower_counts(codes: bytes) -> np.ndarray:
    return np.array([decode_lower(code) for code in codes], dtype=np.float64).reshape(
        SHAPE
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-block-spec", type=int, default=630)
    parser.add_argument("--train-first", type=int, default=638)
    parser.add_argument("--train-last", type=int, default=668)
    parser.add_argument("--holdout", type=int, default=653)
    parser.add_argument("--dead-time-seconds", type=float, default=DEAD_TIME_SECONDS)
    args = parser.parse_args()

    wanted = set(range(args.train_first, args.train_last + 1))
    wanted.add(args.holdout)
    spectra = load_spectra(args.data, wanted)
    blocks = reconstruct_blocks(read_frames(args.telemetry))

    response_samples = []
    for spec_no in range(args.train_first, args.train_last + 1):
        if spec_no == args.holdout:
            continue
        block_index = spec_no - args.first_block_spec
        codes = np.frombuffer(blocks[block_index].codes, dtype=np.uint8).reshape(SHAPE)
        lower = lower_counts(blocks[block_index].codes)
        energies, flux = spectra[spec_no]
        rate = measured_rate(flux, energies, args.dead_time_seconds)
        valid = (codes < 0xFF) & (lower > 0.0) & np.isfinite(rate) & (rate > 0.0)
        response_sample = np.full(SHAPE, np.nan)
        np.divide(lower, rate, out=response_sample, where=valid)
        response_samples.append(response_sample)

    samples = np.stack(response_samples)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        response = np.nanmedian(samples, axis=0)
        absolute_deviation = np.abs(samples - response[None, :, :])
        mad = np.nanmedian(absolute_deviation, axis=0)
    n_train = np.sum(np.isfinite(samples), axis=0)

    holdout_index = args.holdout - args.first_block_spec
    holdout_block = blocks[holdout_index]
    holdout_codes = np.frombuffer(holdout_block.codes, dtype=np.uint8).reshape(SHAPE)
    holdout_lower = lower_counts(holdout_block.codes)
    holdout_energies, holdout_flux = spectra[args.holdout]
    holdout_rate = measured_rate(holdout_flux, holdout_energies, args.dead_time_seconds)
    predicted = response * holdout_rate

    contained = np.zeros(SHAPE, dtype=bool)
    for index in np.ndindex(SHAPE):
        contained[index] = count_interval(int(holdout_codes[index])).contains(
            float(predicted[index])
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "energy_index",
                "direction_index",
                "n_train",
                "effective_response_s",
                "relative_mad",
                "holdout_code_hex",
                "holdout_count_lower",
                "holdout_count_upper",
                "holdout_predicted_count",
                "holdout_in_interval",
            ),
        )
        writer.writeheader()
        for energy_index, direction_index in np.ndindex(SHAPE):
            code = int(holdout_codes[energy_index, direction_index])
            interval = count_interval(code)
            rel_mad = (
                mad[energy_index, direction_index]
                / response[energy_index, direction_index]
            )
            writer.writerow(
                {
                    "energy_index": energy_index,
                    "direction_index": direction_index,
                    "n_train": int(n_train[energy_index, direction_index]),
                    "effective_response_s": (
                        f"{response[energy_index, direction_index]:.9g}"
                    ),
                    "relative_mad": f"{rel_mad:.9g}",
                    "holdout_code_hex": f"0x{code:02x}",
                    "holdout_count_lower": int(
                        holdout_lower[energy_index, direction_index]
                    ),
                    "holdout_count_upper": (
                        "open" if interval.upper is None else interval.upper
                    ),
                    "holdout_predicted_count": (
                        f"{predicted[energy_index, direction_index]:.9g}"
                    ),
                    "holdout_in_interval": contained[energy_index, direction_index],
                }
            )

    usable = np.isfinite(predicted)
    stable = np.isfinite(response) & (response > 0.0)
    relative_mad = np.divide(mad, response, where=stable)
    print(f"training_spectra={len(response_samples)}")
    print(f"response_available_fraction={np.mean(stable):.6f}")
    print(f"median_relative_mad={np.nanmedian(relative_mad[stable]):.6f}")
    print(f"holdout_interval_fraction={np.mean(contained[usable]):.6f}")
    print(f"wrote={args.output}")


if __name__ == "__main__":
    main()
