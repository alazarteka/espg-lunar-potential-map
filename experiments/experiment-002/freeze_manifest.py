"""Freeze calibrated spectrum-653 bins before raw-code comparison.

Selection uses only the Level-1 calibrated product and documented nominal
exposure/dead-time equations.  Raw telemetry codes are intentionally not read.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

N_ENERGY = 15
N_DIRECTION = 88
GEOMETRIC_FACTOR = 0.02
NOMINAL_SPIN_SECONDS = 5.0
DEAD_TIME_SECONDS = 0.3e-6


def load_spectrum(path: Path, spec_no: int) -> tuple[np.ndarray, np.ndarray, str]:
    rows = []
    with path.open(encoding="ascii") as source:
        for line in source:
            fields = line.split()
            if int(fields[3]) == spec_no:
                rows.append(fields)
    if len(rows) != N_ENERGY:
        raise ValueError(
            f"Expected {N_ENERGY} rows for spectrum {spec_no}, got {len(rows)}"
        )
    energies = np.array([float(row[2]) for row in rows])
    flux = np.array([[float(value) for value in row[7:95]] for row in rows])
    return energies, flux, rows[0][1].rstrip(".")


def nominal_exposure(thetas: np.ndarray) -> np.ndarray:
    bins = np.array([np.count_nonzero(thetas == theta) for theta in thetas])
    multiplicity = 16.0 / bins
    return NOMINAL_SPIN_SECONDS / 256.0 * multiplicity


def inverse_calibrated_counts(
    flux: np.ndarray, energies: np.ndarray, exposure: np.ndarray
) -> np.ndarray:
    corrected_rate = flux * GEOMETRIC_FACTOR * energies[:, None]
    measured_rate = corrected_rate / (1.0 + DEAD_TIME_SECONDS * corrected_rate)
    return measured_rate * exposure[None, :]


def choose_samples(
    counts: np.ndarray, energies: np.ndarray
) -> list[tuple[int, int, str]]:
    selected: dict[tuple[int, int], str] = {}

    def add(flat_index: int, reason: str) -> None:
        energy_index, direction_index = np.unravel_index(flat_index, counts.shape)
        selected.setdefault((int(energy_index), int(direction_index)), reason)

    flat = counts.ravel()
    zero = np.flatnonzero(flat == 0.0)
    for index in zero[np.linspace(0, len(zero) - 1, 4, dtype=int)]:
        add(int(index), "zero_flux")

    positive = np.flatnonzero(np.isfinite(flat) & (flat > 0.0))
    order = positive[np.argsort(flat[positive])]
    for quantile in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        location = round(quantile * (len(order) - 1))
        add(int(order[location]), f"positive_q{quantile:.2f}")

    for energy_index, label in ((0, "highest_energy"), (N_ENERGY - 1, "lowest_energy")):
        row = counts[energy_index]
        finite = np.flatnonzero(np.isfinite(row))
        for direction_index in (finite[0], finite[len(finite) // 2], finite[-1]):
            flat_index = np.ravel_multi_index(
                (energy_index, direction_index), counts.shape
            )
            add(int(flat_index), label)

    return [
        (energy, direction, reason) for (energy, direction), reason in selected.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--spec-no", type=int, default=653)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    energies, flux, spectrum_time = load_spectrum(args.data, args.spec_no)
    thetas = np.loadtxt(args.theta, dtype=np.float64)
    exposure = nominal_exposure(thetas)
    counts = inverse_calibrated_counts(flux, energies, exposure)
    samples = choose_samples(counts, energies)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "spec_no",
                "spectrum_time_unix",
                "energy_index",
                "direction_index",
                "energy_ev",
                "theta_deg",
                "flux",
                "nominal_exposure_s",
                "inverse_count_proxy",
                "selection_reason",
            ),
        )
        writer.writeheader()
        for energy_index, direction_index, reason in samples:
            writer.writerow(
                {
                    "spec_no": args.spec_no,
                    "spectrum_time_unix": spectrum_time,
                    "energy_index": energy_index,
                    "direction_index": direction_index,
                    "energy_ev": f"{energies[energy_index]:.6f}",
                    "theta_deg": f"{thetas[direction_index]:.2f}",
                    "flux": f"{flux[energy_index, direction_index]:.9g}",
                    "nominal_exposure_s": f"{exposure[direction_index]:.9f}",
                    "inverse_count_proxy": (
                        f"{counts[energy_index, direction_index]:.9g}"
                    ),
                    "selection_reason": reason,
                }
            )
    print(f"Wrote {len(samples)} frozen samples to {args.output}")


if __name__ == "__main__":
    main()
