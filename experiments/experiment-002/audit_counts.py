"""Compare frozen Level-1 bins with candidate Level-0 counter-code blocks.

The script scores adjacent raw blocks as timing controls, then writes raw code
intervals for the already-frozen sample manifest.  It does not optimize an
exposure or calibration multiplier.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from compression import count_interval, decode_lower
from freeze_manifest import (
    inverse_calibrated_counts,
    load_spectrum,
    nominal_exposure,
)
from raw_telemetry import read_frames, reconstruct_blocks


def _rank(values: np.ndarray) -> np.ndarray:
    """Return stable ordinal ranks for a control-only rank correlation."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def block_score(codes: np.ndarray, count_proxy: np.ndarray) -> dict[str, float]:
    """Score block agreement without fitting any free scale parameter."""
    lower = np.array([decode_lower(int(code)) for code in codes.ravel()]).reshape(
        codes.shape
    )
    finite = np.isfinite(count_proxy)
    log_proxy = np.log1p(count_proxy[finite])
    log_lower = np.log1p(lower[finite])
    pearson = float(np.corrcoef(log_proxy, log_lower)[0, 1])
    spearman = float(np.corrcoef(_rank(log_proxy), _rank(log_lower))[0, 1])

    contained = np.zeros(codes.shape, dtype=bool)
    for index in np.ndindex(codes.shape):
        contained[index] = count_interval(int(codes[index])).contains(
            float(count_proxy[index])
        )
    return {
        "log_pearson": pearson,
        "rank_correlation": spearman,
        "interval_fraction": float(np.mean(contained[finite])),
    }


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="ascii", newline="") as source:
        return list(csv.DictReader(source))


def write_audit(
    path: Path,
    manifest: list[dict[str, str]],
    codes: np.ndarray,
    frozen_proxy: np.ndarray,
    duty_proxy: np.ndarray,
    production_proxy: np.ndarray,
) -> None:
    fieldnames = (
        *manifest[0],
        "raw_code_hex",
        "raw_count_lower",
        "raw_count_upper",
        "proxy_in_raw_interval",
        "proxy_to_lower_ratio",
        "duty_cycle_count_proxy",
        "duty_proxy_in_raw_interval",
        "production_count_proxy",
        "production_to_lower_ratio",
    )
    with path.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fieldnames)
        writer.writeheader()
        for row in manifest:
            energy_index = int(row["energy_index"])
            direction_index = int(row["direction_index"])
            code = int(codes[energy_index, direction_index])
            interval = count_interval(code)
            proxy = float(frozen_proxy[energy_index, direction_index])
            duty_count = float(duty_proxy[energy_index, direction_index])
            production_count = float(production_proxy[energy_index, direction_index])
            ratio = proxy / interval.lower if interval.lower else np.nan
            production_ratio = (
                production_count / interval.lower if interval.lower else np.nan
            )
            writer.writerow(
                row
                | {
                    "raw_code_hex": f"0x{code:02x}",
                    "raw_count_lower": interval.lower,
                    "raw_count_upper": (
                        "open" if interval.upper is None else interval.upper
                    ),
                    "proxy_in_raw_interval": interval.contains(proxy),
                    "proxy_to_lower_ratio": f"{ratio:.9g}",
                    "duty_cycle_count_proxy": f"{duty_count:.9g}",
                    "duty_proxy_in_raw_interval": interval.contains(duty_count),
                    "production_count_proxy": f"{production_count:.9g}",
                    "production_to_lower_ratio": f"{production_ratio:.9g}",
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--theta", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-no", type=int, default=653)
    parser.add_argument(
        "--first-block-spec",
        type=int,
        default=630,
        help="Level-1 spectrum aligned to the first complete raw block",
    )
    parser.add_argument("--control-radius", type=int, default=3)
    args = parser.parse_args()

    energies, flux, _ = load_spectrum(args.data, args.spec_no)
    thetas = np.loadtxt(args.theta, dtype=np.float64)
    exposure = nominal_exposure(thetas)
    proxy = inverse_calibrated_counts(flux, energies, exposure)
    bins = np.array([np.count_nonzero(thetas == theta) for theta in thetas])
    duty_exposure = 5.0 / (256.0 * bins)
    production_exposure = 2.5 / bins
    rate = flux * 0.02 * energies[:, None]
    duty_proxy = rate * duty_exposure[None, :]
    production_proxy = rate * production_exposure[None, :]

    blocks = reconstruct_blocks(read_frames(args.telemetry))
    target_index = args.spec_no - args.first_block_spec
    if not 0 <= target_index < len(blocks):
        raise ValueError(f"Target block index {target_index} is out of range")

    start = max(0, target_index - args.control_radius)
    stop = min(len(blocks), target_index + args.control_radius + 1)
    print("block,spectrum,first_vcdu,log_pearson,rank_corr,interval_fraction")
    for block_index in range(start, stop):
        block = blocks[block_index]
        codes = np.frombuffer(block.codes, dtype=np.uint8).reshape(15, 88)
        score = block_score(codes, proxy)
        print(
            f"{block_index},{args.first_block_spec + block_index},"
            f"{block.first_vcdu},{score['log_pearson']:.6f},"
            f"{score['rank_correlation']:.6f},"
            f"{score['interval_fraction']:.6f}"
        )

    target_codes = np.frombuffer(blocks[target_index].codes, dtype=np.uint8).reshape(
        15, 88
    )
    write_audit(
        args.output,
        load_manifest(args.manifest),
        target_codes,
        proxy,
        duty_proxy,
        production_proxy,
    )
    lower = np.array(
        [decode_lower(int(code)) for code in target_codes.ravel()]
    ).reshape(target_codes.shape)
    positive = lower > 0.0
    production_ratio = production_proxy[positive] / lower[positive]
    duty_contained = np.array(
        [
            count_interval(int(code)).contains(float(value))
            for code, value in zip(
                target_codes.ravel(), duty_proxy.ravel(), strict=True
            )
        ]
    )
    print(f"duty_proxy_interval_fraction={np.mean(duty_contained):.6f}")
    print(f"median_production_to_raw_lower={np.median(production_ratio):.6f}")
    print(f"Wrote frozen-bin audit to {args.output}")


if __name__ == "__main__":
    main()
