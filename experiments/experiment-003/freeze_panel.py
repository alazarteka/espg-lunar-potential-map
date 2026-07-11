"""Freeze the Experiment 003 benchmark, discovery, and confirmation panels.

Selection is temporal and was defined before evaluating any Experiment 003
candidate objective. Run from the repository root:

    .venv/bin/python experiments/experiment-003/freeze_panel.py
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

BENCHMARKS = (652, 653, 654)
DISCOVERY = (635, 640, 645, 650, 655, 660, 665)
CONFIRMATION = tuple(range(669, 719))
FIRST_RAW_SPEC = 630


def spectrum_times(path: Path, wanted: set[int]) -> dict[int, tuple[str, str]]:
    found: dict[int, tuple[str, str]] = {}
    with path.open(encoding="ascii") as source:
        for line in source:
            fields = line.split()
            spec_no = int(fields[3])
            if spec_no in wanted and spec_no not in found:
                found[spec_no] = (fields[0], fields[1].rstrip("."))
    missing = wanted - found.keys()
    if missing:
        raise ValueError(f"Missing spectra: {sorted(missing)}")
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data/1999/091_120APR/3D990429.TAB"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/experiment-003/panel_manifest.csv"),
    )
    args = parser.parse_args()

    panels = (
        ("benchmark", BENCHMARKS),
        ("discovery", DISCOVERY),
        ("confirmation", CONFIRMATION),
    )
    wanted = set().union(*(set(specs) for _, specs in panels))
    times = spectrum_times(args.data, wanted)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="ascii", newline="") as target:
        writer = csv.DictWriter(
            target,
            fieldnames=(
                "panel",
                "panel_index",
                "spec_no",
                "utc",
                "spectrum_time_unix",
                "raw_block_index",
                "is_published_primary",
            ),
        )
        writer.writeheader()
        for panel, specs in panels:
            for panel_index, spec_no in enumerate(specs):
                utc, unix_time = times[spec_no]
                writer.writerow(
                    {
                        "panel": panel,
                        "panel_index": panel_index,
                        "spec_no": spec_no,
                        "utc": utc,
                        "spectrum_time_unix": unix_time,
                        "raw_block_index": spec_no - FIRST_RAW_SPEC,
                        "is_published_primary": spec_no == 653,
                    }
                )
    print(f"Wrote {len(wanted)} frozen spectra to {args.output}")


if __name__ == "__main__":
    main()
