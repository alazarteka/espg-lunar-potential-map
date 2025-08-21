import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import src.config as config
from src.flux import ERData
from src.potential_mapper.pipeline import DataLoader
from src.potential_mapper.spice import load_spice_files
from src.spacecraft_potential import calculate_potential


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot spacecraft potential over one day using spacecraft_potential module",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year", type=int, required=True, help="Year (e.g., 1998)")
    parser.add_argument("--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("--day", type=int, required=True, help="Day (1-31)")
    parser.add_argument("--output", type=str, default=None, help="Path to save plot")
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose logs"
    )
    parser.add_argument(
        "-d", "--display", action="store_true", default=False, help="Show plot"
    )
    return parser.parse_args()


def select_day_file(year: int, month: int, day: int) -> Path:
    files = DataLoader.discover_flux_files(year=year, month=month, day=day)
    if not files:
        raise FileNotFoundError("No ER file found for the requested date")
    if len(files) > 1:
        # Pick the first file deterministically; log a warning for multiple
        print(f"Warning: multiple files matched; using {files[0]}")
    return files[0]


def plot_timeseries(times: np.ndarray, potentials: np.ndarray, output: str | None, display: bool) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(times, potentials, marker="o", linestyle="-", markersize=2)
    ax.set_xlabel("Index (spectrum order)")
    ax.set_ylabel("Spacecraft potential U (V)")
    ax.set_title("Spacecraft potential over one day")
    ax.grid(True, alpha=0.3)

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print(f"Saved plot to {out_path}")
    if display:
        print("Displaying plot...")
        plt.show()


def main() -> None:
    args = parse_args()
    load_spice_files()

    day_file = select_day_file(args.year, args.month, args.day)
    er_data = ERData(str(day_file))

    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
    potentials: List[float] = []
    valid_mask = []

    for spec_no in spec_nos:
        try:
            res = calculate_potential(er_data, int(spec_no))
            if res is None:
                potentials.append(np.nan)
                valid_mask.append(False)
            else:
                _params, U = res
                potentials.append(float(U.magnitude))
                valid_mask.append(True)
        except Exception as e:
            if args.verbose:
                print(f"Spec {spec_no}: {e}")
            potentials.append(np.nan)
            valid_mask.append(False)

    # For now, use spectrum index as x-axis; UTC parsing can be added later
    idx = np.arange(len(spec_nos))
    plot_timeseries(idx, np.array(potentials), args.output, args.display)


if __name__ == "__main__":
    main()
