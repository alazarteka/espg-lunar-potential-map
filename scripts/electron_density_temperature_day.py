import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import src.config as config
from src.flux import ERData
from src.kappa import Kappa
from src.potential_mapper.pipeline import DataLoader
from src.spacecraft_potential import theta_to_temperature_ev
from src.utils.units import ureg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot electron density (m^-3) and temperature (eV) over one day",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--year", type=int, required=True, help="Year (e.g., 1998)")
    parser.add_argument("--month", type=int, required=True, help="Month (1-12)")
    parser.add_argument("--day", type=int, required=True, help="Day (1-31)")
    parser.add_argument(
        "--output", type=str, default=None, help="Path to save plot PNG"
    )
    parser.add_argument(
        "-d", "--display", action="store_true", default=False, help="Show plot"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", default=False, help="Verbose logs"
    )
    return parser.parse_args()


def select_day_file(year: int, month: int, day: int) -> Path:
    files = DataLoader.discover_flux_files(year=year, month=month, day=day)
    if not files:
        raise FileNotFoundError("No ER file found for the requested date")
    if len(files) > 1:
        print(f"Warning: multiple files matched; using {files[0]}")
    return files[0]


def compute_series(er_data: ERData, verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
    densities: List[float] = []
    temperatures_ev: List[float] = []

    for spec_no in spec_nos:
        try:
            k = Kappa(er_data, spec_no=int(spec_no))
            # Density estimate is available without a full fit
            n_m3 = k.density_estimate.to(ureg.particle / (ureg.centimeter ** 3)).magnitude
            densities.append(float(n_m3))

            fit = k.fit()
            if fit is None or not fit.is_good_fit:
                temperatures_ev.append(np.nan)
                continue

            density_mag, kappa, theta = fit.params.to_tuple()
            Te_ev = theta_to_temperature_ev(theta, kappa)
            temperatures_ev.append(float(Te_ev))
        except Exception as e:
            if verbose:
                print(f"Spec {spec_no}: {e}")
            densities.append(np.nan)
            temperatures_ev.append(np.nan)

    return np.array(densities), np.array(temperatures_ev)


def plot_series(dens: np.ndarray, temp_ev: np.ndarray, output: str | None, display: bool) -> None:
    x = np.arange(len(dens))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, constrained_layout=True)

    ax1.plot(x, dens, lw=1)
    ax1.set_ylabel("Density (cm$^{-3}$)")
    ax1.set_yscale('log')
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, temp_ev, color="tab:orange", lw=1)
    ax2.set_ylabel("Temperature (eV)")
    ax2.set_xlabel("Index (spectrum order)")
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Electron density and temperature over one day")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=180)
        print(f"Saved plot to {out_path}")
    if display:
        plt.show()


def main() -> None:
    args = parse_args()
    day_file = select_day_file(args.year, args.month, args.day)
    er_data = ERData(str(day_file))

    dens, temp_ev = compute_series(er_data, verbose=args.verbose)
    plot_series(dens, temp_ev, args.output, args.display)


if __name__ == "__main__":
    main()
