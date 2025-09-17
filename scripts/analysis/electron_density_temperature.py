"""Plot electron density and temperature for a selected day."""

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

import src.config as config
from src.flux import ERData
from src.kappa import Kappa
from src.potential_mapper.spice import load_spice_files
from src.spacecraft_potential import theta_to_temperature_ev
from src.utils.flux_files import select_flux_day_file
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
    # Weighted vs unweighted fits
    parser.add_argument(
        "--weighted",
        dest="use_weights",
        action="store_true",
        default=True,
        help="Use count-derived weights during kappa fits (default)",
    )
    parser.add_argument(
        "--unweighted",
        dest="use_weights",
        action="store_false",
        help="Disable weighting; use unweighted least squares in log-space",
    )
    return parser.parse_args()


def compute_series(
    er_data: ERData, verbose: bool = False, use_weights: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spec_nos = er_data.data[config.SPEC_NO_COLUMN].unique()
    dens_orig: List[float] = []
    temp_orig_ev: List[float] = []
    dens_corr: List[float] = []
    temp_corr_ev: List[float] = []

    for spec_no in spec_nos:
        try:
            k = Kappa(er_data, spec_no=int(spec_no))
            # Original density (cm^-3)
            n_cm3 = k.density_estimate.to(
                ureg.particle / (ureg.centimeter**3)
            ).magnitude
            dens_orig.append(float(n_cm3))

            # Original fit temperature (eV)
            fit = k.fit(use_weights=use_weights)
            if fit is None or not fit.is_good_fit:
                temp_orig_ev.append(np.nan)
            else:
                density_mag, kappa_val, theta_val = fit.params.to_tuple()
                Te_ev = theta_to_temperature_ev(theta_val, kappa_val)
                temp_orig_ev.append(float(Te_ev))

            # Corrected fit (day/night aware)
            cfit, _U = k.corrected_fit(use_weights=use_weights)
            if cfit is None or not cfit.is_good_fit:
                dens_corr.append(np.nan)
                temp_corr_ev.append(np.nan)
            else:
                d_mag, kappa_val, theta_val = cfit.params.to_tuple()
                n_corr_cm3 = (
                    (d_mag * ureg.particle / (ureg.meter**3))
                    .to(ureg.particle / (ureg.centimeter**3))
                    .magnitude
                )
                dens_corr.append(float(n_corr_cm3))
                Te_corr_ev = theta_to_temperature_ev(theta_val, kappa_val)
                temp_corr_ev.append(float(Te_corr_ev))
        except Exception as e:
            if verbose:
                print(f"Spec {spec_no}: {e}")
            dens_orig.append(np.nan)
            temp_orig_ev.append(np.nan)
            dens_corr.append(np.nan)
            temp_corr_ev.append(np.nan)

    return (
        np.array(dens_orig),
        np.array(temp_orig_ev),
        np.array(dens_corr),
        np.array(temp_corr_ev),
    )


def plot_series(
    dens_orig: np.ndarray,
    temp_orig_ev: np.ndarray,
    dens_corr: np.ndarray,
    temp_corr_ev: np.ndarray,
    output: str | None,
    display: bool,
) -> None:
    x = np.arange(len(dens_orig))
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), sharex=True, constrained_layout=True
    )

    # Density
    ax1.plot(x, dens_orig, lw=1, label="density (orig)")
    ax1.plot(x, dens_corr, lw=1, label="density (corrected)")
    ax1.set_ylabel("Density (cm$^{-3}$)")
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="best")

    # Temperature
    ax2.plot(x, temp_orig_ev, color="tab:orange", lw=1, label="Te (orig)")
    ax2.plot(x, temp_corr_ev, color="tab:green", lw=1, label="Te (corrected)")
    ax2.set_ylabel("Temperature (eV)")
    ax2.set_xlabel("Index (spectrum order)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="best")

    fig.suptitle("Electron density and temperature over one day (orig vs corrected)")

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
    # Load SPICE kernels for corrected fits (illumination, day/night branching)
    try:
        load_spice_files()
    except Exception as e:
        if args.verbose:
            print(f"SPICE load failed: {e}")
    day_file = select_flux_day_file(args.year, args.month, args.day)
    er_data = ERData(str(day_file))

    dens_orig, temp_orig_ev, dens_corr, temp_corr_ev = compute_series(
        er_data, verbose=args.verbose, use_weights=args.use_weights
    )
    plot_series(
        dens_orig, temp_orig_ev, dens_corr, temp_corr_ev, args.output, args.display
    )


if __name__ == "__main__":
    main()
