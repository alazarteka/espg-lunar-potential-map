import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.flux import PitchAngle
from tests.test_kappa_fitter import prepare_synthetic_er


def compute_pitch_sets(b_hat: np.ndarray, look: np.ndarray) -> dict[str, np.ndarray]:
    """Return pitch angles (deg) for different sign conventions.

    - base:  angle between +B and +look ("going to")
    - inv_look:  +B and -look ("coming from")
    - inv_B:  -B and +look
    """

    def ang(cosv):
        return np.degrees(np.arccos(np.clip(cosv, -1.0, 1.0)))

    cos_base = np.einsum("ij,ij->i", b_hat, look)
    cos_inv_look = np.einsum("ij,ij->i", b_hat, -look)
    cos_inv_b = np.einsum("ij,ij->i", -b_hat, look)

    return {
        "base": ang(cos_base),
        "inv_look": ang(cos_inv_look),
        "inv_B": ang(cos_inv_b),
    }


def main():
    # 1) Build the synthetic ER used by the tests (isotropic per channel)
    er = prepare_synthetic_er()

    # 2) Compute pitch angles with the repo implementation
    pa = PitchAngle(er, str(config.DATA_DIR / config.THETA_FILE))

    # Choose one energy row to visualize
    row = 0
    flux_row = er.data[config.FLUX_COLS].to_numpy(dtype=float)[row]

    # Look vectors used by PitchAngle
    phis = np.deg2rad(er.data[config.PHI_COLS].to_numpy(dtype=float)[row])
    thetas = np.deg2rad(np.loadtxt(config.DATA_DIR / config.THETA_FILE))
    X = np.cos(phis) * np.cos(thetas)
    Y = np.sin(phis) * np.cos(thetas)
    Z = np.sin(thetas)
    look = np.stack([X, Y, Z], axis=-1)  # (88, 3)

    # Unit magnetic field for this row
    B = er.data[config.MAG_COLS].to_numpy(dtype=float)[row]
    B_hat = B / (np.linalg.norm(B) + 1e-20)
    B_hat = np.tile(B_hat, (config.CHANNELS, 1))  # (88, 3)

    # 3) Compute pitch angles under three conventions
    pitch_sets = compute_pitch_sets(B_hat, look)

    # 4) Create a slightly beamed version of the synthetic flux to make
    #     angular trends visible in the plot (the base synthetic is isotropic)
    alpha = pitch_sets["base"]  # degrees for +B & +look
    # Gaussian beam around 20° (arbitrary) for visualization
    A, alpha0, w = 3.0, 20.0, 15.0
    beam = 1.0 + A * np.exp(-((alpha - alpha0) ** 2) / (2 * w * w))
    flux_beamed = flux_row * beam

    # Normalize for plotting
    f0 = np.maximum(1e-30, np.nanmean(flux_row))
    flux_norm = flux_row / f0
    flux_beamed_norm = flux_beamed / np.maximum(1e-30, np.nanmean(flux_beamed))

    # 5) Plot
    plt.figure(figsize=(12, 4))
    for i, (label, angles) in enumerate(pitch_sets.items(), start=1):
        order = np.argsort(angles)
        plt.subplot(1, 3, i)
        plt.plot(angles[order], flux_norm[order], "o", ms=4, label="synthetic (flat)")
        plt.plot(
            angles[order],
            flux_beamed_norm[order],
            "-",
            lw=2,
            alpha=0.9,
            label="synthetic + beam",
        )
        plt.xlabel("Pitch angle (deg)")
        if i == 1:
            plt.ylabel("Normalized flux")
        plt.title(label)
        plt.grid(True, alpha=0.3)
        if i == 1:
            plt.legend(loc="best", fontsize=9)

    plt.tight_layout()
    out = config.PROJECT_ROOT / "temp" / "synthetic_er_pitch_angle.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"Saved plot → {out}")


if __name__ == "__main__":
    main()
