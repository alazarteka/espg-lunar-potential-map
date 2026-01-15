"""
Synthetic injection tests for loss-cone retrieval validation.

This module generates synthetic electron distributions with known surface potentials,
adds realistic noise, runs the fitting pipeline, and quantifies retrieval bias and
uncertainty. This is essential for validating the methodology before publication.

Usage:
    uv run python -m src.utils.injection_tests

References:
    - Halekas et al. 2008 for loss-cone physics
    - GPT-5 Pro consultation (Q12) recommending synthetic injection tests
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from src import config
from src.model import synth_losscone


@dataclass
class InjectionResult:
    """Results from a single synthetic injection test."""

    true_u_surface: float
    true_bs_over_bm: float
    fitted_u_surface: float
    fitted_bs_over_bm: float
    fitted_beam_amp: float
    chi2: float

    @property
    def u_error(self) -> float:
        """Absolute error in U_surface."""
        return self.fitted_u_surface - self.true_u_surface

    @property
    def u_relative_error(self) -> float:
        """Relative error in U_surface (%)."""
        if abs(self.true_u_surface) < 1e-6:
            return np.nan
        return 100 * self.u_error / abs(self.true_u_surface)

    @property
    def bs_error(self) -> float:
        """Absolute error in bs_over_bm."""
        return self.fitted_bs_over_bm - self.true_bs_over_bm


@dataclass
class InjectionTestSummary:
    """Summary statistics from multiple injection tests."""

    n_tests: int
    n_successful: int
    u_bias: float  # Mean error in U_surface
    u_rmse: float  # RMS error in U_surface
    u_mae: float  # Mean absolute error in U_surface
    bs_bias: float  # Mean error in bs_over_bm
    bs_rmse: float  # RMS error in bs_over_bm
    results: list[InjectionResult]

    def __str__(self) -> str:
        success_rate = 100 * self.n_successful / self.n_tests
        return f"""Injection Test Summary:
  Tests: {self.n_successful}/{self.n_tests} successful ({success_rate:.1f}%)
  U_surface:
    Bias:  {self.u_bias:+.2f} V
    RMSE:  {self.u_rmse:.2f} V
    MAE:   {self.u_mae:.2f} V
  bs_over_bm:
    Bias:  {self.bs_bias:+.4f}
    RMSE:  {self.bs_rmse:.4f}"""


def generate_synthetic_spectrum(
    u_surface: float,
    bs_over_bm: float,
    u_spacecraft: float = 10.0,
    noise_level: float = 0.1,
    beam_amp: float = 0.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic loss-cone spectrum with known parameters.

    Args:
        u_surface: True surface potential in volts (should be negative)
        bs_over_bm: True B_surface / B_spacecraft ratio
        u_spacecraft: Spacecraft potential in volts
        noise_level: Relative noise level (0.1 = 10% Gaussian noise in log space)
        beam_amp: Secondary electron beam amplitude (0 = no beam)
        seed: Random seed for reproducibility

    Returns:
        energies: (nE,) energy grid in eV
        pitches: (nP,) pitch angle grid in degrees
        flux: (nE, nP) normalized flux values with noise
    """
    if seed is not None:
        np.random.seed(seed)

    # Standard LP energy and pitch grids
    energies = np.geomspace(20, 20000, config.SWEEP_ROWS)
    pitch_1d = np.linspace(0, 180, config.CHANNELS)
    # synth_losscone expects pitch_grid to be (nE, nPitch)
    pitches = np.broadcast_to(pitch_1d, (len(energies), len(pitch_1d)))

    # Generate clean synthetic spectrum using the forward model
    beam_width = config.LOSS_CONE_BEAM_WIDTH_EV
    model = synth_losscone(
        energy_grid=energies,
        pitch_grid=pitches,
        U_surface=u_surface,
        U_spacecraft=u_spacecraft,
        bs_over_bm=bs_over_bm,
        beam_width_eV=beam_width,
        beam_amp=beam_amp,
        beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
        background=config.LOSS_CONE_BACKGROUND,
    )

    # Add log-normal noise to simulate counting statistics
    if noise_level > 0:
        log_noise = np.random.normal(0, noise_level, model.shape)
        model = model * np.exp(log_noise)
        model = np.clip(model, config.LOSS_CONE_BACKGROUND, 1.0)

    return energies, pitch_1d, model


def run_injection_test(
    u_surface: float,
    bs_over_bm: float,
    u_spacecraft: float = 10.0,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> InjectionResult | None:
    """
    Run a single synthetic injection test.

    Generates a synthetic spectrum with known parameters, fits it, and returns
    the comparison of true vs fitted values.

    Args:
        u_surface: True surface potential
        bs_over_bm: True magnetic field ratio
        u_spacecraft: Spacecraft potential
        noise_level: Relative noise level
        seed: Random seed

    Returns:
        InjectionResult or None if fitting failed
    """
    from scipy.optimize import differential_evolution

    # Generate synthetic data
    energies, pitches, flux = generate_synthetic_spectrum(
        u_surface=u_surface,
        bs_over_bm=bs_over_bm,
        u_spacecraft=u_spacecraft,
        noise_level=noise_level,
        seed=seed,
    )

    # Prepare data for fitting (matching LossConeFitter logic)
    eps = 1e-6
    data_mask = flux > 0
    log_data = np.log(flux + eps)

    # Build 2D pitch grid for forward model
    pitch_2d = np.broadcast_to(pitches, (len(energies), len(pitches)))

    # Chi-squared objective function
    def chi2_scalar(params):
        U_surf, bs, beam_amp = params
        beam_amp = float(
            np.clip(
                beam_amp, config.LOSS_CONE_BEAM_AMP_MIN, config.LOSS_CONE_BEAM_AMP_MAX
            )
        )
        beam_width = config.LOSS_CONE_BEAM_WIDTH_EV

        model = synth_losscone(
            energy_grid=energies,
            pitch_grid=pitch_2d,
            U_surface=U_surf,
            U_spacecraft=u_spacecraft,
            bs_over_bm=bs,
            beam_width_eV=beam_width,
            beam_amp=beam_amp,
            beam_pitch_sigma_deg=config.LOSS_CONE_BEAM_PITCH_SIGMA_DEG,
            background=config.LOSS_CONE_BACKGROUND,
        )

        if not np.all(np.isfinite(model)) or (model <= 0).all():
            return 1e30

        log_model = np.log(model + eps)
        diff = (log_data - log_model) * data_mask
        chi2 = np.sum(diff * diff)

        if not np.isfinite(chi2):
            return 1e30
        return chi2

    # Bounds
    bounds = [
        (config.LOSS_CONE_U_SURFACE_MIN, config.LOSS_CONE_U_SURFACE_MAX),
        (0.1, 1.1),
        (config.LOSS_CONE_BEAM_AMP_MIN, config.LOSS_CONE_BEAM_AMP_MAX),
    ]

    try:
        result = differential_evolution(
            chi2_scalar,
            bounds,
            maxiter=100,
            tol=1e-4,
            seed=42,
            workers=1,
            updating="deferred",
        )

        if not result.success:
            return None

        fitted_u, fitted_bs, fitted_amp = result.x
        chi2 = result.fun

        return InjectionResult(
            true_u_surface=u_surface,
            true_bs_over_bm=bs_over_bm,
            fitted_u_surface=fitted_u,
            fitted_bs_over_bm=fitted_bs,
            fitted_beam_amp=fitted_amp,
            chi2=chi2,
        )

    except Exception as e:
        logging.debug(f"Injection test failed: {e}")
        return None


def run_injection_test_suite(
    u_range: tuple[float, float] = (-500, -20),
    bs_range: tuple[float, float] = (0.7, 1.0),
    n_u_samples: int = 10,
    n_bs_samples: int = 5,
    noise_level: float = 0.1,
    n_repeats: int = 3,
) -> InjectionTestSummary:
    """
    Run a suite of injection tests across parameter space.

    Args:
        u_range: (min, max) range for U_surface sampling
        bs_range: (min, max) range for bs_over_bm sampling
        n_u_samples: Number of U_surface values to test
        n_bs_samples: Number of bs_over_bm values to test
        noise_level: Relative noise level for synthetic data
        n_repeats: Number of noise realizations per parameter combination

    Returns:
        InjectionTestSummary with statistics and individual results
    """
    u_values = np.linspace(u_range[0], u_range[1], n_u_samples)
    bs_values = np.linspace(bs_range[0], bs_range[1], n_bs_samples)

    results: list[InjectionResult] = []
    n_tests = 0

    total_tests = n_u_samples * n_bs_samples * n_repeats

    for u in tqdm(u_values, desc="U_surface", leave=False):
        for bs in bs_values:
            for rep in range(n_repeats):
                n_tests += 1
                seed = hash((u, bs, rep)) % (2**31)

                result = run_injection_test(
                    u_surface=u,
                    bs_over_bm=bs,
                    noise_level=noise_level,
                    seed=seed,
                )

                if result is not None:
                    results.append(result)

    if not results:
        return InjectionTestSummary(
            n_tests=n_tests,
            n_successful=0,
            u_bias=np.nan,
            u_rmse=np.nan,
            u_mae=np.nan,
            bs_bias=np.nan,
            bs_rmse=np.nan,
            results=[],
        )

    # Compute statistics
    u_errors = np.array([r.u_error for r in results])
    bs_errors = np.array([r.bs_error for r in results])

    return InjectionTestSummary(
        n_tests=n_tests,
        n_successful=len(results),
        u_bias=float(np.mean(u_errors)),
        u_rmse=float(np.sqrt(np.mean(u_errors**2))),
        u_mae=float(np.mean(np.abs(u_errors))),
        bs_bias=float(np.mean(bs_errors)),
        bs_rmse=float(np.sqrt(np.mean(bs_errors**2))),
        results=results,
    )


def main():
    """Run injection test suite and print results."""
    logging.basicConfig(level=logging.INFO)

    print("Running synthetic injection test suite...")
    print("=" * 60)

    summary = run_injection_test_suite(
        u_range=(-500, -20),
        bs_range=(0.7, 1.0),
        n_u_samples=10,
        n_bs_samples=5,
        noise_level=0.1,
        n_repeats=3,
    )

    print(summary)
    print("=" * 60)

    # Print some example results
    if summary.results:
        print("\nExample results (first 5):")
        print("-" * 60)
        header = (
            f"{'True U':>10} {'Fitted U':>10} {'Error':>10} "
            f"{'True bs':>8} {'Fitted bs':>10}"
        )
        print(header)
        print("-" * 60)
        for r in summary.results[:5]:
            print(
                f"{r.true_u_surface:>10.1f} {r.fitted_u_surface:>10.1f} "
                f"{r.u_error:>+10.1f} {r.true_bs_over_bm:>8.3f} "
                f"{r.fitted_bs_over_bm:>10.3f}"
            )


if __name__ == "__main__":
    main()
