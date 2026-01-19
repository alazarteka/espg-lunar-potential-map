
# Kappa Fitter: Design and Analysis Summary

**Date:** 2025-08-09
**Updated:** 2025-11-16

## 1. Overview

This document provides a comprehensive summary of the design, implementation, and analysis of the `Kappa` fitter in `src/kappa.py`. The primary purpose of this tool is to derive the physical parameters of the near-lunar plasma environment by fitting a kappa distribution model to the electron flux data measured by the Lunar Prospector's Electron Reflectometer.

## 2. Fitter Design and Logic

The fitting process follows these core steps:

1.  **Data Preparation:** For a given energy spectrum, the tool calculates the omnidirectional flux from the instrument's directional measurements, filtering for valid pitch angles.
2.  **Density Estimation:** A direct estimation of the electron number density (`n`) is calculated from the measured flux data *before* the main fitting procedure.
3.  **Parameter Fitting:** The `kappa` (κ) and `theta` (Θ) parameters are determined by finding the values that best fit the measured data. This is achieved by minimizing a chi-squared objective function, which quantifies the difference between the measured flux and the flux predicted by the kappa distribution model.
4.  **Optimization:** The minimization is performed using the `L-BFGS-B` algorithm, a robust optimization method. To avoid local minima, the fit is initiated from multiple starting points using Latin Hypercube Sampling.

## 3. Key Improvements and Features

Significant analysis and several key improvements were made to ensure the robustness and scientific validity of the fitter's results.

### 3.1. Sophisticated Error Model

The chi-squared calculation relies on a proper estimation of the uncertainty for each data point. The fitter uses a comprehensive error model based on the findings in the `technical_report.md`:

`σ² = N + (ε_gain * N)² + (ε_G * N)² + N_bg`

This model accounts not only for the statistical Poisson noise (`N`) but also for systematic uncertainties from instrument gain (`ε_gain`) and the geometric factor (`ε_G`). This ensures that the weights used in the fitting process accurately reflect the true uncertainty of each measurement.

### 3.2. Data-Driven Fit Quality Threshold

To programmatically distinguish between reliable and questionable fits, a statistical analysis was performed on the distribution of chi-squared errors across the entire dataset (see `docs/archive/analysis/fitter_error_analysis.md` for details).

Based on this analysis, a **fit quality threshold** was initially proposed at the 95th percentile of the filtered error distribution (215,000).

**Current Implementation (as of 2025-11-16):**

The code currently uses `FIT_ERROR_THRESHOLD = 21_500_000_000` (2.15×10¹⁰), which is substantially more permissive than the statistically-derived value. This threshold accepts nearly all fits except catastrophic failures.

A boolean flag, `is_good_fit`, is returned with every fit. **Note:** The interpretation of this flag depends on which threshold value is ultimately adopted. See `docs/archive/analysis/fitter_error_analysis.md` Section 5 for ongoing discussion of threshold selection.

### 3.3. Uncertainty Calculation for Fitted Parameters

The fitter now calculates and returns the 1-sigma uncertainties for the fitted parameters, `kappa` and `log(theta)`.

This is achieved by using the **inverse of the Hessian matrix**, which is provided by the `scipy` optimizer. The Hessian describes the curvature of the chi-squared surface at the best-fit minimum, and its inverse (the covariance matrix) directly yields the uncertainties of the parameters. This provides a crucial measure of how well-constrained the fitted parameters are by the data.

## 4. The `FitResults` Object

To provide a clean and structured output, the `fit` method returns a single `FitResults` dataclass object with the following attributes:

*   `params: KappaParams`: The best-fit parameters (`density`, `kappa`, `theta`).
*   `params_uncertainty: KappaParams`: The calculated 1-sigma uncertainties for the fitted parameters. The `density` field is 0, as its uncertainty is not determined by the fit.
*   `error: float`: The final chi-squared value of the fit.
*   `is_good_fit: bool`: The flag indicating if the fit quality is within the established threshold.

This object encapsulates all the necessary information from the fitting process in a clear and easily accessible way, improving the overall code clarity and maintainability.
