# Physics and Algorithms

This document details the physical models, algorithms, and assumptions used in the ESPG Lunar Potential Map project.

## 1. Electron Reflectometer (ER) Data Processing

The Lunar Prospector Electron Reflectometer (ER) provides electron flux measurements ($F$) across varying energies and directions. The raw data is processed to determine the electron pitch angle distribution, which is crucial for identifying the loss cone.

### 1.1 Pitch Angle Calculation

The pitch angle $\alpha$ is the angle between the electron's velocity vector and the local magnetic field vector $\mathbf{B}$.

*   **Magnetic Field Convention**: The ER data provides the magnetic field vector in the instrument frame. By convention, the ER magnetic field vector points roughly sunward. However, for loss cone tracing (looking at electrons coming *from* the lunar surface), we define the field direction towards the Moon.
    *   **Implementation**: `unit_magnetic_field = -magnetic_field / |magnetic_field|`
*   **Pitch Angle**:
    $$ \alpha = \arccos(\hat{\mathbf{B}} \cdot \hat{\mathbf{r}}_{look}) $$
    where $\hat{\mathbf{r}}_{look}$ is the instrument look direction derived from the detector's $\theta$ (polar) and $\phi$ (azimuth) angles.

### 1.2 Flux Normalization

To identify the loss cone, the electron flux is normalized. The underlying assumption is that the **incident** flux (electrons moving towards the Moon, pitch $\alpha < 90^\circ$) represents the source population, and the **reflected** flux (electrons returning from the Moon, pitch $\alpha > 90^\circ$) is modified by the surface potential.

Several normalization modes are supported (`src.flux.LossConeFitter`):

1.  **Global** (`global`): The entire energy-pitch distribution is divided by the maximum incident flux value across all energies and angles. This preserves the relative energy spectrum shape but can be sensitive to outliers.
2.  **Ratio** (`ratio`): Each energy bin is normalized independently by dividing by the mean (or max) incident flux at that energy. This removes the energy spectral shape, focusing purely on the angular anisotropy (the loss cone). This is the default mode for some analyses.
3.  **Pairwise Ratio** (`ratio2`): Explicitly normalizes each reflected bin by its symmetric incident counterpart (mirroring around $90^\circ$).
4.  **Rescaled Ratio** (`ratio_rescaled`): Same as `ratio`, but rescaled so the maximum value is 1.0.

**Assumption**: The incident plasma distribution is approximately isotropic over the incident hemisphere ($\alpha < 90^\circ$).

## 2. Spacecraft Potential Estimation

The spacecraft potential ($U_{sc}$) biases the measured electron energies ($E_{meas} = E_{plasma} + e U_{sc}$) and must be determined before fitting the surface potential. The estimation strategy depends on whether the spacecraft is in sunlight or shadow.

### 2.1 Dayside (Sunlit)

In sunlight, photoelectric emission usually dominates, charging the spacecraft positive.

1.  **Initial Kappa Fit**: Fit a Kappa distribution to the raw (uncorrected) electron spectrum.
2.  **Current Balance**: Calculate the electron current density $J_e$ from the fitted distribution.
3.  **J-U Inversion**: Invert a pre-computed or empirical $J-U$ curve (Photoemission vs. Electron collection balance) to estimate $U_{sc}$.
    *   **Implementation**: `src.physics.jucurve.U_from_J`.
4.  **Correction**:
    *   Shift energy bins: $E_{corrected} = E_{meas} - e U_{sc}$.
    *   Refit the Kappa distribution to the corrected energies to refine parameters.

### 2.2 Nightside (Shadow)

In shadow, photoemission is absent. The spacecraft floats to a potential where the net current is zero:
$$ J_e(U_{sc}) + J_i(U_{sc}) - J_{see}(U_{sc}) = 0 $$

*   **Electron Current ($J_e$)**: Integrated flux of the Kappa-distributed electrons reaching the spacecraft.
    *   $$ J_e = e \int_{|U_{sc}|}^{\infty} F(E) dE $$
*   **Ion Current ($J_i$)**: Modeled using Orbit Motion Limited (OML) theory (or a simplified thermal approximation) for attractive potentials (since $U_{sc}$ is typically negative in shadow).
    *   $$ J_i = J_{i0} \sqrt{1 - \frac{U_{sc}}{T_i}} $$
    *   Assumption: $T_i \approx T_e$ (Ion temperature equals electron temperature).
*   **Secondary Electron Emission ($J_{see}$)**: Electrons released from the spacecraft surface due to electron impact. Modeled using the **Sternglass equation**:
    *   $$ \delta(E) = 7.4 \delta_{max} \frac{E}{E_{max}} \exp\left(-2 \sqrt{\frac{E}{E_{max}}}\right) $$
    *   Parameters: $E_{max} = 500$ eV, $\delta_{max} = 1.5$.
*   **Algorithm**: The code solves for $U_{sc}$ using a root-finding algorithm (`brentq`) on the current balance equation.

## 3. Surface Potential Fitting (Loss Cone Analysis)

The lunar surface potential ($U_{surf}$) is determined by analyzing the "loss cone" in the reflected electron population.

### 3.1 Physics Model

Electrons originating from infinity (incident) can only reflect back from the surface if their magnetic mirror point lies above the surface. The presence of a surface electric field modifies this condition.

The critical pitch angle $\alpha_c$ (loss cone angle) is defined by conservation of energy and the first adiabatic invariant (magnetic moment). For a retarding potential (negative $U_{surf}$):

$$ \sin^2(\alpha_c) = \frac{B_s}{B_m} \left( 1 + \frac{e U_{surf}}{E_{plasma}} \right) $$

*   $B_m$: Magnetic field at the spacecraft (measured).
*   $B_s$: Magnetic field at the surface (unknown, fitted as ratio $B_s/B_m$).
*   $U_{surf}$: Surface potential relative to plasma (unknown, fitted).
*   $E_{plasma}$: Electron energy in plasma frame ($E_{meas} - e U_{sc}$).

**Flux Model**:
*   **Inside Loss Cone** ($\alpha > 180^\circ - \alpha_c$): Flux is low (absorbed by surface). Modeled as `background` level.
*   **Outside Loss Cone** ($\alpha < 180^\circ - \alpha_c$): Flux is high (reflected). Modeled as `1.0` (normalized).

### 3.2 Secondary Electron Beam

A Gaussian beam component is added to the model to account for secondary electrons or backscattered electrons accelerated upward by the sheath potential.

*   **Beam Energy**: Centered at $|U_{surf} - U_{sc}|$.
*   **Beam Width**: Proportional to $|U_{surf}|$.
*   **Beam Amplitude**: Fitted parameter (or fixed).

### 3.3 Fitting Algorithm

The code fits the model to the measured normalized flux map (Energy vs. Pitch) by minimizing the $\chi^2$ statistic:

$$ \chi^2 = \sum_{E, \alpha} \left( \log(F_{data}) - \log(F_{model}) \right)^2 $$

1.  **Initialization**: Latin Hypercube Sampling (LHS) generates initial guesses for $U_{surf}$, $B_s/B_m$, and Beam Amplitude.
2.  **Optimization**: `scipy.optimize.differential_evolution` (or Nelder-Mead refinement) finds the global minimum.

## 4. Kappa Distribution Fitting

The incident electron population is modeled as a Kappa distribution, which describes suprathermal tails better than a Maxwellian.

$$ f(v) = n \left(\frac{m}{2\pi \kappa E_{th}}\right)^{3/2} \frac{\Gamma(\kappa+1)}{\Gamma(\kappa-1/2)} \left(1 + \frac{v^2}{\kappa v_{th}^2}\right)^{-(\kappa+1)} $$

*   **Fitted Parameters**: Density ($n$), Kappa ($\kappa$), Thermal Velocity ($v_{th}$ or Temperature $T$).
*   **Instrument Response**: The model includes convolution with the instrument's energy response function (Gaussian in log-energy).

## 5. Temporal Reconstruction

To generate global maps from sparse orbital tracks, the surface potential is modeled as a time-varying field using Spherical Harmonics ($Y_{lm}$) and temporal basis functions ($T_k$).

### 5.1 Model

$$ \Phi(\theta, \phi, t) = \sum_{l=0}^{L_{max}} \sum_{m=-l}^{l} a_{lm}(t) Y_{lm}(\theta, \phi) $$

The coefficients $a_{lm}(t)$ are expanded in a temporal basis:

$$ a_{lm}(t) = \sum_{k} b_{lmk} T_k(t) $$

### 5.2 Temporal Basis Functions

The project supports various basis functions ($T_k$) to capture periodic variations (e.g., lunar day/night cycle):
*   **Constant**: $\Phi_0$
*   **Synodic**: Harmonics of the lunar synodic period (~29.5 days).
    *   $\cos(n \omega_{syn} t)$, $\sin(n \omega_{syn} t)$
*   **Sidereal**: Harmonics of the lunar sidereal period (~27.3 days).

### 5.3 Reconstruction

The coefficients $b_{lmk}$ are solved via regularized least-squares fitting to the aggregate set of instantaneous potential measurements.

## 6. Validation & Reproduction Notes

The project aims to reproduce results from *Halekas et al. (2008)*.

*   **Discrepancies**: Experiments (documented in `docs/experimental/`) have shown systematic differences in fitted $U_{surf}$ magnitude compared to the paper (factor of ~1.5x - 5x depending on normalization).
*   **Normalization**: The exact normalization method used in the literature ("dividing reflected by incident") has ambiguities. The `global` mode preserves energy structure best but yields higher potentials. The `ratio` mode yields lower $\chi^2$ but physically inconsistent parameters in some test cases.
*   **Status**: The code implements the physics faithfully as described, but parameter tuning (e.g., SEY yields, exact normalization logic) may still differ from legacy analyses.
