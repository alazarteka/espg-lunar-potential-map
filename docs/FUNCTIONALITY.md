# Project Functionality & Architecture

The **ESPG Lunar Potential Map** project is a Python-based scientific analysis pipeline designed to map the electrostatic potential of the lunar surface using historical data from the Lunar Prospector mission.

## 1. System Overview

The system operates in a pipeline fashion:
1.  **Data Acquisition**: Download raw data (PDS) and geometry kernels (SPICE).
2.  **Data Loading & Preprocessing**: Read Electron Reflectometer (ER) data, clean anomalies, and compute physical coordinates.
3.  **Physics Modeling**:
    *   Fit plasma distribution functions (Kappa).
    *   Estimate spacecraft potential ($U_{sc}$).
    *   Fit surface potential ($U_{surf}$) using loss cone analysis.
4.  **Mapping**: Project results onto the lunar surface (Lat/Lon) and visualize.

## 2. Core Modules

### `src.data_acquisition`
*   **Purpose**: Manages the local data cache.
*   **Functionality**:
    *   Downloads SPICE kernels (generic and mission-specific).
    *   Downloads ER flux data (PDS archives).
    *   Verifies file integrity (SHA-1).

### `src.flux` & `src.model`
*   **Purpose**: Implements the core physics of electron transport and surface interaction.
*   **Key Classes**:
    *   `ERData`: Container for raw electron spectrometer data.
    *   `PitchAngle`: Computes magnetic pitch angles using SPICE geometry.
    *   `LossConeFitter`: Performs the non-linear least squares fit to determine surface potential.
    *   `src.model.synth_losscone`: The forward model generating synthetic flux maps.

### `src.potential_mapper`
*   **Purpose**: The main orchestration engine.
*   **Key Components**:
    *   `pipeline.py`: Orchestrates the processing flow. Handles file discovery, parallel processing (multiprocessing), and result aggregation.
    *   `coordinates.py`: Handles complex coordinate transformations (Instrument -> Spacecraft -> ME -> Moon Fixed) and magnetic field line tracing.
    *   `spice`: SPICE kernel management wrapper.

### `src.kappa` & `src.spacecraft_potential`
*   **Purpose**: Plasma characterization and spacecraft charging.
*   **Functionality**:
    *   Fits Kappa distributions to measured spectra.
    *   Solves the current balance equation (Photoemission + Plasma Current + Secondary Emission = 0) to find the floating potential of the spacecraft.

### `src.temporal`
*   **Purpose**: Time-dependent global mapping.
*   **Functionality**:
    *   Fits Spherical Harmonics ($Y_{lm}$) to the potential data.
    *   Models time-variation using temporal basis functions (Fourier series).
    *   Allows reconstruction of global potential maps from sparse orbital tracks.

## 3. Data Flow

1.  **Input**:
    *   `*.TAB` files: Electron counts/flux (PDS).
    *   `*.bc`, `*.bsp`, `*.tls`: SPICE kernels (Trajectory, Orientation, Time).
2.  **Processing (per spectrum)**:
    *   **ERData** loads the counts.
    *   **PitchAngle** computes look directions relative to $\mathbf{B}$.
    *   **SpacecraftPotential** estimates $U_{sc}$.
    *   **LossConeFitter** normalizes flux and fits $U_{surf}$ and $B_s/B_m$.
3.  **Output**:
    *   `PotentialResults`: Structured object containing $U_{sc}$, $U_{surf}$, Lat/Lon coordinates, and illumination flags.
    *   **NPZ Caches**: Intermediate batch results stored for analysis.
    *   **Plots**: 2D maps, Time-series, and Loss Cone visualisations.

## 4. Parallel Processing

To handle the large volume of data (high-resolution spectra over months/years), the pipeline uses `multiprocessing`.

*   **Strategy**: Data is chunked by file or by groups of spectra.
*   **Context**: Uses `spawn` context to ensure SPICE kernel state is thread-safe and isolated per worker.
*   **Optimization**: Batch processing sets BLAS/OMP thread counts to 1 to avoid oversubscription when running many parallel processes.

## 5. Temporal Reconstruction

Beyond instantaneous mapping, the project includes a `src.temporal` module to create continuous global maps.

*   **Method**: Least-squares fitting of Spherical Harmonics coefficients $C_{lm}(t)$ to the aggregate dataset.
*   **Basis**: $V(\theta, \phi, t) = \sum_{l,m} Y_{lm}(\theta, \phi) \cdot \sum_k (A_k \cos(\omega_k t) + B_k \sin(\omega_k t))$.
*   **Usage**: Fills gaps between orbital tracks and smooths noise.
