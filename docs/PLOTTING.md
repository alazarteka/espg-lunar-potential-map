# Plotting & Analysis Scripts

The `scripts/plots/` directory contains tools for visualizing data, model fits, and global maps. These scripts are designed to produce publication-quality figures (`_paper` suffix) or interactive analyses.

## Key Scripts

### 1. `plot_losscone_fit_paper.py`
*   **Purpose**: Visualizes the loss cone fit for a specific spectrum.
*   **Output**: A side-by-side comparison of Measured Flux vs. Model Flux (Energy vs. Pitch Angle).
*   **Key Features**:
    *   Shows the fitted loss cone curve (white line).
    *   Displays fitted parameters ($U_{surf}$, $B_s/B_m$, $\chi^2$).
    *   Supports "Paper Mode" to emulate Halekas et al. (2008) settings.
*   **Usage**:
    ```bash
    uv run python scripts/plots/plot_losscone_fit_paper.py --input data/.../FILE.TAB --spec-no 653 --output plot.png
    ```

### 2. `plot_timeseries_potential.py`
*   **Purpose**: Reconstructs surface potential time-series at specific lunar locations.
*   **Output**: Time-series plots of surface potential ($V$) vs Time.
*   **Key Features**:
    *   Uses spherical harmonic coefficients ($C_{lm}(t)$) to evaluate potential at any lat/lon.
    *   Supports plotting multiple locations simultaneously.
    *   Useful for analyzing temporal variability at fixed surface points.
*   **Usage**:
    ```bash
    uv run python scripts/plots/plot_timeseries_potential.py --input artifacts/harmonics.npz --lat 0 45 --lon 0 90 --output timeseries.png
    ```

### 3. `plot_harmonic_reconstruction_paper.py`
*   **Purpose**: Visualizes a single global map reconstruction from spherical harmonics.
*   **Output**: Global map (Rectangular or Mollweide) of surface potential at a specific time snapshot.
*   **Key Features**:
    *   Reconstructs the global field from coefficients for a single time index.
    *   Shows global structure of the electric field.
*   **Usage**:
    ```bash
    uv run python scripts/plots/plot_harmonic_reconstruction_paper.py --input artifacts/harmonics.npz --time-index 10 --output map.png
    ```

### 4. `plot_harmonic_reconstruction_multiday.py`
*   **Purpose**: Visualizes the evolution of the global potential map.
*   **Output**: A 2x3 grid of global maps corresponding to different times.
*   **Key Features**:
    *   Auto-selects or accepts manual time indices.
    *   Uses a shared colorbar for direct comparison across time.
*   **Usage**:
    ```bash
    uv run python scripts/plots/plot_harmonic_reconstruction_multiday.py --input artifacts/harmonics.npz --auto-select --output grid.png
    ```

### 5. `plot_terminator_profile_paper.py`
*   **Purpose**: Analyzes the transition of surface potential across the terminator.
*   **Output**: Plot of Surface Potential vs Solar Zenith Angle (SZA).
*   **Key Features**:
    *   Bins data by SZA to show trends (Dayside ~positive, Nightside ~negative).
    *   Highlights the terminator region ($88^\circ - 92^\circ$).
    *   Displays Sunlit vs Shadowed statistics.

### 6. `plot_measurements_paper.py`
*   **Purpose**: Visualizes the spatial distribution of raw measurements.
*   **Output**: Global map scatter plot of measurement points.
*   **Key Features**:
    *   Filters by date range.
    *   Useful for showing orbital coverage.

## Common Arguments

Most plotting scripts support:
*   `--input`: Path to data file (TAB, NPZ, or Cache Directory).
*   `--output`: Path to save the figure.
*   `--title`: Override the default figure title.
*   `--dpi`: Resolution (default 150 or 300).
*   `--style`: (Implicitly uses `src.visualization.style` for consistent aesthetics).

## Visualization Style

The project uses a centralized style definition in `src.visualization`.
*   **Colormap**: `viridis` is the default for scalar fields (flux, potential).
*   **Theme**: Light background ("Paper" style) is preferred for publication.
*   **Fonts**: Sans-serif (Arial/Helvetica) for readability.
