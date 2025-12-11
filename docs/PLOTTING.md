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
*   **Purpose**: Plots the time evolution of Spacecraft and Surface potentials.
*   **Output**: Stacked time-series plots.
*   **Key Features**:
    *   Visualizes correlation between potentials and other parameters.
    *   Useful for identifying charging events (e.g., crossing the terminator, entering the wake).

### 3. `plot_harmonic_reconstruction_paper.py` & `_multiday.py`
*   **Purpose**: Visualizes the results of the Spherical Harmonic temporal reconstruction.
*   **Output**: Global maps (Mollweide or Orthographic projection) of the lunar potential.
*   **Key Features**:
    *   Compares reconstructed maps with raw data tracks.
    *   Visualizes the global structure of the electric field.

### 4. `plot_measurements_paper.py`
*   **Purpose**: General plotting of raw measurement distributions.
*   **Output**: Histograms or scatter plots of key quantities (Flux, Energy).

## Common Arguments

Most plotting scripts support:
*   `--input`: Path to data file.
*   `--output`: Path to save the figure.
*   `--title`: Override the default figure title.
*   `--dpi`: Resolution (default 150 or 300).
*   `--style`: (Implicitly uses `src.visualization.style` for consistent aesthetics).

## Visualization Style

The project uses a centralized style definition in `src.visualization`.
*   **Colormap**: `viridis` is the default for scalar fields (flux, potential).
*   **Theme**: Light background ("Paper" style) is preferred for publication.
*   **Fonts**: Sans-serif (Arial/Helvetica) for readability.
