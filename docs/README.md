# ESPG Lunar Potential Map – Documentation

User-facing documentation for the command-line interfaces and technical details.

## Technical Documentation

*   **[Functionality & Architecture](FUNCTIONALITY.md)**: High-level overview of the system, modules, and data flow.
*   **[Physics & Algorithms](PHYSICS.md)**: Detailed explanation of the physical models (Loss Cone, Kappa, Spacecraft Charging) and assumptions.
*   **[Plotting & Scripts](PLOTTING.md)**: Guide to the visualization scripts in `scripts/plots/`.

## CLI Reference

| Module | Command | Description |
|--------|---------|-------------|
| [Data Acquisition](data_acquisition.md) | `uv run python -m src.data_acquisition` | Download SPICE kernels and ER flux data |
| [Potential Mapper](potential_mapper.md) | `uv run python -m src.potential_mapper` | Interactive potential mapping |
| [Batch Processing](potential_mapper_batch.md) | `uv run python -m src.potential_mapper.batch` | Parallel batch processing with NPZ output |
| [Temporal Harmonics](temporal.md) | `uv run python -m src.temporal` | Spherical harmonic coefficient fitting |
| [Visualization Style](visualization.md) | — | Colormap categories, fonts, and paper styling |

## Quick Start

```bash
# 1. Download all required data (run once)
uv run python -m src.data_acquisition

# 2. Process a day of data interactively
uv run python -m src.potential_mapper --year 1998 --month 1 --day 15 -d

# 3. Batch process a full month
uv run python -m src.potential_mapper.batch --year 1998 --month 1
```

## Experimental & Archive

*   **[Experimental Notes](experimental/)**: Documentation of experiments (e.g., normalization studies) and decisions made during development.
*   **[Old Docs](old/)**: Archived analysis notes and technical deep-dives.
