# ESPG Lunar Potential Map – Documentation

User-facing documentation for the command-line interfaces.

## CLI Reference

| Module | Command | Description |
|--------|---------|-------------|
| [Data Acquisition](cli/data_acquisition.md) | `uv run python -m src.data_acquisition` | Download SPICE kernels and ER flux data |
| [Potential Mapper](cli/potential_mapper.md) | `uv run python -m src.potential_mapper` | Interactive potential mapping |
| [Batch Processing](cli/potential_mapper_batch.md) | `uv run python -m src.potential_mapper.batch` | Parallel batch processing with NPZ output |
| [Temporal Harmonics](cli/temporal.md) | `uv run python -m src.temporal` | Spherical harmonic coefficient fitting |
| [Engineering Products](cli/engineering.md) | `uv run python -m src.engineering` | Engineering maps and site summaries |
| [Diagnostics Tools](cli/diagnostics.md) | — | Beam detection and loss cone analysis |
| [Visualization Style](visualization/visualization.md) | — | Colormap categories, fonts, and paper styling |
| [Development](dev/development.md) | — | Build, test, lint workflow reference |

## Quick Start

```bash
# 1. Download all required data (run once)
uv run python -m src.data_acquisition

# 2. Process a day of data interactively
uv run python -m src.potential_mapper --year 1998 --month 1 --day 15 -d

# 3. Batch process a full month
uv run python -m src.potential_mapper.batch --year 1998 --month 1
```

## Architecture & Physics

- [Pipeline Overview](architecture/pipeline_overview.md)
- [Coordinate Frames & SPICE](architecture/coordinate_frames.md)
- [GPU Acceleration](architecture/gpu_acceleration.md)
- [Spacecraft Potential](physics/spacecraft_potential.md)

## Archived Documentation

Previous documentation (analysis notes, technical deep-dives) is preserved in
[`archive/`](archive/).
