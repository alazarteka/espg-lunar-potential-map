# Potential Mapper

Maps the lunar surface electrostatic potential using Lunar Prospector ER data.

## Usage

```bash
uv run python -m src.potential_mapper [OPTIONS]
```

## Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--year` | int | None | Filter to specific year (1998 or 1999) |
| `--month` | int | None | Filter to specific month (1-12) |
| `--day` | int | None | Filter to specific day (1-31) |
| `--output` | str | None | Path to output file |
| `-v`, `--verbose` | flag | False | Enable DEBUG-level logging |
| `-d`, `--display` | flag | False | Show the plot after processing |
| `--illumination` | str | None | Filter by `day` or `night` for display |

## What It Does

1. **Loads SPICE kernels** for spacecraft ephemeris
2. **Discovers ER flux files** matching date filters
3. **For each file**:
   - Loads attitude data and builds coordinate transforms
   - Projects magnetic field lines to lunar surface
   - Determines illumination (day/night) for each measurement
   - Fits surface potential using loss-cone model (χ² minimization)
4. **Aggregates results** with lat/lon coordinates
5. **Optionally plots** the potential map

## Output

Results include per-row arrays:
- `spacecraft_latitude`, `spacecraft_longitude` – LP position
- `projection_latitude`, `projection_longitude` – surface footpoint
- `spacecraft_potential` – fitted potential at spacecraft
- `projected_potential` – surface potential (primary output)
- `spacecraft_in_sun`, `projection_in_sun` – illumination flags

## Examples

```bash
# Process a single day
uv run python -m src.potential_mapper --year 1998 --month 3 --day 15

# Process and display an entire month
uv run python -m src.potential_mapper --year 1998 --month 6 -d

# Show only dayside measurements
uv run python -m src.potential_mapper --year 1998 --month 6 -d --illumination day

# Verbose mode
uv run python -m src.potential_mapper --year 1998 --month 1 --day 1 -v
```

## Performance Note

Processing a full month takes several hours due to the per-spectrum fitting. For large-scale processing, use the batch interface instead:

```bash
uv run python -m src.potential_mapper.batch --year 1998 --month 6
```

## Prerequisites

Run `src.data_acquisition` first to download required data files.
