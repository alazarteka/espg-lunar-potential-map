# Data Acquisition

Downloads SPICE kernels and Lunar Prospector Electron Reflectometer (ER) flux data from NASA PDS archives.

## Usage

```bash
uv run python -m src.data_acquisition [-v]
```

## Options

| Flag | Description |
|------|-------------|
| `-v`, `--verbose` | Enable DEBUG-level logging |

## What It Downloads

1. **SPICE Kernels** (to `data/spice_kernels/`):
   - LP ephemeris kernels (`lp_ask_*.bsp`)
   - Generic leap seconds (`latest_leapseconds.tls`)
   - Planetary constants (`pck00011.tpc`)
   - LP ephemeris emulator (`lpephemu.bsp`)

2. **Attitude Data** (to `data/`):
   - `attitude.tab` – spacecraft pointing table

3. **ER Flux Data** (to `data/{year}/{julian_day}/`):
   - `.TAB` files containing 3D electron flux measurements
   - `theta.tab` – detector look direction angles
   - `solid_angles.tab` – generated from theta values

## Directory Structure After Run

```
data/
├── spice_kernels/
│   ├── lp_ask_980111-980531.bsp
│   ├── lp_ask_980601-981031.bsp
│   ├── lp_ask_981101-990331.bsp
│   ├── lp_ask_990401-990730.bsp
│   ├── lpephemu.bsp
│   ├── latest_leapseconds.tls
│   └── pck00011.tpc
├── attitude.tab
├── theta.tab
├── solid_angles.tab
├── 1998/
│   ├── 001/
│   │   ├── LP_ER_3D19980101_V01.TAB
│   │   └── ...
│   └── ...
└── 1999/
    └── ...
```

## Notes

- Downloads are parallelized (up to 20 threads by default)
- Existing files are skipped automatically
- Partial downloads are resumed
- Connection pooling is used for efficiency

## Example

```bash
# First-time setup: download all required data
uv run python -m src.data_acquisition

# Verbose mode for debugging download issues
uv run python -m src.data_acquisition -v
```
