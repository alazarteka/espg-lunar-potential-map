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
| `--max-workers N` | Parallel download threads (default `20`). Lower it if the server throttles. |

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

- Downloads are parallelized (up to 20 threads by default; tune with `--max-workers`).
- Fully downloaded files are skipped automatically.
- **Partial downloads are truly resumed.** In-progress files are written to a
  `<name>.part` sidecar and only atomically renamed to the final name once the
  full download completes and its size matches the server's advertised length.
  Interrupted files keep their `.part` and continue via an HTTP `Range` request
  on the next run — killing and restarting the job is safe and loses no progress.
- The completed size is verified before the rename, so a truncated or corrupt
  download is never promoted to a final filename.
- Transient failures and throttling (HTTP 429/500/502/503/504) are retried with
  exponential backoff that honours the server's `Retry-After` header, so the
  downloader backs off politely instead of hammering NASA/PDS.
- Connection pooling is used for efficiency.

## Reliability & Politeness

For a large (~30 GB) redownload:

- Run inside `tmux` or `screen` so a dropped SSH session doesn't kill the job.
- The job is idempotent and safe to kill/restart at any time — it resumes
  partial files and skips completed ones.
- If the server starts throttling, re-run with a lower `--max-workers` (e.g.
  `--max-workers 8`); the automatic backoff already handles transient 429/503s.

## Example

```bash
# First-time setup: download all required data
uv run python -m src.data_acquisition

# Verbose mode for debugging download issues
uv run python -m src.data_acquisition -v

# Gentler on the server (fewer parallel connections)
uv run python -m src.data_acquisition --max-workers 8
```
