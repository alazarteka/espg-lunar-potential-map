# April 1998 Harmonics Animation

Animated spherical harmonic reconstruction of lunar surface potential for April 1998, with terminator overlay showing day/night boundary.

## Files

- `hemisphere_potential.gif` — North/South polar view animation with terminator lines
- `global_potential.gif` — Equirectangular global view animation with terminator lines and subsolar point

## Generation

```bash
uv run python scripts/dev/temporal_harmonics_animate_terminator.py \
  --input artifacts/paper/harmonics/april_1998_lmax10.npz \
  --output-dir artifacts/paper/plots/harmonics/apr_animation \
  --dpi 200
```

### Key options

| Option | Description |
|--------|-------------|
| `--input` | NPZ file with temporal harmonic coefficients |
| `--lat-steps` | Latitude grid resolution (default: 181) |
| `--lon-steps` | Longitude grid resolution (default: 361) |
| `--fps` | Frames per second (default: 10) |
| `--limit-frames` | Cap number of frames for quick preview |

