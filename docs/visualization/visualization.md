# Visualization Style Guide

This document describes the shared styling conventions for all plots in the ESPG project.

## Colormap Categories

The project uses a semantic colormap scheme based on data type:

| Category | Constant | Colormap | Use Case |
|----------|----------|----------|----------|
| **Measurement** | `CMAP_MEASUREMENT` | `viridis` | Raw data: fluxes, voltages, harmonic reconstructions |
| **Magnitude** | `CMAP_MAGNITUDE` | `plasma` | Derived magnitudes (engineering default) |
| **Residual** | `CMAP_RESIDUAL` | `RdBu_r` | Errors: model residuals (diverging, centered at 0) |

### Engineering Module Colormaps

The engineering module uses distinct colormaps per map type for visual differentiation:

| Map | Colormap | Rationale |
|-----|----------|-----------|
| Power Density | `plasma` | Warm tones for energy/power |
| P95 Potential | `inferno` | High-intensity scale for peak values |
| Risk (>1kV fraction) | `Reds` | Intuitive red = danger/risk |

### Rationale

- **viridis**: Perceptually uniform sequential colormap, ideal for measured values
- **plasma**: Warm-toned sequential colormap, visually distinct for derived quantities
- **RdBu_r**: Diverging colormap with intuitive red/blue semantics for positive/negative errors

## Font Sizes

| Constant | Value | Usage |
|----------|-------|-------|
| `FONT_SIZE_TITLE` | 13 | Figure and subplot titles |
| `FONT_SIZE_LABEL` | 12 | Axis labels |
| `FONT_SIZE_TEXT` | 9 | Annotations, colorbar ticks, stat boxes |

## Grid Style

All plots use a consistent grid via `apply_paper_style(ax)`:
- Alpha: 0.3
- Linestyle: dashed (`--`)
- Linewidth: 0.4

## Usage

```python
from src.visualization import style

# Colormaps
ax.pcolormesh(x, y, flux_data, cmap=style.CMAP_MEASUREMENT)
ax.pcolormesh(x, y, power_data, cmap=style.CMAP_MAGNITUDE)
ax.pcolormesh(x, y, residuals, cmap=style.CMAP_RESIDUAL)

# Apply standard paper styling
style.apply_paper_style(ax)

# Font sizes
ax.set_xlabel("X", fontsize=style.FONT_SIZE_LABEL)
ax.set_title("Title", fontsize=style.FONT_SIZE_TITLE)
```

## See Also

- [`src/visualization/style.py`](../../src/visualization/style.py) — implementation
- [`src/visualization/utils.py`](../../src/visualization/utils.py) — helper functions (e.g., `add_stats_box`)
