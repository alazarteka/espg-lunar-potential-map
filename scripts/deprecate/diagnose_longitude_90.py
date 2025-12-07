"""
Visualize measurement coverage and potential values to diagnose ±90° anomalies.
"""


import matplotlib.pyplot as plt
import numpy as np

# Load one month of data
print("Loading temporal coefficients and reconstructing...")

with np.load('data/temporal_coeffs_lmax10.npz') as data:
    coeffs = data['coeffs']
    times = data['times']
    lmax = int(data['lmax'])

# Reconstruct one timestamp
from scipy.special import sph_harm


def reconstruct_map(coeffs_single, lmax):
    """Reconstruct potential map."""
    lat_grid = np.linspace(-90, 90, 181)
    lon_grid = np.linspace(-180, 180, 361)
    lons, lats = np.meshgrid(lon_grid, lat_grid)

    lat_rad = np.deg2rad(lats.ravel())
    lon_rad = np.deg2rad(lons.ravel())
    colat = np.pi/2 - lat_rad

    # Build design matrix
    n_points = lat_rad.size
    n_coeffs = (lmax + 1)**2
    design = np.empty((n_points, n_coeffs), dtype=np.complex128)

    idx = 0
    for l in range(lmax + 1):
        for m in range(-l, l + 1):
            design[:, idx] = sph_harm(m, l, lon_rad, colat)
            idx += 1

    potential = np.real(design @ coeffs_single).reshape(lats.shape)
    return lats, lons, potential

# Reconstruct middle timestamp
mid_idx = len(times) // 2
print(f"Reconstructing map for {np.datetime_as_string(times[mid_idx], unit='D')}")

lats, lons, potential = reconstruct_map(coeffs[mid_idx], lmax)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 10))

# 1. Full map
im1 = axes[0, 0].contourf(lons, lats, potential, levels=20, cmap='RdBu_r')
axes[0, 0].axvline(-90, color='red', linestyle='--', linewidth=2, label='±90° longitude')
axes[0, 0].axvline(90, color='red', linestyle='--', linewidth=2)
plt.colorbar(im1, ax=axes[0, 0], label='Potential (V)')
axes[0, 0].set_xlabel('Longitude (°)')
axes[0, 0].set_ylabel('Latitude (°)')
axes[0, 0].set_title('Full Reconstructed Map')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Longitude slices at different latitudes
for lat_slice in [0, 30, 60]:
    lat_idx = np.argmin(np.abs(lats[:, 0] - lat_slice))
    axes[0, 1].plot(lons[0, :], potential[lat_idx, :], label=f'Lat = {lat_slice}°', alpha=0.7)

axes[0, 1].axvline(-90, color='red', linestyle='--', alpha=0.5)
axes[0, 1].axvline(90, color='red', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Longitude (°)')
axes[0, 1].set_ylabel('Potential (V)')
axes[0, 1].set_title('Longitude Profiles at Different Latitudes')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 3. Zoom near -90°
lon_mask = (lons[0, :] > -120) & (lons[0, :] < -60)
im3 = axes[1, 0].contourf(
    lons[:, lon_mask], lats[:, lon_mask], potential[:, lon_mask],
    levels=20, cmap='RdBu_r'
)
axes[1, 0].axvline(-90, color='red', linestyle='--', linewidth=2)
plt.colorbar(im3, ax=axes[1, 0], label='Potential (V)')
axes[1, 0].set_xlabel('Longitude (°)')
axes[1, 0].set_ylabel('Latitude (°)')
axes[1, 0].set_title('Zoom: -120° to -60° Longitude')
axes[1, 0].grid(True, alpha=0.3)

# 4. Zoom near +90°
lon_mask = (lons[0, :] > 60) & (lons[0, :] < 120)
im4 = axes[1, 1].contourf(
    lons[:, lon_mask], lats[:, lon_mask], potential[:, lon_mask],
    levels=20, cmap='RdBu_r'
)
axes[1, 1].axvline(90, color='red', linestyle='--', linewidth=2)
plt.colorbar(im4, ax=axes[1, 1], label='Potential (V)')
axes[1, 1].set_xlabel('Longitude (°)')
axes[1, 1].set_ylabel('Latitude (°)')
axes[1, 1].set_title('Zoom: +60° to +120° Longitude')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('artifacts/plots/longitude_90_diagnostic.png', dpi=150, bbox_inches='tight')
print("Saved diagnostic plot to artifacts/plots/longitude_90_diagnostic.png")

# Print statistics near ±90°
print("\n" + "="*60)
print("Potential statistics near ±90° longitude")
print("="*60)

for lon_target in [-90, 90]:
    lon_idx = np.argmin(np.abs(lons[0, :] - lon_target))
    slice_pot = potential[:, lon_idx]

    print(f"\nAt {lon_target:+d}° longitude:")
    print(f"  Mean:   {slice_pot.mean():+8.1f} V")
    print(f"  Std:    {slice_pot.std():8.1f} V")
    print(f"  Range:  {slice_pot.min():+8.1f} to {slice_pot.max():+8.1f} V")

    # Check for discontinuities
    if lon_idx > 0 and lon_idx < len(lons[0, :]) - 1:
        left_slice = potential[:, lon_idx - 1]
        right_slice = potential[:, lon_idx + 1]
        jump = np.abs(slice_pot - (left_slice + right_slice) / 2).mean()
        print(f"  Discontinuity: {jump:.1f} V (mean deviation from neighbors)")

print("="*60)
