# Spacecraft Potential (U_sc)

Estimate Lunar Prospector floating potential per spectrum by branching on
illumination.

## Overview

**Daylight branch**
1. Fit kappa distribution to the observed spectrum
2. Invert the JU photoemission curve to estimate U_sc
3. Shift energies by -U_sc and refit kappa for corrected ambient parameters

**Nightside branch**
Solve the current balance equation:

```
F(U) = Ji(U) + Jsee(U) - Je(U) = 0
```

using a Brent root finder. The currents are derived from:
- **Je**: kappa-distribution electron collection
- **Ji**: OML-like ion collection
- **Jsee**: Sternglass secondary electron yield

## API

Primary entry point:

```python
from src.spacecraft_potential import calculate_potential

result = calculate_potential(er_data, spec_no)
# -> (kappa_params, u_sc) or None
```

## Key Parameters

- `spacecraft_potential_low/high`: initial bracket for nightside root finding
- `sey_E_m`, `sey_delta_m`: Sternglass secondary yield parameters
- `E_min_ev`, `E_max_ev`, `n_steps`: integration grid for current balance

## Related Modules

- `src/physics/charging.py` (current balance)
- `src/physics/jucurve.py` (JU inversion)
- `src/physics/kappa.py` (kappa distribution utilities)
