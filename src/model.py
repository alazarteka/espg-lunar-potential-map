
import numpy as np

###########
# Constants
###########

EPS = 1e-12  # Small epsilon to avoid division by zero
DEFAULT_BACKGROUND = 0.05  # Small non-zero value outside loss cone (match config)
e_0 = 1.602e-19  # Elementary charge in Coulombs


def synth_losscone(
    energy_grid: np.ndarray,
    pitch_grid: np.ndarray,
    U_surface: float | np.ndarray,
    U_spacecraft: float | np.ndarray = 0.0,
    bs_over_bm: float | np.ndarray = 1.0,
    beam_width_eV: float | np.ndarray = 0.0,
    beam_amp: float | np.ndarray = 0.0,
    beam_pitch_sigma_deg: float | np.ndarray = 0.0,
    background: float | np.ndarray = DEFAULT_BACKGROUND,
) -> np.ndarray:
    """
    Build a loss-cone model that never returns NaN/Inf.

    Supports broadcasting for vectorized LHS:
    - energy_grid: (nE,)
    - pitch_grid: (nE, nPitch)
    - U_surface: scalar or (nParams,) - lunar surface potential in volts
    - bs_over_bm: scalar or (nParams,)

    Returns:
    - If params are scalar: (nE, nPitch)
    - If params are arrays: (nParams, nE, nPitch)
    """
    # Ensure inputs are arrays
    energy_grid = np.asarray(energy_grid)
    pitch_grid = np.asarray(pitch_grid)

    # Convert parameters to arrays and detect if inputs were scalar
    U_surface = np.asarray(U_surface)
    U_spacecraft = np.asarray(U_spacecraft)
    bs_over_bm = np.asarray(bs_over_bm)
    beam_amp = np.asarray(beam_amp)
    beam_width_eV = np.asarray(beam_width_eV)
    background = np.asarray(background)

    # Remember if we need to squeeze output (all params were scalar)
    squeeze_output = (
        U_surface.ndim == 0
        and bs_over_bm.ndim == 0
        and beam_amp.ndim == 0
        and beam_width_eV.ndim == 0
        and background.ndim == 0
    )

    # Always promote to 1D for unified batch processing
    if U_surface.ndim == 0:
        U_surface = U_surface[None]
    if bs_over_bm.ndim == 0:
        bs_over_bm = bs_over_bm[None]
    if beam_amp.ndim == 0:
        beam_amp = beam_amp[None]
    if beam_width_eV.ndim == 0:
        beam_width_eV = beam_width_eV[None]
    if background.ndim == 0:
        background = background[None]

    # Determine batch size
    n_params = max(
        U_surface.size,
        bs_over_bm.size,
        beam_amp.size,
        beam_width_eV.size,
        background.size,
    )

    # Reshape params to (nParams, 1, 1) for broadcasting
    U_surface = U_surface.reshape(-1, 1, 1)
    bs_over_bm = bs_over_bm.reshape(-1, 1, 1)
    beam_amp = beam_amp.reshape(-1, 1, 1)
    beam_width_eV = beam_width_eV.reshape(-1, 1, 1)
    background = background.reshape(-1, 1, 1)

    # Guard against E <= 0 (mask invalid energies)
    valid_E = energy_grid > 0
    E_safe = np.where(valid_E, energy_grid, 1.0)  # Avoid div by zero

    # Reshape grids to (1, nE, nPitch)
    pitch_grid_exp = pitch_grid[None, :, :]
    # E_safe is (nE,) -> (1, nE, 1)
    E_safe_exp = E_safe[None, :, None]
    valid_E_exp = valid_E[None, :, None]

    # Handle U_spacecraft: can be scalar or (nE,)
    if U_spacecraft.ndim == 0:
        # Scalar: broadcast to all energies -> (1, 1, 1)
        U_spacecraft = U_spacecraft.reshape(1, 1, 1)
    else:
        # Array (nE,): reshape to (1, nE, 1) for broadcasting
        U_spacecraft = U_spacecraft[None, :, None]

    # Calculate x = B_s/B_m * (1 + U_surface / E - U_spacecraft)
    # (nParams, 1, 1) * (1 + (nParams, 1, 1) / (1, nE, 1)) -> (nParams, nE, 1)
    x = bs_over_bm * (
        1.0 + U_surface / np.maximum(E_safe_exp - U_spacecraft, EPS)
    )

    model = np.broadcast_to(
        background, (n_params, pitch_grid.shape[0], pitch_grid.shape[1])
    ).copy()

    x_clipped = np.clip(x, 0.0, 1.0)
    ac_rad = np.arcsin(np.sqrt(x_clipped))
    ac_deg = np.degrees(ac_rad)  # (nParams, nE, 1)

    # Mask: pitch <= 180 - ac
    # (1, nE, nPitch) <= (nParams, nE, 1) -> (nParams, nE, nPitch)
    mask = pitch_grid_exp <= (180.0 - ac_deg)

    # Apply mask only where energies are valid
    final_mask = mask & valid_E_exp
    model[final_mask] = 1.0

    # Add beam component
    if np.any(beam_width_eV > 0):
        beam_center = np.maximum(np.abs(U_surface - U_spacecraft), beam_width_eV)
        beam = beam_amp * np.exp(
            -0.5 * ((E_safe_exp - beam_center) / beam_width_eV) ** 2
        )

        if beam_pitch_sigma_deg > 0:
            pitch_weight = np.exp(
                -0.5 * ((pitch_grid_exp - 180.0) / beam_pitch_sigma_deg) ** 2
            )
        else:
            pitch_weight = np.ones_like(pitch_grid_exp)

        model += beam * pitch_weight

    # Return squeezed output for scalar inputs, batch output otherwise
    if squeeze_output:
        return model[0]  # (nE, nPitch)
    else:
        return model  # (nParams, nE, nPitch)


def _chi2(params, energies, pitches, data, eps):
    U_surface, bs_over_bm = params
    model = synth_losscone(energies, pitches, U_surface, bs_over_bm)

    # Bail-out if the model went pathological
    if not np.all(np.isfinite(model)) or (model <= 0).all():
        return 1e30  # huge penalty

    diff = np.log(data + eps) - np.log(model + eps)
    return np.sum(diff * diff)
