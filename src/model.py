"""
Loss-cone model for lunar surface potential fitting.

Physics basis from Halekas 2008 (doi:10.1029/2008JA013194), paragraph 37.
"""

from dataclasses import dataclass

import numpy as np

# Small epsilon to avoid division by zero
EPS = 1e-12

# Default flux value outside loss cone
DEFAULT_BACKGROUND = 0.05


@dataclass
class LossConeParams:
    """
    Parameters for the loss-cone model.

    Attributes:
        U_surface: Lunar surface potential in volts.
        bs_over_bm: B_spacecraft / B_Moon_surface ratio.
            Uses 's' for spacecraft and 'm' for Moon, matching Halekas 2008.
            This is NOT B_surface/B_measurement.
        beam_amp: Secondary electron beam amplitude (dimensionless).
        beam_width_eV: Secondary electron beam width in eV.
        beam_pitch_sigma_deg: Beam angular spread in degrees.
        background: Flux value outside loss cone (default 0.05).
    """

    U_surface: float
    bs_over_bm: float = 1.0
    beam_amp: float = 0.0
    beam_width_eV: float = 0.0
    beam_pitch_sigma_deg: float = 0.0
    background: float = DEFAULT_BACKGROUND


def _compute_loss_cone_angle(
    energy: np.ndarray,
    U_surface: np.ndarray,
    U_spacecraft: np.ndarray,
    bs_over_bm: np.ndarray,
) -> np.ndarray:
    """
    Compute loss cone angle from Halekas 2008 formula.

    Formula: sin²(αc) = (BS/BM) × (1 + UM / (E - U_spacecraft))

    Args:
        energy: Electron energies in eV, shape (1, nE, 1)
        U_surface: Surface potential in V, shape (nParams, 1, 1)
        U_spacecraft: Spacecraft potential in V, shape (1, 1, 1) or (1, nE, 1)
        bs_over_bm: Field ratio, shape (nParams, 1, 1)

    Returns:
        Loss cone angle in degrees, shape (nParams, nE, 1)
    """
    # TODO: Review behavior when E < U_spacecraft (physically impossible case).
    # Current: clamp to EPS → x large negative → clips to 0 → full loss cone.
    # Alternative: let E_corrected go negative → x > 1 → clips to 1 → closed cone.
    # The alternative may be more physically correct (no electrons at impossible energies).
    E_corrected = np.maximum(energy - U_spacecraft, EPS)
    x = bs_over_bm * (1.0 + U_surface / E_corrected)
    x_clipped = np.clip(x, 0.0, 1.0)
    return np.degrees(np.arcsin(np.sqrt(x_clipped)))


def _compute_beam(
    energy: np.ndarray,
    pitch: np.ndarray,
    U_surface: np.ndarray,
    U_spacecraft: np.ndarray,
    beam_amp: np.ndarray,
    beam_width_eV: np.ndarray,
    beam_pitch_sigma_deg: float,
) -> np.ndarray:
    """
    Compute secondary electron beam component.

    The beam is a Gaussian in energy centered at |U_surface - U_spacecraft|,
    with angular concentration near pitch = 180°.

    Args:
        energy: Electron energies, shape (1, nE, 1)
        pitch: Pitch angles, shape (1, nE, nPitch)
        U_surface: Surface potential, shape (nParams, 1, 1)
        U_spacecraft: Spacecraft potential, shape (1, 1, 1) or (1, nE, 1)
        beam_amp: Beam amplitude, shape (nParams, 1, 1)
        beam_width_eV: Beam width, shape (nParams, 1, 1)
        beam_pitch_sigma_deg: Angular spread in degrees (scalar)

    Returns:
        Beam contribution, shape (nParams, nE, nPitch)
    """
    beam_center = np.maximum(np.abs(U_surface - U_spacecraft), beam_width_eV)
    energy_profile = beam_amp * np.exp(
        -0.5 * ((energy - beam_center) / np.maximum(beam_width_eV, EPS)) ** 2
    )

    if beam_pitch_sigma_deg > 0:
        pitch_profile = np.exp(-0.5 * ((pitch - 180.0) / beam_pitch_sigma_deg) ** 2)
    else:
        pitch_profile = 1.0

    return energy_profile * pitch_profile


def synth_losscone_batch(
    energy_grid: np.ndarray,
    pitch_grid: np.ndarray,
    U_surface: np.ndarray,
    U_spacecraft: float | np.ndarray = 0.0,
    bs_over_bm: np.ndarray | None = None,
    beam_width_eV: np.ndarray | None = None,
    beam_amp: np.ndarray | None = None,
    beam_pitch_sigma_deg: float = 0.0,
    background: np.ndarray | None = None,
) -> np.ndarray:
    """
    Vectorized batch loss-cone model for multiple parameter sets.

    Computes loss-cone models for multiple parameter combinations simultaneously.
    All array parameters must have the same length (n_params).

    Args:
        energy_grid: (nE,) electron energies in eV
        pitch_grid: (nE, nPitch) pitch angles in degrees
        U_surface: (n_params,) lunar surface potentials in volts
        U_spacecraft: spacecraft potential in volts (scalar or (nE,))
        bs_over_bm: (n_params,) B_spacecraft/B_surface ratios (default: all 1.0)
        beam_width_eV: (n_params,) beam widths in eV (default: all 0.0)
        beam_amp: (n_params,) beam amplitudes (default: all 0.0)
        beam_pitch_sigma_deg: beam angular spread in degrees (scalar)
        background: (n_params,) flux outside loss cone (default: all 0.05)

    Returns:
        Model flux array of shape (n_params, nE, nPitch)
    """
    U_surface = np.atleast_1d(np.asarray(U_surface))
    n_params = U_surface.size

    # Set defaults for optional arrays
    if bs_over_bm is None:
        bs_over_bm = np.ones(n_params)
    else:
        bs_over_bm = np.atleast_1d(np.asarray(bs_over_bm))

    if beam_width_eV is None:
        beam_width_eV = np.zeros(n_params)
    else:
        beam_width_eV = np.atleast_1d(np.asarray(beam_width_eV))

    if beam_amp is None:
        beam_amp = np.zeros(n_params)
    else:
        beam_amp = np.atleast_1d(np.asarray(beam_amp))

    if background is None:
        background = np.full(n_params, DEFAULT_BACKGROUND)
    else:
        background = np.atleast_1d(np.asarray(background))

    # Reshape for broadcasting: params -> (nParams, 1, 1)
    U_surface = U_surface.reshape(-1, 1, 1)
    bs_over_bm = bs_over_bm.reshape(-1, 1, 1)
    beam_amp = beam_amp.reshape(-1, 1, 1)
    beam_width_eV = beam_width_eV.reshape(-1, 1, 1)
    background = background.reshape(-1, 1, 1)

    # Handle energy grid: guard against E <= 0
    energy_grid = np.asarray(energy_grid)
    valid_E = energy_grid > 0
    E_safe = np.where(valid_E, energy_grid, 1.0)

    # Reshape grids for broadcasting
    pitch_exp = pitch_grid[None, :, :]  # (1, nE, nPitch)
    E_exp = E_safe[None, :, None]  # (1, nE, 1)
    valid_E_exp = valid_E[None, :, None]  # (1, nE, 1)

    # Handle U_spacecraft: scalar -> (1,1,1), array(nE,) -> (1,nE,1)
    U_spacecraft = np.asarray(U_spacecraft)
    if U_spacecraft.ndim == 0:
        U_spacecraft = U_spacecraft.reshape(1, 1, 1)
    else:
        U_spacecraft = U_spacecraft[None, :, None]

    # Compute loss cone angle
    ac_deg = _compute_loss_cone_angle(E_exp, U_surface, U_spacecraft, bs_over_bm)

    # Build model: background everywhere, 1.0 inside loss cone
    nE, nPitch = pitch_grid.shape
    model = np.broadcast_to(background, (n_params, nE, nPitch)).copy()

    # Inside loss cone: pitch <= 180 - αc
    inside_cone = (pitch_exp <= (180.0 - ac_deg)) & valid_E_exp
    model[inside_cone] = 1.0

    # Add secondary electron beam if enabled
    if np.any(beam_width_eV > 0) and np.any(beam_amp > 0):
        beam = _compute_beam(
            E_exp,
            pitch_exp,
            U_surface,
            U_spacecraft,
            beam_amp,
            beam_width_eV,
            beam_pitch_sigma_deg,
        )
        model += beam

    return model


def synth_losscone(
    energy_grid: np.ndarray,
    pitch_grid: np.ndarray,
    U_surface: float,
    U_spacecraft: float | np.ndarray = 0.0,
    bs_over_bm: float = 1.0,
    beam_width_eV: float = 0.0,
    beam_amp: float = 0.0,
    beam_pitch_sigma_deg: float = 0.0,
    background: float = DEFAULT_BACKGROUND,
) -> np.ndarray:
    """
    Build a loss-cone model for a single parameter set.

    Physics basis from Halekas 2008 (doi:10.1029/2008JA013194), paragraph 37:

        sin²(αc) = (BS/BM) × (1 + e·UM/E)

    where BS is magnetic field at spacecraft, BM is field at lunar surface,
    UM is lunar surface potential, and E is electron energy. This describes
    the boundary between electrons adiabatically reflected by combined
    magnetic and electrostatic forces, and those lost by impact with surface.

    The parameter `bs_over_bm` follows the paper's convention:
        bs_over_bm = BS/BM = B_spacecraft / B_Moon_surface

    Note: The naming uses 's' for spacecraft and 'm' for Moon (surface),
    matching the paper's notation. This is NOT B_surface/B_measurement.

    The code extends the formula to account for spacecraft potential:
        x = bs_over_bm × (1 + U_surface / (E - U_spacecraft))

    where (E - U_spacecraft) corrects measured energy for spacecraft charging.

    Args:
        energy_grid: (nE,) electron energies in eV
        pitch_grid: (nE, nPitch) pitch angles in degrees
        U_surface: lunar surface potential in volts
        U_spacecraft: spacecraft potential in volts (scalar or (nE,))
        bs_over_bm: B_spacecraft/B_surface ratio
        beam_width_eV: secondary electron beam width in eV
        beam_amp: secondary electron beam amplitude
        beam_pitch_sigma_deg: beam angular spread in degrees
        background: flux value outside loss cone (default 0.05)

    Returns:
        Model flux array of shape (nE, nPitch)
    """
    result = synth_losscone_batch(
        energy_grid,
        pitch_grid,
        np.array([U_surface]),
        U_spacecraft=U_spacecraft,
        bs_over_bm=np.array([bs_over_bm]),
        beam_width_eV=np.array([beam_width_eV]),
        beam_amp=np.array([beam_amp]),
        beam_pitch_sigma_deg=beam_pitch_sigma_deg,
        background=np.array([background]),
    )
    return result[0]
