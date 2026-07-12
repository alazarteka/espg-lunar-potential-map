"""Batched projected Levenberg--Marquardt search for Experiment 006.

The optimizer works in linearly scaled ``[0, 1]`` coordinates and returns
candidate minima only. Scientific quality control and certification belong to
the CPU reference backend described in ``d1-backend-protocol.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import beam_torch
import torch
from torch import Tensor
from torch.func import jacrev, vmap

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class LMIterationDiagnostics:
    """Final iteration diagnostics for each case/start trajectory."""

    iterations: Tensor
    accepted_steps: Tensor
    rejected_steps: Tensor
    damping: Tensor
    gradient_inf_norm: Tensor
    step_norm: Tensor
    termination_code: Tensor


@dataclass(frozen=True)
class BatchedLMResult:
    """Candidate parameters, objectives, and optimizer-only diagnostics."""

    parameters: Tensor
    objectives: Tensor
    converged: Tensor
    diagnostics: LMIterationDiagnostics


def parameter_bounds(
    *, dtype: torch.dtype = torch.float64, device: torch.device | str | None = None
) -> tuple[Tensor, Tensor]:
    """Return the frozen D1 bounds in canonical parameter order."""
    lower = torch.tensor([0.0, 20.0, 3.0, *([-12.0] * 12)], dtype=dtype, device=device)
    upper = torch.tensor(
        [40.0, 2000.0, 50.0, *([12.0] * 12)], dtype=dtype, device=device
    )
    return lower, upper


def _as_float64(value: Tensor, *, device: torch.device) -> Tensor:
    return torch.as_tensor(value, dtype=torch.float64, device=device)


def _validate_shapes(
    energies_ev: Tensor,
    channel_low_ev: Tensor,
    channel_high_ev: Tensor,
    row_ids: Tensor,
    pitches_deg: Tensor,
    observed: Tensor,
    initial_parameters: Tensor,
    row_valid_mask: Tensor,
    sample_valid_mask: Tensor,
) -> None:
    if initial_parameters.ndim != 3 or initial_parameters.shape[-1] != 15:
        raise ValueError("Initial parameters must have shape [cases, starts, 15]")
    cases = initial_parameters.shape[0]
    if energies_ev.ndim != 2 or energies_ev.shape[0] != cases:
        raise ValueError("Energy rows must have shape [cases, rows]")
    if channel_low_ev.shape != energies_ev.shape or channel_high_ev.shape != (
        energies_ev.shape
    ):
        raise ValueError("Channel edges must match energy rows")
    expected_samples = (cases, row_ids.shape[-1])
    for value in (row_ids, pitches_deg, observed, sample_valid_mask):
        if value.shape != expected_samples:
            raise ValueError("Sample arrays must have shape [cases, samples]")
    if row_valid_mask.shape != energies_ev.shape:
        raise ValueError("Row mask must match energy rows")


def projected_levenberg_marquardt(
    energies_ev: Tensor,
    channel_low_ev: Tensor,
    channel_high_ev: Tensor,
    row_ids: Tensor,
    pitches_deg: Tensor,
    observed: Tensor,
    initial_parameters: Tensor,
    *,
    row_valid_mask: Tensor | None = None,
    sample_valid_mask: Tensor | None = None,
    lower_bounds: Sequence[float] | Tensor | None = None,
    upper_bounds: Sequence[float] | Tensor | None = None,
    energy_domain_ev: tuple[float, float] = (20.0, 2000.0),
    fwhm_fraction: float = 0.48,
    physical_lower: Sequence[float] = (0.01, 0.01, 90.0, 0.5),
    physical_upper: Sequence[float] = (2.0, 3.0, 180.0, 45.0),
    max_iterations: int = 100,
    initial_damping: float = 1e-2,
    minimum_damping: float = 1e-10,
    maximum_damping: float = 1e10,
    accepted_step_damping_divisor: float = 3.0,
    rejected_step_damping_multiplier: float = 10.0,
    gradient_tolerance: float = 1e-8,
    step_tolerance: float = 1e-8,
    relative_objective_tolerance: float = 1e-10,
    exact_objective_tolerance: float = 1e-24,
) -> BatchedLMResult:
    """Fit all case/start trajectories with independent projected LM steps.

    Floating inputs are evaluated in float64 on the initial-parameter device.
    Invalid padded samples may contain arbitrary nonfinite values when masked.
    Termination codes are 0 for exhausted iterations, 1 for gradient, 2 for
    step size, 3 for objective change, and 4 for an initially exact objective.
    """
    if max_iterations < 0:
        raise ValueError("max_iterations must be nonnegative")
    if initial_damping <= 0.0:
        raise ValueError("Initial damping must be positive")
    if not 0.0 < minimum_damping <= initial_damping <= maximum_damping:
        raise ValueError("Damping bounds must contain the initial damping")
    if accepted_step_damping_divisor <= 1.0:
        raise ValueError("Accepted-step damping divisor must exceed one")
    if rejected_step_damping_multiplier <= 1.0:
        raise ValueError("Rejected-step damping multiplier must exceed one")
    device = initial_parameters.device
    energies = _as_float64(energies_ev, device=device)
    low = _as_float64(channel_low_ev, device=device)
    high = _as_float64(channel_high_ev, device=device)
    pitches = _as_float64(pitches_deg, device=device)
    values = _as_float64(observed, device=device)
    initial = _as_float64(initial_parameters, device=device)
    rows = (
        torch.ones_like(energies, dtype=torch.bool)
        if row_valid_mask is None
        else (row_valid_mask.to(device=device, dtype=torch.bool))
    )
    samples = (
        torch.ones_like(row_ids, dtype=torch.bool)
        if sample_valid_mask is None
        else sample_valid_mask.to(device=device, dtype=torch.bool)
    )
    indices = row_ids.to(device=device, dtype=torch.long)
    _validate_shapes(
        energies, low, high, indices, pitches, values, initial, rows, samples
    )

    default_lower, default_upper = parameter_bounds(device=device)
    lower = (
        default_lower
        if lower_bounds is None
        else _as_float64(torch.as_tensor(lower_bounds), device=device)
    )
    upper = (
        default_upper
        if upper_bounds is None
        else _as_float64(torch.as_tensor(upper_bounds), device=device)
    )
    if lower.shape != (15,) or upper.shape != (15,):
        raise ValueError("Parameter bounds must each contain 15 values")
    if not bool(torch.all(torch.isfinite(lower) & torch.isfinite(upper))):
        raise ValueError("Parameter bounds must be finite")
    if not bool(torch.all(upper > lower)):
        raise ValueError("Every upper bound must exceed its lower bound")

    cases, starts, _ = initial.shape
    trajectories = cases * starts
    width = upper - lower
    scaled = ((initial - lower) / width).clamp(0.0, 1.0).reshape(trajectories, 15)

    def repeat_cases(value: Tensor) -> Tensor:
        return (
            value[:, None]
            .expand(cases, starts, *value.shape[1:])
            .reshape(trajectories, *value.shape[1:])
        )

    flat_energies = repeat_cases(energies)
    flat_low = repeat_cases(low)
    flat_high = repeat_cases(high)
    flat_indices = repeat_cases(indices)
    flat_pitches = repeat_cases(pitches)
    flat_values = repeat_cases(values)
    flat_rows = repeat_cases(rows)
    flat_samples = repeat_cases(samples)

    def residual_single(
        scaled_parameters: Tensor,
        energy: Tensor,
        channel_low: Tensor,
        channel_high: Tensor,
        sample_rows: Tensor,
        pitch: Tensor,
        observed_values: Tensor,
        valid_rows: Tensor,
        valid_samples: Tensor,
    ) -> Tensor:
        parameters = lower + width * scaled_parameters
        modeled = beam_torch.model_values(
            energy,
            channel_low,
            channel_high,
            sample_rows,
            pitch,
            parameters,
            energy_domain_ev=energy_domain_ev,
            fwhm_fraction=fwhm_fraction,
            physical_lower=physical_lower,
            physical_upper=physical_upper,
            row_valid_mask=valid_rows,
            sample_valid_mask=valid_samples,
        )
        return beam_torch.masked_log_residuals(observed_values, modeled, valid_samples)

    batched_residual = vmap(residual_single)
    batched_jacobian = vmap(jacrev(residual_single, argnums=0))
    fixed = (
        flat_energies,
        flat_low,
        flat_high,
        flat_indices,
        flat_pitches,
        flat_values,
        flat_rows,
        flat_samples,
    )

    residuals = batched_residual(scaled, *fixed)
    objectives = residuals.square().sum(dim=-1)
    damping = torch.full_like(objectives, initial_damping)
    active = torch.ones(trajectories, dtype=torch.bool, device=device)
    converged = torch.zeros_like(active)
    iterations = torch.zeros(trajectories, dtype=torch.long, device=device)
    accepted = torch.zeros_like(iterations)
    rejected = torch.zeros_like(iterations)
    termination = torch.zeros_like(iterations)
    gradient_norm = torch.full_like(objectives, torch.inf)
    step_norm = torch.zeros_like(objectives)

    exact = objectives <= exact_objective_tolerance
    converged |= exact
    active &= ~exact
    termination[exact] = 4

    for iteration in range(max_iterations):
        if not bool(active.any()):
            break
        jacobian = batched_jacobian(scaled, *fixed)
        finite_jacobian = torch.isfinite(jacobian).all(dim=(-2, -1))
        jacobian = torch.where(
            finite_jacobian[:, None, None], jacobian, torch.zeros_like(jacobian)
        )
        jt = jacobian.transpose(-2, -1)
        gradient = torch.bmm(jt, residuals.unsqueeze(-1)).squeeze(-1)
        at_lower = scaled <= 0.0
        at_upper = scaled >= 1.0
        projected_gradient = torch.where(
            (at_lower & (gradient > 0.0)) | (at_upper & (gradient < 0.0)),
            torch.zeros_like(gradient),
            gradient,
        )
        gradient_norm = projected_gradient.abs().amax(dim=-1)
        gradient_done = active & finite_jacobian & (gradient_norm <= gradient_tolerance)
        converged |= gradient_done
        active &= ~gradient_done
        termination[gradient_done] = 1
        if not bool(active.any()):
            break

        normal = torch.bmm(jt, jacobian)
        diagonal = normal.diagonal(dim1=-2, dim2=-1).clamp_min(1e-12)
        system = normal + torch.diag_embed(damping[:, None] * diagonal)
        solution = torch.linalg.solve_ex(system, -gradient.unsqueeze(-1))
        delta = solution.result.squeeze(-1)
        finite_step = (solution.info == 0) & torch.isfinite(delta).all(dim=-1)
        trial = (scaled + delta).clamp(0.0, 1.0)
        projected_step = trial - scaled
        step_norm = projected_step.norm(dim=-1)
        trial_residuals = batched_residual(trial, *fixed)
        trial_objectives = trial_residuals.square().sum(dim=-1)
        improves = (
            active
            & finite_jacobian
            & finite_step
            & torch.isfinite(trial_objectives)
            & (trial_objectives < objectives)
        )
        old_objectives = objectives
        scaled = torch.where(improves[:, None], trial, scaled)
        residuals = torch.where(improves[:, None], trial_residuals, residuals)
        objectives = torch.where(improves, trial_objectives, objectives)
        accepted += improves.to(torch.long)
        rejected += (active & ~improves).to(torch.long)
        damping = torch.where(
            improves,
            (damping / accepted_step_damping_divisor).clamp_min(minimum_damping),
            (damping * rejected_step_damping_multiplier).clamp_max(maximum_damping),
        )
        iterations[active] = iteration + 1

        step_done = (
            active
            & improves
            & (step_norm <= step_tolerance * (step_tolerance + scaled.norm(dim=-1)))
        )
        objective_done = (
            active
            & improves
            & (
                (old_objectives - objectives)
                <= relative_objective_tolerance * (1.0 + objectives)
            )
        )
        termination[step_done] = 2
        termination[objective_done & ~step_done] = 3
        newly_done = step_done | objective_done
        converged |= newly_done
        active &= ~newly_done

    parameters = (lower + width * scaled).reshape(cases, starts, 15)
    shape = (cases, starts)
    diagnostics = LMIterationDiagnostics(
        iterations=iterations.reshape(shape),
        accepted_steps=accepted.reshape(shape),
        rejected_steps=rejected.reshape(shape),
        damping=damping.reshape(shape),
        gradient_inf_norm=gradient_norm.reshape(shape),
        step_norm=step_norm.reshape(shape),
        termination_code=termination.reshape(shape),
    )
    return BatchedLMResult(
        parameters=parameters,
        objectives=objectives.reshape(shape),
        converged=converged.reshape(shape),
        diagnostics=diagnostics,
    )
