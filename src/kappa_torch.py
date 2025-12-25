"""
PyTorch-accelerated Kappa distribution fitter.

Provides batched fitting of kappa distribution parameters across multiple
spectra simultaneously using vectorized tensor operations.

The key insight is that we can evaluate the forward model for all spectra
and all parameter candidates in parallel, then use Differential Evolution
to find the best parameters for each spectrum independently.
"""

from __future__ import annotations

import math

import numpy as np

try:
    import torch
    from torch import Tensor

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    Tensor = None  # type: ignore[misc, assignment]


# Physical constant: electron mass in eV·s²/m²
ELECTRON_MASS_EV_S2_M2 = 5.685630e-12


def _torch_lgamma(x: Tensor) -> Tensor:
    """Compute log-gamma function."""
    return torch.lgamma(x)


def _torch_gamma_ratio(kappa: Tensor) -> Tensor:
    """
    Compute Γ(κ+1) / (π^1.5 · κ^1.5 · Γ(κ-0.5)) in log space for stability.

    Args:
        kappa: (N, P) or (P,) kappa values

    Returns:
        Gamma ratio with same shape as kappa
    """
    # log(Γ(κ+1)) - log(π^1.5) - 1.5*log(κ) - log(Γ(κ-0.5))
    log_ratio = (
        _torch_lgamma(kappa + 1)
        - 1.5 * math.log(math.pi)
        - 1.5 * torch.log(kappa)
        - _torch_lgamma(kappa - 0.5)
    )
    return torch.exp(log_ratio)


def omnidirectional_flux_batch_torch(
    density: Tensor,
    kappa: Tensor,
    theta: Tensor,
    energy: Tensor,
) -> Tensor:
    """
    Compute omnidirectional flux for batched parameters.

    Args:
        density: (N,) density estimates per spectrum [particles/m³]
        kappa: (N, P) kappa values for each spectrum and candidate
        theta: (N, P) thermal speed values [m/s]
        energy: (E,) energy grid [eV]

    Returns:
        (N, P, E) omnidirectional flux [particles/(cm² s eV)]
    """
    N, P = kappa.shape
    E = energy.shape[0]
    device = kappa.device
    dtype = kappa.dtype

    # velocity = sqrt(2E/m_e) in m/s
    # Shape: (E,)
    velocity = torch.sqrt(2.0 * energy / ELECTRON_MASS_EV_S2_M2)

    # Reshape for broadcasting:
    # density: (N,) -> (N, 1, 1)
    # kappa: (N, P) -> (N, P, 1)
    # theta: (N, P) -> (N, P, 1)
    # velocity: (E,) -> (1, 1, E)
    density_exp = density.view(N, 1, 1)
    kappa_exp = kappa.view(N, P, 1)
    theta_exp = theta.view(N, P, 1)
    velocity_exp = velocity.view(1, 1, E)

    # Compute gamma ratio prefactor: Γ(κ+1) / (π^1.5 · κ^1.5 · Γ(κ-0.5))
    # Shape: (N, P, 1)
    prefactor = _torch_gamma_ratio(kappa_exp)

    # Core: n / θ³
    # Shape: (N, P, 1)
    core = density_exp / (theta_exp ** 3)

    # Tail: (1 + (v/θ)² / κ)^(-κ-1)
    # Shape: (N, P, E)
    velocity_ratio_sq = (velocity_exp / theta_exp) ** 2
    tail = torch.pow(1.0 + velocity_ratio_sq / kappa_exp, -kappa_exp - 1.0)

    # Phase space density f(v)
    # Shape: (N, P, E)
    distribution = prefactor * core * tail

    # Directional flux: j(E) = f(v) · v² / m_e
    # Shape: (N, P, E)
    directional_flux = distribution * (velocity_exp ** 2) / ELECTRON_MASS_EV_S2_M2

    # Omnidirectional flux: J(E) = 4π · j(E) · (1e-4 for cm²->m²)
    # Shape: (N, P, E)
    omni_flux = 4.0 * math.pi * 1e-4 * directional_flux

    return omni_flux


def build_response_matrix_torch(
    energy: Tensor,
    energy_window_width_relative: float = 0.5,
) -> Tensor:
    """
    Build log-energy response matrix for instrumental resolution effects.

    Args:
        energy: (E,) energy centers [eV]
        energy_window_width_relative: relative energy resolution (FWHM/E)

    Returns:
        (E, E) response matrix W
    """
    ln_energy = torch.log(energy)

    # Convert relative width to Gaussian sigma in log-energy space
    s = math.asinh(0.5 * energy_window_width_relative) / math.sqrt(2.0 * math.log(2.0))

    # W[i,j] = exp(-0.5 * ((ln(E_i) - ln(E_j)) / s)²)
    # Shape: (E, E)
    diff = ln_energy.unsqueeze(0) - ln_energy.unsqueeze(1)
    W = torch.exp(-0.5 * (diff / s) ** 2)

    # Normalize rows
    W = W / W.sum(dim=1, keepdim=True)

    return W


def compute_kappa_chi2_batch_torch(
    model_flux: Tensor,
    data_flux: Tensor,
    weights: Tensor,
    response_matrix: Tensor | None = None,
    eps: float = 1e-10,
) -> Tensor:
    """
    Compute chi² for batched kappa model fits.

    Args:
        model_flux: (N, P, E) model omnidirectional flux
        data_flux: (N, E) measured flux
        weights: (N, E) fitting weights (1/σ)
        response_matrix: (E, E) optional convolution matrix
        eps: small value to avoid log(0)

    Returns:
        (N, P) chi² values
    """
    N, P, E = model_flux.shape

    # Apply response matrix if provided
    if response_matrix is not None:
        # model_flux: (N, P, E) @ W.T: (E, E) -> (N, P, E)
        model_flux = torch.einsum('npe,ef->npf', model_flux, response_matrix)

    # Log-space comparison
    log_model = torch.log(model_flux + eps)
    log_data = torch.log(data_flux.unsqueeze(1) + eps)  # (N, 1, E)

    # Weighted squared difference
    # weights: (N, E) -> (N, 1, E)
    weights_exp = weights.unsqueeze(1)

    diff = (log_model - log_data) * weights_exp
    chi2 = (diff ** 2).sum(dim=2)  # Sum over energy: (N, P)

    return chi2


class KappaDifferentialEvolution:
    """
    Batched Differential Evolution for Kappa fitting.

    Optimizes kappa parameters for N spectra simultaneously,
    with P candidates per spectrum.
    """

    def __init__(
        self,
        bounds: list[tuple[float, float]],
        n_spectra: int,
        popsize: int = 30,
        mutation: float = 0.8,
        crossover: float = 0.9,
        maxiter: int = 100,
        atol: float = 0.01,
        seed: int = 42,
        device: str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
        """
        Initialize batched DE optimizer.

        Args:
            bounds: [(min, max), ...] for each parameter
            n_spectra: Number of spectra to optimize in parallel
            popsize: Population size per spectrum
            mutation: DE mutation factor F
            crossover: DE crossover probability CR
            maxiter: Maximum iterations
            atol: Convergence tolerance
            seed: Random seed
            device: torch device
            dtype: torch dtype
        """
        self.bounds = bounds
        self.n_params = len(bounds)
        self.n_spectra = n_spectra
        self.popsize = popsize
        self.mutation = mutation
        self.crossover = crossover
        self.maxiter = maxiter
        self.atol = atol
        self.device = torch.device(device)
        self.dtype = dtype

        torch.manual_seed(seed)

        # Convert bounds to tensors
        self.lower = torch.tensor(
            [b[0] for b in bounds], device=self.device, dtype=self.dtype
        )
        self.upper = torch.tensor(
            [b[1] for b in bounds], device=self.device, dtype=self.dtype
        )

    def _init_population(self) -> Tensor:
        """
        Initialize population using Sobol sequence.

        Returns:
            (N, P, n_params) initial population
        """
        N, P = self.n_spectra, self.popsize

        try:
            from torch.quasirandom import SobolEngine
            sobol = SobolEngine(dimension=self.n_params, scramble=True)
            # Generate P samples for each of N spectra
            # For simplicity, use same initial samples for all spectra
            unit_samples = sobol.draw(P).to(device=self.device, dtype=self.dtype)
            # Broadcast to (N, P, n_params)
            unit_samples = unit_samples.unsqueeze(0).expand(N, -1, -1)
        except ImportError:
            unit_samples = torch.rand(N, P, self.n_params, device=self.device, dtype=self.dtype)

        # Scale to bounds
        pop = self.lower + unit_samples * (self.upper - self.lower)
        return pop

    def _mutate(self, population: Tensor, best: Tensor) -> Tensor:
        """
        DE/best/1 mutation.

        Args:
            population: (N, P, n_params) current population
            best: (N, n_params) best solution per spectrum

        Returns:
            (N, P, n_params) mutant vectors
        """
        N, P, D = population.shape

        # Random indices for r0, r1 (excluding self)
        # For simplicity, just use random pairs
        idx = torch.randint(0, P, (N, P, 2), device=self.device)

        r0 = torch.gather(population, 1, idx[:, :, 0:1].expand(-1, -1, D))
        r1 = torch.gather(population, 1, idx[:, :, 1:2].expand(-1, -1, D))

        # mutant = best + F * (r0 - r1)
        best_exp = best.unsqueeze(1).expand(-1, P, -1)
        mutant = best_exp + self.mutation * (r0 - r1)

        # Clip to bounds
        mutant = torch.clamp(mutant, self.lower, self.upper)

        return mutant

    def _crossover(self, population: Tensor, mutant: Tensor) -> Tensor:
        """
        Binomial crossover.

        Args:
            population: (N, P, n_params) current population
            mutant: (N, P, n_params) mutant vectors

        Returns:
            (N, P, n_params) trial vectors
        """
        N, P, D = population.shape

        # Crossover mask
        mask = torch.rand(N, P, D, device=self.device) < self.crossover

        # Ensure at least one parameter from mutant
        j_rand = torch.randint(0, D, (N, P), device=self.device)
        for i in range(D):
            mask[:, :, i] = mask[:, :, i] | (j_rand == i)

        trial = torch.where(mask, mutant, population)
        return trial

    def optimize(
        self,
        objective_fn,
    ) -> tuple[Tensor, Tensor, int]:
        """
        Run batched DE optimization.

        Args:
            objective_fn: Function (N, P, n_params) -> (N, P) fitness

        Returns:
            best_params: (N, n_params) best solution per spectrum
            best_fitness: (N,) best fitness per spectrum
            n_iterations: iterations run
        """
        N, P = self.n_spectra, self.popsize

        # Initialize
        population = self._init_population()  # (N, P, n_params)
        fitness = objective_fn(population)     # (N, P)

        # Track best per spectrum
        best_idx = torch.argmin(fitness, dim=1)  # (N,)
        best_fitness = fitness.gather(1, best_idx.unsqueeze(1)).squeeze(1)  # (N,)
        best_params = population.gather(
            1, best_idx.view(N, 1, 1).expand(-1, -1, self.n_params)
        ).squeeze(1)  # (N, n_params)

        prev_best = best_fitness.clone()

        for iteration in range(self.maxiter):
            # Mutation
            mutant = self._mutate(population, best_params)

            # Crossover
            trial = self._crossover(population, mutant)

            # Evaluate trials
            trial_fitness = objective_fn(trial)

            # Selection: keep better of trial or original (per individual)
            improved = trial_fitness < fitness
            population = torch.where(
                improved.unsqueeze(-1), trial, population
            )
            fitness = torch.where(improved, trial_fitness, fitness)

            # Update best
            current_best_idx = torch.argmin(fitness, dim=1)
            current_best_fitness = fitness.gather(1, current_best_idx.unsqueeze(1)).squeeze(1)

            # Update where improved
            improved_mask = current_best_fitness < best_fitness
            best_fitness = torch.where(improved_mask, current_best_fitness, best_fitness)

            new_best_params = population.gather(
                1, current_best_idx.view(N, 1, 1).expand(-1, -1, self.n_params)
            ).squeeze(1)
            best_params = torch.where(
                improved_mask.unsqueeze(-1), new_best_params, best_params
            )

            # Check convergence (average improvement across spectra)
            improvement = torch.abs(prev_best - best_fitness).mean()
            if improvement < self.atol and iteration > 10:
                break

            prev_best = best_fitness.clone()

        return best_params, best_fitness, iteration + 1


class KappaFitterTorch:
    """
    PyTorch-accelerated batched Kappa fitter.

    Fits kappa distribution parameters for multiple spectra simultaneously
    using vectorized tensor operations.
    """

    DEFAULT_BOUNDS = [
        (2.5, 6.0),   # kappa
        (6.0, 8.0),   # log10(theta) where theta is in m/s
    ]

    def __init__(
        self,
        device: str = "cpu",
        dtype: str = "float64",
        popsize: int = 30,
        maxiter: int = 100,
        use_convolution: bool = True,
        energy_window_width_relative: float = 0.5,
    ):
        """
        Initialize batched Kappa fitter.

        Args:
            device: torch device
            dtype: 'float32' or 'float64'
            popsize: DE population size
            maxiter: DE max iterations
            use_convolution: Apply energy response matrix
            energy_window_width_relative: Energy resolution (FWHM/E)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required for KappaFitterTorch")

        self.device = torch.device(device)
        self.dtype = torch.float64 if dtype == "float64" else torch.float32
        self.popsize = popsize
        self.maxiter = maxiter
        self.use_convolution = use_convolution
        self.energy_window_width_relative = energy_window_width_relative

    def fit_batch(
        self,
        energy: np.ndarray,
        flux_data: np.ndarray,
        density_estimates: np.ndarray,
        weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fit kappa parameters for multiple spectra in batch.

        Args:
            energy: (E,) energy grid [eV]
            flux_data: (N, E) measured omnidirectional flux
            density_estimates: (N,) density estimates [particles/m³]
            weights: (N, E) optional fitting weights

        Returns:
            kappa: (N,) fitted kappa values
            theta: (N,) fitted theta values [m/s]
            chi2: (N,) chi² values
        """
        N, E = flux_data.shape

        # Convert to tensors
        energy_t = torch.tensor(energy, device=self.device, dtype=self.dtype)
        flux_t = torch.tensor(flux_data, device=self.device, dtype=self.dtype)
        density_t = torch.tensor(density_estimates, device=self.device, dtype=self.dtype)

        if weights is None:
            # Use log-flux uncertainty as weights
            weights = np.ones_like(flux_data)
        weights_t = torch.tensor(weights, device=self.device, dtype=self.dtype)

        # Build response matrix if needed
        response_matrix = None
        if self.use_convolution:
            response_matrix = build_response_matrix_torch(
                energy_t, self.energy_window_width_relative
            )

        # Create optimizer
        de = KappaDifferentialEvolution(
            bounds=self.DEFAULT_BOUNDS,
            n_spectra=N,
            popsize=self.popsize,
            maxiter=self.maxiter,
            device=str(self.device),
            dtype=self.dtype,
        )

        # Define objective function
        def objective(params: Tensor) -> Tensor:
            """
            Evaluate chi² for all spectra and candidates.

            Args:
                params: (N, P, 2) [kappa, log_theta]

            Returns:
                (N, P) chi² values
            """
            kappa = params[:, :, 0]  # (N, P)
            log_theta = params[:, :, 1]  # (N, P)
            theta = 10.0 ** log_theta  # (N, P)

            # Compute model flux
            model_flux = omnidirectional_flux_batch_torch(
                density_t, kappa, theta, energy_t
            )  # (N, P, E)

            # Compute chi²
            chi2 = compute_kappa_chi2_batch_torch(
                model_flux, flux_t, weights_t, response_matrix
            )  # (N, P)

            # Penalize invalid
            chi2 = torch.where(torch.isfinite(chi2), chi2, torch.tensor(1e30, device=self.device))

            return chi2

        # Run optimization
        best_params, best_chi2, n_iter = de.optimize(objective)

        # Extract results
        kappa = best_params[:, 0].cpu().numpy()
        log_theta = best_params[:, 1].cpu().numpy()
        theta = 10.0 ** log_theta
        chi2 = best_chi2.cpu().numpy()

        return kappa, theta, chi2
