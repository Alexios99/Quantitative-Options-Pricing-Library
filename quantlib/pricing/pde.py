from quantlib.pricing.analytical import PricingResult, PricingEngine, MethodUsed
from quantlib.core.payoffs import OptionContract, OptionType, CallPayoff, PutPayoff, ExerciseStyle
import numpy as np
from datetime import datetime
from typing import Tuple

# Default number of initial Crank-Nicolson steps replaced by two fully implicit half-steps each
DEFAULT_RANNACHER_STEPS = 2
# Allowed breach of the no-arbitrage bounds (call <= S, put <= K) as a fraction of the strike.
# Far larger than the small oscillations a stable scheme produces near the kink, far smaller
# than the exponential growth of an unstable one.
BOUND_TOLERANCE = 1e-2


class PDEStabilityError(RuntimeError):
    """Raised when the finite-difference solution becomes non-finite or breaks no-arbitrage bounds."""


class PDEEngine(PricingEngine):
    """Black-Scholes PDE solved with a theta-scheme on a uniform spot grid.

    theta = 0.5 is Crank-Nicolson. Because the payoff has a kink at the strike, plain
    Crank-Nicolson leaves undamped high-frequency oscillations that pollute the price and,
    much more visibly, gamma. Rannacher start-up replaces the first `rannacher_steps` time
    steps with two fully implicit half-steps each, which damps those modes before switching
    to Crank-Nicolson, keeping second-order convergence (Rannacher 1984; Giles & Carter 2006).
    """

    def __init__(self, grid_params, solver_params):
        self.n_space = grid_params.n_space
        self.n_time = grid_params.n_time
        self.s_max = grid_params.s_max
        self.theta = grid_params.theta
        self.rannacher_steps = getattr(grid_params, "rannacher_steps", DEFAULT_RANNACHER_STEPS)
        if self.rannacher_steps < 0:
            raise ValueError(f"rannacher_steps must be >= 0, got {self.rannacher_steps}")
        self.psor_omega = solver_params.psor_omega
        self.psor_tol = solver_params.psor_tol
        self.psor_max_iter = solver_params.psor_max_iter

    def price(self, contract: OptionContract) -> PricingResult:
        self._build_grid(contract)
        payoff = CallPayoff(contract.strike) if contract.option == OptionType.CALL else PutPayoff(contract.strike)
        values = payoff.calculate_payoff(self.s_grid).astype(float)
        intrinsic = values.copy()
        upper_bound = self._upper_bound(contract)

        tau = 0.0
        for time_step in range(1, self.n_time + 1):
            if time_step <= self.rannacher_steps:
                sub_steps = [(1.0, 0.5 * self.dt)] * 2   # two fully implicit half-steps
            else:
                sub_steps = [(self.theta, self.dt)]
            for theta, dt in sub_steps:
                values = self._theta_step(contract, values, tau, tau + dt, theta, dt)
                tau += dt
                if contract.style == ExerciseStyle.AMERICAN:
                    values = self._apply_psor(values, intrinsic)
                self._check_stability(values, upper_bound, contract, time_step)

        self.solution = values  # option values on s_grid today
        price_value = self._interp_at_spot(values, contract.spot)
        return PricingResult(price=price_value, method=MethodUsed.PDE, time=datetime.now())

    def _build_grid(self, contract: OptionContract):
        self.s_grid = np.linspace(0, self.s_max, self.n_space)
        self.t_grid = np.linspace(contract.time_to_expiry, 0, self.n_time + 1)
        self.dx = self.s_max / (self.n_space - 1)
        self.dt = contract.time_to_expiry / self.n_time
        self.spot_index = np.searchsorted(self.s_grid, contract.spot)

    def _setup_boundary_conditions(self, contract, time_index: int):
        # t_grid is calendar time (t_grid[0] = T at expiry, t_grid[-1] = 0 today); the solver
        # marches from the payoff, so after time_index steps the time to expiry is T - t.
        tau = contract.time_to_expiry - self.t_grid[time_index]
        self.S_min_boundary, self.S_max_boundary = self._boundary_values(contract, tau)

    def _boundary_values(self, contract: OptionContract, tau: float) -> Tuple[float, float]:
        """Dirichlet values (V at S = 0, V at S = S_max) for time to expiry tau."""
        disc_k = contract.strike * np.exp(-contract.risk_free_rate * tau)
        if contract.option == OptionType.CALL:
            # S -> 0: worthless. S -> S_max (no dividends): S - K e^{-r tau}, same for American
            return 0.0, max(self.s_max - disc_k, 0.0)
        # Put: S -> 0 worth K (American, immediate exercise) or K e^{-r tau} (European)
        lower = float(contract.strike) if contract.style == ExerciseStyle.AMERICAN else disc_k
        return lower, 0.0

    def _upper_bound(self, contract: OptionContract) -> np.ndarray:
        """No-arbitrage upper bound on the grid: call <= S, put <= K."""
        if contract.option == OptionType.CALL:
            return self.s_grid
        return np.full_like(self.s_grid, float(contract.strike))

    def _theta_step(self, contract: OptionContract, v_prev: np.ndarray, tau_prev: float,
                    tau_new: float, theta: float, dt: float) -> np.ndarray:
        """Advance V from tau_prev to tau_new = tau_prev + dt: (I - theta dt L) V_new = (I + (1-theta) dt L) V_prev."""
        r = contract.risk_free_rate
        sigma = contract.volatility
        dS = self.dx

        # Interior nodes
        S = self.s_grid[1:-1]                      # shape (M,)
        Acoef = 0.5 * sigma**2 * S**2 / dS**2      # diffusion
        Bcoef = 0.5 * r * S / dS                   # convection

        # Spatial operator L (tridiagonal): l_lower, l_diag, l_upper
        l_lower = Acoef - Bcoef                    # couples to V_{i-1}
        l_diag  = -2.0 * Acoef - r                 # couples to V_i
        l_upper = Acoef + Bcoef                    # couples to V_{i+1}

        # Build A = I - θΔt L and B = I + (1−θ)Δt L (as tri-diagonals)
        Ad = 1.0 - theta * dt * l_diag
        Au =       - theta * dt * l_upper[:-1]      # superdiag length M-1
        Al =       - theta * dt * l_lower[1:]       # subdiag length M-1

        Bd = 1.0 + (1.0 - theta) * dt * l_diag
        Bu =        (1.0 - theta) * dt * l_upper[:-1]
        Bl =        (1.0 - theta) * dt * l_lower[1:]

        Vprev = v_prev[1:-1]

        # RHS = B * Vprev  (tri-diagonal multiplication)
        rhs = Bd * Vprev
        rhs[1:]  += Bl * Vprev[:-1]
        rhs[:-1] += Bu * Vprev[1:]

        # Dirichlet boundary contributions at the old and new time levels
        VL_prev, VR_prev = v_prev[0], v_prev[-1]
        VL_new, VR_new = self._boundary_values(contract, tau_new)
        rhs[0]  += (1.0 - theta) * dt * l_lower[0] * VL_prev + theta * dt * l_lower[0] * VL_new
        rhs[-1] += (1.0 - theta) * dt * l_upper[-1] * VR_prev + theta * dt * l_upper[-1] * VR_new

        # Solve A * V_new = rhs
        A = np.diag(Ad) + np.diag(Au, 1) + np.diag(Al, -1)
        v_new = np.empty_like(v_prev)
        v_new[0], v_new[-1] = VL_new, VR_new
        v_new[1:-1] = np.linalg.solve(A, rhs)

        # Kept for PSOR, which re-solves the same system with the early-exercise constraint
        self.psor_A_matrix = A
        self.psor_rhs = rhs
        return v_new

    def _apply_psor(self, values: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
        """Projected SOR for the American constraint V >= intrinsic, starting from the unconstrained solution."""
        values = values.copy()
        A, rhs = self.psor_A_matrix, self.psor_rhs
        m = len(rhs)
        for _ in range(self.psor_max_iter):
            old_values = values.copy()
            for i in range(1, self.n_space - 1):
                k = i - 1  # matrix index
                a_left = A[k, k - 1] if k > 0 else 0.0
                a_right = A[k, k + 1] if k < m - 1 else 0.0
                v_computed = (rhs[k] - a_left * values[i - 1] - a_right * values[i + 1]) / A[k, k]
                v_sor = values[i] + self.psor_omega * (v_computed - values[i])
                values[i] = max(v_sor, intrinsic[i])
            if np.max(np.abs(values[1:-1] - old_values[1:-1])) < self.psor_tol:
                break
        return values

    def _check_stability(self, values: np.ndarray, upper_bound: np.ndarray,
                         contract: OptionContract, time_step: int) -> None:
        """Raise instead of silently returning a price from a solution that has blown up."""
        if not np.all(np.isfinite(values)):
            raise PDEStabilityError(
                f"Non-finite PDE values at time step {time_step}/{self.n_time} "
                f"(theta={self.theta}, n_space={self.n_space}, n_time={self.n_time})")
        tolerance = BOUND_TOLERANCE * contract.strike
        breach = max(np.max(values - upper_bound), np.max(-values))
        if breach > tolerance:
            raise PDEStabilityError(
                f"PDE solution breaks no-arbitrage bounds by {breach:.4g} at time step "
                f"{time_step}/{self.n_time} (theta={self.theta}, dt={self.dt:.3g}, dS={self.dx:.3g}); "
                f"the scheme is unstable for this grid - use theta >= 0.5 or a smaller time step")

    def _interp_at_spot(self, values_at_t, spot):
        i = np.searchsorted(self.s_grid, spot)
        if i == 0:
            return float(values_at_t[0])
        if i >= self.n_space:
            return float(values_at_t[-1])
        s0, s1 = self.s_grid[i-1], self.s_grid[i]
        v0, v1 = values_at_t[i-1], values_at_t[i]
        w = (spot - s0) / (s1 - s0)
        return float(v0 + w * (v1 - v0))

    def greeks(self, contract: OptionContract):
        """Greeks calculation not implemented yet."""
        raise NotImplementedError("Greeks calculation for the PDE engine not yet implemented")
