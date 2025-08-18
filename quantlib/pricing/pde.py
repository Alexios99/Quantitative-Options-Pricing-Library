from quantlib.pricing.analytical import PricingResult, PricingEngine, MethodUsed
from quantlib.core.payoffs import OptionContract, OptionType, CallPayoff, PutPayoff
import numpy as np
from datetime import datetime
from scipy import linalg
from typing import Optional, Tuple


class PDEEngine(PricingEngine):
    def __init__(self, grid_params, solver_params):
        self.n_space = grid_params.n_space
        self.n_time = grid_params.n_time
        self.s_max = grid_params.s_max
        self.theta = grid_params.theta
        self.psor_omega = solver_params.psor_omega
        self.psor_tol = solver_params.psor_tol
        self.psor_max_iter = solver_params.psor_max_iter
    def price(self, contract: OptionContract) -> PricingResult:
        self._build_grid(contract)
        # Initialize solution array [space_points, time_points]
        sol_array = np.zeros((self.n_space, self.n_time + 1))
        
        if contract.option == OptionType.CALL:
            payoff_func = CallPayoff(contract.strike) 
            sol_array[:, 0] = payoff_func.calculate_payoff(self.s_grid)
        else:
            payoff_func = PutPayoff(contract.strike)
            sol_array[:, 0] = payoff_func.calculate_payoff(self.s_grid)
        
        for time_step in range(1, self.n_time + 1):
            self._setup_boundary_conditions(contract, time_step)
            sol_array[0, time_step] = self.S_min_boundary
            sol_array[-1, time_step] = self.S_max_boundary
            self._crank_nicolson_step(contract, sol_array, time_step)
            max_val = np.max(np.abs(sol_array[:, time_step]))
            if max_val > 1000:  # Reasonable threshold
                break

        # Extract price at current spot
        price_value = sol_array[self.spot_index, -1]
        
        return PricingResult(price=price_value, method=MethodUsed.PDE, time=datetime.now())

        

    def _build_grid(self, contract: OptionContract):
        self.s_grid = np.linspace(0, self.s_max, self.n_space)
        self.t_grid = np.linspace(contract.time_to_expiry, 0, self.n_time + 1)
        self.dx = self.s_max / (self.n_space - 1)
        self.dt = contract.time_to_expiry / self.n_time
        self.spot_index = np.searchsorted(self.s_grid, contract.spot)
        
    def _setup_boundary_conditions(self, contract: OptionContract, time_index):
        tau = self.t_grid[time_index]
        if contract.option == OptionType.CALL:
            self.S_min_boundary = 0
            self.S_max_boundary = max(self.s_max - contract.strike * np.exp(-contract.risk_free_rate * tau), 0)
        else:
            self.S_min_boundary = contract.strike * np.exp(-contract.risk_free_rate * tau)
            self.S_max_boundary = 0
        
        
    def _crank_nicolson_step(self, contract: OptionContract, V: np.ndarray, time_index: int):
        r = contract.risk_free_rate
        sigma = contract.volatility
        dt = self.dt
        dS = self.dx
        theta = self.theta  # 0.5 for CN

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

        Vprev = V[1:-1, time_index - 1]

        # RHS = B * Vprev  (tri-diagonal multiplication)
        rhs = Bd * Vprev
        rhs[1:]  += Bl * Vprev[:-1]
        rhs[:-1] += Bu * Vprev[1:]

        # Add boundary contributions (Dirichlet)
        VL_n   = V[0,  time_index]
        VR_n   = V[-1, time_index]
        VL_nm1 = V[0,  time_index - 1]
        VR_nm1 = V[-1, time_index - 1]

        # first interior row (couples to left boundary)
        rhs[0]  +=  (1.0 - theta) * dt * l_lower[0] * VL_nm1 \
                +   theta        * dt * l_lower[0] * VL_n
        # last interior row (couples to right boundary)
        rhs[-1] +=  (1.0 - theta) * dt * l_upper[-1] * VR_nm1 \
                +   theta        * dt * l_upper[-1] * VR_n

        # Solve A * V^n = rhs
        M = len(Ad)
        A = np.diag(Ad) + np.diag(Au, 1) + np.diag(Al, -1)
        V[1:-1, time_index] = np.linalg.solve(A, rhs)

    def _apply_psor(self):  # For American exercise
        """Apply PSOR for American early exercise constraint"""
        pass


    def greeks(self, contract: OptionContract):
        """Greeks calculation not implemented yet."""
        raise NotImplementedError("Greeks calculation for trees not yet implemented")