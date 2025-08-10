from enum import Enum
from dataclasses import dataclass, field
from typing import List
from abc import ABC, abstractmethod
import numpy as np
import scipy
from datetime import datetime
from core.payoffs import OptionContract, OptionType, PayoffFunction, ExerciseStyle
from pricing.analytical import PricingEngine, PricingResult, GreeksResult, MethodUsed

@dataclass
class ProbResult:
    p: float
    u: float
    d: float


class TreeEngine(PricingEngine, ABC):
    def __init__(self, n_steps: int):
        if not isinstance(n_steps, int) or n_steps <= 0:
            raise ValueError(f"Invalid number of steps: {n_steps}")
        self.n_steps = n_steps

    @property
    @abstractmethod
    def method_used(self) -> MethodUsed:
        raise NotImplementedError("Greeks calculation for trees not yet implemented")

    # universal intrinsic payoff 
    def _intrinsic(self, c: OptionContract) -> float:
        return max(0.0, (c.spot - c.strike) if c.option == OptionType.CALL else (c.strike - c.spot))

    def price(self, contract: OptionContract) -> PricingResult:
        # 0 time → intrinsic
        if contract.time_to_expiry == 0.0:
            return PricingResult(price=float(self._intrinsic(contract)), method=self.method_used, time=datetime.now())

        # Compute baseline lattice price 
        base_price = float(self._price_impl(contract))

        # Allow engine-specific refinement (e.g., Richardson)
        refined = float(self._refine_price(contract, base_price))

        return PricingResult(price=refined, method=self.method_used, time=datetime.now())

    @abstractmethod
    def _price_impl(self, contract: OptionContract) -> float:
        """Concrete engines (binomial/trinomial) """
        pass

    # Default: no refinement. Engines can override.
    def _refine_price(self, contract: OptionContract, base_price: float) -> float:
        return base_price
    
    def greeks(self, contract: OptionContract) -> GreeksResult:
        pass
       
        

class BinomialTreeEngine(TreeEngine):
    def __init__(self, n_steps: int, richardson: str = "auto"):
        """
        richardson: "off" | "on" | "auto"
        - "on": always use Richardson for European
        - "off": never
        - "auto": use a small heuristic
        """
        super().__init__(n_steps)
        self.richardson = richardson

    @property
    def method_used(self) -> MethodUsed:
        return MethodUsed.BINOMIAL_TREE

    # Core CRR 
    def _calculate_probabilities(self, contract: OptionContract, n: int):
        dt = contract.time_to_expiry / n
        u = np.exp(contract.volatility * np.sqrt(dt))
        d = 1.0 / u
        p = (np.exp(contract.risk_free_rate * dt) - d) / (u - d)
        if p < 0 or p > 1:
            raise ValueError(f"Invalid risk-neutral probability {p}")
        return p, u, d, dt

    def _build(self, S0: float, u: float, d: float, n: int) -> np.ndarray:
        tree = np.zeros((n + 1, n + 1))
        for i in range(n + 1):
            for j in range(i + 1):
                # j = number of down moves, (i - j) up moves
                tree[i, j] = S0 * (u ** (i - j)) * (d ** j)
        return tree

    def _rollback(self, stock: np.ndarray, contract: OptionContract, p: float, disc: float, n: int) -> float:
        opt = np.zeros_like(stock)
        # terminal payoffs
        if contract.option == OptionType.CALL:
            for j in range(n + 1):
                opt[n, j] = max(0.0, stock[n, j] - contract.strike)
        else:
            for j in range(n + 1):
                opt[n, j] = max(0.0, contract.strike - stock[n, j])
        # backward induction
        for i in range(n - 1, -1, -1):
            for j in range(i + 1):
                cont = disc * (p * opt[i + 1, j] + (1.0 - p) * opt[i + 1, j + 1])
                if contract.style == ExerciseStyle.AMERICAN:
                    intrinsic = (stock[i, j] - contract.strike) if contract.option == OptionType.CALL \
                                else (contract.strike - stock[i, j])
                    opt[i, j] = max(cont, max(0.0, intrinsic))
                else:
                    opt[i, j] = cont
        return float(opt[0, 0])

    # one-shot price with a specific N (Richardson)
    def _price_with_n(self, contract: OptionContract, n: int) -> float:
        p, u, d, dt = self._calculate_probabilities(contract, n)
        disc = np.exp(-contract.risk_free_rate * dt)
        stock = self._build(contract.spot, u, d, n)
        return self._rollback(stock, contract, p, disc, n)

    # the normal single-run implementation the base will call first
    def _price_impl(self, contract: OptionContract) -> float:
        return self._price_with_n(contract, self.n_steps)

    # optional refinement (Richardson averaging for European only)
    def _refine_price(self, contract: OptionContract, base_price: float) -> float:
        # turn off for non-European
        if contract.style != ExerciseStyle.EUROPEAN:
            return base_price

        # config
        if self.richardson == "off":
            return base_price
        if self.richardson == "on":
            pN   = base_price
            pNp1 = self._price_with_n(contract, self.n_steps + 1)
            return 0.5 * (pN + pNp1)

        # "auto" heuristic: enable when base error is likely larger
        # (modest N, wide diffusion, near-ATM)
        N = self.n_steps
        lam = contract.volatility * np.sqrt(max(contract.time_to_expiry, 0.0))
        mny = abs(np.log(contract.spot / contract.strike)) if contract.spot > 0 and contract.strike > 0 else 1e9
        use_rich = (N < 200) or (lam > 0.35) or (mny < 0.1)

        if not use_rich:
            return base_price

        pN   = base_price
        pNp1 = self._price_with_n(contract, N + 1)
        return 0.5 * (pN + pNp1)
    
    def greeks(self, contract: OptionContract):
        """Greeks calculation not implemented yet."""
        raise NotImplementedError("Greeks calculation for trees not yet implemented")
