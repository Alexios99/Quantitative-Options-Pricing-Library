from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
from datetime import datetime
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle, CallPayoff, PutPayoff
from quantlib.pricing.analytical import PricingEngine, PricingResult, GreeksResult, MethodUsed, BlackScholesEngine
from quantlib.core.stochastic_processes import GeometricBrownianMotion

@dataclass
class MonteCarloPricingResult(PricingResult):
    """Pricing result class for Monte Carlo"""
    standard_error: float
    confidence_interval: float
    n_paths: int

class MonteCarloEngine(PricingEngine):
    def __init__(self, n_steps, n_paths, variance_reduction=None, seed=None):
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.variance_reduction = variance_reduction
        self.seed = seed

    def price(self, contract: OptionContract) -> MonteCarloPricingResult:
        
        payoffs = self._generate_payoffs(contract)
        discounted_payoffs = payoffs * np.exp(-contract.risk_free_rate * contract.time_to_expiry)

        mean_price, std_error, ci_half_width = self._calculate_statistics(discounted_payoffs)

        return MonteCarloPricingResult(price=mean_price, 
                             method=MethodUsed.MONTE_CARLO, 
                             time=datetime.now(), 
                             standard_error=std_error, 
                             confidence_interval=ci_half_width,
                             n_paths=self.n_paths)

    def _generate_payoffs(self, contract: OptionContract) -> np.ndarray:
        """Handle path generation and variance reduction"""
        stochastic_process = GeometricBrownianMotion(drift=contract.risk_free_rate, volatility=contract.volatility)
        if self.variance_reduction == "antithetic":
            # Set seed for reproducibility
            if self.seed:
                np.random.seed(self.seed)
            
            half_paths = self.n_paths // 2
            
            # Generate random numbers yourself
            Z = np.random.normal(size=(half_paths, self.n_steps))
            
            # Create normal and antithetic pairs
            normal_final_prices = self._simulate_gbm_with_randoms(contract, Z)
            antithetic_final_prices = self._simulate_gbm_with_randoms(contract, -Z)  # Key: use -Z
            
            # Calculate payoffs
            normal_payoffs = self._calculate_payoffs_from_prices(contract, normal_final_prices)
            antithetic_payoffs = self._calculate_payoffs_from_prices(contract, antithetic_final_prices)
            
            # Average the pairs
            paired_payoffs = (normal_payoffs + antithetic_payoffs) / 2
            return paired_payoffs
        else:
            paths = stochastic_process.simulate_paths(contract.spot, contract.time_to_expiry, self.n_paths, self.n_steps, self.seed)
            return self._calculate_payoffs_for_paths(contract, paths)
        
    def _calculate_payoffs_from_prices(self, contract: OptionContract, prices: np.ndarray) -> np.ndarray:
        """Calculate payoffs from terminal prices"""
        if contract.option == OptionType.CALL:
            payoff_func = CallPayoff(contract.strike)
            return payoff_func.calculate_payoff(prices)
        else:
            payoff_func = PutPayoff(contract.strike)
            return payoff_func.calculate_payoff(prices)

    def _calculate_payoffs_for_paths(self, contract: OptionContract, paths: np.ndarray) -> np.ndarray:
        """Calculate payoffs for given paths"""
        final_prices = paths[:, -1]
        
        if contract.option == OptionType.CALL:
            payoff_func = CallPayoff(contract.strike)
            return payoff_func.calculate_payoff(final_prices)
        else:
            payoff_func = PutPayoff(contract.strike)
            return payoff_func.calculate_payoff(final_prices)
        
    def _simulate_gbm_with_randoms(self, contract: OptionContract, Z: np.ndarray) -> np.ndarray:
        """Simulate GBM terminal prices using provided random numbers"""
        dt = contract.time_to_expiry / self.n_steps
        drift_term = (contract.risk_free_rate - contract.volatility**2 / 2) * dt
        vol_term = contract.volatility * np.sqrt(dt)
        
        log_returns = np.cumsum(drift_term + vol_term * Z, axis=1)
        
        terminal_prices = contract.spot * np.exp(log_returns[:, -1])
        return terminal_prices

    def _calculate_statistics(self, payoffs: np.ndarray) -> tuple:
        """Calculate mean, std error and confidence interval"""
        mean_payoff = np.mean(payoffs)
        std_error = np.std(payoffs)/ np.sqrt(len(payoffs))

        confidence_level = 1.96  # for 95% CI
        ci_half_width = confidence_level * std_error

        return mean_payoff, std_error, ci_half_width
   
    def greeks(self, contract: OptionContract):
        """Greeks calculation not implemented yet."""
        raise NotImplementedError("Greeks calculation for trees not yet implemented")