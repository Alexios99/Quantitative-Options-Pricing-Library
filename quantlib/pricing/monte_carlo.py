from dataclasses import dataclass
from datetime import datetime
import numpy as np

from quantlib.core.payoffs import OptionContract,OptionType,CallPayoff, PutPayoff
from quantlib.pricing.analytical import PricingEngine, PricingResult, GreeksResult, MethodUsed, BlackScholesEngine
from quantlib.core.stochastic_processes import GeometricBrownianMotion


@dataclass
class MonteCarloPricingResult(PricingResult):
    """Pricing result class for Monte Carlo."""
    standard_error: float
    confidence_interval: float
    n_paths: int


class MonteCarloEngine(PricingEngine):
    def __init__(self, n_steps: int, n_paths: int, variance_reduction: str | None = None, seed: int | None = None):
        self.n_steps = n_steps
        self.n_paths = n_paths
        self.variance_reduction = variance_reduction  # "none" | "antithetic" | "control"
        self.seed = seed

    def price(self, contract: OptionContract) -> MonteCarloPricingResult:
        payoffs = self._generate_payoffs(contract)
        df = np.exp(-contract.risk_free_rate * contract.time_to_expiry)
        discounted = payoffs * df
        mean_price, std_error, ci_half_width = self._calculate_statistics(discounted)

        return MonteCarloPricingResult(
            price=mean_price,
            method=MethodUsed.MONTE_CARLO,
            time=datetime.now(),
            standard_error=std_error,
            confidence_interval=ci_half_width,
            n_paths=len(discounted)
        )

    def greeks(self, contract: OptionContract) -> GreeksResult:
        """
        Monte Carlo Greeks using the pathwise method for Delta.
        Uses the *same terminal prices* as the selected variance reduction.
        Applies control variates to Delta samples if variance_reduction == "control".
        """
        vr = (self.variance_reduction or "none").lower()

        if vr == "antithetic":
            s1, s2 = self._generate_antithetic_terminal_prices(contract)
            call_1, put_1 = self._pathwise_delta_samples(contract, s1)
            call_2, put_2 = self._pathwise_delta_samples(contract, s2)
            call_samples = 0.5 * (call_1 + call_2)
            put_samples = 0.5 * (put_1 + put_2)

        else:
            final_prices = self._generate_plain_terminal_prices(contract)
            call_samples, put_samples = self._pathwise_delta_samples(contract, final_prices)

            if vr == "control":
                # Control variate on delta using C = S_T/S_0 with E[C] = exp(rT)
                call_samples = self._apply_control_to_delta(contract, final_prices, call_samples)
                put_samples = self._apply_control_to_delta(contract, final_prices, put_samples)

        delta_call = float(np.mean(call_samples))
        delta_put = float(np.mean(put_samples))

        return GreeksResult(
            delta_call=delta_call,
            delta_put=delta_put,
            gamma=0.0,
            theta_call=0.0,
            theta_put=0.0,
            vega=0.0,
            rho_call=0.0,
            rho_put=0.0,
        )


    def _generate_payoffs(self, contract: OptionContract) -> np.ndarray:
        """Handle path generation and variance reduction for pricing."""
        sp = GeometricBrownianMotion(drift=contract.risk_free_rate, volatility=contract.volatility)
        vr = (self.variance_reduction or "none").lower()

        if vr == "antithetic":
            s1, s2 = self._generate_antithetic_terminal_prices(contract)
            p1 = self._payoff_from_terminal(contract, s1)
            p2 = self._payoff_from_terminal(contract, s2)
            return 0.5 * (p1 + p2)

        paths = sp.simulate_paths(
            contract.spot, contract.time_to_expiry, self.n_paths, self.n_steps, self.seed
        )
        payoffs = self._calculate_payoffs_for_paths(contract, paths)

        if vr == "control":
            final_prices = paths[:, -1]
            return self._apply_control_variates_on_price(contract, payoffs, final_prices)

        return payoffs


    def _generate_plain_terminal_prices(self, contract: OptionContract) -> np.ndarray:
        """Plain MC terminal prices (no antithetic pairing)."""
        sp = GeometricBrownianMotion(drift=contract.risk_free_rate, volatility=contract.volatility)
        paths = sp.simulate_paths(
            contract.spot, contract.time_to_expiry, self.n_paths, self.n_steps, self.seed
        )
        return paths[:, -1]

    def _generate_antithetic_terminal_prices(self, contract: OptionContract) -> tuple[np.ndarray, np.ndarray]:
        """Generate paired antithetic terminal prices."""
        half_paths = self.n_paths // 2
        rng = np.random.default_rng(self.seed)
        Z = rng.normal(size=(half_paths, self.n_steps))
        s1 = self._simulate_gbm_with_randoms(contract, Z)
        s2 = self._simulate_gbm_with_randoms(contract, -Z)
        return s1, s2


    def _payoff_from_terminal(self, contract: OptionContract, terminal_prices: np.ndarray) -> np.ndarray:
        if contract.option == OptionType.CALL:
            payoff_func = CallPayoff(contract.strike)
        else:
            payoff_func = PutPayoff(contract.strike)
        return payoff_func.calculate_payoff(terminal_prices)

    def _calculate_payoffs_for_paths(self, contract: OptionContract, paths: np.ndarray) -> np.ndarray:
        final_prices = paths[:, -1]
        return self._payoff_from_terminal(contract, final_prices)


    def _calculate_statistics(self, payoffs: np.ndarray) -> tuple[float, float, float]:
        """Mean, standard error, and 95% half-width CI."""
        n = len(payoffs)
        mean_payoff = float(np.mean(payoffs))
        std_error = float(np.std(payoffs, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        ci_half_width = 1.96 * std_error  # 95% CI
        return mean_payoff, std_error, ci_half_width


    def _simulate_gbm_with_randoms(self, contract: OptionContract, Z: np.ndarray) -> np.ndarray:
        """Simulate GBM terminal prices using provided random numbers (shape: [n_paths, n_steps])."""
        dt = contract.time_to_expiry / self.n_steps
        drift_term = (contract.risk_free_rate - 0.5 * contract.volatility**2) * dt
        vol_term = contract.volatility * np.sqrt(dt)
        log_prices = np.cumsum(drift_term + vol_term * Z, axis=1)
        terminal_prices = contract.spot * np.exp(log_prices[:, -1])
        return terminal_prices


    def _apply_control_variates_on_price(self, contract: OptionContract, payoffs: np.ndarray, final_prices: np.ndarray) -> np.ndarray:
        """
        Apply control variate variance reduction on *payoffs* using S_T with known E[S_T] = S0 * e^{rT}.
        Beta is taken from BS delta sign (kept for compatibility with original code).
        """
        control_expectation = contract.spot * np.exp(contract.risk_free_rate * contract.time_to_expiry)
        control_deviations = final_prices - control_expectation
        beta = self._get_control_beta_from_bs(contract)
        return payoffs + beta * control_deviations

    def _get_control_beta_from_bs(self, contract: OptionContract) -> float:
        """Get beta coefficient from Black–Scholes delta (matching original design)."""
        bs_engine = BlackScholesEngine()
        bs_greeks = bs_engine.greeks(contract)
        return -bs_greeks.delta_call if contract.option == OptionType.CALL else -bs_greeks.delta_put


    def _pathwise_delta_samples(self, contract: OptionContract, final_prices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Return *per-path* discounted pathwise delta samples for call and put.
        These are averaged to produce Monte Carlo delta estimates.
        """
        df = np.exp(-contract.risk_free_rate * contract.time_to_expiry)
        price_deriv = final_prices / contract.spot  # dS_T/dS_0 under GBM pathwise
        call_samples = df * price_deriv * (final_prices > contract.strike).astype(float)
        put_samples = df * price_deriv * -(final_prices < contract.strike).astype(float)
        return call_samples, put_samples

    def _apply_control_to_delta(self, contract: OptionContract, final_prices: np.ndarray, delta_samples: np.ndarray) -> np.ndarray:
        """
        Control variate for delta samples using C = S_T/S_0 with E[C] = exp(rT).
        Adjusts each sample: delta_i + beta*(C_i - E[C]).
        """
        C = final_prices / contract.spot
        C_mean = np.exp(contract.risk_free_rate * contract.time_to_expiry)
        # Robust to tiny variance
        var_C = np.var(C, ddof=1)
        if var_C == 0.0:
            return delta_samples
        cov = np.cov(delta_samples, C, ddof=1)[0, 1]
        beta = cov / var_C
        return delta_samples + beta * (C - C_mean)

    # (Optional) legacy mean-only API retained for compatibility
    def _calculate_pathwise_delta(self, contract: OptionContract, final_prices: np.ndarray) -> tuple[float, float]:
        """Compute mean pathwise deltas (kept for compatibility; greeks() uses sample-based version)."""
        call_samples, put_samples = self._pathwise_delta_samples(contract, final_prices)
        return float(np.mean(call_samples)), float(np.mean(put_samples))
