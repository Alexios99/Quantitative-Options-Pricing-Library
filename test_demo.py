# Add this to your test_demo.py
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.monte_carlo import MonteCarloEngine
from quantlib.core.stochastic_processes import GeometricBrownianMotion
from quantlib.pricing.analytical import BlackScholesEngine
import numpy as np
from dataclasses import replace

# Create engines
mc_normal = MonteCarloEngine(n_steps=50, n_paths=10000, variance_reduction=None, seed=42)
mc_control = MonteCarloEngine(n_steps=50, n_paths=10000, variance_reduction="control", seed=42)
bs_engine = BlackScholesEngine()
# Standard call option
contract = OptionContract(
    spot=100.0, strike=100.0, time_to_expiry=1.0,
    risk_free_rate=0.05, volatility=0.2,
    option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
)


mc_greeks = mc_normal.greeks(contract)
mc_greeks_control = mc_control.greeks(contract)
bs_greeks = bs_engine.greeks(contract)

# Add Monte Carlo engine with antithetic variance reduction
mc_antithetic = MonteCarloEngine(n_steps=50, n_paths=10000, variance_reduction="antithetic", seed=42)
# Test script idea
american_put = OptionContract(
    spot=100, strike=110, time_to_expiry=0.25,
    risk_free_rate=0.05, volatility=0.2,
    option=OptionType.PUT, style=ExerciseStyle.AMERICAN
)

# Should be > European equivalent
american_price = mc_normal.price(american_put).price
european_price = mc_normal.price(replace(american_put, style=ExerciseStyle.EUROPEAN)).price


print(american_price)
print(european_price)
print(f'ABS Diff of EURO VS AMERICAN {american_price-european_price}')
print(f'RELATIVE Diff of EURO VS AMERICAN {(american_price-european_price)/american_price}')

print(mc_greeks)