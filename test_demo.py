# Add this to your test_demo.py
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.monte_carlo import MonteCarloEngine
from quantlib.core.stochastic_processes import GeometricBrownianMotion
from quantlib.pricing.analytical import BlackScholesEngine
import numpy as np

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

# Calculate Greeks using the antithetic Monte Carlo engine
mc_greeks_antithetic = mc_antithetic.greeks(contract)

# Print and compare delta_call and delta_put for all methods
print("Black-Scholes Engine:")
print(f"Delta Call: {bs_greeks.delta_call}, Delta Put: {bs_greeks.delta_put}")

print("\nMonte Carlo (Normal):")
print(f"Delta Call: {mc_greeks.delta_call}, Delta Put: {mc_greeks.delta_put}")

print("\nMonte Carlo (Control Variate):")
print(f"Delta Call: {mc_greeks_control.delta_call}, Delta Put: {mc_greeks_control.delta_put}")

print("\nMonte Carlo (Antithetic):")
print(f"Delta Call: {mc_greeks_antithetic.delta_call}, Delta Put: {mc_greeks_antithetic.delta_put}")

# Compare Monte Carlo results to Black-Scholes results as percentage differences
def compare_greeks_percentage(bs_greeks, mc_greeks, method_name):
    delta_call_diff = abs(bs_greeks.delta_call - mc_greeks.delta_call) / abs(bs_greeks.delta_call) * 100
    delta_put_diff = abs(bs_greeks.delta_put - mc_greeks.delta_put) / abs(bs_greeks.delta_put) * 100
    print(f"\nComparison ({method_name}):")
    print(f"Delta Call Difference: {delta_call_diff:.2f}%")
    print(f"Delta Put Difference: {delta_put_diff:.2f}%")

# Perform comparisons
compare_greeks_percentage(bs_greeks, mc_greeks, "Monte Carlo (Normal)")
compare_greeks_percentage(bs_greeks, mc_greeks_control, "Monte Carlo (Control Variate)")
compare_greeks_percentage(bs_greeks, mc_greeks_antithetic, "Monte Carlo (Antithetic)")


