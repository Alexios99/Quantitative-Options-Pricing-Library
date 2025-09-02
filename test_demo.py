# Add this to your test_demo.py
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.monte_carlo import MonteCarloEngine
import numpy as np

# Test WITHOUT antithetic variates first
print("=== WITHOUT ANTITHETIC ===")
engine_small = MonteCarloEngine(n_steps=50, n_paths=1000, variance_reduction=None, seed=42)
engine_large = MonteCarloEngine(n_steps=50, n_paths=10000, variance_reduction=None, seed=123)

standard_call = OptionContract(
    spot=100.0, strike=100.0, time_to_expiry=1.0,
    risk_free_rate=0.05, volatility=0.2,
    option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
)

result_small = engine_small.price(standard_call)
result_large = engine_large.price(standard_call)

print(f"Small: {result_small.n_paths} paths, SE: {result_small.standard_error:.6f}")
print(f"Large: {result_large.n_paths} paths, SE: {result_large.standard_error:.6f}")
print(f"Ratio: {result_small.standard_error / result_large.standard_error:.6f}")
print(f"Expected: {np.sqrt(1000/10000):.6f}")

# Test WITH antithetic variates
print("\n=== WITH ANTITHETIC ===")
engine_small_ant = MonteCarloEngine(n_steps=50, n_paths=1000, variance_reduction="antithetic", seed=42)
engine_large_ant = MonteCarloEngine(n_steps=50, n_paths=10000, variance_reduction="antithetic", seed=123)

result_small_ant = engine_small_ant.price(standard_call)
result_large_ant = engine_large_ant.price(standard_call)

print(f"Small: {result_small_ant.n_paths} paths, SE: {result_small_ant.standard_error:.6f}")
print(f"Large: {result_large_ant.n_paths} paths, SE: {result_large_ant.standard_error:.6f}")
print(f"Ratio: {result_small_ant.standard_error / result_large_ant.standard_error:.6f}")
print(f"Expected: {np.sqrt(result_small_ant.n_paths/result_large_ant.n_paths):.6f}")