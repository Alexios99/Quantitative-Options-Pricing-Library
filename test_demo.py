from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.monte_carlo import MonteCarloEngine

# Create a simple call option
contract = OptionContract(
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.2,
    option=OptionType.CALL,
    style=ExerciseStyle.EUROPEAN
)

# Test basic Monte Carlo
mc_engine = MonteCarloEngine(n_steps=50, n_paths=10000, seed=42)
result = mc_engine.price(contract)

print(f"MC Price: {result.price:.4f}")
print(f"Std Error: {result.standard_error:.4f}")
print(f"CI Half-Width: {result.confidence_interval:.4f}")

# Compare standard errors
mc_normal = MonteCarloEngine(n_steps=50, n_paths=10000, seed=42)
mc_antithetic = MonteCarloEngine(n_steps=50, n_paths=10000, variance_reduction="antithetic", seed=42)

result_normal = mc_normal.price(contract)
result_antithetic = mc_antithetic.price(contract)

print(f"Normal Std Error: {result_normal.standard_error:.6f}")
print(f"Antithetic Std Error: {result_antithetic.standard_error:.6f}")
print(f"Variance Reduction: {(1 - result_antithetic.standard_error/result_normal.standard_error)*100:.1f}%")