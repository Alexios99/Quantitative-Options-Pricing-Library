"""
Demo 1: Standard Option Pricing
------------------------------
This script demonstrates how to price European and American options using
various engines available in the library:
1. Analytical (Black-Scholes)
2. Monte Carlo Simulation
3. Binomial Trees (CRR)
4. Finite Difference (PDE)
"""

from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.pricing.monte_carlo import MonteCarloEngine
from quantlib.pricing.trees import BinomialTreeEngine
from dataclasses import replace
from quantlib.pricing.pde import PDEEngine
from types import SimpleNamespace

# ==========================================
# 1. Define the Option Contract
# ==========================================
# European Call Option
# Spot=100, Strike=100, T=1 year, r=5%, vol=20%
european_call = OptionContract(
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
    option=OptionType.CALL,
    style=ExerciseStyle.EUROPEAN
)

print(f"--- Option Contract ---")
print(f"Type: {european_call.option.value} {european_call.style.value}")
print(f"Spot: {european_call.spot}, Strike: {european_call.strike}")
print(f"Expiry: {european_call.time_to_expiry} yrs, Vol: {european_call.volatility:.1%}")
print("-" * 30)

# ==========================================
# 2. Analytical Pricing (Black-Scholes)
# ==========================================
bs_engine = BlackScholesEngine()
bs_result = bs_engine.price(european_call)
bs_greeks = bs_engine.greeks(european_call)

print(f"\n[1] Analytical (Black-Scholes)")
print(f"Price: {bs_result.price:.4f}")
print(f"Delta: {bs_greeks.delta_call:.4f}")
print(f"Gamma: {bs_greeks.gamma:.4f}")
print(f"Vega:  {bs_greeks.vega:.4f}")

# ==========================================
# 3. Monte Carlo Simulation
# ==========================================
# 100,000 simulations, 50 time steps
mc_engine = MonteCarloEngine(n_paths=100000, n_steps=50)
mc_result = mc_engine.price(european_call)

print(f"\n[2] Monte Carlo Simulation")
print(f"Price: {mc_result.price:.4f} +/- {mc_result.standard_error:.4f}")


# ==========================================
# 4. Binomial Tree (Cox-Ross-Rubinstein)
# ==========================================
# Useful for American options, but we test on European first for comparison
tree_engine = BinomialTreeEngine(n_steps=500)
tree_result = tree_engine.price(european_call)

print(f"\n[3] Binomial Tree (CRR, N=500)")
print(f"Price: {tree_result.price:.4f}")

# ==========================================
# 5. Pricing an American Put (Early Exercise)
# ==========================================
# American Put often has early exercise value (Premium > European)
american_put = OptionContract(
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.20,
    option=OptionType.PUT,
    style=ExerciseStyle.AMERICAN
)

# Compare European Put (BS) vs American Put (Tree)
euro_put = replace(american_put, style=ExerciseStyle.EUROPEAN)
euro_price = bs_engine.price(euro_put).price
amer_price = tree_engine.price(american_put).price

print(f"\n[4] American vs European Put")
print(f"European Put (BS):   {euro_price:.4f}")
print(f"American Put (Tree): {amer_price:.4f}")
print(f"Early Exercise Premium: {amer_price - euro_price:.4f}")

# ==========================================
# 6. Finite Difference Method (PDE)
# ==========================================
# Solving the Black-Scholes PDE numerically
grid_params = SimpleNamespace(
    n_space=100,
    n_time=100,
    s_max=200,   # Needs to be sufficient (e.g. 2*Spot)
    theta=0.5    # Crank-Nicolson
)

solver_params = SimpleNamespace(
    psor_omega=1.2,
    psor_tol=1e-6,
    psor_max_iter=100
)

pde_engine = PDEEngine(grid_params=grid_params, solver_params=solver_params)
pde_result = pde_engine.price(american_put)

print(f"\n[5] Finite Difference (PDE)")
print(f"American Put Price: {pde_result.price:.4f}")
