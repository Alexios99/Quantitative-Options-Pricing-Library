# test_demo.py (in project root)
from quantlib.core.payoffs import OptionContract, ExerciseStyle, OptionType, CallPayoff
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.pricing.trees import BinomialTreeEngine
from quantlib.utils.convergence import run_study, UseEngineRichardson
from quantlib.utils.visualization import plot_convergence_summary, plot_convergence, plot_log_convergence, plot_richardson_comparison
from quantlib.pricing.pde import PDEEngine
from types import SimpleNamespace

# 1. Fake parameter objects
grid_params = SimpleNamespace(
    n_space=50,
    n_time=1000,
    s_max=150,
    theta=0.5
)
solver_params = SimpleNamespace(
    psor_omega=1.2,
    psor_tol=1e-6,
    psor_max_iter=1000
)

# 2. Option contract for a European call
contract = OptionContract(
    option=OptionType.CALL,
    strike=100,
    spot=100,
    risk_free_rate=0.05,
    volatility=0.2,
    time_to_expiry=1.0,
    style=ExerciseStyle.EUROPEAN
)

# 3. Instantiate engine and run pricing
engine = PDEEngine(grid_params, solver_params)


# Test both engines
bs_engine = BlackScholesEngine()
bs_result = bs_engine.price(contract)
pde_result = engine.price(contract)

print(f"Black-Scholes: {bs_result.price:.6f}")
print(f"PDE:          {pde_result.price:.6f}")
print(f"Difference:   {abs(bs_result.price - pde_result.price):.6f}")
