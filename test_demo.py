# test_demo.py (in project root)
from quantlib.core.payoffs import OptionContract, ExerciseStyle, OptionType
from quantlib.pricing.trees import BinomialTreeEngine
from quantlib.utils.convergence import run_study, UseEngineRichardson
from quantlib.utils.visualization import plot_convergence_summary, plot_convergence, plot_log_convergence, plot_richardson_comparison

# Create a test contract
test_contract = OptionContract(
    spot=100.0,
    strike=100.0,
    time_to_expiry=1.0,
    risk_free_rate=0.05,
    volatility=0.2,
    option=OptionType.CALL,
    style=ExerciseStyle.EUROPEAN
)

# Run convergence study with external Richardson
grid = [25, 50, 100, 200, 400]
report = run_study(
    test_contract, 
    BinomialTreeEngine, 
    grid, 
    UseEngineRichardson.OFF  # This enables external Richardson
)



plot_convergence_summary(report, include_stats=True, save_path="graph/convergence_summary")
plot_convergence(report, save_path="graph/convergence")
plot_log_convergence(report, save_path="graph/log_convergence")
plot_richardson_comparison(report, save_path="graph/richardson_comparison")