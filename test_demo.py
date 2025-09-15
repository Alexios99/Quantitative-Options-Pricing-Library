from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.utils.convergence import run_monte_carlo_study
from quantlib.utils.visualization import plot_monte_carlo_convergence
import matplotlib.pyplot as plt

# Create option contract
contract = OptionContract(
    spot=100.0,
    strike=110.0,
    time_to_expiry=0.25,
    risk_free_rate=0.05,
    volatility=0.2,
    option=OptionType.PUT,
    style=ExerciseStyle.EUROPEAN  # Start with European
)

# Run convergence study
report = run_monte_carlo_study(
    contract=contract,
    path_counts=[1000, 2000, 5000, 10000, 20000, 50000],
    n_steps=50,
    variance_reduction=None,  # Plain MC
    n_trials=5
)

# Plot results
fig, axes = plot_monte_carlo_convergence(report, save_path="mc_convergence.png")
plt.show()

# Check results
print(f"Final error: {report.df['abs_err'].iloc[-1]:.6f}")
print(f"Convergence order: {report.order_estimate:.3f} (should be ~0.5)")
print(f"Passed: {report.passed}")

from quantlib.utils.visualization import compare_monte_carlo_methods

# European option (all methods should work)
european_contract = OptionContract(
    spot=100.0, strike=105.0, time_to_expiry=0.5,
    risk_free_rate=0.05, volatility=0.25,
    option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
)

fig, axes = compare_monte_carlo_methods(
    contract=european_contract,
    path_counts=[1000, 2000, 5000, 10000, 20000],
    save_path="mc_variance_reduction.png"
)
plt.show()

# American option (only plain and antithetic should work)
american_put = OptionContract(
    spot=100.0, strike=110.0, time_to_expiry=1.0,
    risk_free_rate=0.05, volatility=0.3,
    option=OptionType.PUT, style=ExerciseStyle.AMERICAN
)

# Test different variance reduction methods
for method in [None, "antithetic"]:
    print(f"\n=== {method or 'Plain'} MC ===")
    report = run_monte_carlo_study(
        contract=american_put,
        path_counts=[2000, 5000, 10000, 20000],
        variance_reduction=method,
        n_trials=3
    )
    
    print(f"Final error: {report.df['abs_err'].iloc[-1]:.4f}")
    print(f"Order: {report.order_estimate:.3f}")
    
    plot_monte_carlo_convergence(report, save_path=f"american_{method or 'plain'}_mc.png")
    plt.show()

# test_mc_convergence.py
import matplotlib.pyplot as plt
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.utils.convergence import run_monte_carlo_study
from quantlib.utils.visualization import plot_monte_carlo_convergence, compare_monte_carlo_methods

def test_monte_carlo_convergence():
    """Quick test of Monte Carlo convergence analysis"""
    
    # Simple European put
    contract = OptionContract(
        spot=100.0, strike=105.0, time_to_expiry=0.25,
        risk_free_rate=0.05, volatility=0.2,
        option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
    )
    
    print("Running Monte Carlo convergence study...")
    
    # Quick study with fewer paths for testing
    report = run_monte_carlo_study(
        contract=contract,
        path_counts=[1000, 2000, 5000, 10000],
        n_steps=25,
        variance_reduction=None,
        n_trials=3
    )
    
    print(f"\nResults:")
    print(f"Reference price: {report.reference_price:.6f}")
    print(f"Final error: {report.df['abs_err'].iloc[-1]:.6f}")
    print(f"Convergence order: {report.order_estimate:.3f}")
    print(f"Test passed: {report.passed}")
    
    # Plot
    fig, axes = plot_monte_carlo_convergence(report)
    plt.show()
    
    return report


report = test_monte_carlo_convergence()

def comprehensive_comparison():
    """Compare MC against other methods"""
    
    contract = OptionContract(
        spot=100.0, strike=100.0, time_to_expiry=0.5,
        risk_free_rate=0.05, volatility=0.2,
        option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
    )
    
    # Monte Carlo
    mc_report = run_monte_carlo_study(contract, path_counts=[5000, 10000, 20000, 50000])
    
    # Trees (if you have them)
    from quantlib.utils.convergence import run_study
    from quantlib.pricing.trees import BinomialTreeEngine
    
    tree_report = run_study(contract, BinomialTreeEngine, [50, 100, 200, 500])
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # MC convergence
    mc_errors = (mc_report.df['abs_err'] / mc_report.reference_price) * 10000
    ax1.loglog(mc_report.df['N'], mc_errors, 'o-', label='Monte Carlo')
    
    # Tree convergence  
    tree_errors = (tree_report.df['abs_err'] / tree_report.reference_price) * 10000
    ax1.loglog(tree_report.df['N'], tree_errors, 's-', label='Binomial Tree')
    
    ax1.set_xlabel('N (Paths/Steps)')
    ax1.set_ylabel('Error (bp)')
    ax1.set_title('Method Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

comprehensive_comparison()