#!/usr/bin/env python3
"""
Interactive PDE convergence visualization example
Run this to see plots displayed in your environment
"""

from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.utils.convergence import run_pde_convergence_study
from quantlib.utils.visualization import plot_pde_convergence, plot_pde_timing_analysis
import matplotlib.pyplot as plt

def main():
    # Create test contract
    contract = OptionContract(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option=OptionType.CALL,
        style=ExerciseStyle.EUROPEAN
    )
    
    print("🔄 Running PDE convergence analysis...")
    
    # Run convergence study
    report = run_pde_convergence_study(
        contract,
        spatial_sizes=[25, 50, 100, 200],
        temporal_sizes=[100, 250, 500, 1000]
    )
    
    print(f"📈 Analysis Results:")
    print(f"   Reference Price: {report.reference_price:.6f}")
    print(f"   Spatial Order: {report.spatial_order:.3f} (expected ~2.0 for O(h²))")
    print(f"   Temporal Order: {report.temporal_order:.3f} (expected ~1.0 for O(Δt))")
    print(f"   Convergence Passed: {report.passed}")
    
    # Create and display plots
    print("\n📊 Displaying convergence analysis plot...")
    fig1, axes1 = plot_pde_convergence(report)
    plt.show()  # This will display the plot if you're in an interactive environment
    
    print("📊 Displaying timing analysis plot...")
    fig2, axes2 = plot_pde_timing_analysis(report)
    plt.show()  # This will display the plot if you're in an interactive environment
    
    print("\n✅ Visualization complete!")
    print("💡 Tip: If plots don't display, they are saved to 'pde_convergence_plots/' directory")

if __name__ == "__main__":
    main()