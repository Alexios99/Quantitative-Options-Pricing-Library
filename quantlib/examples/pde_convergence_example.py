#!/usr/bin/env python3
"""
Example demonstrating PDE convergence analysis integration
"""

from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.utils.convergence import run_pde_convergence_study
from quantlib.utils.visualization import plot_pde_convergence, compare_methods_convergence
import matplotlib.pyplot as plt

def main():
    # Create test option contract
    contract = OptionContract(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option=OptionType.CALL,
        style=ExerciseStyle.EUROPEAN
    )
    
    print("Running PDE convergence analysis...")
    
    # Run PDE convergence study
    pde_report = run_pde_convergence_study(
        contract,
        spatial_sizes=[25, 50, 100, 200],
        temporal_sizes=[100, 250, 500, 1000]
    )
    
    print(f"Reference price: {pde_report.reference_price:.6f}")
    print(f"Spatial convergence order: {pde_report.spatial_order:.3f}")
    print(f"Temporal convergence order: {pde_report.temporal_order:.3f}")
    print(f"Convergence test passed: {pde_report.passed}")
    
    # Plot PDE convergence
    fig1, axes1 = plot_pde_convergence(pde_report)
    plt.show()
    
    # Compare methods
    print("\nComparing PDE vs Tree methods...")
    fig2, axes2, tree_report, pde_report2 = compare_methods_convergence(
        contract,
        tree_ns=[50, 100, 200, 500],
        pde_spatial=[50, 100, 200],
        pde_temporal=[250, 500, 1000]
    )
    plt.show()
    
    print(f"Tree final error: {tree_report.df['abs_err'].iloc[-1]:.6f}")
    print(f"PDE spatial final error: {pde_report2.spatial_df['abs_err'].iloc[-1]:.6f}")
    print(f"PDE temporal final error: {pde_report2.temporal_df['abs_err'].iloc[-1]:.6f}")

if __name__ == "__main__":
    main()