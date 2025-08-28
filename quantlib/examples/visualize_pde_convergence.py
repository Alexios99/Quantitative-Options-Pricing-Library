#!/usr/bin/env python3
"""
Generate and save PDE convergence visualization plots
"""

from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.utils.convergence import run_pde_convergence_study
from quantlib.utils.visualization import (
    plot_pde_convergence, 
    plot_pde_timing_analysis, 
    compare_methods_convergence
)
import matplotlib.pyplot as plt
import os

def main():
    # Create output directory
    output_dir = "pde_convergence_plots"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create test contracts
    european_call = OptionContract(
        spot=100.0,
        strike=100.0,
        time_to_expiry=1.0,
        risk_free_rate=0.05,
        volatility=0.2,
        option=OptionType.CALL,
        style=ExerciseStyle.EUROPEAN
    )
    
    american_put = OptionContract(
        spot=100.0,
        strike=110.0,
        time_to_expiry=0.5,
        risk_free_rate=0.05,
        volatility=0.3,
        option=OptionType.PUT,
        style=ExerciseStyle.AMERICAN
    )
    
    print("🔄 Running PDE convergence analysis...")
    
    # 1. European Call PDE Convergence
    print("   📊 European Call convergence...")
    eu_report = run_pde_convergence_study(
        european_call,
        spatial_sizes=[25, 50, 100, 200],
        temporal_sizes=[100, 250, 500, 1000]
    )
    
    # Plot main convergence analysis
    fig1, axes1 = plot_pde_convergence(eu_report)
    fig1.savefig(f"{output_dir}/european_call_pde_convergence.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/european_call_pde_convergence.png")
    plt.close(fig1)
    
    # Plot timing analysis
    fig2, axes2 = plot_pde_timing_analysis(eu_report)
    fig2.savefig(f"{output_dir}/european_call_timing.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/european_call_timing.png")
    plt.close(fig2)
    
    # 2. American Put PDE Convergence
    print("   📊 American Put convergence...")
    am_report = run_pde_convergence_study(
        american_put,
        spatial_sizes=[30, 60, 120],
        temporal_sizes=[150, 300, 600]
    )
    
    fig3, axes3 = plot_pde_convergence(am_report)
    fig3.savefig(f"{output_dir}/american_put_pde_convergence.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/american_put_pde_convergence.png")
    plt.close(fig3)
    
    # 3. Method Comparison
    print("   📊 Method comparison...")
    fig4, axes4, tree_report, pde_report = compare_methods_convergence(
        european_call,
        tree_ns=[50, 100, 200, 500],
        pde_spatial=[50, 100, 200],
        pde_temporal=[250, 500, 1000]
    )
    fig4.savefig(f"{output_dir}/method_comparison.png", dpi=300, bbox_inches='tight')
    print(f"   ✅ Saved: {output_dir}/method_comparison.png")
    plt.close(fig4)
    
    # Print summary
    print("\n📈 CONVERGENCE SUMMARY:")
    print(f"European Call (PDE):")
    print(f"  Reference Price: {eu_report.reference_price:.6f}")
    print(f"  Spatial Order: {eu_report.spatial_order:.3f} (expected ~2.0)")
    print(f"  Temporal Order: {eu_report.temporal_order:.3f} (expected ~1.0)")
    print(f"  Final Spatial Error: {eu_report.spatial_df['abs_err'].iloc[-1]:.6f}")
    print(f"  Final Temporal Error: {eu_report.temporal_df['abs_err'].iloc[-1]:.6f}")
    
    print(f"\nAmerican Put (PDE):")
    print(f"  Reference Price: {am_report.reference_price:.6f}")
    print(f"  Spatial Order: {am_report.spatial_order:.3f}")
    print(f"  Temporal Order: {am_report.temporal_order:.3f}")
    
    print(f"\n🎯 All plots saved to '{output_dir}/' directory")
    print("   You can now view the PNG files to see the convergence visualizations!")

if __name__ == "__main__":
    main()