# In quantlib/utils/visualization.py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Optional, List, Tuple
from quantlib.utils.convergence import ConvergenceReport, PDEConvergenceReport


def plot_convergence(report: ConvergenceReport, save_path: Optional[str] = None):
    """Plot convergence error vs N"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ns = report.df['N'].values
    abs_errors = report.df['abs_err'].values
    
    bp_errors = (abs_errors / report.reference_price) * 10000
    
    ax.plot(ns, bp_errors, 'o-', linewidth=2, markersize=6, color='steelblue', label='Tree Error')
    
    # Richardson extrapolation if available
    if 'price_RE' in report.df.columns and report.df['price_RE'].notna().any():
        richardson_mask = report.df['price_RE'].notna()
        re_errors = np.abs(report.df.loc[richardson_mask, 'price_RE'] - report.reference_price)
        re_bp_errors = (re_errors / report.reference_price) * 10000
        re_ns = report.df.loc[richardson_mask, 'N'].values
        
        ax.plot(re_ns, re_bp_errors, 's--', linewidth=2, markersize=6, color='darkred', label='Richardson Extrapolation')
    
    ax.set_xlabel('Number of Steps (N)', fontsize=12)
    ax.set_ylabel('Absolute Error (basis points)', fontsize=12)
    ax.set_title(f'Convergence Analysis\nReference Price: {report.reference_price:.4f}', fontsize=14, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    ax.text(0.98, 0.02, 'bp = 1e4 × |P_num - P_ref| / P_ref', 
            transform=ax.transAxes, fontsize=8, 
            horizontalalignment='right', style='italic', alpha=0.7)
    
    plt.tight_layout()
    
    textstr = f'Order Estimate: {report.order_estimate:.2f}\n'
    if report.richardson_gain is not None:
        textstr += f'Richardson Gain: {report.richardson_gain:.1f}x'
    
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def plot_log_convergence(report: ConvergenceReport, save_path: Optional[str] = None):
    """Log-log plot to visualize order of convergence"""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ns = report.df['N'].values
    abs_errors = report.df['abs_err'].values
    bp_errors = (abs_errors / report.reference_price) * 10000
    
    ax.loglog(ns, bp_errors, 'o-', linewidth=2, markersize=6, color='steelblue', label='Tree Error')
    
    if len(ns) >= 2:
        # Fit line to last few points
        last_k = min(4, len(ns))
        fit_ns = ns[-last_k:]
        fit_errors = bp_errors[-last_k:]
        
        # Power law fit in log space
        log_ns = np.log(fit_ns)
        log_errors = np.log(fit_errors)
        slope, intercept = np.polyfit(log_ns, log_errors, 1)
        
        extended_ns = np.logspace(np.log10(ns[0]), np.log10(ns[-1]), 100)
        fitted_errors = np.exp(intercept) * (extended_ns ** slope)
        
        ax.loglog(extended_ns, fitted_errors, '--', color='gray', alpha=0.7, label=f'Fitted: O(N^{slope:.2f})')
    
    ax.set_xlabel('Number of Steps (N)', fontsize=12)
    ax.set_ylabel('Absolute Error (basis points)', fontsize=12)
    ax.set_title('Log-Log Convergence Analysis', fontsize=14, pad=20)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)

    ax.text(0.98, 0.02, 'bp = 1e4 × |P_num - P_ref| / P_ref', 
            transform=ax.transAxes, fontsize=8, 
            horizontalalignment='right', style='italic', alpha=0.7)
    
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def format_bp_error(abs_error: float, reference_price: float) -> str:
    """Format absolute error as basis points with appropriate precision"""
    bp_error = (abs_error / reference_price) * 10000
    if bp_error < 0.01:
        return f"{bp_error:.3f} bp"
    elif bp_error < 1:
        return f"{bp_error:.2f} bp"
    else:
        return f"{bp_error:.1f} bp"
    
def plot_richardson_comparison(report: ConvergenceReport, save_path: Optional[str] = None):
    """Compare base tree vs Richardson extrapolation errors"""
    
    fig, ax = plt.subplots(figsize=(10, 6))

    ns = report.df['N'].values
    prices_base = report.df['price'].values
    
    abs_errors_base = np.abs(prices_base - report.reference_price)
    bp_errors_base = (abs_errors_base / report.reference_price) * 10000
    
    ax.plot(ns, bp_errors_base, 'o-', linewidth=2, markersize=6, color='steelblue', label='Base Tree')

    if 'price_RE' in report.df.columns and report.df['price_RE'].notna().any():
        richardson_mask = report.df['price_RE'].notna()
        re_ns = report.df.loc[richardson_mask, 'N'].values
        re_prices = report.df.loc[richardson_mask, 'price_RE'].values
        
        re_abs_errors = np.abs(re_prices - report.reference_price)
        re_bp_errors = (re_abs_errors / report.reference_price) * 10000
        
        ax.plot(re_ns, re_bp_errors, 's--', linewidth=2, markersize=6,
                color='darkred', label='Richardson Extrapolation')
      
    ax.set_xlabel('Number of Steps (N)', fontsize=12)
    ax.set_ylabel('Absolute Error (basis points)', fontsize=12)
    ax.set_title('Richardson Extrapolation Comparison\n' +  f'Reference Price: {report.reference_price:.4f}', fontsize=14, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    ax.text(0.98, 0.02, 'bp = 1e4 × |P_num - P_ref| / P_ref', 
            transform=ax.transAxes, fontsize=8, 
            horizontalalignment='right', style='italic', alpha=0.7)
    
    textstr = f'Order Estimate: {report.order_estimate:.2f}\n'
    if report.richardson_gain is not None:
        textstr += f'Richardson Gain: {report.richardson_gain:.1f}x\n'
        textstr += '(Error reduction at largest N)'
    
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def _plot_convergence_subplot(ax, report):
    """Plot convergence on given axis"""
    ns = report.df['N'].values
    abs_errors = report.df['abs_err'].values
    bp_errors = (abs_errors / report.reference_price) * 10000
    
    ax.plot(ns, bp_errors, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.set_title('Linear Convergence')
    ax.set_xlabel('N')
    ax.set_ylabel('Error (bp)')
    ax.grid(True, alpha=0.3)

def _plot_loglog_subplot(ax, report):
    """Plot log-log convergence with fitted trend line"""
    ns = report.df['N'].values
    abs_errors = report.df['abs_err'].values
    bp_errors = (abs_errors / report.reference_price) * 10000
    
    # Log-log plot
    ax.loglog(ns, bp_errors, 'o-', linewidth=2, markersize=6, color='steelblue')
    
    if len(ns) >= 2:
        last_k = min(4, len(ns))
        fit_ns = ns[-last_k:]
        fit_errors = bp_errors[-last_k:]
        
        log_ns = np.log(fit_ns)
        log_errors = np.log(fit_errors)
        slope, intercept = np.polyfit(log_ns, log_errors, 1)
        
        extended_ns = np.logspace(np.log10(ns[0]), np.log10(ns[-1]), 100)
        fitted_errors = np.exp(intercept) * (extended_ns ** slope)
        
        ax.loglog(extended_ns, fitted_errors, '--', color='gray', alpha=0.7)
    
    ax.set_title('Log-Log Convergence', fontsize=11)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Error (bp)', fontsize=10)
    ax.grid(True, alpha=0.3)

def _plot_richardson_subplot(ax, report):
    """Plot Richardson comparison on given axis"""
    ns = report.df['N'].values
    abs_errors = report.df['abs_err'].values
    bp_errors = (abs_errors / report.reference_price) * 10000
    
    # Base tree errors
    ax.plot(ns, bp_errors, 'o-', linewidth=2, markersize=5, 
            color='steelblue', label='Base')
    
    # Richardson if available
    if 'price_RE' in report.df.columns and report.df['price_RE'].notna().any():
        richardson_mask = report.df['price_RE'].notna()
        re_ns = report.df.loc[richardson_mask, 'N'].values
        re_prices = report.df.loc[richardson_mask, 'price_RE'].values
        re_abs_errors = np.abs(re_prices - report.reference_price)
        re_bp_errors = (re_abs_errors / report.reference_price) * 10000
        
        ax.plot(re_ns, re_bp_errors, 's--', linewidth=2, markersize=5,
                color='darkred', label='Richardson')
    
    ax.set_title('Richardson Comparison', fontsize=11)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Error (bp)', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

def _plot_stats_subplot(ax, report):
    """Display key statistics as text"""
    ax.axis('off')
    
    final_error_bp = (report.df.abs_err.iloc[-1] / report.reference_price) * 10000
    
    status, status_color = _determine_status(report, final_error_bp)
    
    stats_text = f"""CONVERGENCE STATISTICS
            
    Reference Price: {report.reference_price:.6f}
    Final Error: {final_error_bp:.2f} bp
    Order Estimate: {report.order_estimate:.3f}
    Odd-Even Gap: {report.odd_even_gap:.6f}

    Engine: {report.meta['engine']}
    Grid Points: {len(report.meta['grid'])}
    Max N: {max(report.meta['grid'])}

    Status: {status}"""

    if report.richardson_gain is not None:
        stats_text += f"\n\nRichardson Gain: {report.richardson_gain:.2f}x"
    
    # Add bp definition caption
    stats_text += f"\n\nbp = 1e4 × |P_num - P_ref| / P_ref"
    
    # Display with colored status box
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor=status_color, alpha=0.8))
    
def _determine_status(report: ConvergenceReport, final_error_bp: float) -> Tuple[str, str]:
    """Determine pass/fail status and color based on Richardson/base error"""
    
    if report.richardson_gain is not None and 'price_RE' in report.df.columns:
        richardson_mask = report.df['price_RE'].notna()
        if richardson_mask.any():
            max_re_idx = report.df.loc[richardson_mask, 'N'].idxmax()
            re_price = report.df.loc[max_re_idx, 'price_RE']
            re_error_bp = abs(re_price - report.reference_price) / report.reference_price * 10000
            
            if re_error_bp <= 1.0:  # Richardson ≤ 1 bp
                return "PASSED (Richardson)", "lightgreen"
    
    if final_error_bp <= 5.0:  # Base ≤ 5 bp
        return "PASSED (Base)", "lightgreen"
    else:
        return "FAILED", "lightcoral"
    
def _plot_timing_subplot(ax, report):
    """Plot computational timing vs N"""
    ns = report.df['N'].values
    timings = report.df['ms'].values
    
    ax.plot(ns, timings, 'o-', linewidth=2, markersize=6, 
            color='green', label='Timing')
    
    ax.set_title('Computational Time', fontsize=11)
    ax.set_xlabel('N', fontsize=10)
    ax.set_ylabel('Time (ms)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Add average timing info
    avg_time = np.mean(timings)
    ax.axhline(avg_time, color='red', linestyle='--', alpha=0.7, 
               label=f'Avg: {avg_time:.1f}ms')
    ax.legend(fontsize=9)

def _check_convergence_warnings(report: ConvergenceReport) -> Optional[str]:
    """Check for convergence issues and return warning message"""
    warnings = []
    
    # Check if errors are decreasing
    errors = report.df['abs_err'].values
    if len(errors) >= 2 and errors[-1] >= errors[0]:
        warnings.append("Non-decreasing errors")
    
    # Check fitted slope range
    if not (0.5 <= report.order_estimate <= 1.2):
        warnings.append(f"Slope {report.order_estimate:.2f} outside [0.5, 1.2]")
    
    return " | ".join(warnings) if warnings else None

def plot_convergence_summary(report: ConvergenceReport, include_stats: bool = True, save_path: Optional[str] = None):
    """Comprehensive convergence analysis dashboard"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Check for convergence warnings
    warning_msg = _check_convergence_warnings(report)
    
    _plot_convergence_subplot(ax1, report)
    _plot_loglog_subplot(ax2, report)
    _plot_richardson_subplot(ax3, report)
    
    if include_stats:
        _plot_stats_subplot(ax4, report)
    else:
        _plot_timing_subplot(ax4, report)
    
    title = f'Convergence Analysis Summary\nEngine: {report.meta["engine"]} | Reference: {report.reference_price:.4f}'
    
    # Add warning banner if needed
    if warning_msg:
        title += f'\n⚠️ {warning_msg}'
        fig.suptitle(title, fontsize=16, y=0.95, color='red')
    else:
        fig.suptitle(title, fontsize=16, y=0.95)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2, ax3, ax4)

def plot_pde_convergence(report: PDEConvergenceReport, save_path: Optional[str] = None):
    """Plot PDE spatial and temporal convergence analysis"""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Spatial convergence (linear)
    spatial_ns = report.spatial_df['n_space'].values
    spatial_errors = report.spatial_df['abs_err'].values
    spatial_bp = (spatial_errors / report.reference_price) * 10000
    
    ax1.plot(spatial_ns, spatial_bp, 'o-', linewidth=2, markersize=6, 
             color='steelblue', label='Spatial Error')
    ax1.set_xlabel('Spatial Grid Points (N_space)')
    ax1.set_ylabel('Error (basis points)')
    ax1.set_title('Spatial Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Temporal convergence (linear)
    temporal_ns = report.temporal_df['n_time'].values
    temporal_errors = report.temporal_df['abs_err'].values
    temporal_bp = (temporal_errors / report.reference_price) * 10000
    
    ax2.plot(temporal_ns, temporal_bp, 's-', linewidth=2, markersize=6, 
             color='darkred', label='Temporal Error')
    ax2.set_xlabel('Time Steps (N_time)')
    ax2.set_ylabel('Error (basis points)')
    ax2.set_title('Temporal Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Log-log spatial convergence with fitted line
    ax3.loglog(spatial_ns, spatial_bp, 'o-', linewidth=2, markersize=6, 
               color='steelblue', label='Spatial')
    
    if len(spatial_ns) >= 2:
        # Fit theoretical O(h²) line
        log_ns = np.log(spatial_ns)
        log_errors = np.log(spatial_bp)
        slope, intercept = np.polyfit(log_ns, log_errors, 1)
        
        extended_ns = np.logspace(np.log10(spatial_ns[0]), np.log10(spatial_ns[-1]), 100)
        fitted_errors = np.exp(intercept) * (extended_ns ** slope)
        
        ax3.loglog(extended_ns, fitted_errors, '--', color='gray', alpha=0.7, 
                   label=f'Fitted: O(N^{slope:.2f})')
        
        # Theoretical O(h²) reference
        h_values = 1.0 / spatial_ns  # h ∝ 1/N for uniform grid
        theoretical = spatial_bp[0] * (h_values / h_values[0])**2
        ax3.loglog(spatial_ns, theoretical, ':', color='green', alpha=0.7, 
                   label='Theoretical O(h²)')
    
    ax3.set_xlabel('Spatial Grid Points')
    ax3.set_ylabel('Error (basis points)')
    ax3.set_title('Log-Log Spatial Convergence')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Statistics and summary
    ax4.axis('off')
    
    final_spatial_bp = spatial_bp[-1]
    final_temporal_bp = temporal_bp[-1]
    
    status = "PASSED" if report.passed else "FAILED"
    status_color = "lightgreen" if report.passed else "lightcoral"
    
    stats_text = f"""PDE CONVERGENCE ANALYSIS

Reference Price: {report.reference_price:.6f}

SPATIAL CONVERGENCE:
Final Error: {final_spatial_bp:.2f} bp
Order Estimate: {report.spatial_order:.3f}
Expected: ~2.0 (O(h²))

TEMPORAL CONVERGENCE:
Final Error: {final_temporal_bp:.2f} bp
Order Estimate: {report.temporal_order:.3f}
Expected: ~1.0 (O(Δt)) for θ={report.meta['theta']}

GRID PARAMETERS:
S_max: {report.meta['s_max']}
Theta: {report.meta['theta']} ({'Crank-Nicolson' if report.meta['theta'] == 0.5 else 'Implicit' if report.meta['theta'] == 1.0 else 'Mixed'})

Status: {status}

bp = 1e4 × |P_PDE - P_ref| / P_ref"""
    
    ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor=status_color, alpha=0.8))
    
    fig.suptitle(f'PDE Convergence Analysis\nReference: {report.reference_price:.4f}', 
                 fontsize=16, y=0.95)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2, ax3, ax4)

def plot_pde_timing_analysis(report: PDEConvergenceReport, save_path: Optional[str] = None):
    """Plot PDE computational timing vs grid size"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Spatial timing
    spatial_ns = report.spatial_df['n_space'].values
    spatial_times = report.spatial_df['ms'].values
    
    ax1.plot(spatial_ns, spatial_times, 'o-', linewidth=2, markersize=6, 
             color='steelblue', label='Spatial')
    ax1.set_xlabel('Spatial Grid Points')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Spatial Grid Timing')
    ax1.grid(True, alpha=0.3)
    
    # Fit timing complexity
    if len(spatial_ns) >= 2:
        log_ns = np.log(spatial_ns)
        log_times = np.log(spatial_times)
        slope, intercept = np.polyfit(log_ns, log_times, 1)
        
        extended_ns = np.logspace(np.log10(spatial_ns[0]), np.log10(spatial_ns[-1]), 100)
        fitted_times = np.exp(intercept) * (extended_ns ** slope)
        
        ax1.loglog(spatial_ns, spatial_times, 'o-', color='steelblue')
        ax1.loglog(extended_ns, fitted_times, '--', color='gray', alpha=0.7, 
                   label=f'O(N^{slope:.2f})')
        ax1.legend()
        ax1.set_title(f'Spatial Timing: O(N^{slope:.2f})')
    
    # Temporal timing
    temporal_ns = report.temporal_df['n_time'].values
    temporal_times = report.temporal_df['ms'].values
    
    ax2.plot(temporal_ns, temporal_times, 's-', linewidth=2, markersize=6, 
             color='darkred', label='Temporal')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Time (ms)')
    ax2.set_title('Temporal Grid Timing')
    ax2.grid(True, alpha=0.3)
    
    if len(temporal_ns) >= 2:
        log_ns = np.log(temporal_ns)
        log_times = np.log(temporal_times)
        slope, intercept = np.polyfit(log_ns, log_times, 1)
        
        extended_ns = np.logspace(np.log10(temporal_ns[0]), np.log10(temporal_ns[-1]), 100)
        fitted_times = np.exp(intercept) * (extended_ns ** slope)
        
        ax2.loglog(temporal_ns, temporal_times, 's-', color='darkred')
        ax2.loglog(extended_ns, fitted_times, '--', color='gray', alpha=0.7, 
                   label=f'O(N^{slope:.2f})')
        ax2.legend()
        ax2.set_title(f'Temporal Timing: O(N^{slope:.2f})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2)

def compare_methods_convergence(contract, tree_ns=None, pde_spatial=None, pde_temporal=None, save_path: Optional[str] = None):
    """Compare convergence of different pricing methods"""
    
    if tree_ns is None:
        tree_ns = [50, 100, 200, 500, 1000]
    if pde_spatial is None:
        pde_spatial = [50, 100, 200, 400]
    if pde_temporal is None:
        pde_temporal = [250, 500, 1000, 2000]
    
    # Import here to avoid circular imports
    from quantlib.pricing.trees import BinomialTreeEngine
    from quantlib.utils.convergence import run_study, run_pde_convergence_study, reference_price
    
    # Get reference price
    ref_price = reference_price(contract)
    
    # Run tree convergence
    tree_report = run_study(contract, BinomialTreeEngine, tree_ns)
    
    # Run PDE convergence
    pde_report = run_pde_convergence_study(contract, 
                                          spatial_sizes=pde_spatial,
                                          temporal_sizes=pde_temporal)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Tree convergence
    tree_errors = (tree_report.df['abs_err'] / ref_price) * 10000
    ax1.plot(tree_report.df['N'], tree_errors, 'o-', linewidth=2, 
             color='green', label='Binomial Tree')
    ax1.set_xlabel('Steps (N)')
    ax1.set_ylabel('Error (basis points)')
    ax1.set_title('Tree Method Convergence')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # PDE spatial convergence
    pde_spatial_errors = (pde_report.spatial_df['abs_err'] / ref_price) * 10000
    ax2.plot(pde_report.spatial_df['n_space'], pde_spatial_errors, 's-', 
             linewidth=2, color='steelblue', label='PDE Spatial')
    ax2.set_xlabel('Spatial Points')
    ax2.set_ylabel('Error (basis points)')
    ax2.set_title('PDE Spatial Convergence')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Combined log-log comparison
    ax3.loglog(tree_report.df['N'], tree_errors, 'o-', 
               color='green', label=f'Tree O(N^{-tree_report.order_estimate:.2f})')
    ax3.loglog(pde_report.spatial_df['n_space'], pde_spatial_errors, 's-', 
               color='steelblue', label=f'PDE Spatial O(N^{-pde_report.spatial_order:.2f})')
    
    pde_temporal_errors = (pde_report.temporal_df['abs_err'] / ref_price) * 10000
    ax3.loglog(pde_report.temporal_df['n_time'], pde_temporal_errors, '^-', 
               color='darkred', label=f'PDE Temporal O(N^{-pde_report.temporal_order:.2f})')
    
    ax3.set_xlabel('Grid Points / Steps')
    ax3.set_ylabel('Error (basis points)')
    ax3.set_title('Method Comparison (Log-Log)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Summary statistics
    ax4.axis('off')
    
    summary_text = f"""METHOD COMPARISON SUMMARY

Reference Price: {ref_price:.6f}

BINOMIAL TREE:
Final Error: {tree_errors.iloc[-1]:.2f} bp
Order: {tree_report.order_estimate:.3f}
Max N: {tree_report.df['N'].iloc[-1]}

PDE METHOD:
Spatial Error: {pde_spatial_errors.iloc[-1]:.2f} bp
Temporal Error: {pde_temporal_errors.iloc[-1]:.2f} bp
Spatial Order: {pde_report.spatial_order:.3f}
Temporal Order: {pde_report.temporal_order:.3f}

PERFORMANCE:
Tree passed: {tree_report.passed}
PDE passed: {pde_report.passed}

bp = 1e4 × |P_num - P_ref| / P_ref"""
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=1', facecolor='lightblue', alpha=0.8))
    
    fig.suptitle(f'Pricing Method Convergence Comparison\n{contract.option.value.title()} Option', 
                 fontsize=16, y=0.95)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, (ax1, ax2, ax3, ax4), tree_report, pde_report