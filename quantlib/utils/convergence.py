from quantlib.core.payoffs import OptionContract, ExerciseStyle, OptionType
from quantlib.pricing.trees import BinomialTreeEngine, TrinomialTreeEngine
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.pricing.pde import PDEEngine
from enum import Enum
import time
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Iterable
from types import SimpleNamespace
from dataclasses import dataclass

class UseEngineSmoothing(str, Enum):
    """Whether run_study asks the engine to apply odd/even averaging internally."""
    OFF = "off"
    ON = "on"


# Deprecated name: the option is odd/even averaging, not Richardson extrapolation
UseEngineRichardson = UseEngineSmoothing

@dataclass
class ConvergenceReport:
    df: pd.DataFrame                       # per-N results
    reference_price: float                 # benchmark
    order_estimate: float                  # slope on loglog |err| vs N
    odd_even_gap: float                    # mean(|err_even|) - mean(|err_odd|)
    smoothing_gain: Optional[float]        # ratio err_raw / err_avg at largest N with an average
    passed: bool                           # basic thresholds met
    meta: Dict[str, Any]                   # params used

@dataclass
class PDEConvergenceReport:
    spatial_df: pd.DataFrame               # spatial convergence results
    temporal_df: pd.DataFrame              # temporal convergence results
    reference_price: float                 # benchmark price
    spatial_order: float                   # spatial convergence order
    temporal_order: float                  # temporal convergence order
    passed: bool                           # convergence criteria met
    meta: Dict[str, Any]                   # grid parameters used


def reference_price(contract: OptionContract, n_steps=5000):
    if contract.style == ExerciseStyle.EUROPEAN:
        ref_price = BlackScholesEngine().price(contract).price
    else:
        ref_price = TrinomialTreeEngine(n_steps, smoothing="off").price(contract).price
    return ref_price

def estimate_order(ns: Iterable[int], abs_err: Iterable[float], last_k=4):
    ns = np.array(list(ns), dtype=float)
    e  = np.array(list(abs_err), dtype=float)
    m  = min(last_k, np.sum(e > 0))
    x  = np.log(ns[-m:])
    y  = np.log(e[-m:])
    slope, _ = np.polyfit(x, y, 1)
    return -slope 

def run_study(contract: OptionContract, engine_cls, n_grid: Iterable[int], use_engine_smoothing: UseEngineSmoothing = UseEngineSmoothing.OFF, ref=None):
    """Tree convergence study. With engine smoothing OFF, also reports the external odd/even
    average 0.5 * (P_N + P_{N+1}) in column price_avg (all but the largest N)."""
    ref_price = ref if ref is not None else reference_price(contract)
    rows = []

    for n in n_grid:
        if use_engine_smoothing == UseEngineSmoothing.ON:
            engine = engine_cls(n_steps=n, smoothing="odd_even")
        else:
            engine = engine_cls(n_steps=n, smoothing="off")
        
        t0 = time.perf_counter()
        pN = engine.price(contract).price
        dt = (time.perf_counter() - t0) * 1000.0

        if n < max(n_grid) and use_engine_smoothing == UseEngineSmoothing.OFF:
            engine_nP1 = engine_cls(n_steps=n+1, smoothing="off")
            pNP1 = engine_nP1.price(contract).price
            p_avg = 0.5 * (pN + pNP1)
        else:
            p_avg = None
        
        rows.append(dict(N=n, price=pN, price_avg=p_avg, abs_err=abs(pN - ref_price), rel_err=abs((pN - ref_price) / ref_price) if ref_price else np.nan, ms=dt))

    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

    order = estimate_order(df.N, df.abs_err.replace(0, np.nan).ffill())
    odd  = df.loc[df.N % 2 == 1, "abs_err"].mean()
    even = df.loc[df.N % 2 == 0, "abs_err"].mean()
    odd_even_gap = (even or 0) - (odd or 0)

    smoothing_gain = None
    if "price_avg" in df and df.price_avg.notna().any():
        mask = df.price_avg.notna() & df.price.notna()
        if mask.any():
            tail = df[mask].iloc[-1]
        err_raw = abs(tail.price - ref_price)
        err_avg = abs(tail.price_avg - ref_price)
        smoothing_gain = (err_raw / err_avg) if err_avg > 0 else None

    passed = bool(
        (df.abs_err.iloc[-1] < 1e-3) and
        (order >= 0.45)              # ≈ O(N^{-1/2}) baseline for binomial
    )

    return ConvergenceReport(
        df=df,
        reference_price=ref_price,
        order_estimate=order,
        odd_even_gap=odd_even_gap if np.isfinite(odd_even_gap) else 0.0,
        smoothing_gain=smoothing_gain,
        passed=passed,
        meta=dict(engine=engine_cls.__name__, grid=list(n_grid))
    )

def run_pde_convergence_study(contract: OptionContract, 
                             reference_engine=None, 
                             spatial_sizes=None, 
                             temporal_sizes=None,
                             s_max=200,
                             theta=0.5,
                             psor_params=None) -> PDEConvergenceReport:
    """Run comprehensive grid refinement study for PDE methods
    
    Args:
        contract: Option contract to price
        reference_engine: Engine for reference price (defaults to analytical)
        spatial_sizes: List of spatial grid sizes to test
        temporal_sizes: List of temporal grid sizes to test
        s_max: Maximum spot price for grid
        theta: Theta parameter for time discretization (0.5 = Crank-Nicolson)
        psor_params: PSOR solver parameters
    
    Returns:
        PDEConvergenceReport with spatial and temporal convergence analysis
    """
    
    if spatial_sizes is None:
        spatial_sizes = [50, 100, 200, 400]
    if temporal_sizes is None:
        temporal_sizes = [250, 500, 1000, 2000]
    if psor_params is None:
        psor_params = SimpleNamespace(
            psor_omega=1.2,
            psor_tol=1e-6,
            psor_max_iter=50
        )
    
    # Get reference price
    if reference_engine is None:
        ref_price = reference_price(contract)
    else:
        ref_price = reference_engine.price(contract).price
    
    # Study spatial convergence (fix temporal)
    spatial_results = []
    for n_space in spatial_sizes:
        grid_params = SimpleNamespace(
            n_space=n_space,
            n_time=1000,  # Fixed high resolution
            s_max=s_max,
            theta=theta
        )
        
        t0 = time.perf_counter()
        engine = PDEEngine(grid_params, psor_params)
        result = engine.price(contract)
        dt = (time.perf_counter() - t0) * 1000.0
        
        abs_err = abs(result.price - ref_price)
        rel_err = abs_err / ref_price if ref_price != 0 else np.nan
        
        spatial_results.append({
            'n_space': n_space,
            'price': result.price,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'ms': dt
        })
    
    # Study temporal convergence (fix spatial)
    temporal_results = []
    
    # Use high spatial resolution to isolate temporal error
    # Use at least 500, or 2x the largest requested spatial size
    fixed_n_space = max(500, max(spatial_sizes) * 2)
    
    for n_time in temporal_sizes:
        grid_params = SimpleNamespace(
            n_space=fixed_n_space,
            n_time=n_time,
            s_max=s_max,
            theta=theta
        )
        
        t0 = time.perf_counter()
        engine = PDEEngine(grid_params, psor_params)
        result = engine.price(contract)
        dt = (time.perf_counter() - t0) * 1000.0
        
        abs_err = abs(result.price - ref_price)
        rel_err = abs_err / ref_price if ref_price != 0 else np.nan
        
        temporal_results.append({
            'n_time': n_time,
            'price': result.price,
            'abs_err': abs_err,
            'rel_err': rel_err,
            'ms': dt
        })
    
    # Convert to DataFrames
    spatial_df = pd.DataFrame(spatial_results)
    temporal_df = pd.DataFrame(temporal_results)
    
    # Estimate convergence orders
    spatial_order = estimate_order(spatial_df['n_space'], spatial_df['abs_err'])
    # Temporal order from successive differences P(n_i) - P(n_{i+1}) on the fixed spatial grid:
    # the spatial error is identical in both prices and cancels, whereas the error against the
    # analytical price is dominated by that fixed spatial floor once the time error is small.
    # Needs at least three time grids; NaN otherwise.
    if len(temporal_df) >= 3:
        temporal_diffs = np.abs(np.diff(temporal_df['price'].to_numpy()))
        temporal_order = estimate_order(temporal_df['n_time'].to_numpy()[:-1], temporal_diffs)
    else:
        temporal_order = np.nan
    
    # Check convergence criteria
    final_spatial_err = spatial_df['abs_err'].iloc[-1]
    final_temporal_err = temporal_df['abs_err'].iloc[-1]
    # Expected: O(h²) in space; O(Δt²) in time for Crank-Nicolson (theta = 0.5), O(Δt) otherwise
    min_temporal_order = 1.5 if theta == 0.5 else 0.8
    passed = (final_spatial_err < 1e-3 and final_temporal_err < 1e-3 and 
              spatial_order > 1.5 and temporal_order > min_temporal_order)
    
    return PDEConvergenceReport(
        spatial_df=spatial_df,
        temporal_df=temporal_df,
        reference_price=ref_price,
        spatial_order=spatial_order,
        temporal_order=temporal_order,
        passed=passed,
        meta={
            'engine': 'PDEEngine',
            'spatial_sizes': spatial_sizes,
            'temporal_sizes': temporal_sizes,
            's_max': s_max,
            'theta': theta,
            'psor_params': psor_params.__dict__
        }
    )

def run_monte_carlo_study(contract: OptionContract, path_counts: Iterable[int] = None, n_steps: int = 50, variance_reduction: str = None, n_trials: int = 5, ref=None) -> ConvergenceReport:
    """Run Monte Carlo convergence study based on number of paths
    
    Args:
        contract: Option contract to price
        path_counts: List of path counts to test
        n_steps: Fixed number of time steps
        variance_reduction: "antithetic", "control", or None
        n_trials: Number of independent runs per path count (for confidence)
        ref: Reference price (uses analytical if None)
    """
    from quantlib.pricing.monte_carlo import MonteCarloEngine
    
    if path_counts is None:
        path_counts = [1000, 2000, 5000, 10000, 20000, 50000]
    
    ref_price = ref if ref is not None else reference_price(contract)
    rows = []
    
    for n_paths in path_counts:
        # Run multiple trials for statistical analysis
        trial_prices = []
        trial_times = []
        
        for trial in range(n_trials):
            engine = MonteCarloEngine(
                n_steps=n_steps, 
                n_paths=n_paths, 
                variance_reduction=variance_reduction,
                seed=42 + trial  # Different seed per trial
            )
            
            t0 = time.perf_counter()
            result = engine.price(contract)
            dt = (time.perf_counter() - t0) * 1000.0
            
            trial_prices.append(result.price)
            trial_times.append(dt)
        
        # Statistics across trials
        mean_price = np.mean(trial_prices)
        std_price = np.std(trial_prices, ddof=1)
        mean_time = np.mean(trial_times)
        
        # Theoretical standard error for MC
        theoretical_se = std_price / np.sqrt(n_paths)
        
        rows.append(dict(
            N=n_paths,
            price=mean_price,
            price_std=std_price,
            theoretical_se=theoretical_se,
            abs_err=abs(mean_price - ref_price),
            rel_err=abs((mean_price - ref_price) / ref_price) if ref_price else np.nan,
            ms=mean_time
        ))
    
    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)
    
    # MC convergence should be O(N^{-1/2})
    order = estimate_order(df.N, df.abs_err.replace(0, np.nan).ffill())
    
    # No odd-even gap for MC
    odd_even_gap = 0.0
    
    # No odd/even averaging for MC
    smoothing_gain = None
    
    # MC-specific passing criteria
    passed = bool(
        (df.abs_err.iloc[-1] < 1e-2) and  # Looser tolerance for MC
        (0.4 <= order <= 0.6)            # Should be ≈ O(N^{-1/2})
    )
    
    return ConvergenceReport(
        df=df,
        reference_price=ref_price,
        order_estimate=order,
        odd_even_gap=odd_even_gap,
        smoothing_gain=smoothing_gain,
        passed=passed,
        meta=dict(
            engine="MonteCarloEngine",
            grid=list(path_counts),
            n_steps=n_steps,
            variance_reduction=variance_reduction,
            n_trials=n_trials
        )
    )
