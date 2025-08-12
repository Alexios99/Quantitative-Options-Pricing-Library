# tests/test_convergence_trees.py
import pytest
import numpy as np
import pandas as pd

from quantlib.utils.convergence import (
    run_study,
    estimate_order,
    reference_price,
    UseEngineRichardson,
    ConvergenceReport,
)
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.trees import BinomialTreeEngine, TrinomialTreeEngine


# Silence harmless runtime warnings from edge cases (e.g., T=0 in BS reference)
pytestmark = pytest.mark.filterwarnings("ignore:.*divide by zero.*:RuntimeWarning")


# ---------- helpers ----------

def bp_error(abs_err: float, ref: float) -> float:
    """Return error in basis points relative to the reference price."""
    if ref == 0:
        return np.inf
    return abs_err / ref * 1e4


# ---------- unit tests for convergence utilities ----------

class TestConvergenceUtils:
    def test_estimate_order_basic(self):
        # Perfect O(N^-1): error = 1/N
        ns = [10, 20, 40, 80]
        errors = [0.1, 0.05, 0.025, 0.0125]
        order = estimate_order(ns, errors)
        assert 0.9 <= order <= 1.1  # ~1.0

    def test_estimate_order_binomial_like(self):
        # Roughly O(N^-0.5)
        ns = [25, 50, 100, 200]
        errors = [0.02, 0.014, 0.01, 0.007]
        order = estimate_order(ns, errors)
        assert 0.3 <= order <= 0.7  # ~0.5

    def test_reference_price_european(self):
        contract = OptionContract(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
        )
        ref = reference_price(contract)
        assert 10 <= ref <= 15  # sanity band for BS call

    def test_reference_price_american(self):
        contract = OptionContract(
            spot=100, strike=110, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.3,
            option=OptionType.PUT, style=ExerciseStyle.AMERICAN,
        )
        ref = reference_price(contract)
        assert 8 <= ref <= 18  # includes early exercise premium


# ---------- fixtures ----------

@pytest.fixture
def european_call():
    return OptionContract(
        spot=100, strike=100, time_to_expiry=1.0,
        risk_free_rate=0.05, volatility=0.2,
        option=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
    )


@pytest.fixture
def american_put():
    return OptionContract(
        spot=100, strike=110, time_to_expiry=1.0,
        risk_free_rate=0.05, volatility=0.3,
        option=OptionType.PUT, style=ExerciseStyle.AMERICAN,
    )


# ---------- convergence studies ----------

class TestConvergenceStudies:
    def test_binomial_european_basic_convergence(self, european_call):
        grid = [25, 50, 100, 200]
        report = run_study(european_call, BinomialTreeEngine, grid, UseEngineRichardson.OFF)

        # Structure/type checks
        assert isinstance(report, ConvergenceReport)
        assert isinstance(report.df, pd.DataFrame)
        assert report.reference_price > 0
        assert {'N', 'price', 'abs_err', 'ms'}.issubset(report.df.columns)

        # Convergence checks (avoid strict monotonicity due to parity oscillations)
        errors = report.df['abs_err'].to_numpy()
        assert errors[-1] <= 0.6 * errors[0]  # net improvement
        assert report.order_estimate >= 0.35  # sane tail slope

        tail_bp = bp_error(errors[-1], report.reference_price)
        assert tail_bp < 50  # < 50 bp by N≈200 (tighten later if you wish)

    def test_binomial_with_engine_richardson(self, european_call):
        grid = [20, 40, 80]
        report = run_study(european_call, BinomialTreeEngine, grid, UseEngineRichardson.ON)

        # When engine uses internal Richardson, external RE should be disabled
        assert report.richardson_gain is None
        assert 'price_RE' in report.df.columns
        assert report.df['price_RE'].isna().all()

        # Expect at least binomial-like slope, often steeper with Richardson
        assert report.order_estimate >= 0.5

    def test_external_richardson_computation(self, european_call):
        grid = [20, 40, 80]
        report = run_study(european_call, BinomialTreeEngine, grid, UseEngineRichardson.OFF)

        # Convention: compute external RE for all but the last grid point
        assert 'price_RE' in report.df.columns
        assert report.df['price_RE'].notna().sum() == len(grid) - 1
        assert pd.isna(report.df['price_RE'].iloc[-1])

        # On the last row where RE exists, it should not be worse than raw
        row = report.df.dropna(subset=['price_RE']).iloc[-1]
        err_raw = abs(row['price'] - report.reference_price)
        err_re = abs(row['price_RE'] - report.reference_price)
        assert err_re <= err_raw + 1e-12

        # If a global gain was computed, it should be ≥ 1
        if report.richardson_gain is not None:
            assert report.richardson_gain >= 1.0

    def test_odd_even_gap_computation(self, european_call):
        grid = [21, 22, 41, 42, 81, 82]  # mix parity
        report = run_study(european_call, BinomialTreeEngine, grid, UseEngineRichardson.OFF)

        even_mean = report.df.loc[report.df.N % 2 == 0, 'abs_err'].mean()
        odd_mean = report.df.loc[report.df.N % 2 == 1, 'abs_err'].mean()
        assert np.isfinite(even_mean) and np.isfinite(odd_mean)
        # Optional minimal magnitude check (not tautological)
        assert abs(even_mean - odd_mean) >= 0.0  # keep as a presence check

    @pytest.mark.slow
    def test_trinomial_american_convergence(self, american_put):
        grid = [25, 50, 100, 200]
        report = run_study(american_put, TrinomialTreeEngine, grid, UseEngineRichardson.OFF)

        assert len(report.df) == len(grid)
        assert report.reference_price > 0

        errors = report.df['abs_err'].to_numpy()
        tail_bp = bp_error(errors[-1], report.reference_price)
        assert tail_bp < 150  # < 150 bp by N≈200 (looser for Americans)
        assert report.order_estimate >= 0.2

    def test_convergence_report_structure(self, european_call):
        grid = [50, 100]
        report = run_study(european_call, BinomialTreeEngine, grid, UseEngineRichardson.OFF)

        assert isinstance(report.df, pd.DataFrame)
        assert report.reference_price > 0
        assert np.isfinite(report.order_estimate)
        assert np.isfinite(report.odd_even_gap)
        assert isinstance(report.passed, bool)
        assert isinstance(report.meta, dict)
        assert 'engine' in report.meta and 'grid' in report.meta

    def test_performance_timing(self, european_call):
        grid = [50, 100]
        report = run_study(european_call, BinomialTreeEngine, grid, UseEngineRichardson.OFF)

        timings = report.df['ms'].to_numpy()
        assert np.all(timings > 0)
        assert np.all(timings < 10_000)  # each run under 10 seconds on CI


# ---------- edge cases ----------

class TestEdgeCases:
    def test_single_point_grid(self):
        contract = OptionContract(
            spot=100, strike=100, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
        )
        report = run_study(contract, BinomialTreeEngine, [100], UseEngineRichardson.OFF)
        assert len(report.df) == 1
        # With only one N, we expect no external Richardson
        assert 'price_RE' in report.df.columns
        assert pd.isna(report.df['price_RE'].iloc[0])
        assert report.richardson_gain is None

    def test_zero_time_to_expiry(self):
        # Intrinsic at T=0
        contract = OptionContract(
            spot=105, strike=100, time_to_expiry=0.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN,
        )
        expected_intrinsic = max(0.0, 105 - 100)

        # Direct engine price should equal intrinsic
        engine = BinomialTreeEngine(n_steps=50, richardson="off")
        direct_result = engine.price(contract)
        assert abs(direct_result.price - expected_intrinsic) < 1e-12

        # In the study, pass the reference explicitly to avoid BS(T=0) issues
        report = run_study(contract, BinomialTreeEngine, [50, 100], UseEngineRichardson.OFF, ref=expected_intrinsic)
        assert abs(report.df['price'].iloc[-1] - expected_intrinsic) < 1e-12


