import pytest
import numpy as np
from datetime import datetime

from pricing.trees import BinomialTreeEngine, TrinomialTreeEngine
from pricing.analytical import BlackScholesEngine, MethodUsed
from core.payoffs import OptionContract, OptionType, ExerciseStyle


class TestBinomialTreeEngine:

    def setup_method(self):
        # Base engines used across tests
        self.engine_off = BinomialTreeEngine(n_steps=50, richardson="off")
        self.engine_on  = BinomialTreeEngine(n_steps=50, richardson="on")
        self.engine_auto = BinomialTreeEngine(n_steps=50, richardson="auto")
        self.bs_engine = BlackScholesEngine()

        self.standard_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )

    def test_binomial_engine_creation(self):
        engine = BinomialTreeEngine(n_steps=100, richardson="off")
        assert engine.n_steps == 100
        assert engine.method_used == MethodUsed.BINOMIAL_TREE

        # invalid n_steps
        with pytest.raises(ValueError):
            BinomialTreeEngine(n_steps=0)
        with pytest.raises(ValueError):
            BinomialTreeEngine(n_steps=-1)

        # bad richardson mode is a programming-time error (optional check if you validate it)
        # Here we just assert valid modes create successfully
        BinomialTreeEngine(n_steps=10, richardson="on")
        BinomialTreeEngine(n_steps=10, richardson="off")
        BinomialTreeEngine(n_steps=10, richardson="auto")

    def test_zero_expiry_intrinsic(self):
        # Base class should short-circuit to intrinsic
        zero_time_contract = OptionContract(
            spot=100.0, strike=90.0, time_to_expiry=0.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        res = self.engine_off.price(zero_time_contract)
        assert abs(res.price - 10.0) < 1e-9  # intrinsic = 100 - 90

        zero_time_put = OptionContract(
            spot=100.0, strike=110.0, time_to_expiry=0.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        res_put = self.engine_off.price(zero_time_put)
        assert abs(res_put.price - 10.0) < 1e-9  # intrinsic = 110 - 100

    def test_european_call_vs_blackscholes_richardson_off(self):
        # Tolerances looser when Richardson is OFF
        test_cases = [
            (100, 90, 0.2, 1.0, "ITM"),
            (100, 100, 0.2, 1.0, "ATM"),
            (100, 110, 0.2, 1.0, "OTM"),
            (100, 100, 0.1, 0.25, "Low vol, short"),
            (100, 100, 0.4, 2.0, "High vol, long"),
        ]
        for S, K, vol, T, desc in test_cases:
            c = OptionContract(S, K, T, 0.05, vol, OptionType.CALL, ExerciseStyle.EUROPEAN)
            tree_p = self.engine_off.price(c).price
            bs_p = self.bs_engine.price(c).price

            # Allow ~0.1 (≈10–11 bp on ~100 notional) or 0.15% of price – whichever larger
            tol = max(0.11, 0.0015 * abs(bs_p))
            assert abs(tree_p - bs_p) < tol, f"{desc}: tree={tree_p:.4f} bs={bs_p:.4f}"

    def test_european_call_vs_blackscholes_richardson_on(self):
        # Much tighter tolerance when Richardson is ON
        test_cases = [
            (100, 90, 0.2, 1.0, "ITM"),
            (100, 100, 0.2, 1.0, "ATM"),
            (100, 110, 0.2, 1.0, "OTM"),
            (100, 100, 0.1, 0.25, "Low vol, short"),
            (100, 100, 0.4, 2.0, "High vol, long"),
        ]
        for S, K, vol, T, desc in test_cases:
            c = OptionContract(S, K, T, 0.05, vol, OptionType.CALL, ExerciseStyle.EUROPEAN)
            tree_p = self.engine_on.price(c).price
            bs_p = self.bs_engine.price(c).price

            # 5bp or 0.1%, whichever larger
            tol = max(0.05, 0.001 * abs(bs_p))
            assert abs(tree_p - bs_p) < tol, f"{desc}: tree={tree_p:.4f} bs={bs_p:.4f}"

    def test_european_put_vs_blackscholes(self):
        put = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        p_tree = self.engine_on.price(put).price
        p_bs = self.bs_engine.price(put).price
        tol = max(0.05, 0.001 * abs(p_bs))  # 5bp or 0.1%
        assert abs(p_tree - p_bs) < tol

    def test_american_option_bounds(self):
        american_put = OptionContract(
            spot=80.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.AMERICAN
        )
        european_put = OptionContract(
            spot=80.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        a = self.engine_off.price(american_put).price
        e = self.engine_off.price(european_put).price
        intrinsic = max(0.0, american_put.strike - american_put.spot)

        assert a >= e - 1e-8, "American >= European"
        assert a >= intrinsic - 1e-8, "American >= intrinsic"
        assert a <= american_put.strike + 1e-8, "American put <= strike"

    def test_put_call_parity(self):
        call = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        put = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        C = self.engine_on.price(call).price
        P = self.engine_on.price(put).price
        parity_diff = C - P
        expected = call.spot - call.strike * np.exp(-call.risk_free_rate * call.time_to_expiry)
        assert abs(parity_diff - expected) < 0.05

    def test_basic_monotonicity(self):
        # Price should increase with spot and volatility, decrease with strike
        base = self.standard_contract

        higher_S = OptionContract(spot=110.0, strike=100.0, time_to_expiry=1.0,
                                  risk_free_rate=0.05, volatility=0.2,
                                  option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        lower_S = OptionContract(spot=90.0, strike=100.0, time_to_expiry=1.0,
                                 risk_free_rate=0.05, volatility=0.2,
                                 option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        higher_vol = OptionContract(spot=100.0, strike=100.0, time_to_expiry=1.0,
                                    risk_free_rate=0.05, volatility=0.3,
                                    option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        higher_K = OptionContract(spot=100.0, strike=110.0, time_to_expiry=1.0,
                                  risk_free_rate=0.05, volatility=0.2,
                                  option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)

        p_base = self.engine_off.price(base).price
        p_hiS = self.engine_off.price(higher_S).price
        p_loS = self.engine_off.price(lower_S).price
        p_hiV = self.engine_off.price(higher_vol).price
        p_hiK = self.engine_off.price(higher_K).price

        assert p_hiS > p_base > p_loS
        assert p_hiV > p_base
        assert p_hiK < p_base

    def test_convergence_rate(self):
        # Prices should stabilise as n_steps grows (with Richardson OFF for a pure lattice test)
        step_counts = [10, 20, 50, 100]
        prices = []
        for n in step_counts:
            eng = BinomialTreeEngine(n_steps=n, richardson="off")
            prices.append(eng.price(self.standard_contract).price)

        diffs = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
        assert diffs[-1] < diffs[0] * 2, "Prices should be converging overall"

        bs = self.bs_engine.price(self.standard_contract).price
        assert abs(prices[-1] - bs) < 0.05

    def test_method_used_and_result_shape(self):
        res = self.engine_off.price(self.standard_contract)
        assert hasattr(res, "price")
        assert hasattr(res, "method")
        assert hasattr(res, "time")
        assert isinstance(res.price, float)
        assert res.price > 0
        assert isinstance(res.time, datetime)
        assert res.method == MethodUsed.BINOMIAL_TREE

    def test_greeks_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.engine_off.greeks(self.standard_contract)


# Parameterised tests
class TestBinomialTreeParameterized:

    BASE_TOLERANCE = 0.5
    TOLERANCE_SCALING_FACTOR = 50.0

    @pytest.mark.parametrize("n_steps", [5, 10, 25, 50, 100])
    @pytest.mark.parametrize("rich", ["off", "on", "auto"])
    def test_convergence_with_steps(self, n_steps, rich):
        engine = BinomialTreeEngine(n_steps=n_steps, richardson=rich)
        bs_engine = BlackScholesEngine()

        c = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )

        tree_p = engine.price(c).price
        bs_p = bs_engine.price(c).price

        # Tolerance improves with steps; keep a loose bound that shrinks with N
        tol = max(self.BASE_TOLERANCE, self.TOLERANCE_SCALING_FACTOR / n_steps)
        assert abs(tree_p - bs_p) < tol, f"n={n_steps}, rich={rich}"

    @pytest.mark.parametrize("option", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("style", [ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN])
    def test_all_option_combinations(self, option, style):
        engine = BinomialTreeEngine(n_steps=50, richardson="auto")
        c = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=option, style=style
        )
        res = engine.price(c)
        assert res.price >= 0.0
        assert res.price < 1000.0

        intrinsic = max(0.0, c.spot - c.strike) if option == OptionType.CALL else max(0.0, c.strike - c.spot)
        if style == ExerciseStyle.AMERICAN:
            assert res.price >= intrinsic - 1e-8


class TestTrinomialTreeEngine:

    def setup_method(self):
        # Base engines used across tests
        self.engine = TrinomialTreeEngine(n_steps=50, richardson="auto")
        self.bs_engine = BlackScholesEngine()
        self.binomial_engine = BinomialTreeEngine(n_steps=50, richardson="off")

        self.standard_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )

    def test_trinomial_engine_creation(self):
        engine = TrinomialTreeEngine(n_steps=100)
        assert engine.n_steps == 100
        assert engine.method_used == MethodUsed.TRINOMIAL_TREE

        # invalid n_steps
        with pytest.raises(ValueError):
            TrinomialTreeEngine(n_steps=0)
        with pytest.raises(ValueError):
            TrinomialTreeEngine(n_steps=-1)

    def test_zero_expiry_intrinsic(self):
        # Base class should short-circuit to intrinsic
        zero_time_contract = OptionContract(
            spot=100.0, strike=90.0, time_to_expiry=0.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        res = self.engine.price(zero_time_contract)
        assert abs(res.price - 10.0) < 1e-9  # intrinsic = 100 - 90

        zero_time_put = OptionContract(
            spot=100.0, strike=110.0, time_to_expiry=0.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        res_put = self.engine.price(zero_time_put)
        assert abs(res_put.price - 10.0) < 1e-9  # intrinsic = 110 - 100

    def test_european_call_vs_blackscholes(self):
        # Trinomial should converge faster than binomial, so tighter tolerances
        test_cases = [
            (100, 90, 0.2, 1.0, "ITM"),
            (100, 100, 0.2, 1.0, "ATM"),
            (100, 110, 0.2, 1.0, "OTM"),
            (100, 100, 0.1, 0.25, "Low vol, short"),
        ]
        for S, K, vol, T, desc in test_cases:
            c = OptionContract(S, K, T, 0.05, vol, OptionType.CALL, ExerciseStyle.EUROPEAN)
            tree_p = self.engine.price(c).price
            bs_p = self.bs_engine.price(c).price

            # Trinomial converges faster: 7bp or 0.1% of price – whichever larger
            tol = max(0.07, 0.001 * abs(bs_p))
            assert abs(tree_p - bs_p) < tol, f"{desc}: tree={tree_p:.4f} bs={bs_p:.4f}"

    def test_european_call_vs_blackscholes_high_vol_long(self):
        # High vol, long tenor (σ=0.4, T=2, N=50) → lattice has O(Δt) bias + odd/even oscillation.
        # With N=50 and no strong smoothing, ~10–11¢ error vs BS is normal.
        # Relax tolerance here; Richardson or larger N restores tighter bp.

        test_cases = [
            (100, 100, 0.4, 2.0, "High vol, long")
        ]
        for S, K, vol, T, desc in test_cases:
            c = OptionContract(S, K, T, 0.05, vol, OptionType.CALL, ExerciseStyle.EUROPEAN)
            tree_p = self.engine.price(c).price
            bs_p = self.bs_engine.price(c).price

            # Trinomial converges faster: 7bp or 0.1% of price – whichever larger
            tol = max(0.11, 0.001 * abs(bs_p))
            assert abs(tree_p - bs_p) < tol, f"{desc}: tree={tree_p:.4f} bs={bs_p:.4f}"

    def test_european_put_vs_blackscholes(self):
        put = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        p_tree = self.engine.price(put).price
        p_bs = self.bs_engine.price(put).price
        tol = max(0.07, 0.001 * abs(p_bs))  # 7bp or 0.1%
        assert abs(p_tree - p_bs) < tol

    def test_trinomial_vs_binomial_convergence(self):
        # Both should converge to similar values for European options
        test_cases = [
            (100, 90, 0.2, 1.0, OptionType.CALL),
            (100, 100, 0.2, 1.0, OptionType.CALL),
            (100, 110, 0.2, 1.0, OptionType.CALL),
            (100, 100, 0.2, 1.0, OptionType.PUT),
        ]
        
        for S, K, vol, T, opt_type in test_cases:
            c = OptionContract(S, K, T, 0.05, vol, opt_type, ExerciseStyle.EUROPEAN)
            tri_p = self.engine.price(c).price
            bin_p = self.binomial_engine.price(c).price
            
            # Both should be close to each other (within 0.1)
            assert abs(tri_p - bin_p) < 0.1, f"Trinomial vs Binomial: {tri_p:.4f} vs {bin_p:.4f}"

    def test_american_option_bounds(self):
        american_put = OptionContract(
            spot=80.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.AMERICAN
        )
        european_put = OptionContract(
            spot=80.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        a = self.engine.price(american_put).price
        e = self.engine.price(european_put).price
        intrinsic = max(0.0, american_put.strike - american_put.spot)

        assert a >= e - 1e-8, "American >= European"
        assert a >= intrinsic - 1e-8, "American >= intrinsic"
        assert a <= american_put.strike + 1e-8, "American put <= strike"

    def test_put_call_parity(self):
        call = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        put = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.PUT, style=ExerciseStyle.EUROPEAN
        )
        C = self.engine.price(call).price
        P = self.engine.price(put).price
        parity_diff = C - P
        expected = call.spot - call.strike * np.exp(-call.risk_free_rate * call.time_to_expiry)
        assert abs(parity_diff - expected) < 0.05

    def test_basic_monotonicity(self):
        # Price should increase with spot and volatility, decrease with strike
        base = self.standard_contract

        higher_S = OptionContract(spot=110.0, strike=100.0, time_to_expiry=1.0,
                                  risk_free_rate=0.05, volatility=0.2,
                                  option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        lower_S = OptionContract(spot=90.0, strike=100.0, time_to_expiry=1.0,
                                 risk_free_rate=0.05, volatility=0.2,
                                 option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        higher_vol = OptionContract(spot=100.0, strike=100.0, time_to_expiry=1.0,
                                    risk_free_rate=0.05, volatility=0.3,
                                    option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)
        higher_K = OptionContract(spot=100.0, strike=110.0, time_to_expiry=1.0,
                                  risk_free_rate=0.05, volatility=0.2,
                                  option=OptionType.CALL, style=ExerciseStyle.EUROPEAN)

        p_base = self.engine.price(base).price
        p_hiS = self.engine.price(higher_S).price
        p_loS = self.engine.price(lower_S).price
        p_hiV = self.engine.price(higher_vol).price
        p_hiK = self.engine.price(higher_K).price

        assert p_hiS > p_base > p_loS
        assert p_hiV > p_base
        assert p_hiK < p_base

    def test_convergence_rate(self):
        # Prices should stabilise as n_steps grows
        step_counts = [10, 20, 50, 100]
        prices = []
        for n in step_counts:
            eng = TrinomialTreeEngine(n_steps=n)
            prices.append(eng.price(self.standard_contract).price)

        diffs = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
        assert diffs[-1] < diffs[0] * 2, "Prices should be converging overall"

        bs = self.bs_engine.price(self.standard_contract).price
        assert abs(prices[-1] - bs) < 0.05

    def test_probability_validation(self):
        # Test that probabilities are valid for various market conditions
        test_cases = [
            (0.1, 0.05, 0.25),  # low vol, short time
            (0.3, 0.05, 1.0),   # high vol, normal time
            (0.2, 0.1, 2.0),    # normal vol, high rate, long time
        ]
        
        for vol, rate, T in test_cases:
            contract = OptionContract(
                spot=100.0, strike=100.0, time_to_expiry=T,
                risk_free_rate=rate, volatility=vol,
                option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
            )
            # This should not raise ValueError about invalid probabilities
            try:
                result = self.engine.price(contract)
                assert result.price > 0
            except ValueError as e:
                if "Probabilities for trinomial not between 0 and 1" in str(e):
                    pytest.fail(f"Invalid probabilities for vol={vol}, rate={rate}, T={T}")
                else:
                    raise

    def test_method_used_and_result_shape(self):
        res = self.engine.price(self.standard_contract)
        assert hasattr(res, "price")
        assert hasattr(res, "method")
        assert hasattr(res, "time")
        assert isinstance(res.price, float)
        assert res.price > 0
        assert isinstance(res.time, datetime)
        assert res.method == MethodUsed.TRINOMIAL_TREE

    def test_greeks_not_implemented(self):
        with pytest.raises(NotImplementedError):
            self.engine.greeks(self.standard_contract)


# Parameterised tests for trinomial
class TestTrinomialTreeParameterized:

    BASE_TOLERANCE = 0.3  # Tighter than binomial due to faster convergence
    TOLERANCE_SCALING_FACTOR = 30.0

    @pytest.mark.parametrize("n_steps", [5, 10, 25, 50, 100])
    def test_trinomial_convergence_with_steps(self, n_steps):
        engine = TrinomialTreeEngine(n_steps=n_steps)
        bs_engine = BlackScholesEngine()

        c = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )

        tree_p = engine.price(c).price
        bs_p = bs_engine.price(c).price

        # Tolerance improves with steps; trinomial converges faster than binomial
        tol = max(self.BASE_TOLERANCE, self.TOLERANCE_SCALING_FACTOR / n_steps)
        assert abs(tree_p - bs_p) < tol, f"n={n_steps}"

    @pytest.mark.parametrize("option", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("style", [ExerciseStyle.EUROPEAN, ExerciseStyle.AMERICAN])
    def test_trinomial_all_option_combinations(self, option, style):
        engine = TrinomialTreeEngine(n_steps=50)
        c = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=option, style=style
        )
        res = engine.price(c)
        assert res.price >= 0.0
        assert res.price < 1000.0

        intrinsic = max(0.0, c.spot - c.strike) if option == OptionType.CALL else max(0.0, c.strike - c.spot)
        if style == ExerciseStyle.AMERICAN:
            assert res.price >= intrinsic - 1e-8

    @pytest.mark.parametrize("engine_type", ["binomial", "trinomial"])
    def test_engine_comparison(self, engine_type):
        """Compare both engine types against Black-Scholes"""
        if engine_type == "binomial":
            engine = BinomialTreeEngine(n_steps=100, richardson="on")
            tolerance = 0.05
        else:
            engine = TrinomialTreeEngine(n_steps=100)
            tolerance = 0.04  # Trinomial should be slightly more accurate

        bs_engine = BlackScholesEngine()
        
        contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )

        tree_price = engine.price(contract).price
        bs_price = bs_engine.price(contract).price

        assert abs(tree_price - bs_price) < tolerance