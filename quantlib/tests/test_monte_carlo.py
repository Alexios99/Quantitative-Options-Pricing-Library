import pytest
import numpy as np
from datetime import datetime
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.monte_carlo import MonteCarloEngine, MonteCarloPricingResult
from quantlib.pricing.analytical import BlackScholesEngine, MethodUsed

class TestMonteCarloEngine:
    
    @pytest.fixture
    def standard_call(self):
        """Standard European call option for testing"""
        return OptionContract(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
    
    @pytest.fixture
    def standard_put(self):
        """Standard European put option for testing"""
        return OptionContract(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.PUT,
            style=ExerciseStyle.EUROPEAN
        )
    
    @pytest.fixture
    def mc_engine(self):
        """Standard Monte Carlo engine"""
        return MonteCarloEngine(n_steps=50, n_paths=10000, seed=42)
    
    @pytest.fixture
    def bs_engine(self):
        """Black-Scholes engine for reference"""
        return BlackScholesEngine()
    
    def test_engine_initialization(self):
        """Test Monte Carlo engine initialization"""
        engine = MonteCarloEngine(n_steps=100, n_paths=50000, variance_reduction="antithetic", seed=123)
        
        assert engine.n_steps == 100
        assert engine.n_paths == 50000
        assert engine.variance_reduction == "antithetic"
        assert engine.seed == 123
    
    def test_pricing_result_type(self, mc_engine, standard_call):
        """Test that pricing returns correct result type"""
        result = mc_engine.price(standard_call)
        
        assert isinstance(result, MonteCarloPricingResult)
        assert result.method == MethodUsed.MONTE_CARLO
        assert isinstance(result.price, float)
        assert isinstance(result.standard_error, float)
        assert isinstance(result.confidence_interval, float)
        assert result.n_paths == 10000
        assert isinstance(result.time, datetime)
    
    def test_reproducibility_with_seed(self, standard_call):
        """Test that same seed produces same results"""
        engine1 = MonteCarloEngine(n_steps=20, n_paths=1000, seed=42)
        engine2 = MonteCarloEngine(n_steps=20, n_paths=1000, seed=42)
        
        result1 = engine1.price(standard_call)
        result2 = engine2.price(standard_call)
        
        assert result1.price == result2.price
        assert result1.standard_error == result2.standard_error
        assert result1.confidence_interval == result2.confidence_interval
    
    def test_different_seeds_give_different_results(self, standard_call):
        """Test that different seeds produce different results"""
        engine1 = MonteCarloEngine(n_steps=20, n_paths=1000, seed=42)
        engine2 = MonteCarloEngine(n_steps=20, n_paths=1000, seed=123)
        
        result1 = engine1.price(standard_call)
        result2 = engine2.price(standard_call)
        
        assert result1.price != result2.price
    
    def test_call_option_pricing(self, mc_engine, bs_engine, standard_call):
        """Test call option pricing against Black-Scholes"""
        mc_result = mc_engine.price(standard_call)
        bs_result = bs_engine.price(standard_call)
        
        # Should be within 3 standard errors (99.7% confidence)
        error = abs(mc_result.price - bs_result.price)
        tolerance = 3 * mc_result.standard_error
        
        assert error <= tolerance, f"MC price {mc_result.price:.4f} vs BS {bs_result.price:.4f}, error {error:.4f} > tolerance {tolerance:.4f}"
        assert mc_result.price > 0  # Call price should be positive
    
    def test_put_option_pricing(self, mc_engine, bs_engine, standard_put):
        """Test put option pricing against Black-Scholes"""
        mc_result = mc_engine.price(standard_put)
        bs_result = bs_engine.price(standard_put)
        
        # Should be within 3 standard errors
        error = abs(mc_result.price - bs_result.price)
        tolerance = 3 * mc_result.standard_error
        
        assert error <= tolerance, f"MC price {mc_result.price:.4f} vs BS {bs_result.price:.4f}, error {error:.4f} > tolerance {tolerance:.4f}"
        assert mc_result.price > 0  # Put price should be positive
    
    def test_call_put_parity(self, mc_engine, standard_call, standard_put):
        """Test call-put parity: Call - Put = S - K*e^(-rT)"""
        call_result = mc_engine.price(standard_call)
        put_result = mc_engine.price(standard_put)
        
        call_put_diff = call_result.price - put_result.price
        theoretical_diff = (standard_call.spot - 
                          standard_call.strike * np.exp(-standard_call.risk_free_rate * standard_call.time_to_expiry))
        
        # Combined standard error for the difference
        combined_se = np.sqrt(call_result.standard_error**2 + put_result.standard_error**2)
        tolerance = 3 * combined_se
        
        error = abs(call_put_diff - theoretical_diff)
        assert error <= tolerance, f"Call-put parity violation: {error:.4f} > {tolerance:.4f}"
    
    def test_confidence_interval_properties(self, mc_engine, standard_call):
        """Test confidence interval properties"""
        result = mc_engine.price(standard_call)
        
        # CI should be positive
        assert result.confidence_interval > 0
        
        # CI should be proportional to standard error
        expected_ci = 1.96 * result.standard_error  # 95% CI
        assert abs(result.confidence_interval - expected_ci) < 1e-10
    
    def test_antithetic_variance_reduction(self, standard_call):
        """Test that antithetic variates reduce variance"""
        engine_normal = MonteCarloEngine(n_steps=50, n_paths=10000, seed=42)
        engine_antithetic = MonteCarloEngine(n_steps=50, n_paths=10000, 
                                           variance_reduction="antithetic", seed=42)
        
        result_normal = engine_normal.price(standard_call)
        result_antithetic = engine_antithetic.price(standard_call)
        
        # Antithetic should reduce standard error
        assert result_antithetic.standard_error < result_normal.standard_error
        
        # Variance reduction should be significant (at least 10%)
        variance_reduction = 1 - (result_antithetic.standard_error / result_normal.standard_error)
        assert variance_reduction > 0.1, f"Variance reduction only {variance_reduction:.1%}"
    
    def test_antithetic_reproducibility(self, standard_call):
        """Test that antithetic variates are reproducible"""
        engine1 = MonteCarloEngine(n_steps=50, n_paths=1000, 
                                 variance_reduction="antithetic", seed=42)
        engine2 = MonteCarloEngine(n_steps=50, n_paths=1000, 
                                 variance_reduction="antithetic", seed=42)
        
        result1 = engine1.price(standard_call)
        result2 = engine2.price(standard_call)
        
        assert result1.price == result2.price
        assert result1.standard_error == result2.standard_error
    
    def test_convergence_with_path_count(self, standard_call):
        """Test that standard error decreases with more paths"""
        paths_small = 1000
        paths_large = 10000
        
        engine_small = MonteCarloEngine(n_steps=50, n_paths=paths_small, 
                                    variance_reduction=None, seed=42)
        engine_large = MonteCarloEngine(n_steps=50, n_paths=paths_large, 
                                    variance_reduction=None, seed=123)
        
        result_small = engine_small.price(standard_call)
        result_large = engine_large.price(standard_call)
        
        # Calculate ratio as large/small (should be < 1)
        actual_ratio = result_large.standard_error / result_small.standard_error
        
        # Fix: Expected ratio should be sqrt(small_paths/large_paths)
        expected_ratio = np.sqrt(result_small.n_paths / result_large.n_paths)  # √(1000/10000) = 0.316
        
        # Now this should work: 0.298 vs expected 0.316
        assert 0.8 * expected_ratio <= actual_ratio <= 1.2 * expected_ratio
    
    def test_moneyness_scenarios(self, mc_engine, bs_engine):
        """Test pricing across different moneyness levels"""
        base_contract = OptionContract(
            spot=100.0,
            strike=100.0,  # Will be modified
            time_to_expiry=0.25,
            risk_free_rate=0.03,
            volatility=0.25,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        
        strikes = [80, 90, 100, 110, 120]  # ITM to OTM
        
        for strike in strikes:
            contract = OptionContract(
                spot=base_contract.spot,
                strike=strike,
                time_to_expiry=base_contract.time_to_expiry,
                risk_free_rate=base_contract.risk_free_rate,
                volatility=base_contract.volatility,
                option=base_contract.option,
                style=base_contract.style
            )
            
            mc_result = mc_engine.price(contract)
            bs_result = bs_engine.price(contract)
            
            # Test within tolerance
            error = abs(mc_result.price - bs_result.price)
            tolerance = 3 * mc_result.standard_error
            
            assert error <= tolerance, f"Strike {strike}: MC {mc_result.price:.4f} vs BS {bs_result.price:.4f}"
            assert mc_result.price >= 0  # Price should be non-negative
    
    def test_time_to_expiry_scenarios(self, mc_engine, bs_engine):
        """Test pricing across different time to expiry"""
        base_contract = OptionContract(
            spot=100.0,
            strike=105.0,
            time_to_expiry=1.0,  # Will be modified
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        
        times = [0.1, 0.25, 0.5, 1.0, 2.0]
        
        for T in times:
            contract = OptionContract(
                spot=base_contract.spot,
                strike=base_contract.strike,
                time_to_expiry=T,
                risk_free_rate=base_contract.risk_free_rate,
                volatility=base_contract.volatility,
                option=base_contract.option,
                style=base_contract.style
            )
            
            mc_result = mc_engine.price(contract)
            bs_result = bs_engine.price(contract)
            
            error = abs(mc_result.price - bs_result.price)
            tolerance = 3 * mc_result.standard_error
            
            assert error <= tolerance, f"Time {T}: MC {mc_result.price:.4f} vs BS {bs_result.price:.4f}"
    
    def test_greeks_not_implemented(self, mc_engine, standard_call):
        """Test that Greeks raises NotImplementedError"""
        with pytest.raises(NotImplementedError, match="Greeks calculation"):
            mc_engine.greeks(standard_call)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Very short expiry
        short_contract = OptionContract(
            spot=100.0,
            strike=100.0,
            time_to_expiry=0.001,  # Almost expired
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        
        engine = MonteCarloEngine(n_steps=10, n_paths=1000, seed=42)
        result = engine.price(short_contract)
        
        # For very short expiry, compare with Black-Scholes (not intrinsic value)
        bs_engine = BlackScholesEngine()
        bs_result = bs_engine.price(short_contract)
        
        # Should be within 3 standard errors
        error = abs(result.price - bs_result.price)
        tolerance = 3 * result.standard_error
        
        assert error <= tolerance, f"Short expiry: MC {result.price:.4f} vs BS {bs_result.price:.4f}"
        assert result.price >= 0  # Price should be non-negative
    
    def test_statistical_properties(self, standard_call):
        """Test statistical properties of the estimator"""
        n_trials = 20
        prices = []
        
        for i in range(n_trials):
            engine = MonteCarloEngine(n_steps=50, n_paths=5000, seed=100+i)
            result = engine.price(standard_call)
            prices.append(result.price)
        
        prices = np.array(prices)
        
        # Get Black-Scholes reference
        bs_engine = BlackScholesEngine()
        bs_result = bs_engine.price(standard_call)
        
        # Sample mean should be close to BS price
        sample_mean = np.mean(prices)
        sample_std = np.std(prices, ddof=1)
        
        # 95% confidence interval for the sample mean
        ci_half_width = 1.96 * sample_std / np.sqrt(n_trials)
        
        assert abs(sample_mean - bs_result.price) <= ci_half_width, \
            f"Sample mean {sample_mean:.4f} not within CI of BS price {bs_result.price:.4f}"

if __name__ == "__main__":
    pytest.main([__file__])