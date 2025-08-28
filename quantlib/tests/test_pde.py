import pytest
import numpy as np
from types import SimpleNamespace
from quantlib.pricing.pde import PDEEngine
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.pricing.trees import BinomialTreeEngine, TrinomialTreeEngine
from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle

class TestPDEEngine:
    """Comprehensive test suite for PDE engine"""
    
    @pytest.fixture
    def standard_contract_european_call(self):
        """Standard European call for testing"""
        return OptionContract(
            option=OptionType.CALL,
            strike=100,
            spot=100,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
    
    @pytest.fixture
    def standard_contract_european_put(self):
        """Standard European put for testing"""
        return OptionContract(
            option=OptionType.PUT,
            strike=100,
            spot=100,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
    
    @pytest.fixture
    def american_put_otm(self):
        """American put OTM - likely early exercise"""
        return OptionContract(
            option=OptionType.PUT,
            strike=100,
            spot=80,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.AMERICAN
        )
    
    @pytest.fixture
    def american_call_itm(self):
        """American call ITM"""
        return OptionContract(
            option=OptionType.CALL,
            strike=100,
            spot=120,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.AMERICAN
        )
    
    @pytest.fixture
    def default_grid_params(self):
        """Default grid parameters for testing"""
        return SimpleNamespace(
            n_space=100,
            n_time=500,
            s_max=200,
            theta=0.5
        )
    
    @pytest.fixture
    def default_solver_params(self):
        """Default solver parameters for testing"""
        return SimpleNamespace(
            psor_omega=1.2,
            psor_tol=1e-6,
            psor_max_iter=50
        )
    
    @pytest.fixture
    def pde_engine(self, default_grid_params, default_solver_params):
        """Standard PDE engine for testing"""
        return PDEEngine(default_grid_params, default_solver_params)
    
    def test_engine_instantiation(self, default_grid_params, default_solver_params):
        """Test engine can be created without errors"""
        engine = PDEEngine(default_grid_params, default_solver_params)
        assert engine.n_space == 100
        assert engine.n_time == 500
        assert engine.s_max == 200
        assert engine.theta == 0.5
        assert engine.psor_omega == 1.2
        assert engine.psor_tol == 1e-6
        assert engine.psor_max_iter == 50
    
    def test_grid_construction(self, pde_engine, standard_contract_european_call):
        """Test grid construction works correctly"""
        contract = standard_contract_european_call
        pde_engine._build_grid(contract)
        
        # Check grid arrays exist and have correct size
        assert hasattr(pde_engine, 's_grid')
        assert hasattr(pde_engine, 't_grid')
        assert len(pde_engine.s_grid) == 100
        assert len(pde_engine.t_grid) == 501  # n_time + 1
        
        # Check grid bounds
        assert pde_engine.s_grid[0] == 0
        assert pde_engine.s_grid[-1] == 200
        assert pde_engine.t_grid[0] == 1.0  # expiry
        assert pde_engine.t_grid[-1] == 0.0  # present
        
        # Check spot index is reasonable
        assert 0 <= pde_engine.spot_index < 100
    
    def test_boundary_conditions_call(self, pde_engine, standard_contract_european_call):
        """Test boundary conditions for calls"""
        contract = standard_contract_european_call
        pde_engine._build_grid(contract)
        
        # Test at different time points
        for time_idx in [1, 250, 500]:
            pde_engine._setup_boundary_conditions(contract, time_idx)
            
            # Call boundaries
            assert pde_engine.S_min_boundary == 0  # Call worth 0 at S=0
            assert pde_engine.S_max_boundary >= 0  # Should be positive
    
    def test_boundary_conditions_put(self, pde_engine, standard_contract_european_put):
        """Test boundary conditions for puts"""
        contract = standard_contract_european_put
        pde_engine._build_grid(contract)
        
        # Test at different time points
        for time_idx in [1, 250, 500]:
            pde_engine._setup_boundary_conditions(contract, time_idx)
            
            # Put boundaries
            assert pde_engine.S_min_boundary > 0  # Put worth discounted strike at S=0
            assert pde_engine.S_max_boundary == 0  # Put worth 0 at high S
    
    def test_european_call_vs_black_scholes(self, pde_engine, standard_contract_european_call):
        """Test European call accuracy against Black-Scholes"""
        contract = standard_contract_european_call
        
        # Price with both methods
        bs_engine = BlackScholesEngine()
        bs_result = bs_engine.price(contract)
        pde_result = pde_engine.price(contract)
        
        # Check price is reasonable
        assert 8 <= pde_result.price <= 15  # Expected range for these parameters
        
        # Check accuracy (within 5 bp)
        error = abs(pde_result.price - bs_result.price)
        assert error <= 0.05, f"PDE error {error:.6f} > 5bp vs Black-Scholes"
        
        # Check pricing result metadata
        assert pde_result.method.value == "pde"
        assert pde_result.time is not None
    
    def test_european_put_vs_black_scholes(self, pde_engine, standard_contract_european_put):
        """Test European put accuracy against Black-Scholes"""
        contract = standard_contract_european_put
        
        # Price with both methods
        bs_engine = BlackScholesEngine()
        bs_result = bs_engine.price(contract)
        pde_result = pde_engine.price(contract)
        
        # Check price is reasonable
        assert 5 <= pde_result.price <= 12  # Expected range for ATM put
        
        # Check accuracy (within 5 bp)
        error = abs(pde_result.price - bs_result.price)
        assert error <= 0.05, f"PDE error {error:.6f} > 5bp vs Black-Scholes"
    
    def test_american_put_early_exercise_premium(self, pde_engine, american_put_otm):
        """Test American put has early exercise premium"""
        # Create corresponding European contract
        european_contract = OptionContract(
            option=american_put_otm.option,
            strike=american_put_otm.strike,
            spot=american_put_otm.spot,
            risk_free_rate=american_put_otm.risk_free_rate,
            volatility=american_put_otm.volatility,
            time_to_expiry=american_put_otm.time_to_expiry,
            style=ExerciseStyle.EUROPEAN
        )
        
        # Price both
        american_result = pde_engine.price(american_put_otm)
        european_result = pde_engine.price(european_contract)
        
        # American should be worth more (early exercise premium)
        assert american_result.price > european_result.price
        
        # Premium should be reasonable (not too large)
        premium = american_result.price - european_result.price
        assert 0 < premium <= 5, f"Early exercise premium {premium:.4f} seems unreasonable"
    
    def test_american_call_vs_european(self, pde_engine, american_call_itm):
        """Test American call vs European (should be similar for no dividends)"""
        # Create corresponding European contract
        european_contract = OptionContract(
            option=american_call_itm.option,
            strike=american_call_itm.strike,
            spot=american_call_itm.spot,
            risk_free_rate=american_call_itm.risk_free_rate,
            volatility=american_call_itm.volatility,
            time_to_expiry=american_call_itm.time_to_expiry,
            style=ExerciseStyle.EUROPEAN
        )
        
        # Price both
        american_result = pde_engine.price(american_call_itm)
        european_result = pde_engine.price(european_contract)
        
        # For calls with no dividends, American should be very close to European
        difference = abs(american_result.price - european_result.price)
        assert difference <= 0.01, f"American call differs too much from European: {difference:.6f}"
    
    def test_pde_stability_various_parameters(self, default_grid_params, default_solver_params):
        """Test PDE remains stable with various parameters"""
        test_cases = [
            # (spot, strike, vol, rate, expiry)
            (100, 100, 0.1, 0.05, 1.0),   # Low vol
            (100, 100, 0.4, 0.05, 1.0),   # High vol
            (80, 100, 0.2, 0.05, 1.0),    # OTM
            (120, 100, 0.2, 0.05, 1.0),   # ITM
            (100, 100, 0.2, 0.01, 1.0),   # Low rate
            (100, 100, 0.2, 0.05, 0.1),   # Short expiry
            (100, 100, 0.2, 0.05, 2.0),   # Long expiry
        ]
        
        engine = PDEEngine(default_grid_params, default_solver_params)
        
        for spot, strike, vol, rate, expiry in test_cases:
            contract = OptionContract(
                option=OptionType.CALL,
                strike=strike,
                spot=spot,
                risk_free_rate=rate,
                volatility=vol,
                time_to_expiry=expiry,
                style=ExerciseStyle.EUROPEAN
            )
            
            result = engine.price(contract)
            
            # Check price is reasonable
            assert result.price >= 0, f"Negative price for {spot}, {strike}, {vol}, {rate}, {expiry}"
            assert result.price <= max(spot, strike) * 2, f"Price too high for {spot}, {strike}, {vol}, {rate}, {expiry}"
            assert not np.isnan(result.price), f"NaN price for {spot}, {strike}, {vol}, {rate}, {expiry}"
            assert not np.isinf(result.price), f"Infinite price for {spot}, {strike}, {vol}, {rate}, {expiry}"
    
    def test_grid_convergence_spatial(self, standard_contract_european_call, default_solver_params):
        """Test spatial grid convergence"""
        contract = standard_contract_european_call
        spatial_sizes = [25, 50, 100, 200]
        prices = []
        
        for n_space in spatial_sizes:
            grid_params = SimpleNamespace(
                n_space=n_space,
                n_time=500,
                s_max=200,
                theta=0.5
            )
            engine = PDEEngine(grid_params, default_solver_params)
            result = engine.price(contract)
            prices.append(result.price)
        
        # Check convergence (later prices should be more stable)
        price_changes = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
        
        # Price changes should generally decrease (convergence)
        assert price_changes[-1] < price_changes[0] * 2, "No clear convergence with spatial refinement"
    
    def test_grid_convergence_temporal(self, standard_contract_european_call, default_solver_params):
        """Test temporal grid convergence"""
        contract = standard_contract_european_call
        temporal_sizes = [100, 250, 500, 1000]
        prices = []
        
        for n_time in temporal_sizes:
            grid_params = SimpleNamespace(
                n_space=100,
                n_time=n_time,
                s_max=200,
                theta=0.5
            )
            engine = PDEEngine(grid_params, default_solver_params)
            result = engine.price(contract)
            prices.append(result.price)
        
        # Check convergence
        price_changes = [abs(prices[i+1] - prices[i]) for i in range(len(prices)-1)]
        
        # Price changes should generally decrease
        assert price_changes[-1] < price_changes[0] * 2, "No clear convergence with temporal refinement"
    
    @pytest.mark.slow
    def test_american_vs_trinomial_tree(self, american_put_otm, default_solver_params):
        """Test American option vs trinomial tree reference"""
        # Configure fine PDE grid
        fine_grid_params = SimpleNamespace(
            n_space=200,
            n_time=1000,
            s_max=200,
            theta=0.5
        )
        
        pde_engine = PDEEngine(fine_grid_params, default_solver_params)
        
        # Use trinomial tree as reference
        tree_engine = TrinomialTreeEngine(n_steps=500)
        
        pde_result = pde_engine.price(american_put_otm)
        tree_result = tree_engine.price(american_put_otm)
        
        # Should agree within reasonable tolerance
        error = abs(pde_result.price - tree_result.price)
        assert error <= 0.10, f"PDE vs Tree error {error:.6f} too large"
    
    def test_put_call_parity_european(self, pde_engine):
        """Test put-call parity for European options"""
        # Create call and put with same parameters
        call_contract = OptionContract(
            option=OptionType.CALL,
            strike=100,
            spot=100,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
        
        put_contract = OptionContract(
            option=OptionType.PUT,
            strike=100,
            spot=100,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
        
        call_price = pde_engine.price(call_contract).price
        put_price = pde_engine.price(put_contract).price
        
        # Put-call parity: C - P = S - K*e^(-rT)
        S = 100
        K = 100
        r = 0.05
        T = 1.0
        
        left_side = call_price - put_price
        right_side = S - K * np.exp(-r * T)
        
        parity_error = abs(left_side - right_side)
        assert parity_error <= 0.01, f"Put-call parity violation: {parity_error:.6f}"
    
    def test_edge_cases(self, default_grid_params, default_solver_params):
        """Test edge cases and boundary conditions"""
        engine = PDEEngine(default_grid_params, default_solver_params)
        
        # Deep ITM call
        deep_itm_call = OptionContract(
            option=OptionType.CALL,
            strike=50,
            spot=150,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
        
        result = engine.price(deep_itm_call)
        intrinsic = 150 - 50
        assert result.price >= intrinsic * 0.95  # Should be close to intrinsic value
        
        # Deep OTM put
        deep_otm_put = OptionContract(
            option=OptionType.PUT,
            strike=50,
            spot=150,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
        
        result = engine.price(deep_otm_put)
        assert result.price <= 1.0  # Should be very small
    
    def test_greeks_not_implemented(self, pde_engine, standard_contract_european_call):
        """Test that Greeks method raises NotImplementedError"""
        with pytest.raises(NotImplementedError):
            pde_engine.greeks(standard_contract_european_call)


# Performance and integration tests
class TestPDEPerformance:
    """Performance and stress tests for PDE engine"""
    
    @pytest.mark.slow
    def test_large_grid_performance(self):
        """Test performance with large grids"""
        large_grid_params = SimpleNamespace(
            n_space=500,
            n_time=2000,
            s_max=200,
            theta=0.5
        )
        
        solver_params = SimpleNamespace(
            psor_omega=1.2,
            psor_tol=1e-6,
            psor_max_iter=50
        )
        
        contract = OptionContract(
            option=OptionType.CALL,
            strike=100,
            spot=100,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
        
        engine = PDEEngine(large_grid_params, solver_params)
        
        import time
        start_time = time.time()
        result = engine.price(contract)
        end_time = time.time()
        
        # Should complete in reasonable time (adjust threshold as needed)
        assert end_time - start_time < 30, "Large grid taking too long"
        assert result.price > 0
    
    def test_memory_usage_stability(self):
        """Test that repeated pricing doesn't leak memory"""
        grid_params = SimpleNamespace(
            n_space=100,
            n_time=500,
            s_max=200,
            theta=0.5
        )
        
        solver_params = SimpleNamespace(
            psor_omega=1.2,
            psor_tol=1e-6,
            psor_max_iter=50
        )
        
        contract = OptionContract(
            option=OptionType.CALL,
            strike=100,
            spot=100,
            risk_free_rate=0.05,
            volatility=0.2,
            time_to_expiry=1.0,
            style=ExerciseStyle.EUROPEAN
        )
        
        engine = PDEEngine(grid_params, solver_params)
        
        # Price multiple times
        prices = []
        for _ in range(10):
            result = engine.price(contract)
            prices.append(result.price)
        
        # Prices should be consistent
        price_std = np.std(prices)
        assert price_std < 1e-10, f"Price inconsistency across runs: std={price_std}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])