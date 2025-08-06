import pytest
import numpy as np
from datetime import datetime
from hypothesis import given, strategies as st
from unittest.mock import patch

from core.payoffs import OptionContract, OptionType, ExerciseStyle
from pricing.analytical import (
    BlackScholesEngine, 
    PricingResult, 
    GreeksResult, 
    MethodUsed
)


class TestBlackScholesEngine:
    
    def setup_method(self):
        self.engine = BlackScholesEngine()
        self.call_contract = OptionContract(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        self.put_contract = OptionContract(
            spot=100.0,
            strike=100.0,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.PUT,
            style=ExerciseStyle.EUROPEAN
        )

    def test_call_option_pricing(self):
        result = self.engine.price(self.call_contract)
        
        assert isinstance(result, PricingResult)
        assert result.price > 0
        assert result.method == MethodUsed.BS
        assert isinstance(result.time, datetime)
        # ATM call with 20% vol, 1Y expiry should be ~10-12
        assert 8.0 < result.price < 15.0

    def test_put_option_pricing(self):
        result = self.engine.price(self.put_contract)
        
        assert isinstance(result, PricingResult)
        assert result.price > 0
        assert result.method == MethodUsed.BS
        # ATM put should be similar to call price
        call_price = self.engine.price(self.call_contract).price
        assert abs(result.price - call_price) < 5.0

    def test_put_call_parity(self):
        call_price = self.engine.price(self.call_contract).price
        put_price = self.engine.price(self.put_contract).price
        
        # Put-Call Parity: C - P = S - K*e^(-rT)
        pv_strike = self.call_contract.strike * np.exp(-self.call_contract.risk_free_rate * self.call_contract.time_to_expiry)
        expected_diff = self.call_contract.spot - pv_strike
        actual_diff = call_price - put_price
        
        assert abs(actual_diff - expected_diff) < 1e-10

    def test_greeks_calculation(self):
        result = self.engine.greeks(self.call_contract)
        
        assert isinstance(result, GreeksResult)
        # Delta should be between 0 and 1 for calls
        assert 0 < result.delta_call < 1
        assert -1 < result.delta_put < 0
        # Gamma should be positive
        assert result.gamma > 0
        # Vega should be positive
        assert result.vega > 0
        # Theta should be negative for long options
        assert result.theta_call < 0
        assert result.theta_put < 0

    def test_deep_itm_call(self):
        itm_contract = OptionContract(
            spot=150.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.2, 
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        result = self.engine.price(itm_contract)
        
        # Deep ITM call should be worth at least intrinsic value
        intrinsic = max(150.0 - 100.0, 0)
        assert result.price > intrinsic
        
        # Delta should be close to 1
        greeks = self.engine.greeks(itm_contract)
        assert greeks.delta_call > 0.8

    def test_deep_otm_call(self):
        otm_contract = OptionContract(
            spot=80.0, strike=120.0, time_to_expiry=0.1,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        result = self.engine.price(otm_contract)
        
        # Deep OTM call should be worth very little
        assert result.price < 2.0
        
        # Delta should be close to 0
        greeks = self.engine.greeks(otm_contract)
        assert greeks.delta_call < 0.2

    def test_zero_time_to_expiry_validation(self):
        invalid_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=0.0001,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        with pytest.raises(ValueError, match="Time to expiry too low"):
            self.engine.price(invalid_contract)

    def test_low_volatility_validation(self):
        invalid_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.0001,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        with pytest.raises(ValueError, match="Volatility too low"):
            self.engine.price(invalid_contract)

    def test_unsupported_option_type(self):
        # This would require modifying the enum, but we can test the logic
        contract = self.call_contract
        # Mock the option type to something invalid
        with patch.object(contract, 'option', 'INVALID'):
            with pytest.raises(ValueError, match="Unsupported option type"):
                self.engine.price(contract)

    @given(
        spot=st.floats(min_value=10.0, max_value=200.0),
        strike=st.floats(min_value=10.0, max_value=200.0),
        time_to_expiry=st.floats(min_value=0.01, max_value=5.0),
        risk_free_rate=st.floats(min_value=0.001, max_value=0.2),
        volatility=st.floats(min_value=0.01, max_value=1.0)
    )
    def test_property_based_pricing(self, spot, strike, time_to_expiry, risk_free_rate, volatility):
        contract = OptionContract(
            spot=spot, strike=strike, time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate, volatility=volatility,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        
        result = self.engine.price(contract)
        # Option price should always be positive
        assert result.price >= 0
        # Option price should be at least intrinsic value
        assert result.price >= max(spot - strike, 0)

    def test_greeks_numerical_properties(self):
        greeks = self.engine.greeks(self.call_contract)
        
        # Test delta bounds
        assert 0 <= greeks.delta_call <= 1
        assert -1 <= greeks.delta_put <= 0
        
        # Test gamma is positive
        assert greeks.gamma >= 0
        
        # Test vega is positive
        assert greeks.vega >= 0

    def test_volatility_sensitivity(self):
        low_vol_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.1,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        high_vol_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=1.0,
            risk_free_rate=0.05, volatility=0.3,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        
        low_vol_price = self.engine.price(low_vol_contract).price
        high_vol_price = self.engine.price(high_vol_contract).price
        
        # Higher volatility should lead to higher option price
        assert high_vol_price > low_vol_price

    def test_time_decay(self):
        short_term_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=0.1,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        long_term_contract = OptionContract(
            spot=100.0, strike=100.0, time_to_expiry=2.0,
            risk_free_rate=0.05, volatility=0.2,
            option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
        )
        
        short_price = self.engine.price(short_term_contract).price
        long_price = self.engine.price(long_term_contract).price
        
        # Longer time to expiry should lead to higher option price
        assert long_price > short_price

    def test_bs_components_calculation(self):
        d1, d2, nd1, nd2, n_neg_d1, n_neg_d2 = self.engine._calculate_bs_components(self.call_contract)
        
        # d1 should be greater than d2
        assert d1 > d2
        
        # Normal CDF values should be between 0 and 1
        assert 0 <= nd1 <= 1
        assert 0 <= nd2 <= 1
        assert 0 <= n_neg_d1 <= 1
        assert 0 <= n_neg_d2 <= 1
        
        # Complementary relationship
        assert abs(nd1 + n_neg_d1 - 1.0) < 1e-10
        assert abs(nd2 + n_neg_d2 - 1.0) < 1e-10