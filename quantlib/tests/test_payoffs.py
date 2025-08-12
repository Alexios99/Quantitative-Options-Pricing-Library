import pytest
import numpy as np
from hypothesis import given, strategies as st
from quantlib.core.payoffs import (
    OptionContract, OptionType, ExerciseStyle, CallPayoff, PutPayoff, 
    BarrierOptionContract, BarrierPayoff, BarrierType
)

class TestOptionContract:
    def test_valid_creation(self):
        contract = OptionContract(
            spot=100,
            strike=105,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        assert contract.spot == 100
        assert contract.strike == 105
        assert contract.time_to_expiry == 1.0
        assert contract.risk_free_rate == 0.05
        assert contract.volatility == 0.2
        assert contract.option == OptionType.CALL
        assert contract.style == ExerciseStyle.EUROPEAN
        
    def test_invalid_spot_price(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=-10,
                strike=105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN
            )

    def test_invalid_strike_price(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=100,
                strike=-105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN
            )

    def test_invalid_time_to_expiry(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=100,
                strike=105,
                time_to_expiry=-1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN
            )

    def test_invalid_volatility(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=100,
                strike=105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=-1,
                option=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN
            )

    def test_american_style_option(self):
        contract = OptionContract(
            spot=100,
            strike=105,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.PUT,
            style=ExerciseStyle.AMERICAN
        )
        assert contract.style == ExerciseStyle.AMERICAN

    @given(
        spot=st.floats(min_value=0.1, max_value=1000),
        strike=st.floats(min_value=0.1, max_value=1000),
        time_to_expiry=st.floats(min_value=0.001, max_value=10),
        volatility=st.floats(min_value=0.0, max_value=2.0)
    )
    def test_valid_parameters_hypothesis(self, spot, strike, time_to_expiry, volatility):
        contract = OptionContract(
            spot=spot,
            strike=strike,
            time_to_expiry=time_to_expiry,
            risk_free_rate=0.05,
            volatility=volatility,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN
        )
        assert contract.spot == spot
        assert contract.strike == strike


class TestCallPayoff:
    def test_in_the_money_call(self):
        call_payoff = CallPayoff(strike=100)
        result = call_payoff.calculate_payoff(110)
        assert result == 10
        
    def test_out_of_the_money_call(self):
        call_payoff = CallPayoff(strike=110)
        result = call_payoff.calculate_payoff(100)
        assert result == 0
        
    def test_at_the_money_call(self):
        call_payoff = CallPayoff(strike=100)
        result = call_payoff.calculate_payoff(100)
        assert result == 0
    
    def test_call_payoff_array(self):
        call_payoff = CallPayoff(strike=100)
        spots = [90, 100, 110] 
        result = call_payoff.calculate_payoff(spots)
        
        expected = np.array([0, 0, 10])  
        np.testing.assert_array_equal(result, expected)

    def test_zero_strike_call(self):
        call_payoff = CallPayoff(strike=0)
        result = call_payoff.calculate_payoff(100)
        assert result == 100

    def test_large_values_call(self):
        call_payoff = CallPayoff(strike=1e6)
        result = call_payoff.calculate_payoff(2e6)
        assert result == 1e6


class TestPutPayoff:
    def test_in_the_money_put(self):
        put_payoff = PutPayoff(strike=110)
        result = put_payoff.calculate_payoff(100)
        assert result == 10
        
    def test_out_of_the_money_put(self):
        put_payoff = PutPayoff(strike=100)
        result = put_payoff.calculate_payoff(110)
        assert result == 0
        
    def test_at_the_money_put(self):
        put_payoff = PutPayoff(strike=100)
        result = put_payoff.calculate_payoff(100)
        assert result == 0
    
    def test_put_payoff_array(self):
        put_payoff = PutPayoff(strike=100)
        spots = [90, 100, 110] 
        result = put_payoff.calculate_payoff(spots)
        
        expected = np.array([10, 0, 0])  
        np.testing.assert_array_equal(result, expected)

    def test_zero_spot_put(self):
        put_payoff = PutPayoff(strike=100)
        result = put_payoff.calculate_payoff(0)
        assert result == 100


class TestBarrierOptionContract:
    def test_valid_barrier_contract_creation(self):
        contract = BarrierOptionContract(
            spot=100,
            strike=105,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL,
            style=ExerciseStyle.EUROPEAN,
            barrier_type=BarrierType.UP_AND_OUT,
            barrier_level=120
        )
        assert contract.barrier_type == BarrierType.UP_AND_OUT
        assert contract.barrier_level == 120

    def test_barrier_contract_inherits_validation(self):
        with pytest.raises(ValueError):
            BarrierOptionContract(
                spot=-100,
                strike=105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN,
                barrier_type=BarrierType.UP_AND_OUT,
                barrier_level=120
            )


class TestBarrierPayoff:
    def test_up_and_out_call_barrier_not_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.UP_AND_OUT,
            strike=100,
            barrier_level=120,
            option=OptionType.CALL
        )
        spot_prices = [110]
        barrier_was_hit = [False]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([10])  # 110 - 100 = 10
        np.testing.assert_array_equal(result, expected)

    def test_up_and_out_call_barrier_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.UP_AND_OUT,
            strike=100,
            barrier_level=120,
            option=OptionType.CALL
        )
        spot_prices = [130]
        barrier_was_hit = [True]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([0])  # Knocked out, no payoff
        np.testing.assert_array_equal(result, expected)

    def test_down_and_out_put_barrier_not_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.DOWN_AND_OUT,
            strike=100,
            barrier_level=80,
            option=OptionType.PUT
        )
        spot_prices = [90]
        barrier_was_hit = [False]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([10])  # 100 - 90 = 10
        np.testing.assert_array_equal(result, expected)

    def test_down_and_out_put_barrier_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.DOWN_AND_OUT,
            strike=100,
            barrier_level=80,
            option=OptionType.PUT
        )
        spot_prices = [75]
        barrier_was_hit = [True]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([0])  # Knocked out, no payoff
        np.testing.assert_array_equal(result, expected)

    def test_up_and_in_call_barrier_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.UP_AND_IN,
            strike=100,
            barrier_level=120,
            option=OptionType.CALL
        )
        spot_prices = [130]
        barrier_was_hit = [True]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([30])  # 130 - 100 = 30
        np.testing.assert_array_equal(result, expected)

    def test_up_and_in_call_barrier_not_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.UP_AND_IN,
            strike=100,
            barrier_level=120,
            option=OptionType.CALL
        )
        spot_prices = [110]
        barrier_was_hit = [False]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([0])  # Not knocked in, no payoff
        np.testing.assert_array_equal(result, expected)

    def test_down_and_in_put_barrier_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.DOWN_AND_IN,
            strike=100,
            barrier_level=80,
            option=OptionType.PUT
        )
        spot_prices = [75]
        barrier_was_hit = [True]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([25])  # 100 - 75 = 25
        np.testing.assert_array_equal(result, expected)

    def test_down_and_in_put_barrier_not_hit(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.DOWN_AND_IN,
            strike=100,
            barrier_level=80,
            option=OptionType.PUT
        )
        spot_prices = [90]
        barrier_was_hit = [False]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([0])  # Not knocked in, no payoff
        np.testing.assert_array_equal(result, expected)

    def test_barrier_payoff_array_mixed(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.UP_AND_OUT,
            strike=100,
            barrier_level=120,
            option=OptionType.CALL
        )
        spot_prices = [110, 130, 90]
        barrier_was_hit = [False, True, False]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([10, 0, 0])  # [110-100, knocked out, 90<100]
        np.testing.assert_array_equal(result, expected)

    def test_barrier_payoff_otm_scenarios(self):
        barrier_payoff = BarrierPayoff(
            barrier_type=BarrierType.UP_AND_OUT,
            strike=100,
            barrier_level=120,
            option=OptionType.CALL
        )
        spot_prices = [95]  # Out of the money
        barrier_was_hit = [False]
        result = barrier_payoff.calculate_payoff(spot_prices, barrier_was_hit)
        expected = np.array([0])  # OTM, no payoff regardless
        np.testing.assert_array_equal(result, expected)
    def test_invalid_volatility(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=100,
                strike=105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=-1,
                option=OptionType.CALL,
                style=ExerciseStyle.EUROPEAN
            )


class TestCallPayoff:
    def test_in_the_money_call(self):
        call_payoff = CallPayoff(strike=100)
        result = call_payoff.calculate_payoff(110)
        assert result == 10
        
    def test_out_of_the_money_call(self):
        call_payoff = CallPayoff(strike=110)
        result = call_payoff.calculate_payoff(100)
        assert result == 0
        
    def test_at_the_money_call(self):
        call_payoff = CallPayoff(strike=100)
        result = call_payoff.calculate_payoff(100)
        assert result == 0
    
    def test_call_payoff_array(self):
        call_payoff = CallPayoff(strike=100)
        spots = [90, 100, 110] 
        result = call_payoff.calculate_payoff(spots)
        
        expected = np.array([0, 0, 10])  
        np.testing.assert_array_equal(result, expected)

class TestPutPayoff:
    def test_in_the_money_put(self):
        put_payoff = PutPayoff(strike=110)
        result = put_payoff.calculate_payoff(100)
        assert result == 10
        
    def test_out_of_the_money_put(self):
        put_payoff = PutPayoff(strike=100)
        result = put_payoff.calculate_payoff(110)
        assert result == 0
        
    def test_at_the_money_put(self):
        put_payoff = PutPayoff(strike=100)
        result = put_payoff.calculate_payoff(100)
        assert result == 0
    
    def test_put_payoff_array(self):
        put_payoff = PutPayoff(strike=100)
        spots = [90, 100, 110] 
        result = put_payoff.calculate_payoff(spots)
        
        expected = np.array([10, 0, 0])  
        np.testing.assert_array_equal(result, expected)