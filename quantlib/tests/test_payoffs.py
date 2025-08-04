import pytest
import numpy as np
from core.payoffs import OptionContract, OptionType, ExerciseStyle, CallPayoff, PutPayoff

class TestOptionContract:
    def test_valid_creation(self):
        contract =  OptionContract(
            spot=100,
            strike=105,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.2,
            option=OptionType.CALL
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
                option=OptionType.CALL
            )

    def test_invalid_strike_price(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=100,
                strike=-105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=0.2,
                option=OptionType.CALL
            )
    def test_invalid_volatility(self):
        with pytest.raises(ValueError):
            OptionContract(
                spot=100,
                strike=105,
                time_to_expiry=1.0,
                risk_free_rate=0.05,
                volatility=-1,
                option=OptionType.CALL
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