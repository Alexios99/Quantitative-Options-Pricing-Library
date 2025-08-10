from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np




class OptionType(Enum):
    CALL = "call"
    PUT = "put"

class ExerciseStyle(Enum):
    EUROPEAN = "european"
    AMERICAN = "american"

class BarrierType(Enum):
    DOWN_AND_OUT = "down-and-out"
    UP_AND_OUT = "up-and-out"
    DOWN_AND_IN = "down-and-in"
    UP_AND_IN = "up-and-in"


@dataclass
class OptionContract:
    """Option contract specification"""
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    option: OptionType
    style: ExerciseStyle


    def __post_init__(self):
        """Validate contract parameters"""
        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.time_to_expiry < 0:
            raise ValueError("Time to expiry must be non-negative")
        if self.volatility < 0:
            raise ValueError("Volatility must be non-negative")
        
@dataclass
class BarrierOptionContract(OptionContract):
    """BarrierOptionContract specification"""
    barrier_type: BarrierType
    barrier_level: float

    def __post_init__(self):
        super().__post_init__()  # Call parent validation
        if self.barrier_level <= 0:
            raise ValueError("Barrier level must be positive")
        if self.barrier_type in [BarrierType.UP_AND_OUT, BarrierType.UP_AND_IN]:
            if self.barrier_level <= self.spot:
                raise ValueError("Up barrier must be above current spot price")
        
        if self.barrier_type in [BarrierType.DOWN_AND_OUT, BarrierType.DOWN_AND_IN]:
            if self.barrier_level >= self.spot:
                raise ValueError("Down barrier must be below current spot price")




class PayoffFunction(ABC):
    """Abstract base class for payoff functions"""
    @abstractmethod
    def calculate_payoff(self, spot_prices):
        pass

class CallPayoff(PayoffFunction):
    """Payoff function for a call option"""
    def __init__(self, strike):
        self.strike = strike
    def calculate_payoff(self, spot_prices):
        spot_prices = np.array(spot_prices)
        return np.maximum(spot_prices - self.strike, 0)


class PutPayoff(PayoffFunction):
    """Payoff function for a put option"""
    def __init__(self, strike):
        self.strike = strike
    def calculate_payoff(self, spot_prices):
        spot_prices = np.array(spot_prices)
        return np.maximum(self.strike - spot_prices, 0)

class BarrierPayoff(PayoffFunction):
    """General Barrier Payoff function"""
    def __init__(self, barrier_type, strike, barrier_level, option):
        self.strike = strike
        self.barrier_type = barrier_type
        self.barrier_level = barrier_level
        self.option = option
    
    def calculate_payoff(self, spot_prices, barrier_was_hit):
        spot_prices = np.array(spot_prices)
        barrier_was_hit = np.array(barrier_was_hit)
        
        if self.option == OptionType.CALL:
            base_payoff = np.maximum(spot_prices - self.strike, 0)
        else:  
            base_payoff = np.maximum(self.strike - spot_prices, 0)
        
        if self.barrier_type in [BarrierType.UP_AND_OUT, BarrierType.DOWN_AND_OUT]:
            # Knock-OUT: payoff is 0 where barrier was hit, base_payoff where it wasn't
            return np.where(barrier_was_hit, 0, base_payoff)
        else:  
            # Knock-IN: payoff is base_payoff where barrier was hit, 0 where it wasn't
            return np.where(barrier_was_hit, base_payoff, 0)

class AmericanPayoff(PayoffFunction):
    """Payoff function for American options with early exercise capability"""
    def __init__(self, strike, option_type):
        self.strike = strike
        self.option_type = option_type
    
    def calculate_payoff(self, spot_prices, exercise_times=None):
        """Calculate payoff for American options"""
        spot_prices = np.array(spot_prices)
        
        if self.option_type == OptionType.CALL:
            intrinsic_value = np.maximum(spot_prices - self.strike, 0)
        elif self.option_type == OptionType.PUT:
            intrinsic_value = np.maximum(self.strike - spot_prices, 0)
        else:
            raise ValueError(f"Unsupported option type: {self.option_type}")
        
        return intrinsic_value
    
    def exercise_value(self, spot_price):
        """Calculate immediate exercise value at given spot price"""
        if self.option_type == OptionType.CALL:
            return max(spot_price - self.strike, 0)
        else:
            return max(self.strike - spot_price, 0)
    
    def should_exercise(self, spot_price, option_value):
        """Determine if early exercise is optimal (simple heuristic)"""
        exercise_val = self.exercise_value(spot_price)
        return exercise_val >= option_value