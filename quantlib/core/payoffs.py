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


@dataclass
class OptionContract:
    """Option contract specification"""
    spot: float
    strike: float
    time_to_expiry: float
    risk_free_rate: float
    volatility: float
    option: OptionType
    style: ExerciseStyle=ExerciseStyle.EUROPEAN


    def __post_init__(self):
        """Validate contract parameters"""
        if self.spot <= 0:
            raise ValueError("Spot price must be positive")
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.time_to_expiry <= 0:
            raise ValueError("Time to expiry must be positive")
        if self.volatility < 0:
            raise ValueError("Volatility must be non-negative")
        
        

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

