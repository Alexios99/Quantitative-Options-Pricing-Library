from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm 
from core.payoffs import OptionContract, OptionType
from datetime import datetime


class MethodUsed(Enum):
    BS = 'blackscholes'
    GREEKS = 'greeks'

@dataclass
class PricingResult:
    """Pricing result class"""
    price: float
    method: MethodUsed
    time: datetime

@dataclass
class GreeksResult:
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float

class PricingEngine(ABC):
    @abstractmethod
    def price(self, contract: OptionContract) -> PricingResult:
        pass
    
    @abstractmethod
    def greeks(self, contract: OptionContract) -> GreeksResult:
        pass




class BlackScholesEngine(PricingEngine):
    def price(self, contract: OptionContract) -> PricingResult:
        d1, d2, nd1, nd2, n_neg_d1, n_neg_d2 = self._calculate_bs_components(contract)
        if contract.option == OptionType.CALL:
            price_value = contract.spot * nd1 - (contract.strike * np.exp(-contract.risk_free_rate * contract.time_to_expiry) * nd2)
        elif contract.option == OptionType.PUT:
            price_value  = contract.strike * np.exp(-contract.risk_free_rate * contract.time_to_expiry) * n_neg_d2 - contract.spot * n_neg_d1 
        else:
            raise ValueError(f"Unsupported option type: {contract.option}")
        return PricingResult(price=price_value, method=MethodUsed.BS, time=datetime.now())

        
    
    def greeks(self, contract: OptionContract) -> GreeksResult:
        pass
        # We'll tackle this next

    def _calculate_bs_components(self, contract: OptionContract):
        self._validate_parameters(contract)
        spot = contract.spot
        strike = contract.strike
        time_to_expiry = contract.time_to_expiry
        risk_free_rate = contract.risk_free_rate
        volatility = contract.volatility
        d1 = (np.log(spot/strike) + (risk_free_rate + (volatility**2)/2)*time_to_expiry)/(volatility * np.sqrt(time_to_expiry))
        d2 = d1 - volatility * np.sqrt(time_to_expiry)
        nd1 = norm.cdf(d1)
        nd2 = norm.cdf(d2)
        n_neg_d1 = norm.cdf(-d1)
        n_neg_d2 = norm.cdf(-d2)
        return (d1, d2, nd1, nd2, n_neg_d1, n_neg_d2)
    
    def _validate_parameters(self, contract: OptionContract):
        """Validate parameters for numerical stability"""
        if contract.volatility < 0.001:
            raise ValueError(f"Volatility too low ({contract.volatility:.6f}), minimum required: 0.001")
        if contract.time_to_expiry <= 1/365:
            raise ValueError(f"Time to expiry too low ({contract.time_to_expiry:.6f}), minimum: {1/365:.6f}")