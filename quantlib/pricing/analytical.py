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
    BINOMIAL_TREE = 'binomial tree'
    TRINOMIAL_TREE = 'trinomial_tree'  

@dataclass
class PricingResult:
    """Pricing result class"""
    price: float
    method: MethodUsed
    time: datetime

@dataclass
class GreeksResult:
    delta_call: float
    delta_put: float
    gamma: float
    theta_call: float
    theta_put: float
    vega: float
    rho_call: float
    rho_put: float

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
        d1, d2, nd1, nd2, n_neg_d1, n_neg_d2 = self._calculate_bs_components(contract)
        delta_call = nd1
        delta_put = nd1 - 1
        gamma = norm.pdf(d1)/(contract.spot * contract.volatility * np.sqrt(contract.time_to_expiry))
        vega = contract.spot * norm.pdf(d1) * np.sqrt(contract.time_to_expiry)
        theta_call = ( -contract.spot * norm.pdf(d1) * contract.volatility / (2 * np.sqrt(contract.time_to_expiry)) )  - contract.risk_free_rate * contract.strike * np.exp(-contract.risk_free_rate * contract.time_to_expiry) * nd2
        theta_put = ( -contract.spot * norm.pdf(d1) * contract.volatility / (2 * np.sqrt(contract.time_to_expiry)) )  + contract.risk_free_rate * contract.strike * np.exp(-contract.risk_free_rate * contract.time_to_expiry) * n_neg_d2
        rho_call = contract.strike * contract.time_to_expiry * np.exp(-contract.risk_free_rate * contract.time_to_expiry) * nd2
        rho_put = -contract.strike * contract.time_to_expiry * np.exp(-contract.risk_free_rate * contract.time_to_expiry) * n_neg_d2
        return GreeksResult(
            delta_call=delta_call,
            delta_put=delta_put,
            gamma=gamma,
            theta_call=theta_call,
            theta_put=theta_put,
            vega=vega,
            rho_call=rho_call,
            rho_put=rho_put
        )



        

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