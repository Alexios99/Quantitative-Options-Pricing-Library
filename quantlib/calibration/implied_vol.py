from quantlib.utils.market_data import OptionChain, MarketQuote
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.core.payoffs import OptionContract, ExerciseStyle
from dataclasses import dataclass
from scipy.optimize import brentq
import numpy as np
from typing import List, Optional, Tuple, Dict
from bisect import bisect_left

@dataclass
class IVPoint:
    strike: float
    expiry: float
    iv: float

class VolatilitySurface:
    def __init__(self, spot_price: float, risk_free_rate: float):
        self.spot = spot_price
        self.risk_free_rate = risk_free_rate
        self.iv_surface = []
    def add_chain(self, chain: OptionChain) -> None:
        for quote in chain:
            try:
                iv = self._calculate_iv(quote, self.spot, self.risk_free_rate)
                if not np.isnan(iv) and 0.001 <= iv <= 5.0: 
                    self.iv_surface.append(IVPoint(quote.strike, quote.time_to_expiry, iv))
            except Exception as e:
                pass


    def get_iv(self, strike: float, expiry: float, method: str = "bilinear") -> float:
        if not self.iv_surface:
            raise ValueError("No data points in surface")
        
        if method == "bilinear":
            neighbors = self._find_bilinear_neighbors(strike, expiry)
            if neighbors is not None:
                return self._bilinear_interpolate(strike, expiry, neighbors)
            else:
                method = "nearest"
        
        if method == "nearest":
            strikes = np.array([p.strike for p in self.iv_surface])
            expiries = np.array([p.expiry for p in self.iv_surface])
            d2 = (strikes - strike)**2 + (expiries - expiry)**2
            i = int(d2.argmin())
            return self.iv_surface[i].iv
    
        raise ValueError(f"Unknown interpolation method: {method}")

    def _calculate_iv(self, market_quote: MarketQuote, spot_price: float, risk_free_rate: float) -> float:
        market_price = market_quote.mid
        bs_engine = BlackScholesEngine()
        def objective(vol):
            contract = OptionContract(spot_price, market_quote.strike, market_quote.time_to_expiry, risk_free_rate, vol, market_quote.option, ExerciseStyle.EUROPEAN)
            bs_result = bs_engine.price(contract)
            bs_price = bs_result.price
        
            return bs_price - market_price
        
        vol_min = 0.001  
        vol_max = 5  

        if market_price <= 0:
            raise ValueError(f"Market Price cannot be negative. Current Market Price: {market_price}") 
        
        
        try:
            return brentq(objective, vol_min, vol_max)
        except (ValueError, RuntimeError):
            return 0.2 #reasonable fallback if convergance fails
        
    
    def get_strikes(self) -> List[float]:
        strikes = sorted(set(p.strike for p in self.iv_surface))
        return strikes
    
    def get_expiries(self) -> List[float]:
        expiries = sorted(set(p.expiry for p in self.iv_surface))
        return expiries
        
    def _get_exact_iv(self, strike: float, expiry: float) -> Optional[float]:
        EPS_K = 1e-12   
        EPS_T = 1e-12   

        for p in self.iv_surface:  
            if abs(p.strike - strike) <= EPS_K and abs(p.expiry - expiry) <= EPS_T:
                return p.iv
        return None
        
        
    def _find_bracketing_values(self, target: float, sorted_values: List[float]) -> Tuple[float, float]:
        if not sorted_values:
            raise ValueError("sorted_values must be non-empty")

        n = len(sorted_values)
        if n == 1:
            v = sorted_values[0]
            return (v, v)

        i = bisect_left(sorted_values, target)

        # target <= first element
        if i == 0:
            v0 = sorted_values[0]
            return (v0, v0)

        # target is greater than all elements
        if i >= n:
            vn = sorted_values[-1]
            return (vn, vn)

        # Now sorted_values[i-1] < = target <= sorted_values[i]
        lo, hi = sorted_values[i-1], sorted_values[i]

        # Exact match (covers duplicates too, since bisect_left gives first index)
        if hi == target:
            return (hi, hi)

        return (lo, hi)
    
    def _find_bilinear_neighbors(self, strike: float, expiry: float) -> Optional[Dict]:
        sorted_strikes = self.get_strikes()
        sorted_expiry = self.get_expiries()

        k1, k2 = self._find_bracketing_values(strike, sorted_strikes)
        t1, t2 = self._find_bracketing_values(expiry, sorted_expiry)

        
        iv11 = self._get_exact_iv(k1, t1)
        iv12 = self._get_exact_iv(k1, t2)
        iv21 = self._get_exact_iv(k2, t1)
        iv22 = self._get_exact_iv(k2, t2)

        if all(v is not None for v in [iv11, iv12, iv21, iv22]):

            return {
                'strikes':  (k1, k2),
                'expiries': (t1, t2),
                'ivs': {
                    (k1, t1): iv11,
                    (k1, t2): iv12,
                    (k2, t1): iv21,
                    (k2, t2): iv22,
                }
            }
        else:
            return None

    def _bilinear_interpolate(self, strike: float, expiry: float, neighbors: Dict) -> float:
        # Extract corner values
        k1, k2 = neighbors['strikes']
        t1, t2 = neighbors['expiries']
        ivs = neighbors['ivs']
        
        if k1 == k2 and t1 == t2:
            return ivs[(k1, t1)]
        elif k1 == k2:
            return ivs[(k1, t1)] + (ivs[(k1, t2)] - ivs[(k1, t1)]) * (expiry - t1)/(t2 - t1)
        elif t1 == t2:
            return ivs[(k1, t1)] + (ivs[(k2, t1)] - ivs[(k1, t1)]) * (strike - k1)/(k2 - k1)
        else:
            iv_at_t1 = ivs[(k1, t1)] + (ivs[(k2, t1)]-ivs[(k1, t1)]) * (strike-k1)/(k2-k1)
            iv_at_t2 = ivs[(k1, t2)] + (ivs[(k2, t2)]-ivs[(k1, t2)]) * (strike-k1)/(k2-k1)
            final_iv = iv_at_t1 + (iv_at_t2-iv_at_t1) * (expiry-t1)/(t2-t1)
            return final_iv

            

