from dataclasses import dataclass
from quantlib.core.payoffs import OptionType
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import numpy as np

@dataclass
class MarketQuote:
    symbol: str                
    option: OptionType      
    strike: float
    time_to_expiry: float      
    bid: float
    ask: float
    timestamp: datetime

    @property
    def mid(self) -> float:
        return 0.5 * (self.bid + self.ask)
    @property
    def spread(self) -> float:
        return abs(self.ask - self.bid)
    

    
    def __post_init__(self):
        """Validate marketquote parameters"""
        if self.strike <= 0:
            raise ValueError("Strike price must be positive")
        if self.time_to_expiry < 0:
            raise ValueError("Time to expiry must be non-negative")
        if self.bid > self.ask:
            raise ValueError(f"Crossed market for {self.symbol}: bid {self.bid} > ask {self.ask}")

    
class OptionChain:
    def __init__(self, symbol: str, expiry: float, quotes: List[MarketQuote] = None):
        self.symbol = symbol
        self.expiry = expiry
        self._book: Dict[Tuple[float, OptionType], MarketQuote] = {}
        if quotes:
            for q in quotes:
                self.add_quote(q)
        
    def add_quote(self, quote: MarketQuote) -> None:
        if quote.symbol != self.symbol:
            raise ValueError(f"Symbol mismatch: chain {self.symbol}, quote {quote.symbol}")
        if quote.time_to_expiry != self.expiry:
            raise ValueError(f"Expiry mismatch: chain {self.expiry}, quote {quote.time_to_expiry}")
        self._book[(quote.strike, quote.option)] = quote
        
    def get_quote(self, strike: float, option_type: OptionType) -> Optional[MarketQuote]:
        return self._book.get((strike, option_type))

    def get_strikes(self) -> List[float]:
        return sorted({k for (k, _opt) in self._book.keys()})
        
    def filter_by_spread(self, max_spread_pct: float) -> 'OptionChain':
        """
        Keep quotes where (ask - bid) / mid <= max_spread_pct.
        If mid <= 0, the quote is dropped.
        """
        def ok(q: MarketQuote) -> bool:
            mid = q.mid
            return mid > 0 and (q.ask - q.bid) / mid <= max_spread_pct

        kept = [q for q in self._book.values() if ok(q)]
        return OptionChain(self.symbol, self.expiry, kept)
    
    def get_atm_strike(self, spot_price: float) -> float:
        strikes = sorted({k for (k, _opt) in self._book.keys()})
        return min(strikes, key=lambda k: (abs(k - spot_price), -k))
    
    def get_calls(self) -> List[MarketQuote]:
        return sorted(
        (q for (_k, opt), q in self._book.items() if opt is OptionType.CALL),
        key=lambda q: q.strike
    )

    def get_puts(self) -> List[MarketQuote]:
        return sorted(
        (q for (_k, opt), q in self._book.items() if opt is OptionType.PUT),
        key=lambda q: q.strike
    )
        
    def __len__(self) -> int:
        return len(self._book)

    def __iter__(self):
        return iter(self._book.values())