import pytest
import numpy as np
from datetime import datetime
from quantlib.core.payoffs import OptionType
from quantlib.utils.market_data import MarketQuote, OptionChain


class TestMarketQuote:
    
    @pytest.fixture
    def valid_quote(self):
        return MarketQuote(
            symbol="AAPL",
            option=OptionType.CALL,
            strike=150.0,
            time_to_expiry=0.25,
            bid=5.0,
            ask=5.5,
            timestamp=datetime.now()
        )
    
    def test_valid_quote_creation(self, valid_quote):
        """Test that valid quote is created without errors"""
        assert valid_quote.symbol == "AAPL"
        assert valid_quote.option == OptionType.CALL
        assert valid_quote.strike == 150.0
        assert valid_quote.time_to_expiry == 0.25
        assert valid_quote.bid == 5.0
        assert valid_quote.ask == 5.5
    
    def test_mid_property(self, valid_quote):
        """Test that mid price is calculated correctly"""
        assert valid_quote.mid == 5.25
        
    def test_spread_property(self, valid_quote):
        """Test that spread is calculated correctly"""
        assert valid_quote.spread == 0.5
    
    def test_negative_strike_raises_error(self):
        """Test that negative strike raises ValueError"""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            MarketQuote(
                symbol="AAPL", option=OptionType.CALL, strike=-150.0,
                time_to_expiry=0.25, bid=5.0, ask=5.5, timestamp=datetime.now()
            )
    
    def test_zero_strike_raises_error(self):
        """Test that zero strike raises ValueError"""
        with pytest.raises(ValueError, match="Strike price must be positive"):
            MarketQuote(
                symbol="AAPL", option=OptionType.CALL, strike=0.0,
                time_to_expiry=0.25, bid=5.0, ask=5.5, timestamp=datetime.now()
            )
    
    def test_negative_time_to_expiry_raises_error(self):
        """Test that negative time to expiry raises ValueError"""
        with pytest.raises(ValueError, match="Time to expiry must be non-negative"):
            MarketQuote(
                symbol="AAPL", option=OptionType.CALL, strike=150.0,
                time_to_expiry=-0.1, bid=5.0, ask=5.5, timestamp=datetime.now()
            )
    
    def test_crossed_market_raises_error(self):
        """Test that bid > ask raises ValueError"""
        with pytest.raises(ValueError, match="Crossed market for AAPL: bid 6.0 > ask 5.0"):
            MarketQuote(
                symbol="AAPL", option=OptionType.CALL, strike=150.0,
                time_to_expiry=0.25, bid=6.0, ask=5.0, timestamp=datetime.now()
            )
    
    def test_zero_time_to_expiry_allowed(self):
        """Test that zero time to expiry is allowed"""
        quote = MarketQuote(
            symbol="AAPL", option=OptionType.CALL, strike=150.0,
            time_to_expiry=0.0, bid=5.0, ask=5.5, timestamp=datetime.now()
        )
        assert quote.time_to_expiry == 0.0
    
    def test_equal_bid_ask_allowed(self):
        """Test that bid == ask is allowed"""
        quote = MarketQuote(
            symbol="AAPL", option=OptionType.CALL, strike=150.0,
            time_to_expiry=0.25, bid=5.0, ask=5.0, timestamp=datetime.now()
        )
        assert quote.spread == 0.0
        assert quote.mid == 5.0


class TestOptionChain:
    
    @pytest.fixture
    def sample_quotes(self):
        """Create sample quotes for testing"""
        timestamp = datetime.now()
        return [
            MarketQuote("AAPL", OptionType.CALL, 140.0, 0.25, 12.0, 12.5, timestamp),
            MarketQuote("AAPL", OptionType.CALL, 150.0, 0.25, 5.0, 5.5, timestamp),
            MarketQuote("AAPL", OptionType.CALL, 160.0, 0.25, 1.0, 1.5, timestamp),
            MarketQuote("AAPL", OptionType.PUT, 140.0, 0.25, 1.0, 1.5, timestamp),
            MarketQuote("AAPL", OptionType.PUT, 150.0, 0.25, 5.0, 5.5, timestamp),
            MarketQuote("AAPL", OptionType.PUT, 160.0, 0.25, 12.0, 12.5, timestamp),
        ]
    
    @pytest.fixture
    def empty_chain(self):
        return OptionChain("AAPL", 0.25)
    
    @pytest.fixture
    def populated_chain(self, sample_quotes):
        return OptionChain("AAPL", 0.25, sample_quotes)
    
    def test_empty_chain_creation(self, empty_chain):
        """Test creating an empty option chain"""
        assert empty_chain.symbol == "AAPL"
        assert empty_chain.expiry == 0.25
        assert len(empty_chain) == 0
    
    def test_populated_chain_creation(self, populated_chain):
        """Test creating chain with initial quotes"""
        assert populated_chain.symbol == "AAPL"
        assert populated_chain.expiry == 0.25
        assert len(populated_chain) == 6
    
    def test_add_quote_success(self, empty_chain):
        """Test successfully adding a quote"""
        quote = MarketQuote("AAPL", OptionType.CALL, 150.0, 0.25, 5.0, 5.5, datetime.now())
        empty_chain.add_quote(quote)
        assert len(empty_chain) == 1
        assert empty_chain.get_quote(150.0, OptionType.CALL) == quote
    
    def test_add_quote_symbol_mismatch(self, empty_chain):
        """Test adding quote with wrong symbol raises error"""
        quote = MarketQuote("MSFT", OptionType.CALL, 150.0, 0.25, 5.0, 5.5, datetime.now())
        with pytest.raises(ValueError, match="Symbol mismatch: chain AAPL, quote MSFT"):
            empty_chain.add_quote(quote)
    
    def test_add_quote_expiry_mismatch(self, empty_chain):
        """Test adding quote with wrong expiry raises error"""
        quote = MarketQuote("AAPL", OptionType.CALL, 150.0, 0.5, 5.0, 5.5, datetime.now())
        with pytest.raises(ValueError, match="Expiry mismatch: chain 0.25, quote 0.5"):
            empty_chain.add_quote(quote)
    
    def test_get_quote_existing(self, populated_chain):
        """Test getting an existing quote"""
        quote = populated_chain.get_quote(150.0, OptionType.CALL)
        assert quote is not None
        assert quote.strike == 150.0
        assert quote.option == OptionType.CALL
    
    def test_get_quote_nonexistent(self, populated_chain):
        """Test getting a non-existent quote returns None"""
        quote = populated_chain.get_quote(170.0, OptionType.CALL)
        assert quote is None
    
    def test_get_strikes(self, populated_chain):
        """Test getting all strikes sorted"""
        strikes = populated_chain.get_strikes()
        assert strikes == [140.0, 150.0, 160.0]
    
    def test_get_strikes_empty_chain(self, empty_chain):
        """Test getting strikes from empty chain"""
        strikes = empty_chain.get_strikes()
        assert strikes == []
    
    def test_filter_by_spread_tight_filter(self, populated_chain):
        """Test filtering with tight spread requirement"""
        # All quotes have 0.5 spread on 5+ mid price, so ~10% or less
        filtered = populated_chain.filter_by_spread(0.05)  # 5% max
        assert len(filtered) < len(populated_chain)
    
    def test_filter_by_spread_loose_filter(self, populated_chain):
        """Test filtering with loose spread requirement"""
        filtered = populated_chain.filter_by_spread(0.40) 
        assert len(filtered) == len(populated_chain)
    
    def test_filter_by_spread_zero_mid_price(self, empty_chain):
        """Test filtering handles zero mid price correctly"""
        # Create quote with zero mid price (bid=ask=0)
        quote = MarketQuote("AAPL", OptionType.CALL, 150.0, 0.25, 0.0, 0.0, datetime.now())
        empty_chain.add_quote(quote)
        
        filtered = empty_chain.filter_by_spread(0.10)
        assert len(filtered) == 0  # Should be dropped due to mid <= 0
    
    def test_get_atm_strike(self, populated_chain):
        """Test finding ATM strike"""
        # Spot at 145, should return 150 (closest strike)
        atm = populated_chain.get_atm_strike(145.0)
        assert atm == 150.0
        
        # Spot at 155, should return 150 (closest strike)
        atm = populated_chain.get_atm_strike(155.0)
        assert atm == 160.0
        
        # Spot exactly at strike
        atm = populated_chain.get_atm_strike(140.0)
        assert atm == 140.0
    
    def test_get_atm_strike_empty_chain(self, empty_chain):
        """Test ATM strike on empty chain raises error"""
        with pytest.raises(ValueError):
            empty_chain.get_atm_strike(150.0)
    
    def test_get_calls(self, populated_chain):
        """Test getting all call options"""
        calls = populated_chain.get_calls()
        assert len(calls) == 3
        assert all(q.option == OptionType.CALL for q in calls)
        # Should be sorted by strike
        strikes = [q.strike for q in calls]
        assert strikes == [140.0, 150.0, 160.0]
    
    def test_get_puts(self, populated_chain):
        """Test getting all put options"""
        puts = populated_chain.get_puts()
        assert len(puts) == 3
        assert all(q.option == OptionType.PUT for q in puts)
        # Should be sorted by strike
        strikes = [q.strike for q in puts]
        assert strikes == [140.0, 150.0, 160.0]
    
    def test_get_calls_empty_chain(self, empty_chain):
        """Test getting calls from empty chain"""
        calls = empty_chain.get_calls()
        assert calls == []
    
    def test_get_puts_empty_chain(self, empty_chain):
        """Test getting puts from empty chain"""
        puts = empty_chain.get_puts()
        assert puts == []
    
    def test_len_method(self, populated_chain, empty_chain):
        """Test __len__ method"""
        assert len(populated_chain) == 6
        assert len(empty_chain) == 0
    
    def test_iter_method(self, populated_chain):
        """Test iteration over chain"""
        quotes = list(populated_chain)
        assert len(quotes) == 6
        assert all(isinstance(q, MarketQuote) for q in quotes)
    
    def test_iter_empty_chain(self, empty_chain):
        """Test iteration over empty chain"""
        quotes = list(empty_chain)
        assert quotes == []
    
    def test_quote_replacement(self, empty_chain):
        """Test that adding quote with same key replaces existing"""
        quote1 = MarketQuote("AAPL", OptionType.CALL, 150.0, 0.25, 5.0, 5.5, datetime.now())
        quote2 = MarketQuote("AAPL", OptionType.CALL, 150.0, 0.25, 4.5, 5.0, datetime.now())
        
        empty_chain.add_quote(quote1)
        assert len(empty_chain) == 1
        assert empty_chain.get_quote(150.0, OptionType.CALL).bid == 5.0
        
        empty_chain.add_quote(quote2)
        assert len(empty_chain) == 1  # Should still be 1
        assert empty_chain.get_quote(150.0, OptionType.CALL).bid == 4.5  # Updated price
    
    def test_filter_preserves_metadata(self, populated_chain):
        """Test that filtering preserves chain metadata"""
        filtered = populated_chain.filter_by_spread(0.20)
        assert filtered.symbol == populated_chain.symbol
        assert filtered.expiry == populated_chain.expiry


class TestOptionChainEdgeCases:
    
    def test_very_large_strikes(self):
        """Test handling very large strike prices"""
        chain = OptionChain("TEST", 1.0)
        quote = MarketQuote("TEST", OptionType.CALL, 1e6, 1.0, 100.0, 105.0, datetime.now())
        chain.add_quote(quote)
        assert chain.get_atm_strike(1e6) == 1e6
    
    def test_very_small_positive_strikes(self):
        """Test handling very small positive strikes"""
        chain = OptionChain("TEST", 1.0)
        quote = MarketQuote("TEST", OptionType.PUT, 0.01, 1.0, 0.005, 0.006, datetime.now())
        chain.add_quote(quote)
        assert len(chain) == 1
    
    def test_floating_point_precision_expiry(self):
        """Test floating point precision in expiry matching"""
        # This tests the potential issue with exact float comparison
        chain = OptionChain("TEST", 0.25)
        quote = MarketQuote("TEST", OptionType.CALL, 100.0, 0.25, 5.0, 5.5, datetime.now())
        # Should work with exact match
        chain.add_quote(quote)
        assert len(chain) == 1
    
    def test_many_strikes_performance(self):
        """Test performance with many strikes"""
        chain = OptionChain("TEST", 1.0)
        timestamp = datetime.now()
        
        # Add 200 strikes
        for i in range(100, 300):
            call_quote = MarketQuote("TEST", OptionType.CALL, float(i), 1.0, 1.0, 1.1, timestamp)
            put_quote = MarketQuote("TEST", OptionType.PUT, float(i), 1.0, 1.0, 1.1, timestamp)
            chain.add_quote(call_quote)
            chain.add_quote(put_quote)
        
        assert len(chain) == 400
        assert len(chain.get_strikes()) == 200
        assert len(chain.get_calls()) == 200
        assert len(chain.get_puts()) == 200
        
        # ATM lookup should still be fast
        atm = chain.get_atm_strike(200.0)
        assert atm == 200.0


if __name__ == "__main__":
    pytest.main([__file__])