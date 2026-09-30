from datetime import datetime

import numpy as np
import pytest

from quantlib.calibration.implied_vol import ImpliedVolatilityError, VolatilitySurface
from quantlib.core.payoffs import ExerciseStyle, OptionContract, OptionType
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.utils.market_data import MarketQuote

SPOT, RATE = 100.0, 0.05


def _quote(strike, expiry, price, option=OptionType.CALL, half_spread=0.0):
    return MarketQuote(symbol="TEST", option=option, strike=strike, time_to_expiry=expiry,
                       bid=price - half_spread, ask=price + half_spread, timestamp=datetime(2025, 1, 1))


def _bs_price(strike, expiry, vol, option):
    contract = OptionContract(SPOT, strike, expiry, RATE, vol, option, ExerciseStyle.EUROPEAN)
    return BlackScholesEngine().price(contract).price


@pytest.fixture
def surface():
    return VolatilitySurface(spot_price=SPOT, risk_free_rate=RATE)


class TestImpliedVolatility:
    @pytest.mark.parametrize("option", [OptionType.CALL, OptionType.PUT])
    @pytest.mark.parametrize("strike, expiry, vol", [(80, 0.25, 0.35), (100, 1.0, 0.2), (130, 2.0, 0.15)])
    def test_round_trip_recovers_volatility(self, surface, option, strike, expiry, vol):
        price = _bs_price(strike, expiry, vol, option)
        iv = surface._calculate_iv(_quote(strike, expiry, price, option), SPOT, RATE)
        assert iv == pytest.approx(vol, abs=1e-8)

    def test_call_below_intrinsic_raises(self, surface):
        intrinsic = SPOT - 80 * np.exp(-RATE * 1.0)
        with pytest.raises(ImpliedVolatilityError, match="no-arbitrage bounds"):
            surface._calculate_iv(_quote(80, 1.0, intrinsic - 0.5), SPOT, RATE)

    def test_call_above_spot_raises(self, surface):
        with pytest.raises(ImpliedVolatilityError, match="no-arbitrage bounds"):
            surface._calculate_iv(_quote(100, 1.0, SPOT + 1.0), SPOT, RATE)

    def test_put_above_discounted_strike_raises(self, surface):
        with pytest.raises(ImpliedVolatilityError, match="no-arbitrage bounds"):
            surface._calculate_iv(_quote(100, 1.0, 100.0, OptionType.PUT), SPOT, RATE)

    def test_vol_beyond_search_range_raises(self, surface):
        # Valid price (inside the bounds) but its implied vol is above the 500% search maximum
        price = _bs_price(100, 1.0, 8.0, OptionType.CALL)
        with pytest.raises(ImpliedVolatilityError, match="above search maximum"):
            surface._calculate_iv(_quote(100, 1.0, price), SPOT, RATE)

    def test_zero_expiry_raises(self, surface):
        with pytest.raises(ImpliedVolatilityError, match="expiry"):
            surface._calculate_iv(_quote(100, 0.0, 5.0), SPOT, RATE)

    def test_no_silent_fallback_value(self, surface):
        """The old implementation returned 0.2 whenever root-finding failed."""
        with pytest.raises(ImpliedVolatilityError):
            surface._calculate_iv(_quote(100, 1.0, 0.001), SPOT, RATE)


class TestAddChain:
    def _chain(self):
        good = [_quote(K, 1.0, _bs_price(K, 1.0, 0.2, OptionType.CALL)) for K in (90, 100, 110)]
        bad = _quote(120, 1.0, SPOT + 5.0)  # above the call upper bound
        return good, bad

    def test_invalid_quote_raises_by_default(self, surface):
        good, bad = self._chain()
        with pytest.raises(ImpliedVolatilityError):
            surface.add_chain(good + [bad])

    def test_skip_invalid_records_rejections(self, surface, caplog):
        good, bad = self._chain()
        with caplog.at_level("WARNING"):
            surface.add_chain(good + [bad], skip_invalid=True)
        assert len(surface.iv_surface) == 3
        assert len(surface.rejected) == 1
        rejected_quote, reason = surface.rejected[0]
        assert rejected_quote is bad and "no-arbitrage bounds" in reason
        assert "Skipped quote" in caplog.text

    def test_valid_chain_interpolates(self, surface):
        good, _ = self._chain()
        surface.add_chain(good)
        assert surface.get_iv(strike=105, expiry=1.0) == pytest.approx(0.2, abs=1e-8)
