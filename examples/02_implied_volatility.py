"""
Demo 2: Volatility Surface & Implied Volatility
----------------------------------------------
This script demonstrates how to:
1. define standard option contracts
2. Calculate Implied Volatility from market prices
3. Construct a Volatility Surface
4. Interpolate volatility for non-quoted strikes
"""

from quantlib.calibration.implied_vol import VolatilitySurface
from quantlib.utils.market_data import MarketQuote, OptionType

# ==========================================
# 1. Mock Market Data
# ==========================================
# We create a list of market quotes mimicking an option chain
# Data format: (Strike, Expiry, Market Price, Type)
market_data_raw = [
    # 3-Month Expiry (T=0.25)
    (90,  0.25, 12.50, OptionType.CALL),
    (95,  0.25, 8.50,  OptionType.CALL),
    (100, 0.25, 5.20,  OptionType.CALL), # ATM
    (105, 0.25, 2.80,  OptionType.CALL),
    (110, 0.25, 1.20,  OptionType.CALL),
    
    # 6-Month Expiry (T=0.50)
    (90,  0.50, 15.20, OptionType.CALL),
    (95,  0.50, 11.50, OptionType.CALL),
    (100, 0.50, 8.10,  OptionType.CALL), # ATM
    (105, 0.50, 5.40,  OptionType.CALL),
    (110, 0.50, 3.20,  OptionType.CALL),
]

spot_price = 100.0
risk_free_rate = 0.05

from datetime import datetime

# ==========================================
# 2. Build Volatility Surface
# ==========================================
print(f"--- Building Volatility Surface ---")
surface = VolatilitySurface(spot_price, risk_free_rate)

chain = []
for K, T, price, opt_type in market_data_raw:
    quote = MarketQuote(
        symbol="DEMO",
        strike=K, 
        bid=price-0.05, 
        ask=price+0.05, 
        time_to_expiry=T, 
        option=opt_type,
        timestamp=datetime.now()
    )
    chain.append(quote)

# Add quotes to surface (calculates IV for each internally)
surface.add_chain(chain)
print(f"Processed {len(chain)} market quotes.")

# ==========================================
# 3. Query Implied Volatility
# ==========================================
# We can now ask for IV at any point
test_points = [
    (100, 0.25),  # Existing point
    (102, 0.25),  # Strike interpolation
    (100, 0.35),  # Time interpolation
    (97, 0.40),   # Bilinear interpolation
]

print(f"\n--- Interpolated Implied Volatilities ---")
print(f"{'Strike':<10} {'Expiry':<10} {'Implied Vol':<15}")
print("-" * 35)

for K, T in test_points:
    iv = surface.get_iv(strike=K, expiry=T)
    print(f"{K:<10.1f} {T:<10.2f} {iv:.4%}")

print(f"\nNote: The surface uses bilinear interpolation between grid points.")
