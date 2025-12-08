"""
Demo 3: SABR Model Calibration
------------------------------
This script demonstrates how to calibrate the SABR stochastic volatility model
to a market volatility smile.

SABR (Stochastic Alpha, Beta, Rho) is widely used to fit implied volatility smiles.
Formula: Hagan et al (2002)
"""

from quantlib.calibration.sabr import SABRCalibrator
import matplotlib.pyplot as plt
import numpy as np

# ==========================================
# 1. Market Data (Volatility Smile)
# ==========================================
# Typical "smile" shape: Higher IV at wings (OTM puts/calls)
# T = 1.0 year, Forward = 100
strikes = [80, 90, 100, 110, 120]
market_ivs = [0.28, 0.24, 0.20, 0.23, 0.27]

market_data = list(zip(strikes, market_ivs))
forward = 100.0
expiry = 1.0

print("--- Market Data (Volatility Smile) ---")
for K, iv in market_data:
    print(f"Strike: {K}, Market IV: {iv:.2%}")

# ==========================================
# 2. Calibrate SABR Model
# ==========================================
# We fit Alpha, Nu, Rho (Beta fixed at 0.5 usually)
calibrator = SABRCalibrator()
params = calibrator.calibrate(market_data, forward, expiry)

print(f"\n--- Calibrated SABR Parameters ---")
print(f"Alpha (Level): {params.alpha:.4f}")
print(f"Beta  (Skew):  {params.beta:.4f} (Fixed/Initial)")
print(f"Rho   (Correl):{params.rho:.4f}")
print(f"Nu    (VolVol):{params.nu:.4f}")

# ==========================================
# 3. Visualize Fit
# ==========================================
print(f"\nGenerating plot 'sabr_smile_fit.png'...")

# Generate smooth smile curve
fine_strikes = np.linspace(70, 130, 100)
model_ivs = []
engine = calibrator.pricing_engine

for K in fine_strikes:
    iv = engine.implied_volatility(forward, K, expiry, params)
    model_ivs.append(iv)

plt.figure(figsize=(10, 6))
plt.plot(strikes, market_ivs, 'ro', label='Market Data')
plt.plot(fine_strikes, model_ivs, 'b-', label='SABR Fit')
plt.title(f'SABR Calibration (T={expiry})')
plt.xlabel('Strike')
plt.ylabel('Implied Volatility')
plt.grid(True)
plt.legend()
plt.savefig('sabr_smile_fit.png')
print("Done.")
