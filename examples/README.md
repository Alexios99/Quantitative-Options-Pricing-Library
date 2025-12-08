# QuantLib Examples

This directory contains demo scripts showcasing the capabilities of the Quantitative Options Pricing Library.

## How to Run

Make sure you are in the project root directory and have the library installed or in your python path.

```bash
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## Available Demos

### 1. `01_option_pricing.py`
Demonstrates pricing of European and American options using:
- Analytical Black-Scholes
- Monte Carlo Simulation
- Binomial Trees (CRR)
- Finite Difference Methods (PDE)

### 2. `02_implied_volatility.py`
Shows how to:
- Construct a Volatility Surface from market quotes
- Interpolate implied volatility for any strike/expiry

### 3. `03_sabr_calibration.py`
Demonstrates:
- Calibrating the SABR stochastic volatility model to a market smile
- Visualizing the fitted volatility curve
