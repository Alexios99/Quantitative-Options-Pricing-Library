# Quantitative Options Pricing Library
Production-quality Python library for options pricing and risk management.

## Installation
```bash
pip install -e .
```

## Quick Start
```python
from quantlib.core.payoffs import OptionContract
from quantlib.pricing.analytical import BlackScholesEngine

# Create option contract
option = OptionContract(spot=100, strike=100, time_to_expiry=1.0,
                       risk_free_rate=0.05, volatility=0.2)

# Price using Black-Scholes
engine = BlackScholesEngine()
result = engine.price(option)
print(f'Option price: {result.value:.4f}')
```
