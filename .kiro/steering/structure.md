# Project Structure

## Directory Organization

```
quantlib/
├── core/                    # Core data structures and base classes
│   ├── payoffs.py          # Option contracts and payoff functions
│   └── stochastic_processes.py
├── pricing/                 # Pricing engines and methods
│   ├── analytical.py       # Black-Scholes and closed-form solutions
│   ├── monte_carlo.py      # Monte Carlo simulation engines
│   ├── pde.py             # PDE-based pricing methods
│   └── trees.py           # Binomial/trinomial tree methods
├── calibration/            # Model calibration utilities
│   ├── heston.py          # Heston model calibration
│   ├── implied_vol.py     # Implied volatility calculations
│   └── sabr.py            # SABR model fitting
├── hedging/               # Risk management and hedging
│   ├── delta_hedging.py   # Delta hedging strategies
│   └── pnl_attribution.py # P&L attribution analysis
├── utils/                 # Utility functions and helpers
│   ├── convergence.py     # Convergence testing utilities
│   ├── market_data.py     # Market data interfaces
│   ├── validation.py      # Input validation helpers
│   └── visualization.py   # Plotting and visualization
├── tests/                 # Test suite
├── examples/              # Usage examples and tutorials
├── docs/                  # Documentation
└── benchmarks/            # Performance benchmarks
```

## Code Organization Patterns

### Core Abstractions
- **OptionContract**: Dataclass for option specifications with validation
- **PricingEngine**: Abstract base class for all pricing methods
- **PayoffFunction**: Abstract base class for payoff calculations
- **PricingResult**: Standardized result container with metadata

### Naming Conventions
- Classes use PascalCase (e.g., `BlackScholesEngine`, `OptionContract`)
- Functions and variables use snake_case (e.g., `calculate_payoff`, `time_to_expiry`)
- Constants use UPPER_CASE (e.g., `OptionType.CALL`)
- Private methods prefixed with underscore (e.g., `_validate_parameters`)

### Module Structure
- Each module should have clear separation of concerns
- Abstract base classes define interfaces
- Concrete implementations inherit from abstractions
- Validation logic centralized in core modules
- Results returned as structured dataclasses with metadata

### Testing Structure
- Test classes mirror source code structure
- Property-based testing with Hypothesis for edge cases
- Comprehensive validation of mathematical properties (e.g., put-call parity)
- Performance benchmarks for numerical methods