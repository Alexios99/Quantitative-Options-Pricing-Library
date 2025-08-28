# Technology Stack

## Core Dependencies
- **Python**: >=3.8 required
- **NumPy**: >=1.24 for numerical computations
- **SciPy**: >=1.10 for statistical functions and optimization
- **Numba**: >=0.57 for JIT compilation and performance optimization
- **Pandas**: >=2.0 for data manipulation
- **Matplotlib/Plotly**: >=3.5/5.0 for visualization

## Development Tools
- **Testing**: pytest >=7.0, pytest-cov >=4.0, hypothesis >=6.0 for property-based testing
- **Code Quality**: black >=22.0, isort >=5.0, mypy >=1.0
- **Market Data**: yfinance >=0.2 for real-time data
- **Documentation**: jupyter >=1.0 for examples and notebooks

## Build System
Uses setuptools with setup.py configuration. No pyproject.toml currently configured.

## Common Commands

### Installation
```bash
# Development installation
cd quantlib
pip install -e .

# Install with dev dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=quantlib

# Run specific test file
pytest tests/test_analytical.py

# Run property-based tests
pytest tests/ -v --hypothesis-show-statistics
```

### Code Quality
```bash
# Format code
black quantlib/ tests/

# Sort imports
isort quantlib/ tests/

# Type checking
mypy quantlib/
```

## Performance Considerations
- Use Numba JIT compilation for computationally intensive functions
- Leverage NumPy vectorization for array operations
- Consider memory usage for large Monte Carlo simulations