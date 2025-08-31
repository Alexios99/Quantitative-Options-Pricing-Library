# Quick test
from quantlib.core.stochastic_processes import GeometricBrownianMotion
import numpy as np

gbm = GeometricBrownianMotion(drift=0.05, volatility=0.2)
paths = gbm.simulate_paths(S0=100, T=1.0, n_paths=10, n_steps=10, seed=42)

print(f"Shape: {paths.shape}")
print(f"First path: {paths[0, :]}")
print(f"All start at 100? {np.allclose(paths[:, 0], 100)}")

# Test reproducibility
paths1 = gbm.simulate_paths(S0=100, T=1.0, n_paths=5, n_steps=5, seed=42)
paths2 = gbm.simulate_paths(S0=100, T=1.0, n_paths=5, n_steps=5, seed=42)
print(paths1[0, :])
print(paths2[0, :])
print(f"Same results with same seed? {np.allclose(paths1, paths2)}")

