from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import norm 

class StochasticProcess(ABC):
    @abstractmethod
    def simulate_paths(self, S0, T, n_paths, n_steps, seed=None):
        """Returns array of shape (n_paths, n_steps+1)"""
        pass

class GeometricBrownianMotion(StochasticProcess):
    def __init__(self, drift, volatility):
        self.drift = drift
        self.volatility = volatility
    
    def simulate_paths(self, S0, T, n_paths, n_steps, seed=None):
        if seed:
            Z = norm.rvs(size=(n_paths, n_steps), random_state=seed)
        else:
            Z = norm.rvs(size=(n_paths, n_steps))
        paths = np.zeros((n_paths, n_steps + 1))
        dt = T/n_steps
        paths[:, 0] = S0
        for i in range(0, n_steps):
            paths[:, i+1] = paths[:, i] * np.exp((self.drift - np.square(self.volatility)/2) * dt + self.volatility * np.sqrt(dt)* Z[:, i])
                
        return paths
        
