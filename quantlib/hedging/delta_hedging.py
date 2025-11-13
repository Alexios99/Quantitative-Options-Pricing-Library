import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class DeltaHedging:
    def __init__(self, S0, K, T, r, sigma, option_type='call'):
        """
        Initialize delta hedging parameters
        
        Args:
            S0: Initial stock price
            K: Strike price
            T: Time to expiration (in years)
            r: Risk-free rate
            sigma: Volatility
            option_type: 'call' or 'put'
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.option_type = option_type.lower()
        
    def black_scholes_price(self, S, t):
        """Calculate Black-Scholes option price"""
        tau = self.T - t  # Time to expiration
        if tau <= 0:
            if self.option_type == 'call':
                return max(S - self.K, 0)
            else:
                return max(self.K - S, 0)
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        d2 = d1 - self.sigma * np.sqrt(tau)
        
        if self.option_type == 'call':
            price = S * norm.cdf(d1) - self.K * np.exp(-self.r * tau) * norm.cdf(d2)
        else:
            price = self.K * np.exp(-self.r * tau) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def calculate_delta(self, S, t):
        """Calculate option delta"""
        tau = self.T - t
        if tau <= 0:
            if self.option_type == 'call':
                return 1.0 if S > self.K else 0.0
            else:
                return -1.0 if S < self.K else 0.0
        
        d1 = (np.log(S / self.K) + (self.r + 0.5 * self.sigma**2) * tau) / (self.sigma * np.sqrt(tau))
        
        if self.option_type == 'call':
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1
    
    def simulate_hedging(self, n_steps=100, n_simulations=1000, hedge_frequency=1):
        """
        Simulate delta hedging strategy
        
        Args:
            n_steps: Number of time steps
            n_simulations: Number of simulation paths
            hedge_frequency: Rehedge every N steps
        """
        dt = self.T / n_steps
        results = []
        
        for sim in range(n_simulations):
            # Initialize
            S = np.zeros(n_steps + 1)
            S[0] = self.S0
            
            # Portfolio tracking
            option_position = -1  # Short one option
            stock_position = 0
            cash_position = 0
            portfolio_values = []
            hedging_errors = []
            
            # Initial hedge
            initial_delta = self.calculate_delta(S[0], 0)
            stock_position = -option_position * initial_delta
            initial_option_price = self.black_scholes_price(S[0], 0)
            cash_position = -option_position * initial_option_price - stock_position * S[0]
            
            # Simulate stock path
            for i in range(1, n_steps + 1):
                # Stock price evolution (geometric Brownian motion)
                dW = np.random.normal(0, np.sqrt(dt))
                S[i] = S[i-1] * np.exp((self.r - 0.5 * self.sigma**2) * dt + self.sigma * dW)
                
                t = i * dt
                
                # Rehedge if necessary
                if i % hedge_frequency == 0 or i == n_steps:
                    current_delta = self.calculate_delta(S[i], t)
                    required_stock = -option_position * current_delta
                    stock_trade = required_stock - stock_position
                    
                    # Update positions
                    cash_position -= stock_trade * S[i]  # Cost of rebalancing
                    stock_position = required_stock
                
                # Calculate portfolio value
                option_value = -option_position * self.black_scholes_price(S[i], t)
                stock_value = stock_position * S[i]
                cash_value = cash_position * np.exp(self.r * t)  # Cash grows at risk-free rate
                
                total_portfolio_value = option_value + stock_value + cash_value
                portfolio_values.append(total_portfolio_value)
                
                # Hedging error (should be close to 0 for perfect hedge)
                hedging_errors.append(total_portfolio_value)
            
            results.append({
                'simulation': sim,
                'stock_path': S.copy(),
                'portfolio_values': portfolio_values.copy(),
                'final_hedging_error': hedging_errors[-1],
                'final_stock_price': S[-1]
            })
        
        return results
    
    def analyze_results(self, results):
        """Analyze hedging simulation results"""
        hedging_errors = [r['final_hedging_error'] for r in results]
        final_prices = [r['final_stock_price'] for r in results]
        
        analysis = {
            'mean_hedging_error': np.mean(hedging_errors),
            'std_hedging_error': np.std(hedging_errors),
            'min_hedging_error': np.min(hedging_errors),
            'max_hedging_error': np.max(hedging_errors),
            'mean_final_price': np.mean(final_prices),
            'hedging_errors': hedging_errors
        }
        
        return analysis
    
    def plot_simulation(self, results, num_paths=10):
        """Plot simulation results"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot stock paths
        time_grid = np.linspace(0, self.T, len(results[0]['stock_path']))
        for i in range(min(num_paths, len(results))):
            ax1.plot(time_grid, results[i]['stock_path'], alpha=0.7)
        ax1.set_title('Stock Price Paths')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Stock Price')
        ax1.grid(True)
        
        # Plot portfolio values
        time_grid_portfolio = np.linspace(0, self.T, len(results[0]['portfolio_values']))
        for i in range(min(num_paths, len(results))):
            ax2.plot(time_grid_portfolio, results[i]['portfolio_values'], alpha=0.7)
        ax2.set_title('Portfolio Values (Hedging Errors)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Portfolio Value')
        ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax2.grid(True)
        
        # Histogram of final hedging errors
        hedging_errors = [r['final_hedging_error'] for r in results]
        ax3.hist(hedging_errors, bins=50, alpha=0.7, edgecolor='black')
        ax3.set_title('Distribution of Final Hedging Errors')
        ax3.set_xlabel('Hedging Error')
        ax3.set_ylabel('Frequency')
        ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
        ax3.grid(True)
        
        # Delta evolution
        S_range = np.linspace(0.5 * self.S0, 1.5 * self.S0, 100)
        deltas = [self.calculate_delta(S, 0) for S in S_range]
        ax4.plot(S_range, deltas)
        ax4.set_title('Option Delta vs Stock Price')
        ax4.set_xlabel('Stock Price')
        ax4.set_ylabel('Delta')
        ax4.axvline(x=self.K, color='red', linestyle='--', alpha=0.5, label='Strike')
        ax4.legend()
        ax4.grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    # Parameters
    S0 = 100      # Initial stock price
    K = 100       # Strike price
    T = 0.25      # 3 months to expiration
    r = 0.05      # 5% risk-free rate
    sigma = 0.2   # 20% volatility
    
    # Create delta hedging instance
    hedger = DeltaHedging(S0, K, T, r, sigma, option_type='call')
    
    # Run simulation
    print("Running delta hedging simulation...")
    results = hedger.simulate_hedging(n_steps=50, n_simulations=1000, hedge_frequency=1)
    
    # Analyze results
    analysis = hedger.analyze_results(results)
    
    print(f"\nDelta Hedging Analysis:")
    print(f"Mean hedging error: ${analysis['mean_hedging_error']:.4f}")
    print(f"Std hedging error: ${analysis['std_hedging_error']:.4f}")
    print(f"Min hedging error: ${analysis['min_hedging_error']:.4f}")
    print(f"Max hedging error: ${analysis['max_hedging_error']:.4f}")
    
    # Plot results
    hedger.plot_simulation(results, num_paths=20)