import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import replace

from quantlib.core.payoffs import OptionContract, OptionType, ExerciseStyle
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.core.stochastic_processes import GeometricBrownianMotion
from quantlib.hedging.pnl_attribution import attribute_pnl

def run_simulation():
    # --- Configuration ---
    S0 = 100.0          # Initial Spot
    K = 100.0           # Strike
    T = 1.0             # Time to Expiry (years)
    r = 0.05            # Risk-free rate
    sigma = 0.20        # Volatility
    n_steps = 1000      # Increased frequency (approx 4x daily) for better hedging
    
    # Create initial contract
    contract = OptionContract(
        spot=S0, strike=K, time_to_expiry=T,
        risk_free_rate=r, volatility=sigma,
        option=OptionType.CALL, style=ExerciseStyle.EUROPEAN
    )
    
    # Engine
    engine = BlackScholesEngine()
    
    # --- 1. Simulate Market Path ---
    print("Simulating market path...")
    process = GeometricBrownianMotion(drift=r, volatility=sigma)
    # simulate_paths returns [n_paths, n_steps+1]
    paths = process.simulate_paths(S0, T, n_paths=1, n_steps=n_steps, seed=None)
    spot_path = paths[0]
    time_grid = np.linspace(0, T, n_steps + 1)
    dt = T / n_steps
    
    # --- 2. Trading Loop ---
    print("Running Delta Hedging simulation...")
    
    # Portfolio State
    cash = 0.0
    shares = 0.0
    portfolio_values = []
    option_prices = []
    
    # PnL Tracking
    pnl_history = []
    
    # Initial Setup (t=0)
    # We SELL the option and HEDGE it (Buy Delta shares)
    res_t0 = engine.price(contract)
    initial_option_price = res_t0.price
    initial_delta = engine.greeks(contract).delta_call
    
    # Sell Option: Receive Premium
    cash += initial_option_price
    
    # Hedge: Buy Delta shares
    shares = initial_delta
    cash -= shares * S0
    
    portfolio_values.append(cash + shares * S0 - initial_option_price) # Should be 0 initially (fair value)
    option_prices.append(initial_option_price)
    
    current_contract = contract
    
    for t_idx in range(1, n_steps + 1):
        # Update Market State
        new_spot = spot_path[t_idx]
        new_time_to_expiry = max(0.0, T - time_grid[t_idx])
        
        # Create new contract state
        new_contract = replace(current_contract, spot=new_spot, time_to_expiry=new_time_to_expiry)
        
        # 1. Accrue Interest on Cash
        # Simple Euler for simplicity: cash * e^(r*dt)
        interest = cash * (np.exp(r * dt) - 1.0)
        cash += interest
        
        # 2. Calculate PnL Attribution (Step t-1 to t)
        # We analyze the OPTION's price change, then compare to our Hedge
        if new_time_to_expiry > 1e-6:
            attribution = attribute_pnl(current_contract, new_contract, engine)
            pnl_history.append(attribution)
        
        # 3. Rebalance Hedge
        if new_time_to_expiry > 1e-6:
            greeks = engine.greeks(new_contract)
            new_delta = greeks.delta_call
            
            # Buy/Sell difference
            delta_change = new_delta - shares
            shares = new_delta
            cash -= delta_change * new_spot
            
            option_price = engine.price(new_contract).price
        else:
            # Expiry
            payoff = max(0.0, new_spot - K)
            option_price = payoff
            
            # Close out position
            cash += shares * new_spot # Sell shares
            shares = 0
            cash -= payoff            # Pay out option holder
            
        option_prices.append(option_price)
        
        # Portfolio Value = Cash + Shares*S - Option_Liability
        pv = cash + shares * new_spot - option_price
        portfolio_values.append(pv)
        
        current_contract = new_contract

    # --- 3. Results ---
    final_pnl = portfolio_values[-1]
    print(f"\nSimulation Complete!")
    print(f"Final Spot: {spot_path[-1]:.2f}")
    print(f"Final Portfolio PnL: {final_pnl:.4f}")
    
    # Aggregate PnL Attribution
    total_delta_pnl = sum(p.delta_pnl for p in pnl_history)
    total_gamma_pnl = sum(p.gamma_pnl for p in pnl_history)
    total_theta_pnl = sum(p.theta_pnl for p in pnl_history)
    total_vega_pnl  = sum(p.vega_pnl for p in pnl_history) # Should be 0 as vol is constant
    total_unexplained = sum(p.unexplained_pnl for p in pnl_history)
    
    print("\n--- Option PnL Attribution (Short Position View) ---")
    # Note: Attribution is for Long Option. We are Short. So we reverse signs.
    print(f"Delta PnL (Hedged by Shares): {-total_delta_pnl:.4f}")
    print(f"Gamma PnL (Realized Vol):     {-total_gamma_pnl:.4f}")
    print(f"Theta PnL (Time Decay):       {-total_theta_pnl:.4f}")
    print(f"Vega PnL:                     {-total_vega_pnl:.4f}")
    print(f"Unexplained:                  {-total_unexplained:.4f}")
    print(f"Net 'Gamma-Theta' PnL:        {-(total_gamma_pnl + total_theta_pnl):.4f}")
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    ax1.plot(time_grid, spot_path, label='Stock Price', color='blue')
    ax1.set_ylabel('Price')
    ax1.set_title('Delta Hedging Simulation')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(time_grid, portfolio_values, label='Hedged Portfolio PnL', color='green')
    ax2.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('PnL')
    ax2.set_xlabel('Time (Years)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig("hedging_simulation.png")
    print("\nPlot saved to hedging_simulation.png")

if __name__ == "__main__":
    run_simulation()
