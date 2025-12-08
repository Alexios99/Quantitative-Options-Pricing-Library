from dataclasses import dataclass
from typing import Optional
import numpy as np

from quantlib.core.payoffs import OptionContract
from quantlib.pricing.analytical import PricingEngine

@dataclass
class PnLComponents:
    """Breakdown of Profit and Loss into Greek-driven components."""
    actual_pnl: float
    delta_pnl: float
    gamma_pnl: float
    theta_pnl: float
    vega_pnl: float
    rho_pnl: float
    unexplained_pnl: float
    
    # Market factor changes for reference
    d_spot: float
    d_vol: float
    d_time: float
    d_rate: float

def attribute_pnl(
    contract_t0: OptionContract, 
    contract_t1: OptionContract, 
    engine: PricingEngine
) -> PnLComponents:
    """
    Explain the PnL between two contract states (t0 and t1) using Taylor series expansion (Greeks).
    
    Args:
        contract_t0: Option contract state at start of period
        contract_t1: Option contract state at end of period
        engine: Pricing engine to use for Greeks and prices (usually Black-Scholes)
        
    Returns:
        PnLComponents object containing the attribution
    """
    
    # 1. Calculate Prices at t0 and t1
    price_t0 = engine.price(contract_t0).price
    price_t1 = engine.price(contract_t1).price
    actual_pnl = price_t1 - price_t0
    
    # 2. Calculate Greeks at t0 (base for Taylor expansion)
    greeks = engine.greeks(contract_t0)
    
    # 3. Calculate changes in market factors
    d_spot = contract_t1.spot - contract_t0.spot
    d_vol = contract_t1.volatility - contract_t0.volatility
    d_time = contract_t0.time_to_expiry - contract_t1.time_to_expiry # Note: dt is positive time elapsed
    d_rate = contract_t1.risk_free_rate - contract_t0.risk_free_rate
    
    # 4. Calculate PnL Components (First & Second Order Taylor Approximation)
    
    # Delta PnL: Linear effect of spot change
    delta_pnl = greeks.delta_call * d_spot if contract_t0.option.name == "CALL" else greeks.delta_put * d_spot
    # Note: Using the correct delta based on option type. 
    # However, greeks object usually separates them. Let's check how greeks are accessed.
    # Assuming greeks.delta_call / delta_put based on previous file views.
    # To be safe and generic, let's pick the right one.
    
    is_call = (contract_t0.option.name == "CALL") or (str(contract_t0.option) == "OptionType.CALL")
    delta = greeks.delta_call if is_call else greeks.delta_put
    delta_pnl = delta * d_spot
    
    # Gamma PnL: Second order effect of spot change (0.5 * Gamma * dS^2)
    gamma_pnl = 0.5 * greeks.gamma * (d_spot ** 2)
    
    # Theta PnL: Effect of time passing (Theta * dt)
    # Theta is usually reported as "per day" or "per year". 
    # In BS engine, theta is typically per year if inputs are annualized.
    # We need to ensure signs match. Theta is usually negative (decay).
    # d_time is positive (elapsed time).
    # So Theta * d_time should be negative PnL for long holders.
    
    theta = greeks.theta_call if is_call else greeks.theta_put
    theta_pnl = theta * d_time
    
    # Vega PnL: Effect of volatility change
    vega_pnl = greeks.vega * d_vol
    
    # Rho PnL: Effect of interest rate change
    rho = greeks.rho_call if is_call else greeks.rho_put
    rho_pnl = rho * d_rate
    
    # 5. Calculate Unexplained PnL (Residual)
    explained_pnl = delta_pnl + gamma_pnl + theta_pnl + vega_pnl + rho_pnl
    unexplained_pnl = actual_pnl - explained_pnl
    
    return PnLComponents(
        actual_pnl=actual_pnl,
        delta_pnl=delta_pnl,
        gamma_pnl=gamma_pnl,
        theta_pnl=theta_pnl,
        vega_pnl=vega_pnl,
        rho_pnl=rho_pnl,
        unexplained_pnl=unexplained_pnl,
        d_spot=d_spot,
        d_vol=d_vol,
        d_time=d_time,
        d_rate=d_rate
    )
