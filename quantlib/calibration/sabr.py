from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
from scipy.optimize import minimize
import math
from quantlib.calibration.implied_vol import VolatilitySurface

NUMERICAL_TOLERANCE = 1e-12

@dataclass
class SABRParameters:
    """SABR model parameters with validation"""
    alpha: float    # Initial volatility level
    beta: float     # CEV exponent (0=normal, 1=lognormal)
    nu: float       # Volatility of volatility  
    rho: float      # Correlation between price and vol
    
    def __post_init__(self):
        """Validate parameter bounds"""
        if self.alpha < 0:
            raise ValueError(f"Alpha is less than 0. Alpha = {self.alpha}")
        if self.beta > 1 or self.beta < 0:
            raise ValueError(f"Beta is {self.beta}")
        if self.nu < 0:
            raise ValueError(f"Nu is {self.nu}")
        if self.rho > 1 or self.rho < -1:
            raise ValueError(f"Rho is {self.rho}")
    
    @property
    def is_normal_sabr(self) -> bool:
        """Check if this is approximately normal SABR (β ≈ 0)"""
        if abs(self.beta - 0) < NUMERICAL_TOLERANCE:
            return True
    
    @property  
    def is_lognormal_sabr(self) -> bool:
        """Check if this is approximately lognormal SABR (β ≈ 1)"""
        if abs(self.beta - 1) < NUMERICAL_TOLERANCE:
            return True

class SABRPricingEngine:
    """SABR implied volatility calculation using Hagan approximation"""
    
    def __init__(self):
        pass
    
    def implied_volatility(self, forward: float, strike: float, time_to_expiry: float, 
                          params: SABRParameters) -> float:
        """Calculate SABR implied volatility using Hagan approximation"""
        # Handle ATM case
        if abs(forward - strike) < NUMERICAL_TOLERANCE:
            return self.atm_volatility(forward, time_to_expiry, params)
        
        # Calculate z and x
        z = self._z_function(params.alpha, params.nu, forward, strike, params.beta)
        x = self._x_function(z, params.rho)
        
        # Base term
        base = self._hagan_base_term(params.alpha, params.beta, forward, strike)
        
        # Correction terms
        corrections = self._hagan_correction_terms(params.beta, params.nu, params.rho, forward, strike, time_to_expiry)
        
        zx_factor = z / x if abs(x) > NUMERICAL_TOLERANCE else 1.0
        
        return base * corrections * zx_factor

    
    def atm_volatility(self, forward: float, time_to_expiry: float, 
                      params: SABRParameters) -> float:
        """Calculate ATM volatility (simplified case F = K)"""
        base_vol = params.alpha * forward**(params.beta - 1)

        time_correction = (1 + params.nu**2/24 * time_to_expiry)

        return base_vol * time_correction
    
    def _z_function(self, alpha: float, nu: float, forward: float, 
                   strike: float, beta: float) -> float:
        """Calculate z parameter in Hagan approximation"""
        if alpha == 0:
            return 0
        z = (nu/alpha) * ((forward * strike)**((1-beta)/2)) * np.log(forward/strike)
        return z
    
    def _x_function(self, z, rho):
        # Handle small z case (use Taylor expansion)
        if abs(z) < NUMERICAL_TOLERANCE:
            return z/(1-rho) if abs(1-rho) > NUMERICAL_TOLERANCE else z
        
        # Handle rho ≈ 1 case
        if abs(1 - rho) < NUMERICAL_TOLERANCE:
            return z
        
        return np.log((np.sqrt(1-2*rho*z + np.square(z)) + z - rho)/(1-rho))
    

    
    def _hagan_base_term(self, alpha: float, beta: float, forward: float, 
                        strike: float) -> float:
        """Calculate base term: α / [(F×K)^((1-β)/2)]"""
        if beta == 1:
            return alpha
        else:
            return alpha / ((forward * strike)**((1-beta)/2))
    
    def _hagan_correction_terms(self, beta, nu, rho, forward, strike, time_to_expiry):
        # Log term corrections
        log_fk = math.log(forward / strike)
        log_corrections = 1 + (1-beta)**2/24 * log_fk**2 
        
        # Time-dependent corrections  
        time_corrections = 1 + nu**2/24 * time_to_expiry
        
        return log_corrections * time_corrections

class SABRCalibrator:
    """SABR parameter calibration to market implied volatilities"""
    
    def __init__(self, pricing_engine: Optional[SABRPricingEngine] = None):
        self.pricing_engine = pricing_engine or SABRPricingEngine()
    
    def calibrate(self, market_data: List[Tuple[float, float]], forward: float, 
                 time_to_expiry: float, initial_guess: Optional[SABRParameters] = None) -> SABRParameters:
        """Calibration using scipy"""
        
        # Default initial guess
        if initial_guess is None:
            initial_guess = SABRParameters(alpha=0.2, beta=0.5, nu=0.3, rho=0.0)
        
        # Convert to array
        x0 = [initial_guess.alpha, initial_guess.beta, initial_guess.nu, initial_guess.rho]
        bounds = [(0.01, 2.0), (0.1, 0.9), (0.01, 1.0), (-0.8, 0.8)]
        
        # Optimize
        from scipy.optimize import minimize
        try:
            result = minimize(
                self._objective_function, x0, 
                args=(market_data, forward, time_to_expiry),
                method='L-BFGS-B', bounds=bounds, 
                options={'maxiter': 100}
            )
            return SABRParameters(*result.x) if result.success else initial_guess
        except:
            return initial_guess
    
    def _objective_function(self, params_array, market_data, forward, time_to_expiry):
        try:
            params = SABRParameters(*params_array)
        except ValueError:
            return 1e6
        
        total_error = 0.0
        for strike, market_iv in market_data:
            try:
                sabr_iv = self.pricing_engine.implied_volatility(forward, strike, time_to_expiry, params)
                error = abs(market_iv - sabr_iv)
                total_error += error**2
            except:
                return 1e6
        return total_error

class SABRSurface:
    """SABR volatility surface for multiple expiries"""
    
    def __init__(self, spot_price: float, risk_free_rate: float):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.sabr_params: Dict[float, SABRParameters] = {}  # expiry -> parameters
        self.pricing_engine = SABRPricingEngine()
        self.calibrator = SABRCalibrator(self.pricing_engine)
    
    def calibrate_to_surface(self, vol_surface: VolatilitySurface) -> None:
        """Calibrate SABR parameters for each expiry in volatility surface"""
        
        # Get all expiries from the IV surface
        expiries = vol_surface.get_expiries()
        
        for expiry in expiries:
            # Extract market data for this expiry
            market_data = []
            strikes = vol_surface.get_strikes()
            
            for strike in strikes:
                try:
                    iv = vol_surface._get_exact_iv(strike, expiry)
                    if iv is not None:
                        market_data.append((strike, iv))
                except:
                    continue
            
            # Calibrate SABR for this expiry
            if len(market_data) >= 4:  # Need minimum points
                forward = self._calculate_forward(expiry)
                params = self.calibrator.calibrate(market_data, forward, expiry)
                self.sabr_params[expiry] = params
        
    def add_expiry(self, expiry: float, market_data: List[Tuple[float, float]], 
               forward: Optional[float] = None) -> SABRParameters:
        """Add SABR calibration for specific expiry"""
        
        if forward is None:
            forward = self._calculate_forward(expiry)
        
        # Calibrate SABR parameters
        params = self.calibrator.calibrate(market_data, forward, expiry)
        self.sabr_params[expiry] = params
        
        return params
    
    def get_implied_volatility(self, strike: float, expiry: float, 
                          forward: Optional[float] = None) -> float:
        """Get SABR implied volatility for any strike/expiry"""
        
        # Get forward if not provided
        if forward is None:
            forward = self._calculate_forward(expiry)
        
        # Get SABR parameters (with interpolation if needed)
        params = self.get_sabr_parameters(expiry)
        
        # Calculate IV using SABR pricing engine
        return self.pricing_engine.implied_volatility(forward, strike, expiry, params)
    
    def get_sabr_parameters(self, expiry: float) -> SABRParameters:
        """Get SABR parameters for specific expiry (with interpolation if needed)"""
        # Exact match
        if expiry in self.sabr_params:
            return self.sabr_params[expiry]
        
        # Interpolate if needed
        return self._interpolate_parameters(expiry)

    def _interpolate_parameters(self, target_expiry: float) -> SABRParameters:
        """Simple linear interpolation of SABR parameters"""
        
        expiries = sorted(self.sabr_params.keys())
        if not expiries:
            raise ValueError("No calibrated parameters available")
        
        # Extrapolation cases
        if target_expiry <= expiries[0]:
            return self.sabr_params[expiries[0]]
        if target_expiry >= expiries[-1]:
            return self.sabr_params[expiries[-1]]
        
        # Find bracketing expiries
        for i in range(len(expiries)-1):
            if expiries[i] <= target_expiry <= expiries[i+1]:
                t1, t2 = expiries[i], expiries[i+1]
                params1 = self.sabr_params[t1]
                params2 = self.sabr_params[t2]
                
                # Linear interpolation weight
                w = (target_expiry - t1) / (t2 - t1)
                
                # Interpolate each parameter
                alpha = params1.alpha + w * (params2.alpha - params1.alpha)
                beta = params1.beta + w * (params2.beta - params1.beta)
                nu = params1.nu + w * (params2.nu - params1.nu)
                rho = params1.rho + w * (params2.rho - params1.rho)
                
                return SABRParameters(alpha, beta, nu, rho)
        
        raise ValueError(f"Cannot interpolate parameters for expiry {target_expiry}")
    
    def _calculate_forward(self, expiry: float) -> float:
        """Calculate forward price F = S * exp(r * T)"""
        return self.spot_price * np.exp(self.risk_free_rate * expiry)
    
    






