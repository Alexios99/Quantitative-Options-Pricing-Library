from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
import cmath
from scipy.optimize import minimize, brentq
from scipy.integrate import quad
from quantlib.calibration.implied_vol import VolatilitySurface
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.core.payoffs import OptionContract, ExerciseStyle, OptionType

NUMERICAL_TOLERANCE = 1e-12

@dataclass
class HestonParameters:
    """Heston model parameters with validation"""
    v0: float      # Initial variance
    kappa: float   # Mean reversion speed  
    theta: float   # Long-term variance
    sigma: float   # Volatility of volatility
    rho: float     # Correlation
    
    def __post_init__(self):
        """Validate Heston parameters"""
        if self.v0 < 0:
            raise ValueError(f"Initial variance must be non-negative: v0 = {self.v0}")
        if self.kappa < 0:
            raise ValueError(f"Mean reversion speed must be positive: kappa = {self.kappa}")
        if self.theta < 0:
            raise ValueError(f"Long-term variance must be non-negative: theta = {self.theta}")
        if self.sigma < 0:
            raise ValueError(f"Vol of vol must be non-negative: sigma = {self.sigma}")
        if abs(self.rho) > 1:
            raise ValueError(f"Correlation must be in [-1,1]: rho = {self.rho}")
    
    @property
    def feller_condition(self) -> bool:
        """Check Feller condition: 2κθ ≥ σ²"""
        return 2 * self.kappa * self.theta >= self.sigma ** 2

class HestonPricingEngine:
    """Hybrid Heston: You implement theory, scipy handles numerics"""
    
    def __init__(self):
        self.integration_limit = 100.0
        
    def price(self, spot: float, strike: float, time_to_expiry: float,
              risk_free_rate: float, params: HestonParameters, 
              option_type: str = "call") -> float:
        """Heston pricing using P1 and P2 probabilities"""
        # Calculate the two probability integrals
        # P1 corresponds to the delta term (measure change -> shift by -i), so j=1
        P1 = self._probability_integral(spot, strike, time_to_expiry, 
                                       risk_free_rate, params, j=0)
        # P2 corresponds to the risk-neutral probability, so j=0
        P2 = self._probability_integral(spot, strike, time_to_expiry, 
                                       risk_free_rate, params, j=1)
        
        # Heston call price formula: S*P1 - K*exp(-r*T)*P2
        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        
        if option_type.lower() == "call":
            price = spot * P1 - strike * discount_factor * P2
        elif option_type.lower() == "put":
            # Put-call parity: Put = Call - S + K*exp(-r*T)
            call_price = spot * P1 - strike * discount_factor * P2
            price = call_price - spot + strike * discount_factor
        else:
            raise ValueError(f"Unknown option type: {option_type}")
            
        return max(price, 0.0)  # Ensure non-negative price
    
    def implied_volatility(self, spot: float, strike: float, time_to_expiry: float,
                          risk_free_rate: float, params: HestonParameters) -> float:
        """Convert Heston price to Black-Scholes implied vol"""
        heston_price = self.price(spot, strike, time_to_expiry, risk_free_rate, params)
        
        # Create BS engine for IV calculation
        bs_engine = BlackScholesEngine()
        
        def objective(vol):
            contract = OptionContract(
                option=OptionType.CALL,
                strike=strike,
                time_to_expiry=time_to_expiry,
                spot=spot,
                risk_free_rate=risk_free_rate,
                volatility=vol,
                style=ExerciseStyle.EUROPEAN
            )
            bs_price = bs_engine.price(contract).price
            return bs_price - heston_price
        
        try:
            iv = brentq(objective, 0.001, 5.0, xtol=1e-6)
            return iv
        except (ValueError, RuntimeError):
            # Fallback: approximate IV using square root of average variance
            avg_var = params.v0 + (params.theta - params.v0) * (1 - np.exp(-params.kappa * time_to_expiry)) / (params.kappa * time_to_expiry)
            return np.sqrt(max(avg_var, 0.001))
    
    def _characteristic_function(self, u: complex, spot: float, time_to_expiry: float,
                                risk_free_rate: float, params: HestonParameters) -> complex:
        """Core Heston characteristic function φ(u)"""
        # Calculate discriminant d
        d = cmath.sqrt((params.kappa - params.rho * params.sigma * 1j * u)**2 + 
                       params.sigma**2 * (1j * u + u**2))
        
        # Calculate g factor
        numerator = params.kappa - params.rho * params.sigma * 1j * u - d
        denominator = params.kappa - params.rho * params.sigma * 1j * u + d
        g = numerator / denominator
        
        # Calculate D(u,T)
        exp_dt = cmath.exp(d * time_to_expiry)
        D = (numerator * (1 - exp_dt)) / (params.sigma**2 * (1 - g * exp_dt))
        
        # Calculate C(u,T)
        if abs(1 - g) < NUMERICAL_TOLERANCE:
            log_term = -d * time_to_expiry  # Limit case
        else:
            log_term = 2 * cmath.log((1 - g * exp_dt) / (1 - g))
        
        C = (params.kappa * params.theta / params.sigma**2) * (numerator * time_to_expiry - log_term)
        
        # Return full characteristic function
        return cmath.exp(C + D * params.v0 + 1j * u * (np.log(spot) + risk_free_rate * time_to_expiry))
    
    def _probability_integral(self, spot: float, strike: float, time_to_expiry: float,
                             risk_free_rate: float, params: HestonParameters, 
                             j: int) -> float:
        """Numerical integration using scipy"""
        def integrand(phi):
            return self._integrand(phi, spot, strike, time_to_expiry, 
                                 risk_free_rate, params, j)
        
        try:
            result, _ = quad(integrand, 0, self.integration_limit, 
                            epsabs=1e-8, limit=1000)
            return 0.5 + result / np.pi
        except Exception as e:
            # Fallback for numerical issues
            print(f"DEBUG: Integration failed for j={j}. Error: {e}")
            return 0.5 if j == 1 else 0.5
    
    def _integrand(self, phi: float, spot: float, strike: float,
                   time_to_expiry: float, risk_free_rate: float,
                   params: HestonParameters, j: int) -> float:
        """Integrand for probability calculation"""
        if abs(phi) < NUMERICAL_TOLERANCE:
            return 0.0
        
        # Calculate characteristic function at (phi - i*j)
        u = phi - 1j * j
        cf = self._characteristic_function(u, spot, time_to_expiry, risk_free_rate, params)
        
        # Calculate integrand: Re[exp(-i*phi*ln(K)) * CF] / phi
        integrand_complex = cmath.exp(-1j * phi * np.log(strike)) * cf / (1j * phi)
        
        return integrand_complex.real

class HestonCalibrator:
    """Calibration using scipy optimization"""
    
    def __init__(self, pricing_engine: Optional[HestonPricingEngine] = None):
        self.pricing_engine = pricing_engine or HestonPricingEngine()
    
    def calibrate(self, market_data: List[Tuple[float, float]], 
                 spot: float, risk_free_rate: float, time_to_expiry: float,
                 initial_guess: Optional[HestonParameters] = None) -> HestonParameters:
        """Calibrate Heston parameters to market data"""
        if not market_data:
            raise ValueError("No market data provided for calibration")
        
        # Default initial guess
        if initial_guess is None:
            initial_guess = HestonParameters(
                v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7
            )
        
        # Parameter bounds: [v0, kappa, theta, sigma, rho]
        bounds = [
            (0.001, 1.0),    # v0
            (0.001, 10.0),   # kappa
            (0.001, 1.0),    # theta
            (0.001, 2.0),    # sigma
            (-0.99, 0.99)    # rho
        ]
        
        # Initial parameter array
        x0 = [initial_guess.v0, initial_guess.kappa, initial_guess.theta, 
              initial_guess.sigma, initial_guess.rho]
        
        # Optimize
        result = minimize(
            self._objective_function,
            x0,
            args=(market_data, spot, risk_free_rate, time_to_expiry),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            print(f"Calibration warning: {result.message}")
        
        # Return calibrated parameters
        optimal_params = result.x
        return HestonParameters(
            v0=optimal_params[0],
            kappa=optimal_params[1],
            theta=optimal_params[2],
            sigma=optimal_params[3],
            rho=optimal_params[4]
        )
    
    def _objective_function(self, params_array, market_data, spot, 
                           risk_free_rate, time_to_expiry):
        """Sum of squared errors between market and model IVs"""
        try:
            # Convert to HestonParameters
            heston_params = HestonParameters(
                v0=params_array[0],
                kappa=params_array[1],
                theta=params_array[2],
                sigma=params_array[3],
                rho=params_array[4]
            )
            
            total_error = 0.0
            for strike, market_iv in market_data:
                try:
                    model_iv = self.pricing_engine.implied_volatility(
                        spot, strike, time_to_expiry, risk_free_rate, heston_params
                    )
                    error = (market_iv - model_iv) ** 2
                    total_error += error
                except:
                    # Heavy penalty for pricing failures
                    total_error += 1000.0
            
            return total_error
            
        except Exception:
            return 1e6  # Large penalty for invalid parameters

class HestonSurface:
    """Heston volatility surface management"""
    
    def __init__(self, spot_price: float, risk_free_rate: float):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.heston_params: Dict[float, HestonParameters] = {}
        self.pricing_engine = HestonPricingEngine()
        self.calibrator = HestonCalibrator(self.pricing_engine)
    
    def calibrate_to_surface(self, vol_surface: VolatilitySurface) -> None:
        """Calibrate Heston parameters for each expiry"""
        expiries = vol_surface.get_expiries()
        
        for expiry in expiries:
            strikes = vol_surface.get_strikes()
            market_data = []
            
            for strike in strikes:
                try:
                    iv = vol_surface.get_iv(strike, expiry)
                    if iv > 0:  # Valid IV
                        market_data.append((strike, iv))
                except:
                    continue
            
            if market_data:
                try:
                    params = self.calibrator.calibrate(
                        market_data, self.spot_price, self.risk_free_rate, expiry
                    )
                    self.heston_params[expiry] = params
                except Exception as e:
                    print(f"Failed to calibrate expiry {expiry}: {e}")
    
    def add_expiry(self, expiry: float, market_data: List[Tuple[float, float]]) -> HestonParameters:
        """Add Heston calibration for specific expiry"""
        params = self.calibrator.calibrate(
            market_data, self.spot_price, self.risk_free_rate, expiry
        )
        self.heston_params[expiry] = params
        return params
    
    def get_implied_volatility(self, strike: float, expiry: float) -> float:
        """Get Heston IV with parameter interpolation if needed"""
        if expiry in self.heston_params:
            params = self.heston_params[expiry]
        else:
            params = self._interpolate_parameters(expiry)
        
        return self.pricing_engine.implied_volatility(
            self.spot_price, strike, expiry, self.risk_free_rate, params
        )
    
    def get_heston_parameters(self, expiry: float) -> HestonParameters:
        """Get Heston parameters with interpolation if needed"""
        if expiry in self.heston_params:
            return self.heston_params[expiry]
        else:
            return self._interpolate_parameters(expiry)
    
    def _interpolate_parameters(self, target_expiry: float) -> HestonParameters:
        """Linear interpolation of Heston parameters"""
        if not self.heston_params:
            raise ValueError("No calibrated parameters available for interpolation")
        
        expiries = sorted(self.heston_params.keys())
        
        if target_expiry <= expiries[0]:
            return self.heston_params[expiries[0]]
        elif target_expiry >= expiries[-1]:
            return self.heston_params[expiries[-1]]
        else:
            # Find bracketing expiries
            for i in range(len(expiries) - 1):
                if expiries[i] <= target_expiry <= expiries[i + 1]:
                    t1, t2 = expiries[i], expiries[i + 1]
                    p1, p2 = self.heston_params[t1], self.heston_params[t2]
                    
                    # Linear interpolation weight
                    w = (target_expiry - t1) / (t2 - t1)
                    
                    return HestonParameters(
                        v0=p1.v0 + w * (p2.v0 - p1.v0),
                        kappa=p1.kappa + w * (p2.kappa - p1.kappa),
                        theta=p1.theta + w * (p2.theta - p1.theta),
                        sigma=p1.sigma + w * (p2.sigma - p1.sigma),
                        rho=p1.rho + w * (p2.rho - p1.rho)
                    )
        
        # Fallback
        return list(self.heston_params.values())[0]