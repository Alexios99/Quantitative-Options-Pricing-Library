from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy.optimize import minimize
from quantlib.calibration.implied_vol import VolatilitySurface

NUMERICAL_TOLERANCE = 1e-12
# Below this |z|, z/x(z) is evaluated by its Taylor series to avoid 0/0
SMALL_Z = 1e-7
BP_PER_VOL_UNIT = 1e4
DEFAULT_BETA = 0.5

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
        if self.rho >= 1 or self.rho <= -1:
            raise ValueError(f"Rho must be in (-1, 1): {self.rho}")

    @property
    def is_normal_sabr(self) -> bool:
        """Check if this is approximately normal SABR (β ≈ 0)"""
        return abs(self.beta) < NUMERICAL_TOLERANCE

    @property
    def is_lognormal_sabr(self) -> bool:
        """Check if this is approximately lognormal SABR (β ≈ 1)"""
        return abs(self.beta - 1) < NUMERICAL_TOLERANCE


@dataclass
class SABRCalibrationResult:
    """Calibrated SABR parameters and the quality of the fit."""
    params: SABRParameters
    rmse: float        # root-mean-square implied-vol error, in vol units (0.0001 = 1 vol bp)
    n_points: int

    @property
    def rmse_bp(self) -> float:
        return self.rmse * BP_PER_VOL_UNIT


class SABRCalibrationError(RuntimeError):
    """Raised when the SABR calibration optimiser fails to converge."""


class SABRPricingEngine:
    """SABR implied volatility from the Hagan et al. (2002) lognormal approximation:

        sigma_B(K) = alpha / { (FK)^((1-b)/2) [1 + (1-b)^2/24 L^2 + (1-b)^4/1920 L^4] }
                     * z / x(z)
                     * { 1 + [ (1-b)^2/24 alpha^2/(FK)^(1-b) + rho b nu alpha / (4 (FK)^((1-b)/2))
                               + (2 - 3 rho^2)/24 nu^2 ] T }

    with L = ln(F/K), z = nu/alpha (FK)^((1-b)/2) L and
    x(z) = ln[ (sqrt(1 - 2 rho z + z^2) + z - rho) / (1 - rho) ].
    At K = F, z/x(z) -> 1 and L = 0, which gives the ATM formula.
    """

    def implied_volatility(self, forward: float, strike: float, time_to_expiry: float,
                          params: SABRParameters) -> float:
        """Calculate SABR implied volatility using the Hagan approximation"""
        if forward <= 0 or strike <= 0:
            raise ValueError(f"Forward and strike must be positive (F={forward}, K={strike})")
        alpha, beta, nu, rho = params.alpha, params.beta, params.nu, params.rho
        one_minus_beta = 1.0 - beta
        log_fk = np.log(forward / strike)
        fk_half = (forward * strike) ** (one_minus_beta / 2.0)   # (FK)^((1-b)/2)

        denominator = fk_half * (1.0
                                 + one_minus_beta ** 2 / 24.0 * log_fk ** 2
                                 + one_minus_beta ** 4 / 1920.0 * log_fk ** 4)
        z = (nu / alpha) * fk_half * log_fk if alpha > 0 else 0.0
        time_correction = 1.0 + (one_minus_beta ** 2 / 24.0 * alpha ** 2 / fk_half ** 2
                                 + rho * beta * nu * alpha / (4.0 * fk_half)
                                 + (2.0 - 3.0 * rho ** 2) / 24.0 * nu ** 2) * time_to_expiry

        return alpha / denominator * self._z_over_x(z, rho) * time_correction

    def atm_volatility(self, forward: float, time_to_expiry: float,
                      params: SABRParameters) -> float:
        """ATM implied volatility (K = F)."""
        return self.implied_volatility(forward, forward, time_to_expiry, params)

    @staticmethod
    def _z_over_x(z: float, rho: float) -> float:
        """z / x(z), with the Taylor expansion 1 - rho*z/2 near z = 0."""
        if abs(z) < SMALL_Z:
            return 1.0 - 0.5 * rho * z
        x = np.log((np.sqrt(1.0 - 2.0 * rho * z + z * z) + z - rho) / (1.0 - rho))
        return z / x


class SABRCalibrator:
    """Calibrate SABR (alpha, rho, nu) to a single-expiry smile with beta held fixed.

    beta is not fitted: on a single smile, beta and rho both control the skew and are
    close to unidentifiable, so beta is chosen up front (0.5 by default; 1 for lognormal
    dynamics, 0 for normal).
    """

    # (alpha, rho, nu)
    BOUNDS = [(1e-8, None), (-0.999, 0.999), (1e-8, 5.0)]

    def __init__(self, pricing_engine: Optional[SABRPricingEngine] = None, beta: float = DEFAULT_BETA):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.pricing_engine = pricing_engine or SABRPricingEngine()
        self.beta = beta

    def calibrate(self, market_data: List[Tuple[float, float]], forward: float,
                 time_to_expiry: float, initial_guess: Optional[SABRParameters] = None) -> SABRCalibrationResult:
        """Fit (alpha, rho, nu) to (strike, implied vol) pairs; raises SABRCalibrationError on failure."""
        if len(market_data) < 3:
            raise ValueError(f"Need at least 3 quotes to fit alpha, rho and nu; got {len(market_data)}")

        if initial_guess is None:
            # alpha from the quote nearest the forward: sigma_ATM ~ alpha / F^(1-beta)
            _, atm_iv = min(market_data, key=lambda q: abs(q[0] - forward))
            x0 = [atm_iv * forward ** (1.0 - self.beta), 0.0, 0.3]
        else:
            x0 = [initial_guess.alpha, initial_guess.rho, initial_guess.nu]

        result = minimize(
            self._objective_function, x0,
            args=(market_data, forward, time_to_expiry),
            method='L-BFGS-B', bounds=self.BOUNDS,
            options={'maxiter': 1000}
        )
        if not result.success:
            raise SABRCalibrationError(f"SABR calibration failed: {result.message}")

        alpha, rho, nu = result.x
        params = SABRParameters(alpha=alpha, beta=self.beta, nu=nu, rho=rho)
        return SABRCalibrationResult(
            params=params,
            rmse=self.rmse(params, market_data, forward, time_to_expiry),
            n_points=len(market_data),
        )

    def rmse(self, params: SABRParameters, market_data: List[Tuple[float, float]],
             forward: float, time_to_expiry: float) -> float:
        """Root-mean-square implied-vol error (vol units)."""
        errors = [self.pricing_engine.implied_volatility(forward, strike, time_to_expiry, params) - iv
                  for strike, iv in market_data]
        return float(np.sqrt(np.mean(np.square(errors))))

    def _objective_function(self, x, market_data, forward, time_to_expiry):
        """Sum of squared implied-vol errors in vol bp^2.

        In raw vol units the objective is ~1e-6 near a good fit, below L-BFGS-B's absolute
        stopping tolerance, so the optimiser would stop before fitting the smile.
        """
        alpha, rho, nu = x
        params = SABRParameters(alpha=alpha, beta=self.beta, nu=nu, rho=rho)
        total_error = 0.0
        for strike, market_iv in market_data:
            sabr_iv = self.pricing_engine.implied_volatility(forward, strike, time_to_expiry, params)
            total_error += ((market_iv - sabr_iv) * BP_PER_VOL_UNIT) ** 2
        return total_error

class SABRSurface:
    """SABR volatility surface for multiple expiries"""

    def __init__(self, spot_price: float, risk_free_rate: float, beta: float = DEFAULT_BETA):
        self.spot_price = spot_price
        self.risk_free_rate = risk_free_rate
        self.sabr_params: Dict[float, SABRParameters] = {}  # expiry -> parameters
        self.fit_rmse: Dict[float, float] = {}                # expiry -> calibration RMSE (vol units)
        self.pricing_engine = SABRPricingEngine()
        self.calibrator = SABRCalibrator(self.pricing_engine, beta=beta)

    def calibrate_to_surface(self, vol_surface: VolatilitySurface) -> None:
        """Calibrate SABR parameters for each expiry with at least 4 quoted strikes"""
        for expiry in vol_surface.get_expiries():
            market_data = []
            for strike in vol_surface.get_strikes():
                iv = vol_surface._get_exact_iv(strike, expiry)
                if iv is not None:
                    market_data.append((strike, iv))
            if len(market_data) >= 4:
                self.add_expiry(expiry, market_data)

    def add_expiry(self, expiry: float, market_data: List[Tuple[float, float]],
               forward: Optional[float] = None) -> SABRParameters:
        """Add SABR calibration for specific expiry"""
        if forward is None:
            forward = self._calculate_forward(expiry)
        result = self.calibrator.calibrate(market_data, forward, expiry)
        self.sabr_params[expiry] = result.params
        self.fit_rmse[expiry] = result.rmse
        return result.params

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
