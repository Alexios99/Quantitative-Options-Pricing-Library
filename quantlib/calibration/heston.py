from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import warnings
import numpy as np
from scipy.optimize import minimize, brentq
from scipy.integrate import quad, IntegrationWarning
from quantlib.calibration.implied_vol import VolatilitySurface
from quantlib.pricing.analytical import BlackScholesEngine
from quantlib.core.payoffs import OptionContract, ExerciseStyle, OptionType

NUMERICAL_TOLERANCE = 1e-12
# Prices more negative than this indicate an integration problem rather than rounding noise
NEGATIVE_PRICE_TOLERANCE = 1e-8
BP_PER_VOL_UNIT = 1e4
# Penalty for a strike with no valid model implied vol: a 100 vol-point error, in bp^2
FAILED_POINT_PENALTY_BP2 = (1.0 * BP_PER_VOL_UNIT) ** 2
# Bounds on s = sigma / sqrt(2*kappa*theta) when the Feller condition is enforced
FELLER_RATIO_BOUNDS = (0.001, 1.0)


class HestonPricingError(RuntimeError):
    """Raised when the Fourier integration does not produce a valid price."""


class HestonCalibrationError(RuntimeError):
    """Raised when the calibration optimiser fails to converge."""


class FellerConditionWarning(UserWarning):
    """Calibrated parameters violate 2*kappa*theta >= sigma^2 (variance can reach zero)."""


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
    """Semi-analytical Heston pricer: numerical integration of the characteristic function.

    Call = S * P1 - K * exp(-rT) * P2, where
        P2 = 1/2 + 1/pi * int_0^inf Re[ exp(-iu ln K) * phi(u)     / (iu)          ] du
        P1 = 1/2 + 1/pi * int_0^inf Re[ exp(-iu ln K) * phi(u - i) / (iu * phi(-i)) ] du
    and phi is the characteristic function of ln S_T under the risk-neutral measure,
    with phi(-i) = E[S_T] = S * exp(rT).
    """

    def price(self, spot: float, strike: float, time_to_expiry: float,
              risk_free_rate: float, params: HestonParameters,
              option_type: str = "call") -> float:
        """Price a European call or put under Heston."""
        forward = spot * np.exp(risk_free_rate * time_to_expiry)
        # P1: stock-measure probability, uses phi(u - i) normalised by phi(-i) = forward
        P1 = self._probability(spot, strike, time_to_expiry, risk_free_rate, params,
                               shift=-1j, normaliser=forward)
        # P2: risk-neutral exercise probability, uses phi(u)
        P2 = self._probability(spot, strike, time_to_expiry, risk_free_rate, params,
                               shift=0.0, normaliser=1.0)

        discount_factor = np.exp(-risk_free_rate * time_to_expiry)
        call_price = spot * P1 - strike * discount_factor * P2

        if option_type.lower() == "call":
            price = call_price
        elif option_type.lower() == "put":
            # Put-call parity: Put = Call - S + K*exp(-r*T)
            price = call_price - spot + strike * discount_factor
        else:
            raise ValueError(f"Unknown option type: {option_type}")

        if price < -NEGATIVE_PRICE_TOLERANCE:
            raise HestonPricingError(
                f"Negative Heston price {price:.3e} (K={strike}, T={time_to_expiry}, {params})")
        return max(price, 0.0)  # clip rounding noise only

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

        return brentq(objective, 0.001, 5.0, xtol=1e-10)

    def _characteristic_function(self, u: complex, spot: float, time_to_expiry: float,
                                risk_free_rate: float, params: HestonParameters) -> complex:
        """Characteristic function of ln S_T, in the Albrecher et al. (2007) "little trap" form.

        Using exp(-dT) and g = (b - d) / (b + d) keeps the complex logarithm on its principal
        branch, avoiding the discontinuities of Heston's original formulation.
        """
        kappa, theta, sigma, rho, v0 = params.kappa, params.theta, params.sigma, params.rho, params.v0
        T = time_to_expiry
        b = kappa - rho * sigma * 1j * u
        d = np.sqrt(b ** 2 + sigma ** 2 * (1j * u + u ** 2))
        g = (b - d) / (b + d)
        exp_neg_dT = np.exp(-d * T)

        C = (kappa * theta / sigma ** 2) * ((b - d) * T - 2.0 * np.log((1.0 - g * exp_neg_dT) / (1.0 - g)))
        D = ((b - d) / sigma ** 2) * (1.0 - exp_neg_dT) / (1.0 - g * exp_neg_dT)

        return np.exp(C + D * v0 + 1j * u * (np.log(spot) + risk_free_rate * T))

    def _probability(self, spot: float, strike: float, time_to_expiry: float,
                     risk_free_rate: float, params: HestonParameters,
                     shift: complex, normaliser: float) -> float:
        """1/2 + 1/pi * int_0^inf Re[exp(-iu ln K) phi(u + shift) / (iu * normaliser)] du"""
        log_strike = np.log(strike)

        def integrand(u: float) -> float:
            if u < NUMERICAL_TOLERANCE:
                u = NUMERICAL_TOLERANCE  # integrand has a finite limit at u = 0
            cf = self._characteristic_function(u + shift, spot, time_to_expiry, risk_free_rate, params)
            return (np.exp(-1j * u * log_strike) * cf / (1j * u * normaliser)).real

        with warnings.catch_warnings():
            warnings.simplefilter("error", IntegrationWarning)
            try:
                result, _ = quad(integrand, 0.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=1000)
            except IntegrationWarning as exc:
                raise HestonPricingError(
                    f"Heston integral did not converge (K={strike}, T={time_to_expiry}, {params}): {exc}"
                ) from exc
        if not np.isfinite(result):
            raise HestonPricingError(f"Non-finite Heston integral (K={strike}, T={time_to_expiry}, {params})")
        return 0.5 + result / np.pi

class HestonCalibrator:
    """Calibrate Heston parameters to a single-expiry implied-volatility smile."""

    # Parameter bounds: [v0, kappa, theta, sigma, rho]
    BOUNDS = [
        (0.001, 1.0),    # v0
        (0.001, 10.0),   # kappa
        (0.001, 1.0),    # theta
        (0.001, 2.0),    # sigma
        (-0.99, 0.99)    # rho
    ]

    def __init__(self, pricing_engine: Optional[HestonPricingEngine] = None):
        self.pricing_engine = pricing_engine or HestonPricingEngine()

    def calibrate(self, market_data: List[Tuple[float, float]],
                 spot: float, risk_free_rate: float, time_to_expiry: float,
                 initial_guess: Optional[HestonParameters] = None,
                 enforce_feller: bool = False) -> HestonParameters:
        """Calibrate Heston parameters to (strike, implied vol) pairs.

        enforce_feller=False: unconstrained fit; warns with FellerConditionWarning if the
            result violates 2*kappa*theta >= sigma^2 (common for equity smiles).
        enforce_feller=True: optimises s in (0, 1] with sigma = s * sqrt(2*kappa*theta), so
            every candidate satisfies the Feller condition exactly.
        """
        if not market_data:
            raise ValueError("No market data provided for calibration")

        # Default initial guess
        if initial_guess is None:
            initial_guess = HestonParameters(
                v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7
            )

        args = (market_data, spot, risk_free_rate, time_to_expiry)
        x0 = [initial_guess.v0, initial_guess.kappa, initial_guess.theta,
              initial_guess.sigma, initial_guess.rho]

        if enforce_feller:
            to_model = self._feller_to_model
            s0 = initial_guess.sigma / np.sqrt(2 * initial_guess.kappa * initial_guess.theta)
            x0[3] = min(max(s0, FELLER_RATIO_BOUNDS[0]), FELLER_RATIO_BOUNDS[1])
            bounds = self.BOUNDS[:3] + [FELLER_RATIO_BOUNDS] + self.BOUNDS[4:]
        else:
            to_model = list
            bounds = self.BOUNDS

        result = minimize(lambda x: self._objective_function(to_model(x), *args), x0,
                          method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000})

        if not result.success:
            raise HestonCalibrationError(f"Heston calibration failed: {result.message}")

        params = HestonParameters(*to_model(result.x))
        if not enforce_feller and not params.feller_condition:
            warnings.warn(
                f"Calibrated parameters violate the Feller condition "
                f"(2*kappa*theta = {2 * params.kappa * params.theta:.4f} < sigma^2 = {params.sigma ** 2:.4f}); "
                f"pass enforce_feller=True to constrain the fit.",
                FellerConditionWarning, stacklevel=2)
        return params

    @staticmethod
    def _feller_to_model(x) -> List[float]:
        """Map [v0, kappa, theta, s, rho] to [v0, kappa, theta, sigma, rho], sigma = s*sqrt(2*kappa*theta)."""
        v0, kappa, theta, s, rho = x
        return [v0, kappa, theta, s * np.sqrt(2.0 * kappa * theta), rho]

    def rmse(self, params: HestonParameters, market_data: List[Tuple[float, float]],
             spot: float, risk_free_rate: float, time_to_expiry: float) -> float:
        """Root-mean-square implied-volatility error of a parameter set (in vol units)."""
        sse_bp2 = self._objective_function(
            [params.v0, params.kappa, params.theta, params.sigma, params.rho],
            market_data, spot, risk_free_rate, time_to_expiry)
        return float(np.sqrt(sse_bp2 / len(market_data))) / BP_PER_VOL_UNIT

    def _objective_function(self, params_array, market_data, spot,
                           risk_free_rate, time_to_expiry):
        """Sum of squared implied-vol errors, in vol basis points squared.

        Measuring errors in bp keeps the objective O(1)-O(100) near a good fit; in raw vol
        units it is ~1e-6, below L-BFGS-B's absolute stopping tolerance, so the optimiser
        would stop before fitting the smile.
        """
        try:
            heston_params = HestonParameters(*params_array)
        except ValueError:
            return FAILED_POINT_PENALTY_BP2 * len(market_data)  # outside the valid parameter region

        total_error = 0.0
        for strike, market_iv in market_data:
            try:
                model_iv = self.pricing_engine.implied_volatility(
                    spot, strike, time_to_expiry, risk_free_rate, heston_params
                )
            except (ValueError, HestonPricingError):
                # No implied vol exists for this price (outside BS bounds) or the
                # integral failed: penalise so the optimiser moves away from this region.
                total_error += FAILED_POINT_PENALTY_BP2
                continue
            total_error += ((market_iv - model_iv) * BP_PER_VOL_UNIT) ** 2
        return total_error

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
        for expiry in vol_surface.get_expiries():
            market_data = []
            for strike in vol_surface.get_strikes():
                iv = vol_surface._get_exact_iv(strike, expiry)
                if iv is not None:
                    market_data.append((strike, iv))
            if market_data:
                self.heston_params[expiry] = self.calibrator.calibrate(
                    market_data, self.spot_price, self.risk_free_rate, expiry
                )

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

        raise ValueError(f"Cannot interpolate parameters for expiry {target_expiry}")
