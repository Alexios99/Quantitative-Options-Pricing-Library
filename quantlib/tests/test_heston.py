import warnings

import numpy as np
import pytest

from quantlib.calibration.heston import (
    FellerConditionWarning,
    HestonCalibrator,
    HestonParameters,
    HestonPricingEngine,
)
from quantlib.core.payoffs import ExerciseStyle, OptionContract, OptionType
from quantlib.pricing.analytical import BlackScholesEngine


@pytest.fixture
def engine():
    return HestonPricingEngine()


def _bs_call(spot, strike, T, r, vol):
    contract = OptionContract(spot, strike, T, r, vol, OptionType.CALL, ExerciseStyle.EUROPEAN)
    return BlackScholesEngine().price(contract).price


class TestHestonPricing:
    def test_published_reference_value(self, engine):
        """Fang, F. & Oosterlee, C. W. (2008), "A Novel Pricing Method for European Options
        Based on Fourier-Cosine Series Expansions", SIAM J. Sci. Comput. 31(2), 826-848,
        Heston test case: S0=100, K=100, T=1, r=0, kappa=1.5768, theta=0.0398,
        sigma=0.5751, rho=-0.5711, v0=0.0175; reference call price 5.785155450."""
        params = HestonParameters(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
        price = engine.price(spot=100, strike=100, time_to_expiry=1.0, risk_free_rate=0.0, params=params)
        assert price == pytest.approx(5.785155450, abs=1e-6)

    @pytest.mark.parametrize("strike", [80, 100, 120])
    @pytest.mark.parametrize(
        "rho, min_ratio_per_decade, max_error_at_1e3",
        [
            (0.0, 50.0, 1e-5),   # no skew term: error is O(sigma^2), falls ~100x per decade
            (-0.5, 5.0, 5e-3),   # skew term rho*sigma dominates: error is O(sigma), ~10x per decade
        ],
    )
    def test_converges_to_black_scholes_as_vol_of_vol_vanishes(
        self, engine, strike, rho, min_ratio_per_decade, max_error_at_1e3
    ):
        """With v0 = theta, variance is deterministic and constant in the sigma -> 0 limit,
        so the Heston price must converge to Black-Scholes with vol sqrt(theta).

        Rates are checked between sigma = 0.01 and 0.001, inside the asymptotic regime. At larger
        sigma the O(sigma) and O(sigma^2) terms can partially cancel (they do near ATM for rho < 0),
        so a single step from 0.1 does not show the asymptotic rate."""
        spot, T, r, theta = 100.0, 1.0, 0.05, 0.04
        bs = _bs_call(spot, strike, T, r, np.sqrt(theta))
        errors = []
        for sigma in [0.01, 0.001]:
            params = HestonParameters(v0=theta, kappa=2.0, theta=theta, sigma=sigma, rho=rho)
            errors.append(abs(engine.price(spot, strike, T, r, params) - bs))
        assert errors[1] < errors[0] / min_ratio_per_decade
        assert errors[1] < max_error_at_1e3

    def test_put_call_parity_and_bounds(self, engine):
        params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
        spot, strike, T, r = 100.0, 110.0, 0.5, 0.03
        call = engine.price(spot, strike, T, r, params, "call")
        put = engine.price(spot, strike, T, r, params, "put")
        assert call - put == pytest.approx(spot - strike * np.exp(-r * T), abs=1e-10)
        assert max(spot - strike * np.exp(-r * T), 0.0) < call < spot
        assert max(strike * np.exp(-r * T) - spot, 0.0) < put < strike * np.exp(-r * T)

    def test_call_price_decreases_with_strike(self, engine):
        params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.3, rho=-0.7)
        prices = [engine.price(100, K, 1.0, 0.05, params) for K in (70, 85, 100, 115, 130)]
        assert all(a > b for a, b in zip(prices, prices[1:]))

    def test_negative_correlation_produces_downward_skew(self, engine):
        params = HestonParameters(v0=0.04, kappa=2.0, theta=0.04, sigma=0.5, rho=-0.7)
        ivs = [engine.implied_volatility(100, K, 1.0, 0.0, params) for K in (80, 100, 120)]
        assert ivs[0] > ivs[1] > ivs[2]


class TestHestonCalibration:
    STRIKES = [80, 90, 100, 110, 120]

    def _synthetic_smile(self, engine, params, spot=100.0, r=0.02, T=1.0):
        return [(K, engine.implied_volatility(spot, K, T, r, params)) for K in self.STRIKES]

    @pytest.mark.slow
    def test_calibration_fits_synthetic_smile(self, engine):
        true = HestonParameters(v0=0.05, kappa=1.5, theta=0.06, sigma=0.4, rho=-0.6)  # satisfies Feller
        smile = self._synthetic_smile(engine, true)
        calibrator = HestonCalibrator(engine)
        fitted = calibrator.calibrate(smile, spot=100.0, risk_free_rate=0.02, time_to_expiry=1.0)
        # A single smile does not identify all five parameters, so test the fit, not the parameters
        assert calibrator.rmse(fitted, smile, 100.0, 0.02, 1.0) < 1e-4  # < 1 vol bp

    def test_feller_violation_warns(self, engine, monkeypatch):
        calibrator = HestonCalibrator(engine)
        violating = HestonParameters(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
        assert not violating.feller_condition

        class Result:  # stand-in optimiser result so the test isolates the Feller check
            success = True
            x = np.array([violating.v0, violating.kappa, violating.theta, violating.sigma, violating.rho])

        monkeypatch.setattr("quantlib.calibration.heston.minimize", lambda *a, **k: Result())
        with pytest.warns(FellerConditionWarning):
            calibrator.calibrate([(100, 0.2)], spot=100.0, risk_free_rate=0.0, time_to_expiry=1.0)

    @pytest.mark.slow
    def test_enforce_feller_returns_feller_compliant_parameters(self, engine):
        # Smile generated from Feller-violating parameters: the constrained fit must still comply
        true = HestonParameters(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
        smile = self._synthetic_smile(engine, true, r=0.0)
        calibrator = HestonCalibrator(engine)
        with warnings.catch_warnings():
            warnings.simplefilter("error", FellerConditionWarning)
            fitted = calibrator.calibrate(smile, spot=100.0, risk_free_rate=0.0, time_to_expiry=1.0,
                                          enforce_feller=True)
        assert 2 * fitted.kappa * fitted.theta - fitted.sigma ** 2 >= -1e-8

    def test_failed_optimisation_raises(self, engine, monkeypatch):
        from quantlib.calibration.heston import HestonCalibrationError

        class Result:
            success = False
            message = "forced failure"
            x = np.array([0.04, 2.0, 0.04, 0.3, -0.7])

        monkeypatch.setattr("quantlib.calibration.heston.minimize", lambda *a, **k: Result())
        with pytest.raises(HestonCalibrationError):
            HestonCalibrator(engine).calibrate([(100, 0.2)], spot=100.0, risk_free_rate=0.0, time_to_expiry=1.0)
