import numpy as np
import pytest

from quantlib.calibration.sabr import (
    SABRCalibrationError,
    SABRCalibrator,
    SABRParameters,
    SABRPricingEngine,
)

F, T = 100.0, 1.0
STRIKES = [70, 80, 90, 95, 100, 105, 110, 120, 130]


@pytest.fixture
def engine():
    return SABRPricingEngine()


class TestHaganFormula:
    def test_zero_vol_of_vol_lognormal_is_flat_at_alpha(self, engine):
        """beta=1, nu=0 is Black-Scholes with constant vol alpha: every correction term vanishes."""
        params = SABRParameters(alpha=0.25, beta=1.0, nu=0.0, rho=0.0)
        for K in STRIKES:
            assert engine.implied_volatility(F, K, T, params) == pytest.approx(0.25, abs=1e-14)

    def test_atm_matches_closed_form(self, engine):
        """At K = F: sigma = alpha/F^(1-b) * [1 + ((1-b)^2/24 alpha^2/F^(2-2b)
        + rho b nu alpha/(4 F^(1-b)) + (2 - 3 rho^2)/24 nu^2) T]."""
        a, b, nu, rho = 2.0, 0.5, 0.4, -0.3
        expected = a / F ** (1 - b) * (1 + ((1 - b) ** 2 / 24 * a ** 2 / F ** (2 - 2 * b)
                                            + rho * b * nu * a / (4 * F ** (1 - b))
                                            + (2 - 3 * rho ** 2) / 24 * nu ** 2) * T)
        params = SABRParameters(alpha=a, beta=b, nu=nu, rho=rho)
        assert engine.atm_volatility(F, T, params) == pytest.approx(expected, rel=1e-14)

    def test_continuous_through_the_money(self, engine):
        params = SABRParameters(alpha=2.0, beta=0.5, nu=0.4, rho=-0.3)
        atm = engine.implied_volatility(F, F, T, params)
        # dSigma/d(ln K) is O(0.1) here, so a relative strike move eps must move the vol by < eps;
        # eps = 1e-9 also exercises the z -> 0 branch that would otherwise give 0/0.
        for eps in (1e-4, 1e-6, 1e-9):
            assert abs(engine.implied_volatility(F, F * (1 + eps), T, params) - atm) < eps
            assert abs(engine.implied_volatility(F, F * (1 - eps), T, params) - atm) < eps

    def test_negative_rho_gives_downward_skew(self, engine):
        params = SABRParameters(alpha=2.0, beta=0.5, nu=0.4, rho=-0.5)
        ivs = [engine.implied_volatility(F, K, T, params) for K in (80, 100, 120)]
        assert ivs[0] > ivs[1] > ivs[2]

    def test_includes_alpha_dependent_time_correction(self, engine):
        """The ATM vol must grow with T through the alpha^2 and rho*beta*nu*alpha terms,
        not only through nu^2: with nu = 0 and beta < 1 the T-dependence is (1-b)^2 alpha^2 / 24."""
        params = SABRParameters(alpha=2.0, beta=0.5, nu=0.0, rho=0.0)
        base = params.alpha / F ** 0.5
        vol_t1 = engine.atm_volatility(F, 1.0, params)
        assert vol_t1 == pytest.approx(base * (1 + 0.25 / 24 * 4.0 / F), rel=1e-14)


class TestCalibration:
    TRUE = SABRParameters(alpha=2.0, beta=0.5, nu=0.45, rho=-0.35)

    def _smile(self, engine, params):
        return [(K, engine.implied_volatility(F, K, T, params)) for K in STRIKES]

    def test_recovers_parameters_from_model_generated_smile(self, engine):
        smile = self._smile(engine, self.TRUE)
        result = SABRCalibrator(engine, beta=0.5).calibrate(smile, F, T)
        assert result.params.alpha == pytest.approx(self.TRUE.alpha, rel=1e-4)
        assert result.params.rho == pytest.approx(self.TRUE.rho, abs=1e-4)
        assert result.params.nu == pytest.approx(self.TRUE.nu, abs=1e-4)
        assert result.rmse_bp < 0.01

    def test_beta_is_fixed_not_fitted(self, engine):
        smile = self._smile(engine, self.TRUE)
        for beta in (0.0, 0.5, 1.0):
            result = SABRCalibrator(engine, beta=beta).calibrate(smile, F, T)
            assert result.params.beta == beta

    def test_default_beta_is_one_half(self):
        assert SABRCalibrator().beta == 0.5

    def test_reports_rmse_consistent_with_parameters(self, engine):
        smile = [(80, 0.28), (90, 0.24), (100, 0.20), (110, 0.23), (120, 0.27)]
        result = SABRCalibrator(engine).calibrate(smile, F, T)
        residuals = [engine.implied_volatility(F, K, T, result.params) - iv for K, iv in smile]
        assert result.rmse == pytest.approx(np.sqrt(np.mean(np.square(residuals))), rel=1e-12)
        assert result.n_points == 5

    def test_failed_optimisation_raises(self, engine, monkeypatch):
        class Result:
            success = False
            message = "forced failure"
            x = np.array([2.0, 0.0, 0.3])

        monkeypatch.setattr("quantlib.calibration.sabr.minimize", lambda *a, **k: Result())
        with pytest.raises(SABRCalibrationError):
            SABRCalibrator(engine).calibrate(self._smile(engine, self.TRUE), F, T)

    def test_rejects_invalid_beta_and_too_few_quotes(self, engine):
        with pytest.raises(ValueError):
            SABRCalibrator(engine, beta=1.5)
        with pytest.raises(ValueError):
            SABRCalibrator(engine).calibrate([(100, 0.2), (110, 0.19)], F, T)
