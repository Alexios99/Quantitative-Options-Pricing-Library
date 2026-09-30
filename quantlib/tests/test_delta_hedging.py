import numpy as np

from quantlib.hedging.delta_hedging import DeltaHedging


def _hedger():
    return DeltaHedging(S0=100, K=100, T=0.25, r=0.05, sigma=0.20)


class TestSeeding:
    def test_same_seed_reproduces_results(self):
        a = _hedger().simulate_hedging(n_steps=20, n_simulations=50, seed=7)
        b = _hedger().simulate_hedging(n_steps=20, n_simulations=50, seed=7)
        for ra, rb in zip(a, b):
            np.testing.assert_array_equal(ra["stock_path"], rb["stock_path"])
            assert ra["final_hedging_error"] == rb["final_hedging_error"]

    def test_different_seeds_give_different_paths(self):
        a = _hedger().simulate_hedging(n_steps=20, n_simulations=5, seed=1)
        b = _hedger().simulate_hedging(n_steps=20, n_simulations=5, seed=2)
        assert not np.array_equal(a[0]["stock_path"], b[0]["stock_path"])

    def test_seed_does_not_touch_global_numpy_state(self):
        np.random.seed(123)
        expected = np.random.random()
        np.random.seed(123)
        _hedger().simulate_hedging(n_steps=10, n_simulations=5, seed=0)
        assert np.random.random() == expected


class TestHedgingAccounting:
    """A delta-hedged short option financed at the risk-free rate should have zero expected
    P&L, with dispersion shrinking like 1/sqrt(number of rebalances)."""

    N_SIMS = 2000

    def _errors(self, n_steps):
        results = _hedger().simulate_hedging(n_steps=n_steps, n_simulations=self.N_SIMS, seed=42)
        return np.array([r["final_hedging_error"] for r in results])

    def test_mean_hedging_error_is_zero(self):
        errors = self._errors(100)
        # The old accounting added the short option instead of subtracting it: mean ~ 2x the payoff
        assert abs(errors.mean()) < 4 * errors.std(ddof=1) / np.sqrt(self.N_SIMS)

    def test_error_dispersion_halves_when_rebalancing_quadruples(self):
        std_25, std_100, std_400 = (self._errors(n).std(ddof=1) for n in (25, 100, 400))
        assert 1.6 < std_25 / std_100 < 2.4
        assert 1.6 < std_100 / std_400 < 2.4

    def test_example_script_does_not_double_count_payoff(self, tmp_path, monkeypatch, capsys):
        """examples/05 closes the position at expiry; the seeded path finishes in the money, so
        subtracting the payoff twice would show up as a loss of about the payoff (~16)."""
        import runpy
        from pathlib import Path

        import matplotlib
        matplotlib.use("Agg")
        script = Path(__file__).resolve().parents[2] / "examples" / "05_delta_hedging_sim.py"
        monkeypatch.chdir(tmp_path)
        runpy.run_path(str(script), run_name="__main__")
        out = capsys.readouterr().out
        final_pnl = float(next(l for l in out.splitlines() if l.startswith("Final Portfolio PnL")).split(":")[1])
        final_spot = float(next(l for l in out.splitlines() if l.startswith("Final Spot")).split(":")[1])
        assert final_spot > 100  # the seeded path ends in the money, so the test exercises settlement
        assert abs(final_pnl) < 1.0
