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
