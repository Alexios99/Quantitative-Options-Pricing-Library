"""Reproducible accuracy benchmarks for the pricing engines.

Every input is synthetic and every random draw is seeded, so re-running the script on the same
code reproduces every number in benchmarks/RESULTS.md (up to floating-point differences across
platforms); only the runtime line in the header changes between runs.

    python benchmarks/benchmark.py            # full run, writes benchmarks/RESULTS.md
    python benchmarks/benchmark.py --quick    # smaller grids and fewer seeds (smoke test)

Error units: "bp" means relative error in basis points of the reference price,
1e4 * |P - P_ref| / P_ref, unless a column says otherwise.
"""
import argparse
import platform
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from quantlib.calibration.heston import HestonParameters, HestonPricingEngine  # noqa: E402
from quantlib.calibration.sabr import SABRCalibrator, SABRParameters, SABRPricingEngine  # noqa: E402
from quantlib.core.payoffs import ExerciseStyle, OptionContract, OptionType  # noqa: E402
from quantlib.hedging.delta_hedging import DeltaHedging  # noqa: E402
from quantlib.pricing.analytical import BlackScholesEngine  # noqa: E402
from quantlib.pricing.monte_carlo import MonteCarloEngine  # noqa: E402
from quantlib.pricing.pde import PDEEngine  # noqa: E402
from quantlib.pricing.trees import BinomialTreeEngine, TrinomialTreeEngine  # noqa: E402

DEFAULT_OUTPUT = ROOT / "benchmarks" / "RESULTS.md"
SPOT, STRIKE, EXPIRY, RATE, VOL = 100.0, 100.0, 1.0, 0.05, 0.20
PSOR = SimpleNamespace(psor_omega=1.2, psor_tol=1e-6, psor_max_iter=50)
S_MAX = 200.0
MC_STEPS = 50

FULL = dict(tree_steps=[50, 100, 200, 500, 1000], pde_grids=[(51, 50), (101, 100), (201, 200), (401, 400)],
            american_tree_steps=[100, 200, 500, 1000], american_pde_grids=[(101, 100), (201, 200), (401, 400)],
            american_ref_steps=2000, mc_paths=[10_000, 100_000], mc_seeds=20, lsm_paths=20_000, hedge_sims=2000)
QUICK = dict(tree_steps=[50, 100], pde_grids=[(51, 50), (101, 100)], american_tree_steps=[100],
             american_pde_grids=[(51, 50)], american_ref_steps=400, mc_paths=[5_000], mc_seeds=3,
             lsm_paths=4_000, hedge_sims=200)


def contract(option=OptionType.CALL, style=ExerciseStyle.EUROPEAN):
    return OptionContract(SPOT, STRIKE, EXPIRY, RATE, VOL, option, style)


def bp(price, ref):
    return 1e4 * abs(price - ref) / ref


def table(headers, rows):
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(lines)


def order_of_convergence(ns, errors):
    """Least-squares slope of -log(error) against log(N)."""
    slope, _ = np.polyfit(np.log(ns), np.log(errors), 1)
    return -slope


LIBRARY_PATHS = ["quantlib", ":(exclude)quantlib/tests"]


def git_state():
    """Last commit that changed library code (tests excluded), plus a flag for uncommitted edits.

    Reporting the last library change rather than HEAD keeps the line valid when RESULTS.md is
    committed afterwards (a commit cannot contain its own hash)."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    try:
        commit = git("log", "-1", "--format=%h", "--", *LIBRARY_PATHS)
        dirty = git("status", "--porcelain", "--", *LIBRARY_PATHS)
        return commit + (" plus uncommitted changes" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown (not a git checkout)"


def european_trees(cfg):
    ref = BlackScholesEngine().price(contract()).price
    engines = [("CRR", BinomialTreeEngine, "off"), ("CRR odd/even avg", BinomialTreeEngine, "odd_even"),
               ("Trinomial", TrinomialTreeEngine, "off"), ("Trinomial odd/even avg", TrinomialTreeEngine, "odd_even")]
    rows = []
    for n in cfg["tree_steps"]:
        row = [n]
        for _, cls, mode in engines:
            price = cls(n, smoothing=mode).price(contract()).price
            row.append(f"{bp(price, ref):.2f}")
        rows.append(row)
    text = (f"European ATM call (S=K={SPOT:g}, T={EXPIRY:g}, r={RATE:.0%}, σ={VOL:.0%}); "
            f"reference: Black-Scholes = {ref:.6f}. Error in bp of price.\n\n")
    return text + table(["Steps N"] + [name for name, _, _ in engines], rows)


def european_pde(cfg):
    sections = []
    for option in (OptionType.CALL, OptionType.PUT):
        c = contract(option)
        ref = BlackScholesEngine().price(c).price
        rows, errors = [], {0: [], 2: []}
        for n_space, n_time in cfg["pde_grids"]:
            row = [f"{n_space} × {n_time}", f"{S_MAX / (n_space - 1):g}"]
            for steps in (0, 2):
                grid = SimpleNamespace(n_space=n_space, n_time=n_time, s_max=S_MAX, theta=0.5, rannacher_steps=steps)
                price = PDEEngine(grid, PSOR).price(c).price
                errors[steps].append(abs(price - ref))
                row.append(f"{bp(price, ref):.2f}")
            rows.append(row)
        ns = [g[0] - 1 for g in cfg["pde_grids"]]
        order = f"{order_of_convergence(ns, errors[2]):.2f}" if len(ns) > 1 else "n/a"
        sections.append(
            f"European ATM {option.value}; reference: Black-Scholes = {ref:.6f}. Crank-Nicolson, S_max = {S_MAX:g}, "
            f"space and time refined together. Error in bp of price. Observed order (with Rannacher): {order}.\n\n"
            + table(["Grid (space × time)", "dS", "CN only (bp)", "CN + Rannacher (bp)"], rows))
    return "\n\n".join(sections)


def rannacher_gamma():
    """Plain CN vs CN + Rannacher on a coarse time grid: the payoff kink's oscillations show up in gamma."""
    c = OptionContract(SPOT, STRIKE, 0.25, RATE, VOL, OptionType.CALL, ExerciseStyle.EUROPEAN)
    ref = BlackScholesEngine().price(c).price
    bs_gamma = BlackScholesEngine().greeks(c).gamma
    rows = []
    for label, steps in (("CN only", 0), ("CN + Rannacher", 2)):
        engine = PDEEngine(SimpleNamespace(n_space=401, n_time=10, s_max=S_MAX, theta=0.5, rannacher_steps=steps), PSOR)
        price = engine.price(c).price
        v, ds, s_mid = engine.solution, engine.dx, engine.s_grid[1:-1]
        gamma = ((v[2:] - 2 * v[1:-1] + v[:-2]) / ds ** 2)[(s_mid > 0.8 * STRIKE) & (s_mid < 1.2 * STRIKE)]
        rows.append([label, f"{bp(price, ref):.2f}", f"{gamma.min():+.4f}", f"{gamma.max():.4f}"])
    text = (f"European ATM call, T = 0.25, fine spot grid (401 nodes, dS = 0.5) but only 10 time steps. Gamma is "
            f"the second difference of the PDE solution for 80 < S < 120; Black-Scholes ATM gamma = {bs_gamma:.4f}. "
            f"A call's gamma is never negative.\n\n")
    return text + table(["Scheme", "Price error (bp)", "Min gamma", "Max gamma"], rows)


def american_put(cfg):
    put = contract(OptionType.PUT, ExerciseStyle.AMERICAN)
    n_ref = cfg["american_ref_steps"]
    ref = TrinomialTreeEngine(n_ref).price(put).price
    ref_half = TrinomialTreeEngine(n_ref // 2).price(put).price
    european = BlackScholesEngine().price(replace(put, style=ExerciseStyle.EUROPEAN)).price
    rows = []
    for n in cfg["american_tree_steps"]:
        price = BinomialTreeEngine(n).price(put).price
        rows.append([f"CRR, N = {n}", f"{price:.5f}", f"{bp(price, ref):.2f}"])
    for n_space, n_time in cfg["american_pde_grids"]:
        grid = SimpleNamespace(n_space=n_space, n_time=n_time, s_max=S_MAX, theta=0.5)
        price = PDEEngine(grid, PSOR).price(put).price
        rows.append([f"PDE CN + Rannacher + PSOR, {n_space} × {n_time}", f"{price:.5f}", f"{bp(price, ref):.2f}"])
    text = (f"American ATM put (same parameters). There is no closed form, so the reference is a "
            f"{n_ref}-step trinomial tree = {ref:.6f} (not analytical). European put (Black-Scholes) = {european:.6f}, "
            f"so the early-exercise premium is {ref - european:.4f}. The reference's own discretisation error is "
            f"roughly |P(N={n_ref // 2}) − P(N={n_ref})| = {bp(ref_half, ref):.2f} bp, so smaller differences "
            f"below are not resolvable.\n\n")
    return text + table(["Method", "Price", "Difference vs reference (bp)"], rows)


def mc_european(cfg):
    c = contract()
    ref = BlackScholesEngine().price(c).price
    rows = []
    for n_paths in cfg["mc_paths"]:
        stats = {}
        for method in (None, "antithetic", "control"):
            results = [MonteCarloEngine(MC_STEPS, n_paths, method, seed=s).price(c) for s in range(cfg["mc_seeds"])]
            prices = np.array([r.price for r in results])
            stats[method] = (prices, np.mean([r.standard_error for r in results]))
        for method, (prices, mean_se) in stats.items():
            rows.append([f"{n_paths:,}", method or "plain", f"{prices.mean():.4f}", f"{bp(prices.mean(), ref):.2f}",
                         f"{mean_se:.5f}", f"{prices.std(ddof=1):.5f}", f"{stats[None][1] / mean_se:.2f}x"])
    text = (f"European ATM call; reference: Black-Scholes = {ref:.6f}. {MC_STEPS} time steps; seeds 0-{cfg['mc_seeds'] - 1}. "
            f"Each method uses the same total number of simulated paths (antithetic: N/2 pairs). "
            f"'Mean SE' is the average reported standard error; 'Std of price' is the spread of the "
            f"estimates across seeds (it should match the mean SE if the SE is honest). SE reduction = "
            f"plain mean SE / method mean SE.\n\n")
    return text + table(["Paths", "Method", "Mean price", "Mean price error (bp)", "Mean SE",
                         "Std of price across seeds", "SE reduction"], rows)


def mc_american(cfg):
    put = contract(OptionType.PUT, ExerciseStyle.AMERICAN)
    ref = TrinomialTreeEngine(cfg["american_ref_steps"]).price(put).price
    n_paths, rows, plain_se = cfg["lsm_paths"], [], None
    for method in (None, "antithetic", "control"):
        results = [MonteCarloEngine(MC_STEPS, n_paths, method, seed=s).price(put) for s in range(cfg["mc_seeds"])]
        prices = np.array([r.price for r in results])
        mean_se = np.mean([r.standard_error for r in results])
        plain_se = plain_se or mean_se
        rows.append([method or "plain", f"{prices.mean():.4f}", f"{prices.mean() - ref:+.4f}",
                     f"{mean_se:.5f}", f"{plain_se / mean_se:.2f}x"])
    text = (f"American ATM put by Longstaff-Schwartz ({n_paths:,} paths, {MC_STEPS} exercise dates, seeds "
            f"0-{cfg['mc_seeds'] - 1}). The exercise policy is fitted on independent paths, so the estimator is "
            f"low-biased; {MC_STEPS} exercise dates also make it Bermudan. Reference: "
            f"{cfg['american_ref_steps']}-step trinomial tree = {ref:.6f}.\n\n")
    return text + table(["Method", "Mean price", "Mean price − reference", "Mean SE", "SE reduction"], rows)


def heston_checks():
    engine = HestonPricingEngine()
    ref_params = HestonParameters(v0=0.0175, kappa=1.5768, theta=0.0398, sigma=0.5751, rho=-0.5711)
    published = 5.785155450
    price = engine.price(100.0, 100.0, 1.0, 0.0, ref_params)
    rows = [["Fang & Oosterlee (2008) test case", f"{published:.9f}", f"{price:.9f}", f"{abs(price - published):.1e}"]]
    bs = BlackScholesEngine().price(contract()).price
    for rho in (0.0, -0.5):
        for sigma in (0.01, 0.001):
            p = engine.price(SPOT, STRIKE, EXPIRY, RATE, HestonParameters(0.04, 2.0, 0.04, sigma, rho))
            rows.append([f"Black-Scholes limit: σ_v = {sigma}, ρ = {rho}", f"{bs:.9f}", f"{p:.9f}", f"{abs(p - bs):.1e}"])
    text = ("Fang & Oosterlee (2008), SIAM J. Sci. Comput. 31(2): S=K=100, T=1, r=0, κ=1.5768, θ=0.0398, "
            "σ=0.5751, ρ=−0.5711, v0=0.0175. Black-Scholes limit: v0 = θ = 0.04 (σ = 20%), κ = 2, "
            "ATM call with r = 5%; the error should vanish as the vol of vol σ_v → 0.\n\n")
    return text + table(["Check", "Reference", "Heston engine", "Absolute difference"], rows)


def sabr_checks():
    engine, forward, expiry = SABRPricingEngine(), 100.0, 1.0
    true = SABRParameters(alpha=2.0, beta=0.5, nu=0.45, rho=-0.35)
    strikes = [70, 80, 90, 95, 100, 105, 110, 120, 130]
    model_smile = [(k, engine.implied_volatility(forward, k, expiry, true)) for k in strikes]
    demo_smile = [(80, 0.28), (90, 0.24), (100, 0.20), (110, 0.23), (120, 0.27)]
    rows = []
    for label, smile in (("Smile generated by SABR (α=2, β=0.5, ρ=−0.35, ν=0.45)", model_smile),
                         ("Hand-made V-shaped smile from examples/03", demo_smile)):
        fit = SABRCalibrator(engine, beta=0.5).calibrate(smile, forward, expiry)
        p = fit.params
        rows.append([label, len(smile), f"{p.alpha:.4f}", f"{p.rho:.4f}", f"{p.nu:.4f}", f"{fit.rmse_bp:.2f}"])
    text = "Calibration of α, ρ, ν with β fixed at 0.5, F = 100, T = 1. RMSE in implied-vol basis points.\n\n"
    return text + table(["Synthetic smile", "Quotes", "α", "ρ", "ν", "Fit RMSE (vol bp)"], rows)


def delta_hedging(cfg):
    hedger = DeltaHedging(S0=SPOT, K=STRIKE, T=0.25, r=RATE, sigma=VOL)
    rows = []
    for n_steps in (25, 100, 400):
        analysis = hedger.analyze_results(hedger.simulate_hedging(n_steps=n_steps, n_simulations=cfg["hedge_sims"], seed=42))
        errors = np.array(analysis["hedging_errors"])
        rows.append([n_steps, f"{errors.mean():+.4f}", f"{errors.std(ddof=1) / np.sqrt(len(errors)):.4f}",
                     f"{errors.std(ddof=1):.4f}"])
    text = (f"Short ATM call (T = 0.25, σ = {VOL:.0%}, r = {RATE:.0%}) delta-hedged with Black-Scholes delta, "
            f"rebalanced every step, no transaction costs; {cfg['hedge_sims']:,} paths, seed 42. "
            f"Theory: mean ≈ 0 and the standard deviation halves when rebalancing frequency quadruples.\n\n")
    return text + table(["Rebalances", "Mean final P&L", "SE of mean", "Std of final P&L"], rows)


SECTIONS = [
    ("Trees vs Black-Scholes", european_trees),
    ("PDE vs Black-Scholes", european_pde),
    ("Rannacher start-up on a coarse time grid", lambda cfg: rannacher_gamma()),
    ("American put", american_put),
    ("Monte Carlo variance reduction (European)", mc_european),
    ("Monte Carlo variance reduction (American, Longstaff-Schwartz)", mc_american),
    ("Heston pricer checks", lambda cfg: heston_checks()),
    ("SABR calibration", lambda cfg: sabr_checks()),
    ("Delta hedging error vs rebalancing frequency", delta_hedging),
]


def run(quick: bool = False, output: Path = DEFAULT_OUTPUT) -> Path:
    cfg = QUICK if quick else FULL
    start = time.perf_counter()
    body = []
    for title, fn in SECTIONS:
        t0 = time.perf_counter()
        body.append(f"## {title}\n\n{fn(cfg)}")
        print(f"{title}: {time.perf_counter() - t0:.1f}s", flush=True)
    header = (
        "# Benchmark results\n\n"
        "Generated by `python benchmarks/benchmark.py`"
        + (" --quick" if quick else "") + ". Do not edit by hand; re-run the script instead.\n\n"
        f"- Library code (quantlib/, excluding tests) last changed in commit {git_state()}\n"
        f"- Python {platform.python_version()}, NumPy {np.__version__}, SciPy {scipy.__version__}\n"
        f"- Runtime: {time.perf_counter() - start:.0f} s\n"
        "- All inputs are synthetic and all random draws are seeded.\n"
        "- \"bp\" = relative error in basis points of the reference price, 1e4 × |P − P_ref| / P_ref, "
        "unless stated otherwise.\n"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n" + "\n\n".join(body) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--quick", action="store_true", help="smaller grids and fewer seeds (smoke test)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to write the markdown results")
    args = parser.parse_args()
    print(f"Wrote {run(quick=args.quick, output=args.output)}")


if __name__ == "__main__":
    main()
