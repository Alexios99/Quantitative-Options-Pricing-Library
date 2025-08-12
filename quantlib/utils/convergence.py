from core.payoffs import OptionContract, ExerciseStyle, OptionType
from pricing.trees import BinomialTreeEngine, TrinomialTreeEngine
from pricing.analytical import BlackScholesEngine
from enum import Enum
import time
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, Iterable
from dataclasses import dataclass

class UseEngineRichardson(str, Enum):
    OFF = "off"
    ON = "on"

@dataclass
class ConvergenceReport:
    df: pd.DataFrame                       # per-N results
    reference_price: float                 # benchmark
    order_estimate: float                  # slope on loglog |err| vs N
    odd_even_gap: float                    # mean(|err_even|) - mean(|err_odd|)
    richardson_gain: Optional[float]       # ratio err_noRE / err_RE at largest N
    passed: bool                           # basic thresholds met
    meta: Dict[str, Any]                   # params used


def reference_price(contract: OptionContract, n_steps=5000):
    if contract.style == ExerciseStyle.EUROPEAN:
        ref_price = BlackScholesEngine().price(contract).price
    else:
        ref_price = TrinomialTreeEngine(n_steps, "off").price(contract).price
    return ref_price

def estimate_order(ns: Iterable[int], abs_err: Iterable[float], last_k=4):
    ns = np.array(list(ns), dtype=float)
    e  = np.array(list(abs_err), dtype=float)
    m  = min(last_k, np.sum(e > 0))
    x  = np.log(ns[-m:])
    y  = np.log(e[-m:])
    slope, _ = np.polyfit(x, y, 1)
    return -slope 

def run_study(contract: OptionContract, engine_cls, n_grid: Iterable[int], use_engine_richardson: UseEngineRichardson=UseEngineRichardson.OFF, ref=None):
    ref_price = ref if ref is not None else reference_price(contract)
    rows = []

    for n in n_grid:
        if use_engine_richardson == UseEngineRichardson.ON:
            engine = engine_cls(n_steps=n, richardson="on")
        else:
            engine = engine_cls(n_steps=n, richardson="off")
        
        t0 = time.perf_counter()
        pN = engine.price(contract).price
        dt = (time.perf_counter() - t0) * 1000.0

        if n < max(n_grid) and use_engine_richardson == UseEngineRichardson.OFF: 
            engine_nP1 = engine_cls(n_steps=n+1, richardson="off")
            pNP1 = engine_nP1.price(contract).price
            pRE = 0.5 * (pN + pNP1)
        else:
            pRE = None  
        
        rows.append(dict(N=n, price=pN, price_RE=pRE, abs_err=abs(pN - ref_price), rel_err=abs((pN - ref_price) / ref_price) if ref_price else np.nan, ms=dt))

    df = pd.DataFrame(rows).sort_values("N").reset_index(drop=True)

    order = estimate_order(df.N, df.abs_err.replace(0, np.nan).ffill())
    odd  = df.loc[df.N % 2 == 1, "abs_err"].mean()
    even = df.loc[df.N % 2 == 0, "abs_err"].mean()
    odd_even_gap = (even or 0) - (odd or 0)

    richardson_gain = None
    if "price_RE" in df and df.price_RE.notna().any():
        mask = df.price_RE.notna() & df.price.notna()
        if mask.any():
            tail = df[mask].iloc[-1]
        err_no = abs(tail.price - ref_price)
        err_re = abs(tail.price_RE - ref_price)
        richardson_gain = (err_no / err_re) if err_re > 0 else None

    passed = bool(
        (df.abs_err.iloc[-1] < 1e-3) and
        (order >= 0.45)              # ≈ O(N^{-1/2}) baseline for binomial
    )

    return ConvergenceReport(
        df=df,
        reference_price=ref_price,
        order_estimate=order,
        odd_even_gap=odd_even_gap if np.isfinite(odd_even_gap) else 0.0,
        richardson_gain=richardson_gain,
        passed=passed,
        meta=dict(engine=engine_cls.__name__, grid=list(n_grid))
    )




        