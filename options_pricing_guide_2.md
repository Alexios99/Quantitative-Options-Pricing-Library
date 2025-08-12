Perfect. Here’s a clean, upgraded guide you can follow as your single source going forward. I’ve folded in fixes and improvements from your earlier guide and aligned it with your current progress. I call out exactly which file to work in for every task.

# QuantLib Mini: Implementation Guide (v2)

## Scope and status

You’ve worked on three files so far:

* `quantlib/core/payoffs.py`
* `quantlib/pricing/analytical.py`
* `quantlib/pricing/trees.py`

Your original guide set sound goals and a phased roadmap, but a few success metrics and patterns need tightening so your tests are stable and CI-friendly.&#x20;

---

# 1) Updated success metrics and definitions of done

## Method-specific accuracy targets

Replace the blanket “1 bp across all methods” with realistic, per-method targets you can hit quickly and verify in CI.&#x20;

* **Analytical (BS)**: serves as the reference for Europeans. No tolerance, this is truth.

  * DoD: parity and symmetry tests pass at machine precision.
  * Files: `pricing/analytical.py`, `tests/`

* **Trees (CRR binomial, KR trinomial)**:

  * European vanilla at N in \[400, 800]: final absolute error ≤ 2–5 bp vs BS on a standard call.
  * Observed order on tail points ≥ 0.45 without Richardson, noticeably steeper with it.
  * Odd–even error gap recorded and small in magnitude.
  * DoD: convergence report produced and tests pass.
  * Files: `pricing/trees.py`, `utils/convergence.py`, `tests/`

* **Monte Carlo** (later): use confidence intervals, not raw bp error.

  * Target: analytical price within 95% CI and CI half-width ≤ 5–10 bp for the configured budget.
  * DoD: variance-reduction on, deterministic seeding, CI reported.
  * Files: `pricing/monte_carlo.py`, `tests/`
  * Note: your earlier antithetic pattern pairs steps, not paths. We will fix that when you get to MC.&#x20;

* **PDE** (later): tuned grid reaches ≤ 2–5 bp and passes a grid-refinement study.

  * DoD: refinement study mirrors Trees.
  * Files: `pricing/pde.py`, `utils/convergence.py`, `tests/`

---

# 2) What to build next (you are at Milestone 2.1)

Break Milestone 2.1 into three crisp sub-steps. This keeps “done” unambiguous.

## 2.1a CRR European baseline

**Goal**: Complete binomial European pricing with a correct backward roll and discounting.

* File: `pricing/trees.py`

  * Finish `BinomialTreeEngine`:

    * Implement terminal payoff vector and backward induction with risk-neutral probability p and discount factor exp(−r·dt). Your `CRR` up/down and p scaffolding is already in place.&#x20;
    * Keep the method pure European here. Early exercise comes in 2.1b.
    * Ensure there is a private helper like `_price_with_n(contract, n)` that performs a full price with a given step count. Your internal Richardson already calls this. If it does not exist yet, add it.&#x20;
  * Keep the Richardson path you added, but set default `richardson="off"` for test purity. You will measure its effect, not hide it.&#x20;

* File: `pricing/analytical.py`

  * Verify `BlackScholesEngine` exposes `price(contract)` with a `PricingResult` that the tree engine already imports. This is your reference for Europeans.&#x20;

* File: `core/payoffs.py`

  * Confirm `OptionContract`, `OptionType`, `ExerciseStyle` match what `trees.py` imports. Keep intrinsic payoff definitions consistent with your base class.&#x20;

**Acceptance**:

* Price a standard European call (S=K=100, T=1, r=5%, σ=20%) at N=\[25, 50, 100, 200, 400, 800] and see monotone error decay into ≤ 5 bp by N≈800.

## 2.1b American exercise and KR trinomial

**Goal**: Add early exercise to binomial and complete a stable trinomial for Americans and as a fine-grid reference.

* File: `pricing/trees.py`

  * **Binomial**: in the rollback, switch from pure continuation to `max(continuation, intrinsic)` when `ExerciseStyle.AMERICAN`.
  * **Trinomial**: complete build and rollback. Your KR factors and probabilities are already scaffolded. Add terminal payoffs and a 3-branch backward pass that supports early exercise. Clamp small probability overshoots as you began to do.&#x20;

**Acceptance**:

* American put at S=K=100, T=1, r=5%, σ=20% converges cleanly with N in low hundreds against a fine-grid trinomial reference.

## 2.1c Convergence toolkit and tests

**Goal**: Measure, not guess. Produce a small report for any engine over an N-grid, with order estimate, odd–even gap, and Richardson gain.

* New file: `utils/convergence.py`

  * Provide a `run_study(contract, engine_cls, n_grid, use_engine_richardson)` that:

    * Computes or accepts a **reference price**:

      * European → BS from `pricing/analytical.py`.
      * American → large-N `TrinomialTreeEngine` with Richardson off.
    * Loops over N in `n_grid` and records price, ms, abs\_err, rel\_err.
    * Optionally does external Richardson using N and N+1 and records an improved price for comparison.
    * Estimates tail **order** from a log-log fit on the last K points.
    * Computes **odd–even gap** (mean even error minus mean odd error).
    * Computes **Richardson gain** at the largest N where both prices exist.
    * Returns a plain dict or tiny object plus a list of rows, so tests can assert on it.

* Optional edits: `utils/visualization.py`

  * Add thin helpers to plot error vs N and log-log error vs N. Keep plotting separate from measurement.

* New file: `tests/test_convergence_trees.py`

  * Add four tests:

    1. **Binomial Euro**: grid \[25, 50, 100, 200, 400, 800], final abs\_err ≤ 5 bp, order ≥ 0.45.
    2. **Richardson helps**: show a gain > 1.3 when enabled or via external averaging.
    3. **Trinomial Euro**: grid \[20, 40, 80, 160, 320], final abs\_err ≤ 1–2 bp, small odd–even gap.
    4. **American put**: binomial vs fine-grid trinomial ref at N\_ref≈8192, final abs\_err ≤ 2–5 bp.
  * Gate any heavy grid behind `@pytest.mark.slow`.

**Why now**: your original roadmap lists “Convergence analysis tools” under 2.1 but did not standardise grids or acceptance thresholds. Do it here and close 2.1 cleanly.&#x20;

---

# 3) File-by-file to-do list

## `quantlib/pricing/trees.py`

* Complete binomial rollback for Europeans, then extend to Americans with early exercise.&#x20;
* Ensure `_price_with_n(contract, n)` exists and is used by your internal Richardson path. Your refine hook already calls it.&#x20;
* Finish KR trinomial build and rollback with probability guards and early exercise.&#x20;
* Keep `richardson` default to `"off"` for repeatable tests; enable in convergence runs.

## `quantlib/pricing/analytical.py`

* Confirm `BlackScholesEngine` and `PricingResult` APIs line up with `trees.py` imports. This module is your European reference across the project.&#x20;
* Add parity and symmetry unit tests referencing this engine.

## `quantlib/core/payoffs.py`

* Ensure `OptionContract`, `OptionType`, `ExerciseStyle` match current imports. Keep intrinsic payoff consistent with `TreeEngine._intrinsic`.&#x20;

## New: `quantlib/utils/convergence.py`

* Implement the neutral measurement utilities described in 2.1c. Keep it engine-agnostic so you can reuse it for PDE and MC later. Your old guide placed convergence under Milestone 2.1 but did not specify structure. This file fixes that.&#x20;

## Optional: `quantlib/utils/visualization.py`

* Add small plotting helpers for error curves if you want quick charts. Your old guide already earmarked `utils/visualization.py` for plots.&#x20;

## New: `quantlib/tests/test_convergence_trees.py`

* Add the four convergence tests and pragmatic thresholds described above. Your previous “integration convergence” example used 1,000,000 paths and tight tolerances which is not CI-friendly. We avoid that for Trees.&#x20;

---

# 4) Updated project timeline (8 weeks template)

You are near the end of 2.1. Here is the rest, tuned for stable delivery.

* **Week 1–2 (done)**: Foundations and Analytical

  * Payoffs and BS engine complete with parity tests.&#x20;

* **Week 3 (now)**: **Milestone 2.1a–c**

  * Finish CRR European, add American exercise, complete KR trinomial.
  * Build `utils/convergence.py`, add convergence tests and optional plots.
  * Definition of Done for 2.1: the four tests pass and a convergence report runs without manual intervention.&#x20;

* **Week 4–5**: PDE engine

  * Crank–Nicolson with PSOR for Americans, plus a PDE refinement study using the same convergence utilities. Your old roadmap already lists these items; now you will enforce study outputs.&#x20;

* **Week 5–6**: Monte Carlo

  * GBM paths, fix antithetic to pair paths not steps, add control variates, report CI. Tests assert the BS price lies within the 95% CI with target half-width. Your earlier example will be updated at this point.&#x20;

* **Week 6–7**: Calibration starter

  * Keep SABR marked “alpha” until full formulas and tests are in. Your earlier guide leaves SABR incomplete. Track that explicitly.&#x20;

* **Week 8**: Integration and docs

  * End-to-end examples, performance benchmarks, and a compact technical report section with convergence tables and plots.&#x20;

---

# 5) Test plan you can implement today

* **Unit**

  * Analytical: call–put parity, put–call symmetry, delta sign checks.&#x20;
  * Trees: probability ranges, monotonicity in N for Europeans.

* **Property-based**

  * Monotonicity of call in spot. Use light ranges to keep it fast.&#x20;

* **Convergence (new)**

  * The four tree tests in `tests/test_convergence_trees.py` described in 2.1c.

* **Slow tier**

  * Fine-grid American reference and any very large N. Mark with `@pytest.mark.slow`.

---

# 6) Deliverables checklist per milestone

## Milestone 2.1 Trees

* `pricing/trees.py`: CRR European, American early exercise, KR trinomial complete.
* `utils/convergence.py`: study runner with order, odd–even gap, and Richardson gain.
* `tests/test_convergence_trees.py`: four tests green on CI hardware.
* Optional plots in `utils/visualization.py`.
* A short text or markdown dump of the convergence table for the standard call.

## Milestone 2.2 PDE

* `pricing/pde.py`: CN scheme, PSOR, and boundary conditions.
* Convergence study mirrored from Trees.
* Tests asserting bp target and slope on tail points.

## Milestone 2.3 MC

* `pricing/monte_carlo.py`: GBM, variance reduction, CI reporting.
* Tests assert BS price within CI with target half-width.
* Fix antithetic pattern from the earlier guide.&#x20;

---

# 7) Notes on patterns from the old guide to retain or adjust

* Keep the clean repo layout and audience focus. Those were strong.&#x20;
* Avoid the factory pattern that eagerly constructs engines. Prefer passing an engine class plus kwargs into your convergence tools. Your earlier example would hide parameters and create circulars.&#x20;
* Mark advanced models as “alpha” until formulas and tests are complete to avoid accidental imports. Your SABR section was left incomplete.&#x20;

---

## Quick start today

1. Finish binomial rollback and early exercise in `pricing/trees.py`.&#x20;
2. Complete trinomial build and rollback in `pricing/trees.py`.&#x20;
3. Add `utils/convergence.py` and wire it to BS for Europeans and fine-grid trinomial for Americans.
4. Write `tests/test_convergence_trees.py` with the four checks.
5. Run the study on a standard call, record order and bp error, and tick off Milestone 2.1.

If you want, I can tailor the exact grids and acceptance numbers to the runtime you see locally once your binomial rollback is in.
