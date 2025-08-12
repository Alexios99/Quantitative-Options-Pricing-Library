gotcha. here’s a practical, followable blueprint you can implement line by line. no code, just steps, structure, formulas, and checks.

# Step-by-step guide: convergence analysis for `trees.py`

## 0) Preconditions to verify in your codebase

1. You have tree engines, for example `BinomialTreeEngine` and `TrinomialTreeEngine`.
2. Each engine can run at an arbitrary number of steps, for example a constructor or method that accepts `n_steps`.
3. Each engine exposes a way to compute a price given a contract, for example `engine.price(contract)` that returns an object with a numeric `price`.
4. For Europeans, you have an analytical Black–Scholes pricer available. If not, plan to use a very fine trinomial result as the reference instead.
5. Your tree engines can optionally enable or disable internal Richardson extrapolation. If the flag does not exist yet, plan to support both “use engine as is” and “force off” modes so you can test external Richardson too.

Tip: if any of these are missing, write a tiny adapter layer rather than editing your engine internals.

---

## 1) Create a small “convergence” utility module

Create a new module in whatever utilities package you use, for example `utils/convergence.py`. This module will do four things:

* compute or accept a trusted reference price,
* run a grid of step counts and collect results,
* estimate the observed order of convergence,
* summarise stability symptoms like even–odd oscillations and the benefit of Richardson.

### 1.1 Define a lightweight “report” container

Plan a simple result object that holds:

* a table of per-N results (columns: N, price, optionally price after external Richardson, absolute error, relative error, elapsed milliseconds),
* the reference price used,
* an estimated order of convergence,
* an even–odd oscillation indicator (for example the difference between mean even-N error and mean odd-N error),
* a “Richardson gain” metric (ratio of error without Richardson to error with Richardson at the largest N where both exist),
* a boolean “passed” flag for your chosen thresholds,
* a metadata dict with things like engine name and grid.

Implementation hint: you can return a pandas DataFrame for the table, but a simple list of dicts is fine if you prefer no extra dependencies.

### 1.2 Reference price strategy

Implement a helper that returns a benchmark number:

* If the contract style is European, compute Black–Scholes. Use the same inputs your trees use: spot, strike, T, r, sigma, dividend if any.
* If the style is American, compute a very fine trinomial tree with a large `n_steps` (pick a default like 8k to 32k, tune later), with Richardson disabled. Alternatively, allow the caller to pass an explicit `ref` value to avoid recalculating.

Tip: expose a parameter like `style_hint="auto"` that you map to “euro” or “amer” based on the contract. Allow overriding for edge cases.

---

## 2) Implement the convergence run

Create a function that accepts:

* `contract`: your option contract object,
* `engine_cls`: the engine class to test, for example Binomial or Trinomial,
* `n_grid`: an increasing iterable of integers, for example \[25, 50, 100, 200, 400, 800],
* `use_engine_richardson`: one of “as\_is”, “on”, or “off”,
* `ref`: optional numeric reference price to use directly.

Inside the function, do the following for each N in the grid:

1. Instantiate the engine with `n_steps = N`. If `use_engine_richardson` is not “as\_is”, explicitly set that flag on the engine.
2. Measure elapsed time around the pricing call. Use a high-resolution clock.
3. Store the raw price.
4. Optionally compute an external Richardson extrapolation using N and N+1:

   * Run the same engine with `n_steps = N+1` under the same Richardson setting.
   * Combine the two prices using the classic first-order Richardson formula: p\_RE = 0.5 × (p\_N + p\_{N+1}). Keep this value alongside the raw price.
     Tip: if N+1 is too slow for your grid tail, you can compute external Richardson only for the last few N.
5. Once you have raw price (and optional p\_RE), compute absolute error and relative error against the reference price.
6. Append a row to your results table with N, price, price\_RE (if available), abs\_err, rel\_err, milliseconds.

Edge cases and tips:

* If the reference price is exactly zero, skip relative error or set it to NaN.
* Wrap external Richardson in a try block so a failure on N+1 does not kill the whole run.
* Sort the results by N at the end, then reset any indices if you are using a DataFrame.

---

## 3) Post-processing: diagnostics and metrics

After collecting all rows, compute the following:

### 3.1 Order of convergence

Goal: estimate p in error ≈ C · N^{-p}.

Steps:

1. Take the last K points where error is positive and finite. Use K between 3 and 6; 4 is a good default.
2. Compute x = log(N), y = log(abs\_err) on those tail points.
3. Fit a straight line y = a + b x by least squares. The observed order is p = −b.
4. Store p as `order_estimate`.

Tips:

* If any of the tail errors are zero due to exact matching at coarse N, replace zero with the previous nonzero error to avoid log(0) or drop that point.
* If you have a meaningful p\_RE column, you may also compute an order estimate on the Richardson-extrapolated errors to quantify improvement. This is optional.

### 3.2 Even–odd oscillation indicator

Steps:

1. Split the rows into even N and odd N.
2. Compute the mean absolute error in each group.
3. Define `odd_even_gap` as mean\_even\_error minus mean\_odd\_error. Positive values indicate even N tends to be worse than odd N, negative means the reverse. Close to zero suggests stability.

Tip: this is a rough indicator that helps detect parity artifacts in binomial trees.

### 3.3 Richardson gain

Steps:

1. Find the last row where both raw price and p\_RE are available.
2. Compute err\_no = |price − reference|.
3. Compute err\_re = |price\_RE − reference|.
4. Define `richardson_gain` = err\_no / err\_re if err\_re > 0. Larger than 1 indicates improvement. Values around 2 are common when Richardson helps.

### 3.4 Pass or fail heuristic

Choose thresholds that reflect your expectations and runtime budget. For example:

* Require final absolute error at the largest N to be below a small tolerance, for example 1e−3 for vanilla calls with S=K=100, T=1, r=5 percent, sigma=20 percent.
* Require `order_estimate` to be at least 0.45 for a vanilla binomial without smoothing. This is roughly consistent with N^{-1/2}.
* Optionally, if `use_engine_richardson` is “on”, require `richardson_gain` to exceed 1.3 to confirm the feature is pulling its weight.

Store a boolean `passed` based on these checks.

---

## 4) Optional plotting helpers

These are entirely optional, but helpful when debugging:

* Error vs N on linear axes. This makes absolute magnitudes intuitive.
* Log-log error vs N with the fitted line and the estimated slope. This visually confirms the order.
* Even vs odd N error markers. This reveals parity oscillations.

Tips:

* One plot per figure, no subplots, so you can drop images in a report easily.
* If you must keep dependencies light, generate simple CSVs and plot elsewhere.

---

## 5) Lightweight CLI or script runner

Create a tiny script you can run from the command line to:

1. Construct a standard European call contract, for example S=100, K=100, T=1, r=0.05, sigma=0.2.
2. Run the convergence study for Binomial with a grid like \[25, 50, 100, 200, 400, 800] and Richardson off.
3. Print a compact table (N, price, abs\_err, ms) and the estimated order.
4. Repeat for Richardson on, or compute external Richardson, and print the gain.

This lets you sanity check quickly without opening a notebook.

---

## 6) Tests to lock behaviour (no code, just what to test)

Create a test file for convergence, for example `tests/test_convergence_trees.py`. Add the following tests:

### 6.1 European call convergence with Binomial

* Grid: \[25, 50, 100, 200, 400, 800], Richardson off.
* Assertions:

  * Final absolute error < 2e−3.
  * Observed order ≥ 0.45.

### 6.2 Richardson improves accuracy on Europeans

* Grid: \[50, 100, 200, 400], engine Richardson on.
* Assertions:

  * `richardson_gain` is either not computable for some engines or, if computable, greater than 1.3.

Tip: if your engine’s internal Richardson changes the notion of “N”, document that, and if needed switch to external Richardson for this test so the comparison is apples to apples.

### 6.3 Trinomial stability on Europeans

* Grid: \[20, 40, 80, 160, 320], Richardson off.
* Assertions:

  * Final absolute error < 1e−3.
  * Optional: even–odd gap close to zero in magnitude compared to the final error.

### 6.4 American put against a fine-grid reference

* Contract: S=100, K=100, T=1, r=0.05, sigma=0.2, American put.
* Reference: very fine trinomial with Richardson off, for example N\_ref = 8192.
* Tested engine: Binomial on grid \[50, 100, 200, 400].
* Assertion:

  * Final absolute error < 2e−3.

General test tips:

* Keep grids as small as possible while still reliable. You can add a “slow” marker with larger grids for local checks if needed.
* If floating point jitter makes a test flaky, widen thresholds slightly or increase K in the slope fit.

---

## 7) Performance and numerical tips

* Use a high-resolution timer and report milliseconds. This gives you cost-to-accuracy context and lets you regress runtime.
* For the slope fit, use only the last K points. Early points often live in the pre-asymptotic region and can mislead the estimate.
* If any log transform would hit zero error, replace zeros with the previous nonzero error or drop those entries from the fit. Never take log(0).
* External Richardson uses N and N+1. If your tree requires special parity for early exercise or dividends, switch to N and N+2 to keep the lattice structure consistent. Update the combination weights accordingly if you change spacing.
* If you observe strong even–odd oscillations, log them in the report and try toggling things like averaging at the terminal step or forward-shoot smoothing. The convergence module should only measure, not “fix”.

---

## 8) How to integrate with `trees.py`

* Do not modify pricing guts. Treat engines as black boxes that accept `n_steps` and return a price.
* Keep the convergence module independent of specific engine internals. Accept an engine class and construct it with keyword arguments. This keeps it reusable for future engines.
* Return plain Python structures so the rest of your stack, including tests and scripts, can consume them without extra dependencies.

---

## 9) Definition of done

You are done when:

1. Running the CLI script prints a per-N table and a summary with order, odd–even gap, Richardson gain, and pass/fail.
2. The four tests above pass consistently on your machine in reasonable time.
3. For a European call, the log-log plot of error vs N shows a near-straight tail with slope close to −0.5 without Richardson and clearly steeper with Richardson enabled.
4. For an American put, the final absolute error at N in the low hundreds sits under your set tolerance with the reference defined.

---

## 10) Quick checklist you can literally follow

* Create `utils/convergence.py`.
* Implement: reference price helper with Euro and Amer branches, accepting an override.
* Implement: convergence runner that loops over N, times pricing, computes errors, and optionally external Richardson.
* Implement: post-processing to estimate order, odd–even gap, Richardson gain, pass/fail.
* Optional: plotting helpers or CSV writer.
* Add CLI script for a European call demo.
* Add four tests described above with pragmatic thresholds.
* Document the expected orders and the meaning of each metric in the module docstring.

If you want, tell me your exact engine constructor signature and the names of your contract fields, and I will tailor the parameter passing and the grids to your codebase specifics without giving code.
