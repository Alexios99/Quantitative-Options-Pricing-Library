Alright — here’s a **complete, corrected, step-by-step guide** for implementing your **trinomial tree engine** so it’s both correct and fits into your existing `TreeEngine` architecture.

I’ll focus on **how** and **why** at each step, so you can write the code yourself rather than me just handing it over.

---

## **1. Class Structure**

Since you already have `TreeEngine` as a base, follow the same approach as `BinomialTreeEngine`:

```python
class TrinomialTreeEngine(TreeEngine):
    def __init__(self, n_steps: int):
        super().__init__(n_steps)

    @property
    def method_used(self) -> MethodUsed:
        return MethodUsed.TRINOMIAL_TREE
```

*Tip:* You could add a parameter for `"kr"` vs `"standard"` formulas if you want to allow both versions.

---

## **2. Parameters & Probabilities (KR version)**

This is where the big difference is. For the **Kamrad–Ritchken** model:

1. **Time step**:

   $$
   \Delta t = \frac{T}{N}
   $$

2. **Factors**:

   $$
   u = e^{\sigma \sqrt{2\Delta t}}
   $$

   $$
   d = \frac{1}{u}
   $$

   $$
   m = 1
   $$

   *(Yes, "middle" is just the current price — the bigger `u` jump helps stability.)*

3. **Drift term**:

   $$
   \nu = r - q - \frac{\sigma^2}{2}
   $$

   If you have no dividends, $q = 0$.

4. **Probabilities**:

   $$
   p_u = \frac{1}{2} \left[ \frac{\sigma^2 \Delta t + (\nu \Delta t)^2}{\sigma^2 \Delta t} + \frac{\nu \Delta t}{\sigma \sqrt{2\Delta t}} \right]
   $$

   $$
   p_d = \frac{1}{2} \left[ \frac{\sigma^2 \Delta t + (\nu \Delta t)^2}{\sigma^2 \Delta t} - \frac{\nu \Delta t}{\sigma \sqrt{2\Delta t}} \right]
   $$

   $$
   p_m = 1 - p_u - p_d
   $$

   *Tip:* Check `p_u`, `p_d`, `p_m` are all between 0 and 1 — raise an error if not.

---

## **3. Building the Stock Price Tree**

For a trinomial tree:

* Indexing is a little trickier than binomial because each step adds **two more possible moves**.
* After `i` steps, the lowest index corresponds to `i` down moves, the highest index to `i` up moves, and everything in between is reachable by a mix of ups/mids/downs.
* You can store it in a 2D NumPy array:
  Size: `(n_steps + 1, 2 * n_steps + 1)`

  * Column index offset: middle column at `n_steps` represents `S0` at step 0.

**Hint:**
For each step:

```python
stock[i, j] = S0 * (u ** up_moves) * (d ** down_moves)
```

But in KR, since `m = 1`, you can compute `u` and `d` powers directly from position relative to centre.

---

## **4. Rollback Logic**

For trinomial rollback:

1. **Initialise terminal values**:

   * At maturity (`i = N`),
     For a call: `max(S_T - K, 0)`
     For a put: `max(K - S_T, 0)`

2. **Backward induction**:
   For each earlier step `i`:

   $$
   V_{i,j} = e^{-r \Delta t} \left( p_u \cdot V_{i+1, j+1} + p_m \cdot V_{i+1, j} + p_d \cdot V_{i+1, j-1} \right)
   $$

   * Notice indexing:
     `j` is the position in the stock tree, `j+1` is up move, `j` is middle, `j-1` is down.
   * For **American options**:

     ```
     intrinsic = payoff(stock[i,j])
     V[i,j] = max(V[i,j], intrinsic)
     ```

3. Continue until `V[0, centre_index]` — that’s your price.

---

## **5. Integration With Your `TreeEngine`**

* Implement `_calculate_probabilities()` using KR formulas.
* Implement `_build()` to generate the 2D stock price tree with a centred index.
* Implement `_rollback()` to do the backward induction with trinomial probabilities.
* `_price_impl()` calls these methods in order:

  ```python
  def _price_impl(self, contract):
      p_u, p_m, p_d, u, d, dt = self._calculate_probabilities(...)
      stock = self._build(...)
      return self._rollback(stock, contract, p_u, p_m, p_d, dt)
  ```

---

## **6. Debugging Tips**

* Start with a small number of steps (e.g., 3) and print the stock tree to check indexing.
* Test with European options against Black–Scholes — if the error is > a few bp for `N=100`, check probabilities.
* Verify `p_u + p_m + p_d == 1` and none are negative.

---

If you follow these steps, you’ll have a **correct KR trinomial engine** that drops into your current design — and if later you want a “normal” trinomial, you only need to change `_calculate_probabilities()`.

---

If you want, I can next make you a **worked example with N=3** so you can see the actual stock tree layout and index mapping. That way you won’t get lost when coding `_build()`. Would you like me to prepare that?
