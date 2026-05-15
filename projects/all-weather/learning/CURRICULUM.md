# All-Weather Engine — Rebuild Curriculum

A six-session walkthrough for rebuilding the production engine from scratch.
Each session produces one Python module in `learning/`, verified against
the production code in `engine/`. By the end, you should be able to defend
every line of the production engine — or improve on it.

---

## Why this exercise

The production engine in `engine/` is ~1,500 lines of math: data fetching,
covariance estimation, SLSQP optimisation, monthly rebalance simulation,
performance statistics, and IS/OOS validation. Reading it once teaches
you the API. Writing it from scratch teaches you the *reasoning*.

The point is not novelty — your final code should converge to numbers
that match production to four decimal places. The value is in the
hours you spend wondering *why* a particular line is the way it is.
When you diff your version against production at the end, every
difference is a learning moment: yours is cleaner, or yours is wrong.

---

## How to use this curriculum

### Workflow per session

1. **Read the session here.** Skim Theory, study Steps and Verification.
2. **Open the production file mentioned in *Diff against*.** Read it
   once for context. Close it.
3. **Write your version in `learning/0X_*.py`.** Use only the function
   signatures in *What you write* — implementation is yours.
4. **Run the verification block.** Iterate until your code passes.
5. **Diff your file against production.** For every difference, classify
   it: *cleaner* (candidate to upstream), *wrong* (production is right —
   fix yours), or *equivalent* (different style, same result).

### Time budget

Each session is ~1.5–2 hours. Three sessions per week = ~3 weeks total.
The diff exercise after each session adds ~15 minutes.

### What this curriculum is not

- **Not a textbook.** Theory blocks are condensed. Look up Sharpe-ratio
  derivations, SLSQP convergence theory, or risk-parity provenance
  separately when you need depth.
- **Not full code.** Function signatures and verification numbers are
  given. Implementations are yours.
- **Not a guide to `live/` or `research/`.** Those are operational glue
  and analyses; learn them by reading once, not by rewriting.

---

## Pre-flight

### Environment

```bash
cd projects/all-weather
conda activate allweather
# or, without activation:
conda run -n allweather python3 your_script.py
```

Run all session code as `python3 -m learning.01_data` etc. from the
project root, so relative imports to `engine/` work.

### Recommended reading before starting

- *All Weather Story* (Bridgewater paper, 12 pages) — the original
  motivation for risk-balanced multi-asset portfolios.
- `engine/__init__.py` and `strategies.json` — what the production
  engine is producing.
- `README.md` — top-level project context.

---

# Session 1 — Data foundation

**Goal.** Build a function that returns aligned, total-return daily
prices for a list of tickers, with quality checks, ready to feed into
covariance estimation.

**Time.** ~1.5 h.

**Prerequisites.** Familiarity with pandas DataFrames and `yfinance`.

**File you will write.** `learning/01_data.py`

**Diff against.** [engine/data.py](../engine/data.py)

### Theory

- **Total return vs price return.** `yfinance.download(..., auto_adjust=True)`
  returns prices that have dividends and splits already reinvested.
  This is the correct input for portfolio backtests, because holding
  an ETF gives you the dividends. Without it, you systematically
  underweight income-producing assets.

- **Log returns vs simple returns.** Production uses log returns
  (`r_t = log(P_t / P_{t-1})`) for covariance estimation. Log
  returns are time-additive (cumulative log return = sum of log
  returns), approximately normal under reasonable assumptions, and
  symmetric (a +50% / −33% round-trip is zero in log space).
  Arithmetic returns lack these properties and bias covariance.

- **Forward-fill on weekends and holidays.** Markets are closed
  Sat/Sun and on US holidays, so those calendar dates have no
  natural price. We fill them with the last known close, which
  correctly reflects the value of your portfolio (you couldn't have
  traded those days anyway). *Backfilling* would introduce
  look-ahead bias — you'd be using next Monday's price on Sunday.

- **Why month-end resampling uses `"ME"`.** pandas ≥ 2.2 changed the
  alias from `"M"` to `"ME"`. Both produce month-end (last calendar
  day) timestamps, then we take the last *trading* day's price for
  that month. We do not use month-start, because we want the value
  *at* the rebalance date, not before.

### What you write

```python
def fetch_prices(tickers: list[str],
                 start_date: str,
                 end_date: str) -> pd.DataFrame:
    """
    Returns a daily DataFrame indexed by date, one column per ticker,
    forward-filled across non-trading days.
    """
```

### Steps

1. Call `yf.download(tickers, start=start_date, end=end_date,
   auto_adjust=True, progress=False)`.
2. yfinance returns a MultiIndex when given more than one ticker;
   it returns a flat-column frame for a single ticker. Handle both
   cases — extract the `Close` column and end up with one column
   per ticker.
3. Drop rows where every column is NaN (pre-listing dates).
4. Forward-fill remaining NaNs (`ffill`) so weekend/holiday gaps
   carry the prior close.
5. Add quality checks:
   - **All-NaN column.** Raise `ValueError` — likely a bad ticker.
   - **Single-day move > 30%.** Print a warning — usually a stock
     split that yfinance failed to adjust.
   - **Negative price.** Raise `AssertionError` — corrupted data.
   - **Stale data.** If the last date is more than 45 calendar
     days before `today`, print a warning. yfinance total-return
     data lags by 30–45 days for some tickers.

### Verification

```python
from learning.01_data import fetch_prices as your_fetch
from engine.data import fetch_prices as prod_fetch

your_df = your_fetch(["SPY"], "2020-01-01", "2020-12-31")
prod_df = prod_fetch(["SPY"], "2020-01-01", "2020-12-31")

assert your_df.shape == prod_df.shape
assert (your_df.index == prod_df.index).all()
assert (your_df["SPY"].round(4) == prod_df["SPY"].round(4)).all()
assert not your_df.isna().any().any()                  # no NaNs after ffill
assert your_df.index.is_monotonic_increasing           # sorted ascending
```

### Common pitfalls

- yfinance returns a `MultiIndex` `(field, ticker)` for multi-ticker
  downloads but a flat `Index` for a single ticker. Easy to forget
  one branch.
- Forgetting `auto_adjust=True` gives you price-only data — your
  cumulative returns will be ~2% / year too low for equity ETFs.
- Calling `.fillna(0)` instead of `.ffill()` corrupts everything.
- Dropping with `dropna(how="any")` instead of `how="all"` makes
  GSG (less data) silently disappear from the result.

---

# Session 2 — Covariance & risk contributions

**Goal.** Compute the covariance matrix of asset log returns and the
percentage risk contribution of each asset under a given weight vector.
This is the kernel that risk parity is built around.

**Time.** ~1.5 h.

**Prerequisites.** Session 1. Working knowledge of NumPy linear algebra.

**File you will write.** `learning/02_risk.py`

**Diff against.** Covariance + RC blocks in
[engine/optimiser.py](../engine/optimiser.py) (lines ~424–437 of the
`compute_risk_parity_weights` function).

### Theory

- **Log returns from price DataFrame.**
  `r_t = log(P_t / P_{t-1})`. In code: `np.log(prices / prices.shift(1)).dropna()`.

- **Sample covariance.** For a return matrix `R` of shape `(T, n)`:
  `Σ_ij = E[(r_i - μ_i)(r_j - μ_j)] = (1/(T-1)) (R - μ)ᵀ (R - μ)`.
  In code: `log_rets.cov().values` (uses `ddof=1` by default — sample
  covariance, not population). Annualisation `× 252` is *not* needed
  for derivation of weights — the optimiser only uses the relative
  structure of `Σ`. We annualise only when reporting volatility.

- **Portfolio variance.** `σ²_p = wᵀ Σ w`. Bilinear form.

- **Risk contribution of asset i.**
  `RC_i = w_i × (Σw)_i / (wᵀ Σ w)`.
  - `(Σw)_i` is the i-th element of the marginal-variance vector
    `Σw`, i.e. the partial derivative of portfolio variance with
    respect to `w_i` (up to a factor of 2).
  - Multiplying by `w_i` gives asset i's contribution to total
    portfolio variance.
  - Dividing by `wᵀΣw` makes the contributions sum to 1, so we can
    interpret them as percentages.

- **Why equal weight ≠ equal risk.** For a 50/50 stock/bond
  portfolio, stocks contribute ~90% of variance because their
  volatility is ~5× higher. Risk parity inverts that: it gives
  bonds a much higher weight so each contributes the same risk.

### What you write

```python
def log_returns(prices: pd.DataFrame) -> pd.DataFrame: ...
def cov_matrix(log_rets: pd.DataFrame) -> np.ndarray: ...
def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray: ...
```

### Steps

1. `log_returns`: `np.log(prices / prices.shift(1)).dropna()`.
2. `cov_matrix`: `log_rets.cov().values` (or implement manually with
   `np.cov(log_rets.T, ddof=1)` — the matrices are identical).
3. `risk_contributions(w, cov)`:
   - `port_var = w @ cov @ w` (scalar).
   - `marginal = cov @ w` (vector, shape `(n,)`).
   - `rc = (w * marginal) / port_var` — element-wise multiply by `w`
     gives variance contributions, then normalise.
   - Edge case: if `port_var < 1e-12`, return `np.ones(n) / n`
     (avoids division by zero with a degenerate weight vector).

### Verification

```python
import numpy as np
from learning.02_risk import log_returns, cov_matrix, risk_contributions
from engine.data import fetch_prices

# Production 6-asset universe, 5y window ending 2020-01-01
tickers = ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]
px = fetch_prices(tickers, "2014-01-01", "2020-01-01")
lr = log_returns(px)
cov = cov_matrix(lr)

# Cov matches numpy
assert np.allclose(cov, np.cov(lr.values.T, ddof=1))

# RCs sum to 1.0
w = np.array([0.134, 0.103, 0.175, 0.348, 0.142, 0.098])
rc = risk_contributions(w, cov)
assert np.isclose(rc.sum(), 1.0, atol=1e-9)

# Equal weight: SPY+QQQ dominate risk
w_eq = np.full(6, 1/6)
rc_eq = risk_contributions(w_eq, cov)
assert rc_eq[0] + rc_eq[1] > 0.50    # stocks contribute > 50% of risk under eq weight
```

### Common pitfalls

- Forgetting `.dropna()` after `shift(1)` — the first row is NaN,
  and `np.log` of NaN propagates.
- Using arithmetic returns (`pct_change()`) instead of log returns:
  the cov matrix is similar but not identical, and the RP weights
  drift slightly. Defend the choice — production uses log.
- Computing `RC_i = w_i × σ_i × correlation` is a different
  decomposition (component CTR). The production formula is
  `w_i × (Σw)_i / (wᵀΣw)`, which is the *percentage* contribution
  to *variance* (not std).

---

# Session 3 — Risk parity via SLSQP

**Goal.** Solve for the weights that equalise risk contributions,
subject to `Σw = 1` and `w_i ≥ 0.02`. This is the heart of the engine.

**Time.** ~2 h.

**Prerequisites.** Sessions 1 & 2. Familiarity with constrained
optimisation; ideally exposure to gradient-based solvers.

**File you will write.** `learning/03_rp.py`

**Diff against.** `compute_risk_parity_weights` in
[engine/optimiser.py](../engine/optimiser.py) (lines 372–478).

### Theory

- **Objective.** Minimise the dispersion of risk contributions. Two
  equivalent forms:
  - `Σ_i Σ_j (RC_i - RC_j)²` (sum of pairwise squared differences),
  - `Var(RC)` (variance of the RC vector — the production form).

  These have the same minimum (zero, when all RCs are equal). The
  variance form is used because it's faster and SLSQP is happy with
  it.

- **Equality constraint.** `Σ w_i = 1` — the portfolio is fully
  invested.

- **Inequality constraint.** `w_i ≥ 0.02` for every asset.

  *Why a 2% floor?* Without it, the optimiser can push the weight of a
  high-vol asset toward zero, trivially equalising RCs (since `w=0` →
  `RC=0`). The 2% floor forces every asset into the portfolio and
  delivers a meaningfully equalised solution.

- **SLSQP.** Sequential Least-Squares Programming. Gradient-based,
  handles both equality and inequality constraints. The objective
  here is smooth and convex *in the relevant region*, so SLSQP
  converges reliably from an equal-weight starting point.

- **Convergence checks.** SLSQP returns `OptimizeResult` with a
  `success` boolean. Production currently *prints a warning* on
  non-convergence and continues (ToDo: should raise). For learning,
  raise.

- **IS/OOS boundary.** When fitting RP weights, every observation
  used must come from before the OOS start date. The function
  takes `end_date` and slices `prices.loc[prices.index < end_date]`
  *before* computing the covariance. If you don't enforce this,
  your weights have look-ahead bias.

### What you write

```python
def compute_risk_parity_weights(prices: pd.DataFrame,
                                tickers: list[str],
                                estimation_years: float = 5.0,
                                min_weight: float = 0.02,
                                end_date: str | None = None) -> dict[str, float]:
    """
    Returns {ticker: weight} summing to 1.0, rounded to 4 d.p.
    Raises if SLSQP does not converge.
    """
```

### Steps

1. Filter `tickers` to those present in `prices.columns`. Raise if
   none.
2. If `end_date` provided, slice `prices.loc[prices.index < end_date]`.
3. Slice to the estimation window: last `estimation_years` of data
   relative to `prices.index[-1]`.
4. Compute log returns + covariance (Session 2).
5. Define `_objective(w)` returning `np.var(_risk_contributions(w))`.
6. Build `constraints` list: one equality `Σw - 1 = 0`.
7. Build `bounds` list: `[(min_weight, 1.0)] * n`.
8. `w0 = np.full(n, 1/n)` — start from equal weight.
9. Call `scipy.optimize.minimize(_objective, w0, method="SLSQP",
   bounds=bounds, constraints=constraints,
   options={"maxiter": 2000, "ftol": 1e-12})`.
10. Raise if `not result.success`.
11. Re-normalise `w = result.x / result.x.sum()` (numerical safety).
12. Round to 4 d.p. and return as `dict`.

### Verification

```python
from learning.03_rp import compute_risk_parity_weights as your_rp
from engine.optimiser import compute_risk_parity_weights as prod_rp
from engine.data import fetch_prices

tickers = ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]
px = fetch_prices(tickers, "2010-01-01", "2020-01-01")

your_w = your_rp(px, tickers, end_date="2020-01-01")
prod_w = prod_rp(px, tickers, end_date="2020-01-01")

# Match production to 4 decimals
for t in tickers:
    assert abs(your_w[t] - prod_w[t]) < 1e-4, (t, your_w[t], prod_w[t])

# Sums to 1.0
assert abs(sum(your_w.values()) - 1.0) < 1e-9

# All weights ≥ 2% floor
assert all(w >= 0.02 - 1e-9 for w in your_w.values())
```

### Common pitfalls

- **Treating SLSQP failure as a warning.** Production does (and the
  `ToDo` flags it). For real money, treat it as an error.
- **No floor / floor too low.** Try `min_weight=0.001` and watch
  the optimiser dump weight onto one asset; it equalises RCs by
  zeroing assets out, which is meaningless.
- **Forgetting to slice by `end_date` before computing the
  covariance window.** If you slice the estimation window first and
  then truncate, you'll have inconsistent bounds.
- **Re-normalising weights *after* the round-to-4dp**: the
  rounding error breaks the equality constraint by ~1e-6. Either
  re-normalise after rounding, or accept the rounding error.

---

# Session 4 — Backtest engine

**Goal.** Simulate the monthly-rebalance strategy over a daily price
history and produce a DataFrame of month-end portfolio values, plus
benchmark and buy-and-hold reference series.

**Time.** ~2 h.

**Prerequisites.** Sessions 1–3. Comfort with pandas resampling.

**File you will write.** `learning/04_backtest.py`

**Diff against.** [engine/backtest.py](../engine/backtest.py) (the
`run_backtest` function).

### Theory

- **Three reference strategies, run together.**
  1. *Rebalanced* — at every month end, sell/buy back to target weights.
  2. *Buy & hold* — same starting weights, never rebalanced. Drifts
     toward the best performer.
  3. *S&P 500 reference* — 100% SPY, never touched. Used as benchmark.

  The production engine optionally runs a fourth: *60/40 SPY/TLT
  annual rebalance*, used as a more realistic reference. Implement
  it if you want (signature takes `tlt_prices` for this).

- **Monthly resampling.** Take daily prices, group by month-end
  (`prices.resample("ME").last()`), align all series on common
  month-end dates.

- **Continuous shares (no integer rounding).** A backtest holds
  fractional shares so that the math is exact. Real Alpaca trading
  rounds to whole shares (or fractional with limits) — that
  rounding cost is small and is handled by transaction-cost
  parameter (`0.001` = 10 bps).

- **Transaction costs.** At each rebalance, compute `|Δshares| ×
  price` per asset, sum across assets, multiply by
  `transaction_cost_pct`, deduct from portfolio value before
  computing new shares.

- **Buy & Hold.** Compute the initial holdings in shares from
  `portfolio_value × allocation / price_at_t0`. Each subsequent
  month, value = `Σ shares × price`. No rebalancing — the
  weights drift.

- **Compounded vs arithmetic returns.** Monthly return columns
  are computed as `value_t / value_{t-1} - 1`. The aggregate
  (CAGR, etc.) is computed in Session 5 from these.

### What you write

```python
def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: dict[str, float],
                 portfolio_value: float = 100_000,
                 tlt_prices: pd.Series | None = None,
                 transaction_cost_pct: float = 0.0,
                 tax_drag_pct: float = 0.0) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by month-end with at least:
      All Weather Value             (rebalanced strategy)
      Buy & Hold All Weather
      S&P 500 Value
      <if tlt_prices> 60/40 Value
      All Weather Value Monthly Ret (%)   etc. for each strategy
    """
```

### Steps

1. Resample `prices` and `benchmark_prices` to month-end.
2. Take the intersection of dates so all series align.
3. Compute starting share counts:
   `shares[t] = portfolio_value × allocation[t] / price_t0[t]`
4. Loop over month-ends from index 1 to end:
   - Compute current value of rebalanced portfolio:
     `Σ shares × price_t`.
   - Compute target shares from new prices and target weights.
   - Compute `Δshares = |new_shares - old_shares|`.
   - `cost = (Σ Δshares × price_t) × transaction_cost_pct`.
   - Deduct `cost` from portfolio value, then recompute target
     shares with the post-cost value.
   - Update buy-and-hold series (no rebalance — just `shares × price`).
   - Update SPY series (`bench_shares × bench_price`).
   - Update 60/40 if `tlt_prices` provided (annual rebalance: same
     formula but only at January each year).
5. Build the output DataFrame with value columns and monthly-return
   columns.

### Suggested refactor

The production loop in `engine/backtest.run_backtest` tracks 18
local variables and is ~250 lines long. It works, but is hard to
read. Consider building a `PortfolioSimulator` class with state
methods (`step`, `rebalance`, `record`) — this is the place where
your version may legitimately be cleaner than production. Just be
sure it produces identical numbers.

### Verification

```python
import pandas as pd
from learning.04_backtest import run_backtest as your_bt
from engine.backtest import run_backtest as prod_bt
from engine.data import fetch_prices

tickers = ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]
px = fetch_prices(tickers + ["TLT"], "2010-01-01", "2024-01-01")
allocation = {"SPY": 0.134, "QQQ": 0.103, "TLT": 0.175,
              "TIP": 0.348, "GLD": 0.142, "GSG": 0.098}

your_df = your_bt(px[tickers], px["SPY"], allocation,
                  tlt_prices=px["TLT"], transaction_cost_pct=0.001)
prod_df = prod_bt(px[tickers], px["SPY"], allocation,
                  tlt_prices=px["TLT"], transaction_cost_pct=0.001)

# Final value within 0.1% of production
assert abs(your_df["All Weather Value"].iloc[-1] /
           prod_df["All Weather Value"].iloc[-1] - 1) < 0.001

# Monthly return column matches to 4 d.p.
your_ret = your_df["All Weather Value Monthly Ret (%)"].round(4)
prod_ret = prod_df["All Weather Value Monthly Ret (%)"].round(4)
assert (your_ret == prod_ret).all()
```

### Common pitfalls

- **Integer share rounding.** A subtle bug — fractional shares are
  required for a clean backtest. Don't `round()` shares anywhere.
- **Forgetting transaction costs.** A 0.1% cost on a 100% turnover
  monthly rebalance shaves ~1.2% / year. Easy to forget; CAGR
  comes out too high.
- **Misaligned index.** `prices.resample("ME").last()` and
  `benchmark.resample("ME").last()` may not have identical indexes
  if one ticker has a longer history. Use
  `monthly.index.intersection(bench.index)` and `.loc[common]`.
- **Year accounting.** `years = (idx[-1] - idx[0]).days / 365.25`,
  not `len(idx) / 12` (the latter is off by holidays).

---

# Session 5 — Performance statistics

**Goal.** Compute the standard set of metrics: CAGR, Max Drawdown,
Sharpe, Sortino, Calmar, Ulcer Index. Each is a small function;
the value is in getting the conventions exactly right.

**Time.** ~1.5 h.

**Prerequisites.** Sessions 1–4.

**File you will write.** `learning/05_stats.py`

**Diff against.** [engine/stats.py](../engine/stats.py).

### Theory

- **CAGR.** `((V_T / V_0) ^ (1 / years) - 1) × 100`. Expressed as
  a percentage. Geometric, not arithmetic — captures compounding.

- **Max Drawdown.** `min((V_t - max(V_0..V_t)) / max(V_0..V_t)) × 100`.
  Peak-to-trough decline as a percentage. Returns a *negative*
  number. Compute via `series.cummax()` for the running peak.

- **Sharpe ratio.** `((mean - rf_per_period) / std) × sqrt(periods_per_year)`.
  - For monthly returns: `sqrt(12)` annualisation factor.
  - `rf_annual` should be converted: `rf_monthly = (1 + rf_annual)^(1/12) - 1`.
  - Returns are passed in as percentages (e.g., `1.5` for 1.5%);
    divide by 100 inside the function.
  - Returns 0.0 if the std is below `1e-10` (degenerate flat series).

- **Sortino ratio.** Like Sharpe but the denominator is the std of
  *only* the below-target returns. `downside = r[r < rf_monthly]`,
  then `(mean(r) - rf) / std(downside) × sqrt(12)`.
  - Choice point: production uses *downside std*, not
    *semi-deviation* (which divides by total length, not just
    downside count). Either is defensible — pick one and document.

- **Calmar ratio.** `CAGR / |MDD|`. A return-per-unit-of-pain measure.
  Higher is better. Calmar of 0.5 means earning 0.5% of CAGR for
  every 1% of MDD accepted. Production uses Calmar as the primary
  evaluation metric.

- **Ulcer Index.** `sqrt(mean(DD_t²))`. The RMS of the drawdown
  series, where DD_t = `(V_t - peak_t) / peak_t × 100`. Penalises
  *both* depth and duration of drawdowns. A single −20% spike that
  recovers immediately has a lower Ulcer than two years at −10%.

### What you write

```python
def compute_cagr(series: pd.Series, years: float) -> float: ...
def compute_max_drawdown(series: pd.Series) -> float: ...
def compute_sharpe(monthly_ret_pct: pd.Series, rf_annual: float = 0.0) -> float: ...
def compute_sortino(monthly_ret_pct: pd.Series, rf_annual: float = 0.0) -> float: ...
def compute_calmar(cagr: float, max_drawdown: float) -> float: ...
def compute_ulcer_index(series: pd.Series) -> float: ...
def compute_stats(backtest_df: pd.DataFrame) -> dict: ...
```

`compute_stats` is the aggregator — call it on a `run_backtest`
output to get all metrics for the rebalanced strategy as a dict.

### Steps

1. Implement each metric per the formulas above.
2. Handle the degenerate cases:
   - Empty series (`return 0.0`).
   - `MDD == 0` in Calmar (`return 0.0`).
   - `std < 1e-10` in Sharpe (`return 0.0`).
   - Empty downside set in Sortino (`return 0.0`).
3. Build `compute_stats(df)` to read the `All Weather Value` and
   `All Weather Value Monthly Ret (%)` columns and return a dict.

### Verification

```python
import numpy as np, pandas as pd
from learning.05_stats import (compute_cagr, compute_max_drawdown,
    compute_sharpe, compute_sortino, compute_calmar, compute_ulcer_index)
from engine.stats import (compute_cagr as p_cagr,
    compute_max_drawdown as p_mdd, compute_sharpe as p_sharpe,
    compute_sortino as p_sortino, compute_calmar as p_calmar,
    compute_ulcer_index as p_ulcer)

# Synthetic series for repeatability
np.random.seed(42)
rets = pd.Series(np.random.normal(0.005, 0.04, 240))   # 20y monthly
values = 100 * (1 + rets).cumprod()
years  = 20.0

assert round(compute_cagr(values, years), 3) == round(p_cagr(values, years), 3)
assert round(compute_max_drawdown(values), 3) == round(p_mdd(values), 3)
assert round(compute_sharpe(rets * 100, 0.035), 3) == round(p_sharpe(rets * 100, 0.035), 3)
assert round(compute_sortino(rets * 100, 0.035), 3) == round(p_sortino(rets * 100, 0.035), 3)
assert round(compute_calmar(8.0, -15.0), 3) == round(p_calmar(8.0, -15.0), 3)
assert round(compute_ulcer_index(values), 3) == round(p_ulcer(values), 3)
```

### Common pitfalls

- **Annualising with 252 vs 12.** For *daily* returns, `sqrt(252)`.
  For *monthly* returns, `sqrt(12)`. The backtest produces monthly
  returns; using 252 inflates Sharpe by ~4.6×.
- **Arithmetic mean of compounded returns.** Don't compute CAGR
  from `mean(monthly_rets) × 12`. Use `(V_T / V_0) ^ (1/years)`.
- **Forgetting to convert rf to monthly.** `rf_annual = 0.035`
  *minus* monthly returns is wrong by a factor of 12.
- **Sign convention.** MDD should be negative. If yours is
  positive, you're returning `|min|` instead of `min`.

---

# Session 6 — IS/OOS validation & 3-window RP averaging

**Goal.** Reproduce the production weights in `strategies.json`
by running the full RP pipeline three times — once per OOS split
— and averaging. The result must match `strategies.json` to four
decimals.

**Time.** ~2 h.

**Prerequisites.** Sessions 1–5.

**File you will write.** `learning/06_validation.py`

**Diff against.** [strategies.json](../strategies.json) — this is the
master numerical truth. Specifically, the `6asset_tip_gsg_rpavg`
allocation block.

### Theory

- **IS/OOS discipline.** Never optimise on data you'll evaluate on.
  When fitting RP weights for evaluation on the 2022→present window,
  you must use only data from before 2022.

- **Why three windows?** A single OOS split can be lucky or
  unlucky. Three independent splits (cutoffs 2018, 2020, 2022)
  produce three weight vectors. Each window has a different IS
  range and a different OOS range. The three RP weight vectors
  agree on the broad structure (TIP-heavy, balanced rest) but
  differ in the margins.

- **Why average?** The averaged vector is more robust than any
  single one. It is what the production strategy uses.

- **No look-ahead.** When computing RP for cutoff 2018, use only
  data from 2013–2017 (5y window ending Dec 2017). The
  `compute_risk_parity_weights(end_date=...)` function from
  Session 3 already enforces this.

### What you write

```python
def compute_rp_averaged_weights(tickers: list[str],
                                oos_dates: list[str],
                                lookback_years: float = 5.0,
                                fetch_start: str = "2010-01-01"
                                ) -> dict[str, float]:
    """
    For each cutoff in oos_dates, computes RP weights using only
    data before the cutoff, then averages the three weight dicts.
    """
```

### Steps

1. Fetch prices for `tickers` from `fetch_start` to today (or to
   the latest cutoff + a little buffer).
2. For each `cutoff` in `oos_dates`, call your Session 3
   `compute_risk_parity_weights(prices, tickers, estimation_years=lookback_years, end_date=cutoff)`.
3. Collect the three weight dicts.
4. Average element-wise: `avg[t] = mean(w_dicts[i][t] for i in 0..2)`.
5. Round to 3 decimals (production convention; this is the actual
   precision in `strategies.json`).
6. Return as dict.

### Verification

```python
from learning.06_validation import compute_rp_averaged_weights

tickers = ["SPY", "QQQ", "TLT", "TIP", "GLD", "GSG"]
weights = compute_rp_averaged_weights(
    tickers,
    oos_dates=["2018-01-01", "2020-01-01", "2022-01-01"],
    lookback_years=5.0,
)

# strategies.json["6asset_tip_gsg_rpavg"]["allocation"]
expected = {
    "SPY": 0.134, "QQQ": 0.103, "TLT": 0.175,
    "TIP": 0.348, "GLD": 0.142, "GSG": 0.098,
}
for t, w in expected.items():
    assert abs(weights[t] - w) < 1e-3, (t, weights[t], w)
```

If your weights match, you have rebuilt the production engine
end-to-end.

### Common pitfalls

- **Letting OOS data leak.** If you fetch data only up to a single
  cutoff and reuse it across windows, you're fine. But a common
  shortcut — fetch once to today, and trust `end_date` slicing —
  *is* correct, because Session 3 enforces it.
- **Averaging in the wrong space.** Average raw weights, then
  round once. Don't round each set of weights to 3dp first and
  then average — accumulated rounding error pushes you off the
  4dp match.
- **5y window vs available history.** GSG has data only from
  mid-2006 (it started July 2006). The 5y window for cutoff
  2018-01-01 is 2013–2017, which is fine. But if you accidentally
  ask for a 12y window for cutoff 2010, you'll silently get
  truncated data and weights will diverge.

---

# Final diff exercise

After all six sessions pass their verification blocks, do a side-by-side
diff of every `learning/0X_*.py` file against its production counterpart
in `engine/`.

```bash
diff learning/01_data.py    engine/data.py
diff learning/02_risk.py    engine/optimiser.py    # cov + RC blocks only
diff learning/03_rp.py      engine/optimiser.py    # compute_risk_parity_weights only
diff learning/04_backtest.py  engine/backtest.py
diff learning/05_stats.py   engine/stats.py
diff learning/06_validation.py    # no direct production analogue; verify via strategies.json
```

For every difference longer than ~3 lines, write one line in
`learning/DIFF_NOTES.md` classifying it:

| Class | Meaning | Action |
|---|---|---|
| **CLEANER** | Yours is more readable / has the same numerics | Candidate to upstream into `engine/` |
| **WRONG** | Yours diverges; production is right | Fix yours; understand why production is the way it is |
| **EQUIVALENT** | Different style, same result | Note and move on |
| **SCOPE** | Production has features your version doesn't (overlay, rolling RP, etc.) | Out of scope — the curriculum focused on the production path |

The DIFF_NOTES file is the deliverable that proves you own the code.

---

# Extension exercises (optional)

These are *not* sessions — they're things to try after the main six
to deepen specific areas.

| Topic | Production file | What to do |
|---|---|---|
| Walk-forward Pareto | `research/validation.py` | Reproduce the walk-forward Calmar Pareto frontier. Quantify how the 3-window RP averaging compares to picking the single best window. |
| Daily MDD | `engine/stats.py:compute_max_drawdown_daily` | Implement daily MDD via daily-return reconstruction. Compare against monthly MDD on production weights — gap is typically 4–8%. |
| Compare ALLW | `research/compare_allw.py` | Read this through. It's a polished comparison artefact; understand the fee-adjustment math and Excel/PNG outputs. |
| Live ETF substitution | `research/compare_live_etfs.py` | Read this through. Note how the GSG→PDBC substitution looks acceptable at the pair level but degrades the portfolio. |
| Alpaca rebalance | `live/alpaca_rebalance.py` | Operational, not pedagogical. Read the preview logic; understand the order-placement contract; do not rewrite. |

---

# Appendix — Minimal API surface

If you want a quick reference for the production API, here are the
exact signatures your `learning/` modules should target:

```python
# engine/data.py
def fetch_prices(tickers: list[str], start_date: str, end_date: str) -> pd.DataFrame

# engine/optimiser.py
def compute_risk_parity_weights(prices: pd.DataFrame,
                                tickers: list[str],
                                estimation_years: float = 5.0,
                                min_weight: float = 0.02,
                                end_date: str | None = None) -> dict[str, float]

# engine/backtest.py
def run_backtest(prices: pd.DataFrame,
                 benchmark_prices: pd.Series,
                 allocation: dict,
                 portfolio_value: float | None = None,
                 tlt_prices: pd.Series | None = None,
                 transaction_cost_pct: float = 0.0,
                 tax_drag_pct: float = 0.0) -> pd.DataFrame

# engine/stats.py
def compute_cagr(series: pd.Series, years: float) -> float
def compute_max_drawdown(series: pd.Series) -> float
def compute_sharpe(monthly_ret_series: pd.Series, rf_annual: float = 0.0) -> float
def compute_sortino(monthly_ret_series: pd.Series, rf_annual: float = 0.0) -> float
def compute_calmar(cagr: float, max_drawdown: float) -> float
def compute_ulcer_index(series: pd.Series) -> float
def compute_stats(backtest: pd.DataFrame, prices=None, allocation=None) -> list[StrategyStats]
```

Production also has `compute_avg_drawdown`, `compute_max_drawdown_duration`,
`compute_avg_recovery_time`, and `compute_max_drawdown_daily` —
add these in Session 5 if you want full feature parity.
