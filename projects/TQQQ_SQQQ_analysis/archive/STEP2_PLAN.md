# Step 2 — Backtest Engine (TQQQ standalone, 2-D RSI window sweep)

Hand-off document for Sonnet. **Read this entire document before writing any code.** Project-wide context lives in `PROJECT_PLAN.md`; Step 1's findings (which informed several of the decisions below) live in `step1_outputs/eda_report.md`. Skim both before starting.

---

## 0. What this step is and is NOT

**Is:** pure computation. Take the TQQQ canonical trade log; for each cell in a 21-cell RSI-window grid (plus a baseline), walk the trades chronologically, decide for each trade whether the leverage sleeve fires, compute net P/L (including borrow cost), build the equity curve, and produce a metrics table and a per-trade equity walk per scenario. Output goes to `runs/`.

**Is NOT:**
- The presentation layer. **No plots, no heatmaps, no markdown reports.** All visualization is Step 3.
- SQQQ standalone. **TQQQ only.**
- Combined TQQQ+SQQQ portfolio. **Deferred to a later phase.**
- Intraday RSI logic / independent sleeve exit. v1 sleeve enters and exits with the underlying trade.
- A re-validation of Step 1's checks. Assume the canonical CSV is correct and well-formed.

---

## 1. Project briefing in two paragraphs

We have a TQQQ trade log from a 2020–2026 Alpaca backtest. We want to evaluate adding a leverage sleeve on top: when `RSI_entry` (the strategy's RSI at trade entry) falls inside a window `[low, high)`, deploy an extra 30 % of current portfolio value into the same trade, sized off pre-trade equity, with the same fill prices as the underlying. The sleeve incurs a borrow cost (^IRX yield + an Alpaca-style tier spread) while held, then closes when the underlying trade closes. We're sweeping 21 different `(low, high)` window cells plus a "no sleeve" baseline, producing a metrics table that downstream code (Step 3) will visualize as heatmaps.

Step 1 EDA established: (a) `pnl_pct` is in percentage form (e.g., `2.33` means 2.33 %) — `/100` is mandatory in the sleeve formula; (b) `RSI_entry` only falls in `[35, 72]` in the canonical data; (c) there is no detectable single-axis RSI signal in the data, so the experiment is really about quantifying the leverage/exposure tradeoff under borrow cost rather than finding an alpha signal. The work should still be done cleanly and honestly so that the resulting metrics speak for themselves.

---

## 2. Working directory

```
/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/RSI_tests/
```

Use absolute paths in code. Create outputs under `runs/`.

---

## 3. Environment

Run all Python in the `quant` conda env:

```bash
conda activate quant
~/opt/anaconda3/envs/quant/bin/python step2_backtest.py
```

Required packages: `pandas`, `numpy`, `sqlite3`, `scipy.stats`. Do NOT import `matplotlib` — Step 2 has no plots.

---

## 4. Inputs

### Trade log (primary)
- `TRADES_TQQQ_canonical.csv` — 1627 trades, single run `2026-05-11 16:23:17`, span 2020-01-02 13:30 → 2026-05-08 14:00.

### Price/rate data (SQLite)
- `DB_TQQQ_historical_data.db` → `candles_1d` table → use `adj_close` for the B&H TQQQ benchmark.
- `DB_^IRX_historical_data.db` → `candles_1d` table → use `close` as the annualized 13-week T-bill yield, in **percent** (e.g., `3.60` means 3.60 %).

### Critical column references in canonical CSV
| Column | Meaning |
|---|---|
| `entry_time` | bar timestamp of entry (ET, no timezone in CSV) |
| `exit_time` | bar timestamp of exit |
| `avg_order_price` | actual entry fill (slippage baked in) |
| `exit_avg_order_price` | actual exit fill (slippage baked in) |
| `pnl_pct` | per-share return **in percent**, e.g. `2.33` means 2.33 % |
| `RSI_entry` | RSI at entry — the sleeve gating variable |
| `capital_before` | equity right before this trade, **in the baseline (no-sleeve) run** |
| `capital_after` | cash remaining right after entry order filled |
| `capital_end` | equity right after this trade closes (baseline) |
| `pnl` | dollar P/L of this trade in the baseline run |
| `exit_reason` | e.g. `TRAIL_STOP` |

`capital_before` for trade #1 in the TQQQ canonical is **$10,000**. This is the starting capital. Each trade deploys ~95 % of available capital, so the strategy is almost fully invested whenever a trade is open.

---

## 5. Outputs (write everything into `runs/`)

### `runs/metrics.csv`
One row per scenario (22 total: 21 window cells + 1 baseline). Required columns:

| Column | Type | Description |
|---|---|---|
| `scenario` | str | `"baseline"` or `"low<L>_high<H>"` (e.g., `"low40_high60"`) |
| `low` | int or NaN | low bound (NaN for baseline) |
| `high` | int or NaN | high bound (NaN for baseline) |
| `sleeve_trigger_rate` | float | fraction of trades where the sleeve fired |
| `n_trades_total` | int | always 1627 |
| `n_trades_triggered` | int | count of trades where sleeve fired |
| `starting_capital` | float | 10000.0 |
| `final_equity` | float | equity at end of last trade |
| `total_return` | float | `(final_equity / starting_capital) - 1` |
| `cagr` | float | annualized return over the span |
| `daily_vol_ann` | float | annualized std of daily returns |
| `sharpe_ann` | float | annualized Sharpe (Rf=0) |
| `sortino_ann` | float | annualized Sortino (Rf=0) |
| `calmar` | float | `cagr / max_dd` |
| `omega` | float | Omega(0) on daily returns |
| `max_dd` | float | maximum drawdown as a positive fraction (e.g. 0.25 = 25 %) |
| `max_dd_duration_cal_days` | int | longest calendar-days underwater |
| `max_dd_duration_trading_days` | int | longest trading-days underwater |
| `ulcer_index` | float | RMS of drawdowns on daily series |
| `time_underwater_pct` | float | fraction of days below previous peak |
| `var_95` | float | 5th-percentile daily return (negative number) |
| `cvar_95` | float | mean of daily returns ≤ var_95 |
| `skew` | float | skewness of daily returns |
| `excess_kurt` | float | excess kurtosis of daily returns |
| `tail_ratio` | float | abs(p95) / abs(p5) of daily returns |
| `win_rate` | float | trade-level fraction of trades with positive net P/L |
| `profit_factor` | float | gross profit / abs(gross loss) |
| `expectancy_per_trade` | float | mean trade P/L in dollars |
| `avg_hold_days` | float | mean trade duration |
| `max_losing_streak` | int | max consecutive losing trades |
| `sleeve_only_total_pnl` | float | sum of net sleeve P/L (in dollars) across all triggered trades |
| `marginal_cagr_vs_baseline` | float | `cagr - cagr_baseline` |
| `marginal_sharpe_vs_baseline` | float | `sharpe - sharpe_baseline` |
| `sharpe_ci_low` / `sharpe_ci_high` | float | 95 % stationary-block-bootstrap CI on `sharpe_ann` |
| `cagr_ci_low` / `cagr_ci_high` | float | 95 % CI on `cagr` |
| `deflated_sharpe` | float | Deflated Sharpe Ratio across the 22-scenario grid (Bailey & López de Prado) — same value duplicated on every row |
| `vs_bh_alpha_ann` | float | annualized alpha vs B&H TQQQ |
| `vs_bh_beta` | float | beta vs B&H TQQQ |
| `vs_bh_info_ratio_ann` | float | annualized Information Ratio vs B&H TQQQ |

Append one more row at the end of the CSV: `scenario="bh_tqqq"` with the same metrics computed on the B&H TQQQ daily equity curve over the same window (most cross-CI / sleeve-only / vs-bh columns will be NaN — that's fine).

### `runs/equity_<scenario>.csv`
22 files, one per scenario. Per-trade equity walk. Required columns:

| Column | Description |
|---|---|
| `trade_id` | from canonical CSV |
| `entry_time` | datetime |
| `exit_time` | datetime |
| `RSI_entry` | float |
| `sleeve_triggered` | bool |
| `equity_before` | equity right before trade opens |
| `baseline_pnl_dollars` | underlying trade P/L applied to current equity |
| `sleeve_notional` | `0.30 × equity_before` if triggered else `0.0` |
| `sleeve_gross_pnl` | `sleeve_notional × (pnl_pct / 100)` if triggered else `0.0` |
| `sleeve_days_held` | calendar days between entry and exit |
| `sleeve_borrow_rate_ann` | the rate used for this sleeve (^IRX% + tier spread) |
| `sleeve_borrow_cost` | dollars |
| `sleeve_net_pnl` | gross − borrow_cost |
| `equity_after` | equity after baseline P/L and any sleeve P/L applied |
| `exit_reason` | from canonical |

These two output families are everything Step 3 needs.

---

## 6. Detailed task spec

### 6.1 Setup
- Create `runs/` directory.
- Load `TRADES_TQQQ_canonical.csv` with `pd.read_csv(..., parse_dates=["entry_time", "exit_time"])`.
- Sort by `entry_time` ascending, reset index.
- Load `^IRX` daily close from SQLite into a `Series` indexed by ET date.
- Load TQQQ `adj_close` from SQLite into a `Series` indexed by ET date.

### 6.2 Sleeve interface

```python
from abc import ABC, abstractmethod

class Sleeve(ABC):
    @abstractmethod
    def should_enter(self, trade_row) -> bool: ...

    @abstractmethod
    def exit_event(self, trade_row, current_time) -> bool: ...

    @property
    @abstractmethod
    def label(self) -> str: ...


class NoSleeve(Sleeve):
    @property
    def label(self): return "baseline"
    def should_enter(self, trade_row): return False
    def exit_event(self, trade_row, current_time): return True  # never used


class WindowEntryRSISleeve(Sleeve):
    def __init__(self, low: float, high: float):
        self.low = low
        self.high = high

    @property
    def label(self): return f"low{int(self.low)}_high{int(self.high)}"

    def should_enter(self, trade_row) -> bool:
        return self.low <= trade_row.RSI_entry < self.high

    def exit_event(self, trade_row, current_time) -> bool:
        # v1: sleeve exits when the underlying trade exits
        return current_time >= trade_row.exit_time
```

`exit_event` is unused in v1 (the engine always closes the sleeve at `exit_time`). It's there so the interface is ready for v2 intraday-RSI exits.

### 6.3 Borrow-cost helper

```python
TIER_SPREAD_SCHEDULE = [
    (25_000,      0.080),
    (50_000,      0.070),
    (100_000,     0.060),
    (250_000,     0.050),
    (500_000,     0.045),
    (float("inf"), 0.040),
]

def tier_spread(equity: float) -> float:
    for cap, spread in TIER_SPREAD_SCHEDULE:
        if equity <= cap:
            return spread
    return TIER_SPREAD_SCHEDULE[-1][1]

def annual_borrow_rate(irx_close_pct: float, equity: float) -> float:
    return irx_close_pct / 100.0 + tier_spread(equity)

def borrow_cost(notional: float, ann_rate: float, days_held: float) -> float:
    return notional * ann_rate * days_held / 365.0
```

Look up `^IRX` close on the trade's `entry_time` ET date. If the date is missing (Treasury holiday on an NYSE trading day), forward-fill from the most recent prior date.

### 6.4 Per-scenario backtest walker

For each scenario (the 21 windows plus baseline):

1. Initialize `equity = 10_000.0` and an empty list of per-trade rows.
2. For each trade in chronological order:
   a. Record `equity_before = equity`.
   b. Compute the baseline return ratio of the trade from the canonical row's `capital_after_baseline / capital_before_baseline`. **Use stored values, not pnl_pct**: `baseline_ratio = capital_end / capital_before`. This correctly captures the strategy's actual sizing (~95 % deployed) and any small accounting frictions baked into the original.
      - `baseline_pnl_dollars = equity_before * (baseline_ratio - 1)`.
      - `equity_after_baseline = equity_before * baseline_ratio`.
   c. Decide sleeve via `sleeve.should_enter(trade_row)`.
   d. If sleeve fires:
      - `sleeve_notional = 0.30 * equity_before`.
      - `sleeve_gross = sleeve_notional * (trade_row.pnl_pct / 100.0)`.
      - `days_held = (trade_row.exit_time - trade_row.entry_time).total_seconds() / 86400.0`.
      - Look up `^IRX` rate on `entry_time` ET date (forward-fill on miss).
      - `ann_rate = annual_borrow_rate(irx_close_pct, equity_before)`.
      - `borrow = borrow_cost(sleeve_notional, ann_rate, days_held)`.
      - `sleeve_net = sleeve_gross - borrow`.
      - `equity_after = equity_after_baseline + sleeve_net`.
    Else: `sleeve_net = 0`, all sleeve fields zero, `equity_after = equity_after_baseline`.
   e. Append row to per-trade list. Update `equity = equity_after`.
3. After the loop, return the per-trade DataFrame.

Note: starting equity is $10 000, but capital_before for trade #1 in the source is also $10 000, so the baseline-ratio walk reproduces the canonical's `capital_end` series for the baseline scenario. **Verify this**: for the baseline scenario, `equity_after` of each trade should match the canonical CSV's `capital_end` (within 1e-6 relative tolerance). This is an integrity check, not a hard assert — print a warning if it fails for >1 % of trades.

### 6.5 Daily-resampled equity curve

Most risk metrics use daily returns, not per-trade returns. After the per-trade walk, build a daily equity series:

- Construct a daily date index covering the analysis window (`2020-01-02` → last `exit_time` ET date).
- At each daily timestamp, the equity is the last-known `equity_after` from a trade whose `exit_time <= daily_timestamp`. Before the first trade closes, equity = starting capital.
- Build as a `pd.Series` indexed by date.
- `daily_returns = equity.pct_change().dropna()`.

This is an approximation (we don't have intra-trade marks), so daily returns are stepwise — change on days a trade closes, zero on days no trade closed. Acceptable for v1; we'll improve in v2 with bar data.

### 6.6 Metrics computation

All metrics take either the per-trade DataFrame or the daily equity series as input.

**Definitions to be precise about:**

- `cagr = (final_equity / starting_capital) ** (365 / span_days) - 1` where `span_days` = calendar days from first trade entry to last trade exit.
- `sharpe_ann = mean(daily_returns) / std(daily_returns) * sqrt(252)`. Risk-free rate set to 0 for consistency across scenarios.
- `sortino_ann = mean(daily_returns) / std(min(daily_returns, 0)) * sqrt(252)`.
- `omega(0)` = `sum(positive_returns) / abs(sum(negative_returns))` on daily returns.
- `max_dd`: walk daily equity, track running max, drawdown = `(equity - running_max) / running_max`. Return `-min(drawdown_series)` (a positive number).
- `max_dd_duration_cal_days`: longest run of consecutive days below the prior peak (calendar days).
- `max_dd_duration_trading_days`: same but counted in trading days (drop weekends).
- `ulcer_index = sqrt(mean(drawdown_series**2))`.
- `time_underwater_pct = mean(drawdown_series < 0)`.
- `var_95 = daily_returns.quantile(0.05)` (a negative number).
- `cvar_95 = daily_returns[daily_returns <= var_95].mean()`.
- `tail_ratio = abs(daily_returns.quantile(0.95)) / abs(daily_returns.quantile(0.05))`.
- `profit_factor = sum(positive_trade_pnls) / abs(sum(negative_trade_pnls))` on the per-trade `(baseline + sleeve_net)` P/L.
- `expectancy_per_trade = mean(total_trade_pnl)` in dollars.
- `avg_hold_days = mean((exit_time - entry_time).total_seconds() / 86400)`.
- `max_losing_streak`: longest run of consecutive trades with total P/L < 0.

### 6.7 B&H TQQQ benchmark

- Load TQQQ `adj_close` daily series for the analysis window.
- `bh_equity = 10_000 * adj_close / adj_close.iloc[0]`.
- Compute every metric on this series (using daily returns from `adj_close.pct_change()`).
- Write as a final row in `metrics.csv` with `scenario="bh_tqqq"`. Many sleeve-specific columns will be NaN.
- Use B&H TQQQ daily returns to compute `vs_bh_alpha_ann`, `vs_bh_beta`, `vs_bh_info_ratio_ann` for each strategy scenario:
  - `beta` = OLS slope of `daily_returns_strategy` on `daily_returns_bh`.
  - `alpha_daily` = mean(`daily_returns_strategy` - `beta * daily_returns_bh`).
  - `alpha_ann` = `alpha_daily * 252`.
  - `excess_returns = daily_returns_strategy - daily_returns_bh`.
  - `info_ratio_ann = mean(excess) / std(excess) * sqrt(252)`.

### 6.8 Bootstrap CIs

For each scenario (not for the benchmark), compute stationary-block-bootstrap 95 % CIs on `sharpe_ann` and `cagr`.

- 1000 iterations.
- Block length: 10 (calendar days). Reasonable for daily data with mild autocorrelation.
- For each iteration: sample blocks with replacement until you have a daily-returns series of the original length; compute the metric on that resampled series; record.
- Return `[2.5th percentile, 97.5th percentile]` as `(ci_low, ci_high)`.
- Implementation: use the standard "stationary bootstrap" (Politis & Romano, 1994) where each block length is drawn from a geometric distribution with mean 10. If implementing from scratch feels heavy, a simpler circular-block bootstrap with fixed length 10 is acceptable — note the choice in a code comment.

### 6.9 Deflated Sharpe Ratio

One value, applied uniformly across all 22 scenarios in `metrics.csv`.

Formula (Bailey & López de Prado, 2014, simplified):

```
DSR = Φ( (SR_max - E[max(SR)]) * sqrt(T - 1) / sqrt(1 - skew*SR_max + (kurt - 1) / 4 * SR_max**2) )

where:
  SR_max = the maximum observed Sharpe across the 22 scenarios
  E[max(SR)] = expected max under null, ≈ (1 - γ) * Φ^-1(1 - 1/N) + γ * Φ^-1(1 - 1/(N*e))
  γ = Euler-Mascheroni constant (~0.5772)
  N = 22 (number of trials)
  T = effective sample size (use number of daily returns)
  skew, kurt = skewness and kurtosis of daily returns of the best-Sharpe scenario
  Φ = standard normal CDF
```

DSR is the probability that the best observed Sharpe is "real" given multiple-comparisons bias. If implementing the exact formula is too involved, a workable approximation is to report `SR_max - E[max(SR)]` as a "deflation adjustment" with a code comment explaining the simplification. Default to writing this column carefully — it's the user's main protection against fooling themselves on the grid search.

### 6.10 Write CSVs

- Order rows in `metrics.csv` so baseline is first, then the 21 cells sorted by `(low, high)` ascending, then `bh_tqqq` last.
- Per-trade equity CSVs: filenames `equity_baseline.csv`, `equity_low35_high45.csv`, etc.
- Use `float_format="%.6f"` to keep file sizes reasonable.
- Print one progress line per scenario completed.

---

## 7. Acceptance criteria

- Script exits 0 with no uncaught exceptions.
- `runs/metrics.csv` exists with **23 rows** (header + 22 scenarios + 1 benchmark) and every column specified in §5.
- `runs/equity_*.csv` — exactly **22 files** (baseline + 21 cells).
- Baseline integrity check: for the baseline scenario, `equity_after` per trade is within 1e-6 relative tolerance of canonical `capital_end`. Print a single summary line: `Baseline integrity: matches canonical capital_end for N/1627 trades`. If N < 1610, surface as a warning.
- Sleeve-trigger sanity: for each window, `sleeve_trigger_rate` should equal `mean(low <= RSI_entry < high)` over all trades. Print one line per scenario.
- `metrics.csv` `bh_tqqq` row: `final_equity` should be roughly `10000 * adj_close[last] / adj_close[first]`. Sanity-check it's positive and not absurd (likely 5x–10x over the window).

---

## 8. Hard constraints / what NOT to do

- **No plots.** No `matplotlib`, no `seaborn`. Step 3 owns visualization.
- **No SQQQ.** Not in scope. Ignore `TRADES_SQQQ_canonical.csv` entirely.
- **No combined portfolio.** Not in scope.
- **No intraday RSI / independent sleeve exit.** Sleeve exits with the underlying trade in v1.
- **No fees beyond borrow cost.** Slippage is already in `avg_order_price` per Step 1; do NOT add commissions or slippage on top.
- **No modifying canonical CSV.** Read-only.
- **No revalidating Step 1's checks.** Assume the canonical is clean.
- **Use `^IRX` close as percent, not as fraction.** `irx_close = 3.60` means 3.60 %. Divide by 100 inside the borrow formula.
- **Use `pnl_pct` divided by 100 in the sleeve formula.** Same scaling as ^IRX.
- **Compute `baseline_ratio` from `capital_end / capital_before`**, not from `pnl_pct` directly. The strategy deploys ~95 % per trade, so the *portfolio* return is ~95 % of the *per-share* return — using `pnl_pct / 100` for the baseline would overstate baseline returns by ~5 %.

---

## 9. Implementation hints

- The 1627-trade per-scenario walk is trivial in Python time. 22 scenarios × 1627 trades = 35 k iterations. Vectorize the borrow-cost math if convenient, but a plain Python loop per scenario is fine.
- Bootstrap is the expensive step: 22 × 1000 × O(daily-resample-and-metric) ≈ low seconds. Cache the daily returns series per scenario instead of recomputing.
- For the daily equity series, `pd.Series(...).resample("D").ffill()` over an index built from trade exit times is the canonical way.
- ^IRX rate lookup: index your `^IRX` Series by date (`et_datetime[:10]`), then on each entry_time take `irx.asof(entry_date)` to forward-fill.

---

## 10. When the user reviews

The user will look at `runs/metrics.csv` and want to know:
1. Did the baseline scenario reproduce the canonical's final equity? (Integrity check.)
2. What's the headline pattern across cells: which window cell has the best Sharpe, the worst Max DD, the highest CAGR?
3. Does adding leverage hurt or help, on net?
4. How wide are the bootstrap CIs, and is the best-cell Sharpe still impressive after the Deflated-Sharpe adjustment?

You don't need to answer those questions in this step — the metrics CSV does. Just make sure the numbers are computed correctly, the file is complete, and the per-trade CSVs are written so the user can spot-check individual trades.

If something in the canonical data unexpectedly breaks the engine (e.g., a `pnl_pct` that produces a negative borrow cost via some weird sign issue), STOP and surface it cleanly rather than silently masking. The user wants correctness over speed.
