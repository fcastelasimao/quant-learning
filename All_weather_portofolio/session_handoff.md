# Session Handoff — 2026-03-26

## Current state
Phase 12: Implementing rolling RP and weekly rebalancing experiments.

## What needs to be implemented

### Rolling RP (backtest.py)
Add `run_backtest_rolling_rp()` function that:
- Takes daily prices and a list of tickers (no fixed allocation)
- At each recomputation date (quarterly by default), calls
  `compute_risk_parity_weights()` with `end_date` set to that date
- Uses the new RP weights as the allocation until next recomputation
- Returns (backtest_df, weight_history) — same format as run_backtest
  plus a log of how weights changed over time

Config additions:
- `RP_LOOKBACK_YEARS = 5.0`
- `RP_RECOMPUTE_FREQ = "QS"` (quarter start)

### run_rolling_rp.py
New experiment script that:
- Runs rolling RP on full period (2006-2026) and each OOS split
- Compares to static RP results
- Saves weight evolution to CSV
- Logs to master log

### Weekly rebalancing test
Change `DATA_FREQUENCY = "W"`, `SHARPE_ANNUALISATION = 52`.
Run with `TRANSACTION_COST_PCT = 0.001`.
Compare monthly vs weekly Calmar on 2020-split OOS.

## ALLW comparison note
compare_allw.py now uses monthly rebalancing (build_daily_series with
rebalance=True). Results are consistent with the 20-year backtest methodology.
Bug fixed: PDBC 0.98 → 0.098 in rpavg_live allocation.
Bug to fix: `cell.font = "C9D1D9"` in save_comparison_excel footer — should
be `cell.font = small_grey`.

## Production allocation
SPY: 13%  QQQ: 11%  TLT: 19%  TIP: 33%  GLD: 14%  GSG: 10%