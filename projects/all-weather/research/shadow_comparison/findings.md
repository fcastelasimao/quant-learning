---
verdict: todo
summary: "Live vs simulated returns reconciliation — not yet started (Section G in handoff)"
promoted: null
---

# Shadow Comparison

**Status:** TODO — planned for Section G of the active work plan.

Will reconcile actual live portfolio performance (from `live/logs/run_summary.jsonl` and `live/logs/runs/*.json`) against simulated backtest returns to detect slippage, fill-price divergence, and execution quality issues.

## Planned deliverables

- Cumulative actual vs simulated with deviation bands
- Per-rebalance fill-price vs simulated-price diff
- `shadow_summary.csv` to bundle for marimo
- `live/logs/WARNINGS.log` when MAE > threshold

## Scripts

- `backtest_shadow.py` — stub, consumes legacy `performance_tracking_*.csv` and new structured logs
