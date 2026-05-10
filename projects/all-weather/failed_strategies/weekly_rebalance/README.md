# Weekly / Threshold Rebalancing

**Status:** Closed — no improvement after transaction costs.

**Question:** Does threshold-based rebalancing (only trade when drift > X%) improve Calmar vs fixed monthly rebalancing? Does `full_on_breach` (rebalance all assets when any drifts) beat `per_asset` (trade only the drifting asset, hold proceeds as cash)?

**Result:** `run_threshold_sensitivity.py` sweeps 0%–20% thresholds on 3 windows. At 0 cost, threshold rebalancing matches monthly. At realistic costs (0.1%), neither mode improves Calmar. Cash drag from `per_asset` hurts Sharpe. Monthly rebalancing remains production default.

**Run:**
```bash
conda run -n allweather python3 failed_strategies/weekly_rebalance/run_rebalance_mode_comparison.py
conda run -n allweather python3 failed_strategies/weekly_rebalance/run_threshold_sensitivity.py
```

**Files:**
- `run_rebalance_mode_comparison.py` — monthly vs per_asset vs full_on_breach head-to-head
- `run_threshold_sensitivity.py` — threshold sweep (0%–20%) on IS/OOS windows
