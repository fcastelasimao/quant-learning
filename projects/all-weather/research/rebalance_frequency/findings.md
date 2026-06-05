---
verdict: reopened
summary: "Weekly/threshold rebalancing showed no improvement pre-tax — reopened and superseded by tax_drift_trigger investigation"
promoted: null
---

# Rebalance Frequency (Pre-Tax)

**Status:** Closed under transaction-cost-only modelling. **Reopened and superseded** by the `tax_drift_trigger` investigation, which showed drift beats monthly under realistic US tax modelling.

**Question:** Does threshold-based rebalancing (only trade when drift > X%) improve Calmar vs fixed monthly rebalancing? Does `full_on_breach` (rebalance all assets when any drifts) beat `per_asset` (trade only the drifting asset, hold proceeds as cash)?

**Original result (pre-tax):** `run_threshold_sensitivity.py` sweeps 0%-20% thresholds on 3 windows. At 0 cost, threshold rebalancing matches monthly. At realistic costs (0.1%), neither mode improves Calmar. Cash drag from `per_asset` hurts Sharpe. Monthly rebalancing remained the production default.

**Why the verdict flipped under tax-aware modelling:** See `../tax_drift_trigger/findings_threshold_sweep.md`. Drift triggers reduce rebalance frequency, fewer realised gains qualify as short-term (taxed at ~37%) instead of long-term (~23.8%). The structural rate advantage dominates transaction costs.

**Run:**
```bash
conda run -n allweather python3 research/rebalance_frequency/run_rebalance_mode_comparison.py
conda run -n allweather python3 research/rebalance_frequency/run_threshold_sensitivity.py
```
