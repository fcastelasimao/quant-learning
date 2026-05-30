# Weekly / Threshold Rebalancing

**Status:** Closed under transaction-cost-only modelling. **To be re-opened**
under tax-aware modelling (Section D, 2026-05-30) — the original kill criterion
ignored realised capital-gains tax, which is the dominant cost for a monthly-
rebalanced US-taxable account and the channel through which drift triggers most
plausibly add value.

**Question:** Does threshold-based rebalancing (only trade when drift > X%) improve Calmar vs fixed monthly rebalancing? Does `full_on_breach` (rebalance all assets when any drifts) beat `per_asset` (trade only the drifting asset, hold proceeds as cash)?

**Original result (pre-tax):** `run_threshold_sensitivity.py` sweeps 0%–20% thresholds on 3 windows. At 0 cost, threshold rebalancing matches monthly. At realistic costs (0.1%), neither mode improves Calmar. Cash drag from `per_asset` hurts Sharpe. Monthly rebalancing remains production default.

**Why the verdict may flip under tax-aware modelling:**
1. Drift triggers reduce rebalance frequency → fewer realised gains → lower tax bill on the rebalanced slice.
2. Under monthly rebalancing every lot is <1 year old, so realised gains are short-term (top federal rate ~37% + state). Under drift triggers, lots routinely age past 365 days and the rate drops to ~23.8% federal (LT + NIIT). Structural rate advantage that the original tx-cost-only model couldn't see.
3. Combined with a tax-optimal (LT-first / HIFO) lot selector, the advantage compounds further.

**Re-test gate (Section D):** the new `research/tax_threshold_sweep.py` must show drift+tax beating monthly+tax on Calmar by ≥5% on ≥2 of 3 OOS windows before the production policy moves to drift-trigger. If it does, this README's `Status` will flip to "Re-opened — drift threshold X% adopted on YYYY-MM-DD". If it doesn't, this README stays closed and the tax model becomes a reporting tool on the existing monthly policy.

**Run:**
```bash
conda run -n allweather python3 failed_strategies/weekly_rebalance/run_rebalance_mode_comparison.py
conda run -n allweather python3 failed_strategies/weekly_rebalance/run_threshold_sensitivity.py
```

**Files:**
- `run_rebalance_mode_comparison.py` — monthly vs per_asset vs full_on_breach head-to-head
- `run_threshold_sensitivity.py` — threshold sweep (0%–20%) on IS/OOS windows
