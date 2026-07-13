---
verdict: closed
summary: "6-asset universe (SPY, QQQ, TLT, TIP, GLD, GSG) confirmed optimal across 50k random subsets and two 8-asset variants"
promoted: "engine/config.py — production ticker list"
---

# Universe Selection

## 100-ETF Universe Scan

**Question:** Is there a better ETF subset than the production 6-asset universe, found by searching ~100 candidate ETFs?

**Method:** 50k random subsets of 3-8 ETFs from ~100 candidates. Score each subset by diversification ratio under RP weights, using only IS data (before 2020). Filter for minimum history, de-duplicate correlated tickers.

**Result:** No consistently superior subset found. The production 6-asset universe (SPY, QQQ, TLT, TIP, GLD, GSG) ranks in the top-3% on diversification ratio. The few subsets that score higher either lack sufficient data history or include ETFs with correlation > 0.85 that collapse under stress.

**Run:**
```bash
conda run -n allweather python3 research/universe_selection/scan_universes.py
```

## 8-Asset Variants

**Question:** Do two 8-asset candidates (scan winner + TIP variant) beat the 6-asset production portfolio?

**Experiments:**
- Experiment A: SPY, QQQ, IJR, TLT, IEF, GLD, CPER, DBA (scan winner, no TIP)
- Experiment B: SPY, QQQ, IJR, TLT, TIP, GLD, CPER, DBA (keeps TIP, drops IEF)

**Result:** RP weights averaged across 3 OOS splits (2018, 2020, 2022). Both 8-asset variants underperform `6asset_tip_gsg_rpavg` on Calmar across all windows. Adding IJR/IEF/CPER/DBA introduces diversification noise that RP cannot exploit without history of inflation regimes.

**Run:**
```bash
conda run -n allweather python3 research/universe_selection/run_8asset_experiments.py
```
