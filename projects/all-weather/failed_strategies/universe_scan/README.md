# 100-ETF Universe Scan

**Status:** Closed — 6-asset universe confirmed optimal.

**Question:** Is there a better ETF subset than the production 6-asset universe, found by searching ~100 candidate ETFs?

**Method:** 50k random subsets of 3–8 ETFs from ~100 candidates. Score each subset by diversification ratio under RP weights, using only IS data (before 2020). Filter for minimum history, de-duplicate correlated tickers.

**Result:** No consistently superior subset found. The production 6-asset universe (SPY, QQQ, TLT, TIP, GLD, GSG) ranks in the top-3% on diversification ratio. The few subsets that score higher either lack sufficient data history or include ETFs with correlation > 0.85 that collapse under stress.

**Run:**
```bash
conda run -n allweather python3 failed_strategies/universe_scan/scan_universes.py
```
