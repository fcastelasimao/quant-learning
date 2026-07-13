---
verdict: closed
summary: "126 parameter combos tested — re-entry timing not reliably learnable from IS data"
promoted: null
---

# SPY Momentum Overlay

**Question:** Can a drawdown-protection overlay on SPY (exit when DD > threshold + D1 < 0 + D2 < 0; re-enter when D1 > 0 + D2 > 0 OR price recovers) improve OOS Calmar?

**Result:** 126 combinations tested (7 d_window x 6 threshold x 3 reduce_pct). Best IS Calmar: 0.72 (d_window=20, threshold=0.10, reduce_pct=1.0). OOS result on the same combo: 0.43 vs 0.48 baseline — overlay hurts on the hardest split. Re-entry timing is not reliably learnable from IS data.

**Run:**
```bash
conda run -n allweather python3 research/momentum_overlay/run_overlay_grid.py
```

**Files:**
- `overlay.py` — `compute_overlay_signal()` + `run_backtest_with_overlay()` (extracted from engine/backtest.py)
- `run_overlay_grid.py` — IS grid search + OOS validation
