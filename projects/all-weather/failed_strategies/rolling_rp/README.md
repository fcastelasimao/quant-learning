# Rolling Risk Parity

**Status:** Closed — converges to static RP.

**Question:** Does recomputing RP weights quarterly from a trailing 5-year covariance produce better OOS Calmar than static RP weights fixed at the IS boundary?

**Result:** On all 3 OOS splits (2018, 2020, 2022), rolling RP converges to approximately the same weights as static RP. No consistent Calmar improvement. Static weights are simpler, more interpretable, and equally good.

**Run:**
```bash
conda run -n allweather python3 failed_strategies/rolling_rp/run_rolling_rp.py
```

**Files:**
- `rolling_rp.py` — `run_backtest_rolling_rp()` function (extracted from engine/backtest.py)
- `run_rolling_rp.py` — experiment runner: rolling vs static on 3 OOS splits
