# All Weather Portfolio — TODO

Last updated: 2026-03-26

---

## ✅ COMPLETED

- [x] RP validated across 3 OOS splits — beats manual on all 3
- [x] Universe scan — confirms 6-asset universe
- [x] ALLW comparison with monthly rebalancing — Calmar 2.80 vs 1.96
- [x] Overlay grid search — does not add value, closed
- [x] Data integrity audit — all OOS claims verified
- [x] Production weights: SPY 13%, QQQ 11%, TLT 19%, TIP 33%, GLD 14%, GSG 10%

---

## 🔴 CURRENT — Phase 12: Rolling RP + Rebalancing frequency

### A1: Implement rolling RP backtest
Add `run_backtest_rolling_rp()` to backtest.py. Recomputes RP weights
quarterly from trailing 5-year covariance. Rebalances to new weights each quarter.

Changes needed:
- `backtest.py`: new `run_backtest_rolling_rp()` function
- `config.py`: add `RP_LOOKBACK_YEARS` and `RP_RECOMPUTE_FREQ`
- `run_rolling_rp.py`: new experiment script (3-split validation)

### A2: Compare rolling RP vs static RP
Run both on 3 OOS splits. If rolling matches or beats static, adopt as production.
Also save weight history CSV to show how allocations evolve over time.

### A3: Weekly vs monthly rebalancing experiment
Change `DATA_FREQUENCY = "W"` and `SHARPE_ANNUALISATION = 52`.
Run with transaction costs (0.001). Compare Calmar to monthly.

### A4: add Energy and oil EFTs to test universe.

### A5: Undestand ALLW comparison: is this just investing in the company or what would have happened if I gave my money to the company?

### A6: Some functions are still dependent on global variables

---

## 🟡 NEXT — Product launch

- [ ] Brand name
- [ ] Blog post (ALLW comparison data ready)
- [ ] Landing page
- [ ] Paper trading (start when rolling RP settled)
- [ ] FCA compliance review

---

## Decision gates

| Gate | Status |
|---|---|
| DE | CLOSED |
| Static RP multi-window | PASSED |
| Overlay | CLOSED |
| Rolling RP vs static | Open — Phase 12 |
| Weekly vs monthly rebalancing | Open — Phase 12 |
| Demand (>100 signups) | Open — after blog |
| Paper trading | Open — after rolling RP |