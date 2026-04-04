# All Weather Portfolio — TODO

Last updated: 2026-04-04

---

## Completed

- [x] RP validated across 3 OOS splits — beats manual on all 3
- [x] Universe scan (100 ETFs, 50k subsets) — confirms 6-asset universe
- [x] 8-asset validation — 6-asset wins on all Calmar windows
- [x] Bond leverage (1.0x–2.5x) — destroys Calmar in rising-rate regime
- [x] ALLW comparison — Calmar 3.03 vs 2.18 (fee-adjusted)
- [x] Overlay grid search — does not add value, closed
- [x] Rolling RP — converges to same weights as static, closed
- [x] Weekly rebalancing — no improvement after costs, closed
- [x] Data integrity audit — all OOS claims verified
- [x] Paper trading launched — April 2026 via Alpaca (two accounts)
- [x] LinkedIn post + comparison plot
- [x] Dead code archived, config.py fixed to production strategy

---

## Current — Product launch

- [ ] Brand name
- [ ] Blog post (ALLW comparison data ready)
- [ ] Landing page
- [ ] FCA compliance review

---

## Someday / Maybe

- [ ] GBP/EUR currency adjustment for non-US investors
- [ ] Threshold-based rebalancing (vs calendar-based)
- [ ] Track live vs backtest ETF performance divergence
- [ ] Some functions still dependent on global variables (config.py)

---

## Decision gates

| Gate | Status |
|---|---|
| DE | CLOSED |
| Static RP multi-window | PASSED |
| Overlay | CLOSED |
| Rolling RP vs static | CLOSED |
| Weekly vs monthly rebalancing | CLOSED |
| Universe scan (100 ETFs) | CLOSED — 6-asset confirmed |
| 8-asset universe | CLOSED — 6-asset wins |
| Bond leverage | CLOSED — destroys Calmar |
| Paper trading | STARTED — April 2026 |
| Demand (>100 signups) | Open — after blog |
