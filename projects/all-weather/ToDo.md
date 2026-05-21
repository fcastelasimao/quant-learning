# All Weather Portfolio — TODO

Last updated: 2026-05-19

---

## Completed

- [x] RP validated across 3 OOS splits — beats manual on all 3
- [x] Universe scan (100 ETFs, 50k subsets) — confirms 6-asset universe
- [x] 8-asset validation — 6-asset wins on all Calmar windows
- [x] Bond leverage (1.0x–2.5x) — destroys Calmar in rising-rate regime
- [x] ALLW comparison — Calmar 3.03 vs 2.18 (fee-adjusted)
- [x] SPY momentum overlay grid search — does not add value, closed
- [x] Rolling RP — converges to same weights as static, closed
- [x] Weekly rebalancing — no improvement after costs, closed
- [x] Data integrity audit — all OOS claims verified
- [x] FMP adjusted-close data backfill — complete for production tickers
- [x] RP rerun across yfinance/FMP total-return and price-return bases — FMP adj_close confirms yfinance total-return
- [x] RSI ETF leverage overlay research bundle — default 20% overlays plus threshold/leverage grid
- [x] Marimo leverage comparison notebook — date filter, overlay selector, threshold/leverage heatmaps
- [x] RSI leverage OOS validation runner — IS-only selection for 2018/2020/2022 windows
- [x] Paper trading launched — April 2026 via Alpaca (two accounts)
- [x] Production strategy named 6 Asset RP Baseline (`6_asset_rp_baseline` alias)
- [x] Live rebalancer defaults to live tickers and blocks non-production strategies
- [x] LinkedIn post + comparison plot
- [x] Dead code archived, config.py fixed to production strategy

---

## Current — RSI leverage overlay research

- [ ] Run and review RSI overlay OOS validation bundle
- [x] Extend GLD overlay leverage grid above 50% in OOS research mode only
- [ ] Add train/test or walk-forward threshold selection before accepting any grid winner
- [ ] Add trade-level diagnostics: episode count, average win/loss, worst episode, recovery time
- [ ] Test combined capped overlays after one-ETF overlays pass OOS checks

---

## Current — Product launch

- [x] Strategy name: 6 Asset RP Baseline
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
| yfinance vs FMP data-source rerun | PASSED — adjusted-close matches total-return |
| SPY momentum overlay | CLOSED |
| Rolling RP vs static | CLOSED |
| Weekly vs monthly rebalancing | CLOSED |
| Universe scan (100 ETFs) | CLOSED — 6-asset confirmed |
| 8-asset universe | CLOSED — 6-asset wins |
| Bond leverage | CLOSED — destroys Calmar |
| RSI ETF leverage overlay | RESEARCH — GLD/SPY promising in-sample, needs OOS gate |
| Paper trading | STARTED — April 2026 |
| Demand (>100 signups) | Open — after blog |
