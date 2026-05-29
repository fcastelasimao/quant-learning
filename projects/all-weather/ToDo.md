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
- [x] **Broker-agnostic rebalancer (`live/rebalance.py`)** — `Broker` Protocol in `live/brokers/base.py`; Alpaca + Tastytrade implementations; `make_broker()` factory. Legacy `live/alpaca_rebalance.py` preserved for back-compat. (2026-05-27)
- [x] **Tastytrade broker** — guarded import, session caching at `~/.allweather/tt_session_*.json`, qty-only orders, login CLI helper
- [x] **Threshold-based / interval-based cadence** — `--min-rebalance-interval-days 31` (default), state in `logs/cadence_*.json`. Replaces month-end calendar check.
- [x] **31-day holding-period gate** — FIFO lot ledger in `live/lots.py`; `logs/lots_*.json` persists acquisition dates per lot; sells blocked while any lot < 31 days
- [x] **Per-account strategy budget** — `live/budget.py` virtual sub-portfolio cap (`--budget`, `--initialize-budget`, `--ignore-budget`). Grows from strategy-symbol dividends + PnL.
- [x] **Dry-execute mode** — `--dry-execute` simulates fills from current prices, writes full RunSummary, never advances cadence/lots/budget state. Lets you test logs without waiting for month-end.
- [x] **Structured run logging** — `live/runlog.py`: JSONL stream (`logs/run_summary.jsonl`), per-run JSON archive (`logs/runs/*.json`, auto-pruned at 200), monthly summary CSV (`logs/monthly_runs.csv`)
- [x] **Slack + email notifications** — `live/notify.py`, never raises. Env-var configured (`ALLW_SLACK_WEBHOOK_URL`, `ALLW_NOTIFY_EMAIL`, `ALLW_SMTP_*`)
- [x] **Pre-flight healthcheck** — `live/healthcheck.py`: env, imports, credentials, strategies.json, logs writable, cadence state
- [x] **launchd scheduling template** — `live/scheduler/com.allweather.rebalance.plist.template` + `scripts/install_launchd.sh` (installs `--dry-execute` only; real execution stays manual)
- [x] **JEPQ benchmark** — added to `compare_allw.py`; `research/compare_jepq.py` head-to-head vs AW since 2022-05-03; `JEPQ_Return%` column in performance CSV
- [x] **Backtest shadow** — `research/backtest_shadow.py` reconciles actual live returns against engine simulation (MAE, RMSE, bias + cumulative-deviation chart)
- [x] `docs/BROKER_SETUP.md` — full setup guide for Alpaca + Tastytrade, budget cap, holding-period, notifications, scheduling, migration from legacy

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
- [ ] Some functions still dependent on global variables (config.py)
- [ ] First live execution via the new broker-agnostic pipeline (`live.rebalance`) — requires manual review of one full `--dry-execute` run first
- [ ] Migrate paper-trading workflow from `live.alpaca_rebalance` to `live.rebalance --broker alpaca` after one full month-end has been observed via the new pipeline

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
| Paper trading | STARTED — April 2026 (Alpaca via `live.alpaca_rebalance`) |
| Broker-agnostic pipeline | PASSED — `live.rebalance` ships with Alpaca + Tastytrade backends (2026-05-27) |
| 31-day holding-period enforcement | PASSED — FIFO lot ledger + `--min-rebalance-interval-days` cadence |
| Per-account budget cap | PASSED — virtual sub-portfolio sizing in `live.budget` |
| Structured run logs | PASSED — JSONL + monthly CSV + per-run JSON in `live.runlog` |
| Demand (>100 signups) | Open — after blog |
