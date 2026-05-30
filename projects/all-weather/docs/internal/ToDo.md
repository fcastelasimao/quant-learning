# All Weather Portfolio — Historical Changelog (RETIRED as active worklist)

> **Retired as the active TODO on 2026-05-30.**
> Active work tracking has moved to **`docs/internal/session_handoff.md`**.
> This file is kept as a chronological record of completed milestones, useful
> for new sessions / new collaborators who want to see how the project got to
> where it is. **Do not add new active items here.** Add closed milestones
> here when they ship.

Last historical update: 2026-05-30

---

## Completed (chronological)

### Phase 1-12 — research foundation
- [x] RP validated across 3 OOS splits — beats manual on all 3
- [x] Universe scan (100 ETFs, 50k subsets) — confirms 6-asset universe
- [x] 8-asset validation — 6-asset wins on all Calmar windows
- [x] Bond leverage (1.0x–2.5x) — destroys Calmar in rising-rate regime
- [x] ALLW comparison — Calmar 3.03 vs 2.18 (fee-adjusted)
- [x] SPY momentum overlay grid search — does not add value, closed
- [x] Rolling RP — converges to same weights as static, closed
- [x] Weekly rebalancing — no improvement after costs, closed (**pre-tax verdict; may be reopened under tax-aware modelling**)
- [x] Data integrity audit — all OOS claims verified
- [x] FMP adjusted-close data backfill — complete for production tickers
- [x] RP rerun across yfinance/FMP total-return and price-return bases — FMP adj_close confirms yfinance total-return
- [x] RSI ETF leverage overlay research bundle — default 20% overlays plus threshold/leverage grid
- [x] Marimo leverage comparison notebook — date filter, overlay selector, threshold/leverage heatmaps
- [x] RSI leverage OOS validation runner — IS-only selection for 2018/2020/2022 windows

### Phase 13 — paper trading + product naming (April 2026)
- [x] Paper trading launched — April 2026 via Alpaca (two accounts)
- [x] Production strategy named 6 Asset RP Baseline (`6_asset_rp_baseline` alias)
- [x] Live rebalancer defaults to live tickers and blocks non-production strategies
- [x] LinkedIn post + comparison plot
- [x] Dead code archived, config.py fixed to production strategy

### Phase 14-15 — broker-agnostic migration (May 2026)
- [x] **Broker-agnostic rebalancer (`live/rebalance.py`)** — `Broker` Protocol in `live/brokers/base.py`; Alpaca + Tastytrade implementations; `make_broker()` factory. (2026-05-27)
- [x] **Tastytrade broker** — guarded import, session caching, qty-only orders, login CLI helper
- [x] **Threshold-based / interval-based cadence** — `--min-rebalance-interval-days 31` (default), state in `logs/cadence_*.json`. Replaces month-end calendar check.
- [x] **31-day holding-period gate** — FIFO lot ledger in `live/lots.py`; `logs/lots_*.json` persists acquisition dates per lot; sells blocked while any lot < 31 days
- [x] **Per-account strategy budget** — `live/budget.py` virtual sub-portfolio cap (`--budget`, `--initialize-budget`, `--ignore-budget`)
- [x] **Dry-execute mode** — `--dry-execute` simulates fills, writes full RunSummary, doesn't advance state
- [x] **Structured run logging** — `live/runlog.py`: JSONL stream, per-run JSON archive (auto-pruned at 200), monthly summary CSV
- [x] **Slack + email notifications** — `live/notify.py`, never raises. Env-var configured (currently disabled by default)
- [x] **Pre-flight healthcheck** — `live/healthcheck.py`: env, imports, credentials, strategies.json, logs writable, cadence state
- [x] **launchd scheduling template** — `live/scheduler/com.allweather.rebalance.plist.template` + install script (installs `--dry-execute` only)
- [x] **JEPQ benchmark** — added to `compare_allw.py`; `research/compare_jepq.py` head-to-head vs AW since 2022-05-03; `JEPQ_Return%` column in performance CSV
- [x] **Backtest shadow** — `research/backtest_shadow.py` reconciles actual live vs simulated returns
- [x] `docs/BROKER_SETUP.md` — full setup guide

### Phase 16 — cleanup + central data fetcher + handoff (2026-05-30)
- [x] **Phase 1A repo cleanup** — `archive/` deleted, `ALPACA_SETUP_MAC.md` folded into `docs/BROKER_SETUP.md`, root MDs moved to `docs/internal/`, `scripts/install_launchd.sh` moved to `live/scheduler/`, `live/alpaca_rebalance.py` renamed to `live/_legacy/alpaca_rebalance.py` (tests + Makefile + docs updated), `portfolio_holdings.json` removed, `CLAUDE.md` rewritten to match reality
- [x] **Live preview enrichment** — `live/rebalance.py` preview / dry-execute prints cadence gate, budget snapshot, lot-ledger ages inline. One-page human-review report before pressing execute.
- [x] **Alpaca lot-level capability check** — verified `alpaca-py 0.43.2` has no order-time lot selection. Documented in `docs/research/alpaca_lot_selection.md`. Implication: tax-optimal lot selector is a research counterfactual on Alpaca; FIFO is broker reality.
- [x] **Central data manager — dividends** — added `fetch_dividends_history`, `dividends` SQLite table, `--with-dividends` / `--dividends-only` CLI flags to `QuantFinance/data_manager.py`. JEPQ added to default symbol set. Historical fetch verified: SPY 134, QQQ 87, TLT 285, TIP 198, JEPQ 48 rows. GLD/GLDM/GSG correctly empty (no cash distributions).
- [x] **`engine.data.fetch_dividends` reader** — long-form DataFrame from the central per-symbol SQLite. Used by the upcoming tax model.
- [x] **JEPQ price history in central store** — 1021 daily rows from inception 2022-05-03 → 2026-05-29.
- [x] **Central data manager contract doc** — `docs/data/central_data_manager.md`.
- [x] **Session handoff scaffolding** — `docs/internal/session_handoff.md` is now the cross-session memory. `CLAUDE.md` points new sessions at it first.

---

## Decision gates (historical reference)

| Gate | Status |
|---|---|
| DE | CLOSED |
| Static RP multi-window | PASSED |
| yfinance vs FMP data-source rerun | PASSED — adjusted-close matches total-return |
| SPY momentum overlay | CLOSED |
| Rolling RP vs static | CLOSED |
| Weekly vs monthly rebalancing | CLOSED (pre-tax) — to be re-opened under tax-aware modelling |
| Universe scan (100 ETFs) | CLOSED — 6-asset confirmed |
| 8-asset universe | CLOSED — 6-asset wins |
| Bond leverage | CLOSED — destroys Calmar |
| RSI ETF leverage overlay | RESEARCH — GLD/SPY promising in-sample, needs OOS gate |
| Paper trading | STARTED — April 2026 (Alpaca via `live._legacy.alpaca_rebalance`) |
| Broker-agnostic pipeline | PASSED — `live.rebalance` ships with Alpaca + Tastytrade backends (2026-05-27) |
| 31-day holding-period enforcement | PASSED — FIFO lot ledger + `--min-rebalance-interval-days` cadence |
| Per-account budget cap | PASSED — virtual sub-portfolio sizing in `live.budget` |
| Structured run logs | PASSED — JSONL + monthly CSV + per-run JSON in `live.runlog` |
| Central data manager dividends | PASSED — populated for all dividend-paying production tickers (2026-05-30) |
| Demand (>100 signups) | Open — after blog |
