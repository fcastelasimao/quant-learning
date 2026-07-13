# Research Log

---

## Project goal

Backtesting and validation engine for risk-balanced portfolios.
Primary metric: Calmar ratio. Positioned against Bridgewater's ALLW ETF.

## Phase 1-8 (2026-03-18 to 2026-03-20)
Explored 8 universes. TIP replaces LQD. 6asset_tip_gsg as best universe.

## Phase 9 — DE optimiser (2026-03-21 to 2026-03-23)
26 experiments, all fail vs manual. Gate 1 closed.

## Phase 10 — RP foundation (2026-03-23 to 2026-03-25)
Rf=0.035. Daily MDD. RP diagnostic: TLT 2x overweighted, TIP 2x underweighted.
RP 5yr OOS Calmar 0.512 vs manual 0.403.

## Phase 11 — Validation (2026-03-25 to 2026-03-26)

### Data integrity: OOS results in strategies.json corrected.

### RP multi-window: Gate 2 PASSED
| Split | Manual | RP | Improvement |
|---|---|---|---|
| 2020 | 0.406 | 0.480 | +18% |
| 2018 | 0.417 | 0.462 | +11% |
| 2022 | 0.345 | 0.385 | +12% |

### Universe scan: 15 ETFs, 16k subsets. Confirms 6-asset universe.

### SPY momentum overlay grid: 126 combos, 3-split OOS. Does not add value. Closed.

### ALLW comparison (monthly rebalanced, fee-adjusted, Mar 2025-Mar 2026)
| Metric | rpavg | ALLW |
|---|---|---|
| CAGR | 16.05% | 17.23% |
| Max DD | -5.74% | -8.79% |
| Calmar | 2.797 | 1.961 |
| Ulcer | 1.134 | 1.845 |

## Phase 12 — Expanded experiments (2026-04-03 to 2026-04-04)

### Rolling RP
Recomputes RP weights quarterly from trailing 5-year covariance.
Result: converges to same weights as static. Closed.

### Weekly rebalancing
No improvement after transaction costs. Closed.

### 100-ETF universe scan
Researched ~100 candidates across 7 macro buckets, added to scan_universes.py.
Random sampling (50k subsets) from 50 post-dedup ETFs. Top ETFs: TLT, CPER, DBA, GLD.
Result: 6-asset universe confirmed optimal.

### 8-asset validation
Tested {SPY, QQQ, IJR, TLT, IEF, GLD, CPER, DBA} and {SPY, QQQ, IJR, TLT, TIP, GLD, CPER, DBA}.
Used proper RP-averaged weights (3 IS windows, averaged, OOS evaluated).
Result: 6-asset production beats both on all Calmar windows. Closed.

### Bond leverage (1.0x–2.5x on TLT+TIP)
Every 0.25x adds ~0.5% CAGR but ~3% deeper drawdowns.
At 2x leverage, 2022 OOS Calmar drops from 0.355 to 0.079.
Result: leverage destroys risk-adjusted returns in rising-rate regime. Closed.

---

## Phase 13 — Paper trading + code cleanup (2026-04-03)

### Paper trading launched
Two Alpaca accounts:
- Backtest ETFs: SPY, QQQ, TLT, TIP, GLD, GSG
- Live ETFs: IVV, QQQM, TLT, TIP, GLDM, PDBC (lower-cost equivalents)

Multi-account support added to `alpaca_monthly_rebalance.py` via `--account` CLI arg.

### compare_allw.py refactored
- Strategy registry with StrategyDef dataclass, reads allocations from strategies.json
- `enabled` flag per strategy for toggling without deletion
- Excel export cleaned: grey headers, no background colors, spacer rows between groups
- Results path changed to `__file__`-relative (no longer depends on CWD)

### Repository reorganised
Moved to `projects/` layout:
- `All_weather_portfolio/` → `projects/all-weather/`
- `wave_rider/` → `projects/wave-rider/`
- Dead projects to `archive/`
- Dead all-weather code to `projects/all-weather/archive/`

### ALLW comparison updated (fee-adjusted, Mar 2025–Apr 2026)
| Metric | rpavg | ALLW |
|---|---|---|
| CAGR | 17.4% | 19.1% |
| Max DD | -5.7% | -8.8% |
| Calmar | 3.03 | 2.18 |

### LinkedIn post + plot
Two-panel figure: equity curves with metrics table inset + worst drawdown zoom.
Post drafted, gated code sharing ("DM me for methodology").

### Dead code archived
Moved completed experiment scripts to archive/:
run_8asset_experiments.py, run_leverage_experiment.py, scan_universes.py, run_rp_validation.py.
Fixed config.py DEFAULT_STRATEGY (was pointing to 8-asset, now 6asset_tip_gsg_rpavg).
Disabled 8-asset entry in compare_allw.py.

---

## Phase 14 — FMP adjusted-close validation rerun (2026-05-09)

### FMP data upgrade
Added `adj_close` support to the local FMP SQLite daily data. Existing OHLCV files were preserved and enriched by backfilling FMP adjusted close. Coverage is complete for SPY, QQQ, TLT, TIP, GLD, GSG, and GLDM through 2026-05-08.

### RP averaging rerun after rebalancing-code bug fix
Ran `research.rerun_rp_validation` across 4 data bases and the same 2018/2020/2022 OOS windows. Each run wrote normal result folders and appended to `results/master_log.xlsx`.

| Data Basis | 2018 Calmar | 2020 Calmar | 2022 Calmar | Notes |
|---|---:|---:|---:|---|
| yfinance total return | 0.487 | 0.503 | 0.452 | Canonical production basis |
| FMP adjusted close | 0.488 | 0.504 | 0.453 | Confirms yfinance total-return result |
| yfinance price return | 0.334 | 0.343 | 0.284 | Excludes distributions |
| FMP close | 0.334 | 0.343 | 0.284 | Matches price-return diagnostic |

Conclusion: the source difference is tiny; dividend/total-return treatment is the material difference. FMP `adj_close` and yfinance total-return produce effectively the same RP weights and OOS performance. Production allocation remains unchanged because the corrected FMP adjusted-close averaged weights only drift modestly: SPY 13.34%, QQQ 10.76%, TLT 18.65%, TIP 33.48%, GLD 13.59%, GSG 10.19%.

---

## Phase 15 — RSI ETF leverage overlay research (2026-05-11)

Built a research-only leverage overlay engine that leaves the production base
portfolio unchanged and compares independent one-ETF overlays. Each overlay uses
that ETF's own RSI-14 signal, applies a one-day execution lag, and adds overlay
returns on top of the base portfolio.

Latest reviewed bundle:
`results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg`

Data window: 2006-07-21 to 2026-05-08. Default rule: entry RSI < 30, exit RSI >
50, +20% overlay. Grid rule: entry 20-36 by 2, exit 40-70 by 2, overlay 15%-50%
by 5%.

### Default 20% overlay results

| Overlay | CAGR | Sharpe | Calmar | Max DD | Active Days | Interpretation |
|---|---:|---:|---:|---:|---:|---|
| Base | 6.96% | 0.471 | 0.306 | -22.74% | 0.00% | Production reference |
| GLD | 7.27% | 0.489 | 0.324 | -22.45% | 10.92% | Strongest default risk-adjusted overlay |
| SPY | 7.64% | 0.503 | 0.323 | -23.62% | 6.46% | Best default return boost, more equity crash risk |
| TLT | 7.02% | 0.463 | 0.309 | -22.74% | 10.38% | Small benefit |
| TIP | 6.94% | 0.459 | 0.291 | -23.86% | 7.97% | Worse than base |
| QQQ | 7.34% | 0.461 | 0.245 | -29.93% | 7.35% | Return up, risk-adjusted quality down |
| GSG | 6.06% | 0.323 | 0.225 | -26.93% | 16.68% | Reject default rule |

### Best in-sample grid candidates

| Selection | ETF | Entry | Exit | Overlay | CAGR | Calmar | Max DD | Active Days | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Best risk-adjusted | GLD | 22 | 46 | 50% | 8.07% | 0.429 | -18.83% | 3.07% | Most interesting overall candidate |
| Best drawdown preservation | GLD | 22 | 64 | 40% | 7.73% | 0.419 | -18.46% | 8.07% | Supports deeper gold overlay research |
| Best selective equity overlay | SPY | 22 | 42 | 50% | 8.20% | 0.426 | -19.27% | 0.56% | Very low activity; sample-dependence risk |
| Best TLT row | TLT | 36 | 40 | 50% | 7.97% | 0.355 | -22.45% | 10.98% | Worth OOS checking |
| Reasonable QQQ row | QQQ | 36 | 40 | 25% | 8.66% | 0.337 | -25.71% | 7.09% | More aggressive portfolio profile |

Gold is the cleanest candidate. SPY can improve return but may add equity crash
exposure. QQQ is tempting but materially weakens drawdown quality. GSG default
is harmful despite helping in parts of the 2022 inflation/rate period.

Key caveat: the 6,912-row grid is in-sample research, not a production
recommendation. The next gate is to rerun overlay threshold/leverage selection
under the same 2018, 2020, and 2022 stress-window discipline used for RP, and to
add walk-forward or train/test validation before treating any threshold as
learned.

OOS validation runner added in `research.validate_leverage_oos`. It exports
`results/leverage_oos_validation/<timestamp>_6asset_tip_gsg_rpavg/`, selects
rules using IS-only data for the 2018/2020/2022 starts, and evaluates OOS base
vs overlay performance with pass/fail gates. GLD receives the extended 15%-100%
leverage sweep; all other ETFs use 15%-50%.

---

## Phase 16 — Repo cleanup + central data plumbing (2026-05-30)

Preparation for tax-aware backtest research (Section D) and for the June 1 live
launch on the client's Alpaca account.

### Repo cleanup (Phase 1A)
- Deleted `archive/` (graveyard of superseded scripts; nothing imported anywhere).
- Folded `ALPACA_SETUP_MAC.md` into `docs/BROKER_SETUP.md` (already pointed there).
- Moved `ToDo.md`, `research_log.md`, `linkedin_post.md` → `docs/internal/`.
- Moved `scripts/install_launchd.sh` → `live/scheduler/install_launchd.sh`.
- Renamed `live/alpaca_rebalance.py` → `live/_legacy/alpaca_rebalance.py`. Updated test imports, Makefile, `docs/BROKER_SETUP.md`, `README.md`, `learning/CURRICULUM.md`, docstring refs in `live/rebalance.py` and `live/brokers/alpaca.py`. Fixed `_PROJECT_ROOT` resolution in the moved module (one extra `dirname` because it now sits two levels deep).
- Refreshed `CLAUDE.md` to match the real tree (11 engine files, 3 notebooks, all `live/` modules including `_legacy/`, `env.py`, `healthcheck.py`).
- Deleted `portfolio_holdings.json` (ad-hoc state, already gitignored).
- All 211 tests still pass.

### Live rebalancer preview enrichment
- Cadence gate is now computed unconditionally (was only on `--execute`). Enforcement still fires only on real execute without `--force-cadence` / `--force`.
- Preview / dry-execute output now prints inline:
  - Cadence status (`ELIGIBLE` / `BLOCKED` + days-since / days-remaining)
  - Budget snapshot (cap, positions value, reserved cash, managed capital)
  - Lot-ledger summary (blocked symbols with days remaining, eligible symbols with days held)
- One-page human-review report before pressing execute on the client's account.

### Central data manager — dividend support
The shared `QuantFinance/data_manager.py` (one directory above this project) now
also fetches per-symbol dividend history from FMP and writes it to a `dividends`
table inside each per-symbol SQLite cache.

- New CLI: `--with-dividends`, `--dividends-only`.
- New schema: `dividends(ex_date PK, amount, adj_amount, record_date, payment_date, declaration_date, label)`.
- Idempotent upsert keyed on `ex_date`. Reruns are safe.
- JEPQ added to `DEFAULT_SYMBOLS`. Price history backfilled to inception 2022-05-03.
- Live-tested coverage:

| Ticker | Dividend rows | First         | Last        |
|--------|--------------:|---------------|-------------|
| SPY    | 134           | 1993-03-19    | 2026-03-20  |
| QQQ    | 87            | 2003-12-24    | 2026-03-23  |
| TLT    | 285           | 2002-09-03    | 2026-05-01  |
| TIP    | 198           | 2003-12-31    | 2026-05-01  |
| JEPQ   | 48            | 2022-06-01    | 2026-05-01  |
| GLD    | 0             | (no distributions — bullion ETF)        |
| GLDM   | 0             | (no distributions — bullion ETF)        |
| GSG    | 0             | (partnership ETF, K-1 instead of cash)  |

### engine/data.py reader
- Added `fetch_dividends(tickers, start, end, data_dir=None)` reading the new `dividends` table from the central per-symbol DB. Long-form output: `Ticker, ExDate, Amount, AdjAmount, RecordDate, PaymentDate, DeclarationDate`.
- The existing `fetch_prices_from_fmp_db` already resolved `data_dir` to `QuantFinance/data/` via `_repo_data_dir()`, so no duplicate FMP fetch logic in this project. The plumbing was already correct — the dividend reader just rides the same path.

### Alpaca lot-level capability check
- Verified `alpaca-py 0.43.2` `MarketOrderRequest` has no `lot_id` / `tax_lot_method` / `cost_basis_method` field. Alpaca uses Compressed FIFO for end-of-day positions. Feature request open since 2020.
- Implication: in the upcoming tax model the **tax-optimal lot selector is a research counterfactual on Alpaca**; FIFO is the broker-reality default. Documented in `docs/research/alpaca_lot_selection.md`.

### Documentation
- New: `docs/data/central_data_manager.md` (contract between projects + the shared fetcher).
- New: `docs/research/alpaca_lot_selection.md`.

### What's next — Section D
Tax model + drift trigger inside `engine/`, gated by a golden regression test on
the current production weights. **Active plan in `docs/internal/session_handoff.md`**
(`ToDo.md` is retired as the active worklist). **Decision gate:** the closed
`failed_strategies/weekly_rebalance/` verdict was made without taxes; under
realistic US tax + tax-optimal lot selection the question re-opens, with kill
criterion ≥5% Calmar improvement on ≥2 of 3 OOS windows required to flip the
production policy.

---

## Open questions
- Brand name, FCA compliance, GBP/EUR adjustment
- Live vs backtest ETF performance divergence over time
- Optimal rebalancing trigger (threshold-based vs calendar-based) — **active under D**
- OOS validation for RSI ETF leverage overlays, especially GLD higher leverage
