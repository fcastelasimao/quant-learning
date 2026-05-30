# CLAUDE.md — Development Context

> **NEW SESSION? READ `docs/internal/session_handoff.md` FIRST.**
> It contains the live work state, the active plan (in order, with kill criteria),
> decisions taken (do not re-litigate), and operating principles. This file
> (CLAUDE.md) is for static repo layout + invariants; the handoff is for what
> we are actively doing.

This file provides context for AI-assisted development (Claude Code, Copilot, etc.).

## Environment

```bash
# Run from projects/all-weather/. The conda env is named `allweather`.
conda run -n allweather python <script>
conda run -n allweather python -m pytest tests/ -v
```

Direct interpreter (if conda shell init isn't available):
`/Users/franciscosimao/opt/anaconda3/envs/allweather/bin/python`.

## Data

Price + dividend history is fetched and cached by the **shared**
`QuantFinance/data_manager.py` (one directory above this project). Per-ticker
SQLite caches live in `QuantFinance/data/DB_<TICKER>_historical_data.db`.
`engine/data.py` is a thin wrapper around it. See `docs/data/central_data_manager.md`.

## Repository layout

```
projects/all-weather/
├── main.py                   entry point — imports from engine/ and live/
├── strategies.json           production strategy registry (gitignored)
├── Makefile                  make test / backtest / compare-allw / rebalance-*
│
├── engine/                   pure backtest math — no live IO
│   ├── analytics.py          monthly rebal series, drawdowns, turnover helpers
│   ├── backtest.py           run_backtest (monthly rebalance simulation)
│   ├── calendar.py           date/frequency helpers (MONTH_END alias)
│   ├── config.py             parameters; loads allocation from strategies.json
│   ├── data.py               yfinance/FMP fetch wrapper, quality checks
│   ├── explorers.py          universe-exploration helpers
│   ├── leverage.py           RSI ETF overlay engine; research-only
│   ├── optimiser.py          compute_risk_parity_weights (SLSQP) + random search
│   ├── plotting.py           dark-theme matplotlib chart helpers
│   └── stats.py              CAGR, MDD, Sharpe, Sortino, Calmar, Ulcer
│
├── live/                     broker-agnostic live execution
│   ├── _legacy/              preserved for back-compat
│   │   └── alpaca_rebalance.py   pre-broker-agnostic Alpaca-only rebalancer
│   ├── brokers/              Broker Protocol + concrete implementations
│   │   ├── base.py           Broker Protocol, PositionSnapshot, OrderResult
│   │   ├── factory.py        make_broker(broker_name, trading_mode, account_label)
│   │   ├── alpaca.py         AlpacaBroker (alpaca-py)
│   │   └── tastytrade.py     TastytradeBroker (community SDK, guarded import)
│   ├── scheduler/            launchd plist template + install_launchd.sh
│   ├── budget.py             per-account virtual sub-portfolio cap
│   ├── env.py                api_keys.env loader
│   ├── healthcheck.py        pre-flight: env, creds, strategies, cadence
│   ├── lots.py               FIFO lot ledger + 31-day hold enforcement
│   ├── notify.py             Slack webhook + SMTP email (never raises)
│   ├── portfolio.py          load/save holdings + rebalancing instructions
│   ├── rebalance.py          broker-agnostic rebalancer
│   └── runlog.py             RunSummary → JSONL + monthly_runs.csv + per-run JSON
│
├── research/                 analyses + report builders (slated for Phase 1B subpackage reorg)
│   ├── compare_allw.py       ALLW ETF head-to-head (includes JEPQ benchmark)
│   ├── compare_jepq.py       JEPQ vs AW since 2022-05-03 inception
│   ├── compare_fmp_rp.py     local FMP SQLite RP-weight comparison
│   ├── compare_live_etfs.py  backtest vs live ETF performance pairwise
│   ├── backtest_shadow.py    actual live vs simulated returns reconciliation
│   ├── build_strategy_comparison_report.py  production_validation bundle builder
│   ├── build_leverage_comparison_report.py  RSI leverage overlay bundle
│   ├── build_mixed_leverage_report.py       SPY+GLD overlay bundle
│   ├── production_validation.py             top-level production bundle runner
│   ├── validate_leverage_oos.py             OOS validation for RSI overlays
│   ├── validate_mixed_leverage_oos.py       OOS for mixed-pair overlays
│   ├── validation.py         walk-forward + Pareto frontier helpers
│   ├── rebalance_thresholds.py  drift-threshold policy sweeps (research-only)
│   ├── rerun_rp_validation.py   yfinance/FMP RP rerun matrix
│   ├── sensitivity.py        sensitivity helpers
│   ├── strategy_plotting.py  matplotlib figure builders for strategy marimo
│   ├── leverage_plotting.py  matplotlib figure builders for leverage marimo
│   ├── leverage_analysis.py  analysis helpers used by leverage plots
│   ├── plot_linkedin.py      two-panel LinkedIn figure
│   └── export.py             Excel master log + results formatting
│
├── failed_strategies/        closed investigations — reproducible but off-prod
│   ├── README.md             summary table
│   ├── strategies_archive.json  demoted entries removed from strategies.json
│   ├── bond_leverage/
│   ├── differential_evolution/
│   ├── eight_asset_universe/
│   ├── momentum_overlay/
│   ├── rolling_rp/
│   ├── universe_scan/
│   └── weekly_rebalance/     (verdict may be reopened under tax-aware modelling)
│
├── notebooks/                marimo notebooks (read CSV artifacts only)
│   ├── data_explorer.py
│   ├── leverage_comparison.py
│   └── strategy_comparison.py
│
├── tests/                    pytest suite (15 test files)
│   ├── conftest.py
│   ├── test_analytics.py        engine/analytics helpers
│   ├── test_broker_protocol.py  Broker Protocol contract
│   ├── test_data.py             engine/data
│   ├── test_env.py              api_keys.env loader
│   ├── test_explorers.py
│   ├── test_leverage_analysis.py
│   ├── test_leverage_oos_validation.py
│   ├── test_leverage_overlay.py
│   ├── test_live_rebalance.py   legacy alpaca_rebalance (imports live._legacy)
│   ├── test_mixed_leverage_oos_validation.py
│   ├── test_rebalance_thresholds.py
│   ├── test_rolling_rp.py       imports from failed_strategies/
│   ├── test_stats.py
│   └── test_strategy_comparison_report.py
│
├── docs/                     public-facing documentation
│   ├── BROKER_SETUP.md       broker credential + scheduling guide
│   ├── IMPLEMENTATION_PLAN.md  historical migration plan (broker-agnostic)
│   ├── bank_pack.md
│   ├── claim_register.md
│   ├── customer_pack.md
│   ├── data/                 data plumbing contracts
│   │   └── central_data_manager.md   contract between projects + shared SQLite store
│   ├── research/             per-investigation decision docs (TQQQ/SQQQ style)
│   │   └── alpaca_lot_selection.md   why backtest tax_optimal is research-only on Alpaca
│   └── internal/             cross-session memory + historical notes
│       ├── session_handoff.md  ← LIVE work state, active plan, decisions (READ FIRST)
│       ├── ToDo.md             historical changelog (retired as active worklist)
│       ├── research_log.md     historical research narrative by phase
│       └── linkedin_post.md    (gitignored)
│
├── learning/                 6-session engine rewrite curriculum
└── logs/                     untracked private paper/live audit logs (gitignored)
```

## Key Constraints

- **IS/OOS discipline:** Never optimise on data after OOS_START. RP covariance uses `end_date` parameter.
- **Calmar ratio** is the primary evaluation metric (CAGR / |max drawdown|).
- **Risk parity** equalises risk contributions via covariance matrix only — does not optimise for returns.
- Production weights (`6asset_tip_gsg_rpavg`): SPY 13.4%, QQQ 10.3%, TLT 17.5%, TIP 34.8%, GLD 14.2%, GSG 9.8%.
- Live ticker mapping only substitutes `GLD → GLDM`. Everything else trades the backtest ticker.
- Data sources are configurable in `engine/config.py`: `DATA_SOURCE="yfinance"` or `"fmp"`; for FMP use `FMP_PRICE_COLUMN="adj_close"` when matching total-return methodology.
- 2026-05-09 rerun: yfinance total-return and FMP `adj_close` are effectively identical (Calmar 2018/2020/2022: 0.487/0.503/0.452 vs 0.488/0.504/0.453). Price-return/close materially understates performance.
- RSI ETF leverage overlay is research-only. Latest reviewed bundles: single-ETF `results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg`; mixed SPY+GLD `results/mixed_leverage/2026-05-15_16-27-59_6asset_tip_gsg_rpavg`; mixed-pair OOS validation `results/mixed_leverage_oos_validation/2026-05-15_17-54-04_6asset_tip_gsg_rpavg`.
- Default RSI overlay rule: ETF's own RSI-14, entry <30, exit >50, +20%, one-day execution lag, one ETF at a time. Default GLD and SPY are strongest; GSG default is rejected.
- Next RSI overlay gate: walk-forward / train-test validation on top of the mixed-pair OOS run before promoting any threshold to production. (Open research direction — leverage track.)
- `results/` and `research/results/` are `.gitignore`'d — all output is regenerated by running scripts.
- `logs/performance_tracking_<broker>_<mode>_<account>.csv` is `.gitignore`'d and kept locally only. JEPQ added as a 4th benchmark column (alongside SPY, ALLW, 60/40).
- **Live execution** uses `live/rebalance.py` (broker-agnostic). The legacy `live/_legacy/alpaca_rebalance.py` is preserved unchanged for backward compatibility with `tests/test_live_rebalance.py`. See `docs/BROKER_SETUP.md`.
- Cadence gate: `--min-rebalance-interval-days 31` replaces the old month-end calendar check. State at `logs/cadence_<broker>_<mode>_<account>_<strategy>.json`.
- Lot ledger: `logs/lots_<broker>_<mode>_<account>_<strategy>.json`. Tracks acquisition dates because Tastytrade doesn't expose them via API.
- Budget cap: `--budget AMOUNT` + `--initialize-budget` for virtual sub-portfolio sizing. State at `logs/budget_*.json`.
- Structured logs: `logs/run_summary.jsonl` (append-only), `logs/monthly_runs.csv`, `logs/runs/<timestamp>_*.json` (auto-pruned at 200 files).

## Active research thread

Tax-aware backtesting and drift-trigger rebalancing.
**Live plan and decisions:** `docs/internal/session_handoff.md`.
The closed `failed_strategies/weekly_rebalance/` verdict was made under
transaction-cost-only modelling; under realistic US tax + tax-optimal lot
selection, drift triggers may reopen the question. New investigations should
land documentation in `docs/research/<topic>.md` (TQQQ/SQQQ project style),
and produce CSV/JSON artifacts the marimos read without recomputation.

## Closed Investigations

All closed investigations live in `failed_strategies/` with a `README.md` explaining what was tested and why it failed. Each script still runs end-to-end with `from engine import ...` imports.

| Investigation | Result |
|---|---|
| Differential Evolution | All 26 experiments fail OOS — structural regime mismatch |
| SPY momentum overlay | Re-entry timing not learnable; no OOS Calmar improvement |
| Rolling RP | Converges to same weights as static RP across all splits |
| Weekly/threshold rebalancing | No improvement after transaction costs (pre-tax — may be reopened) |
| 100-ETF universe scan | 6-asset universe confirmed optimal |
| 8-asset universe | 6-asset wins on all Calmar windows |
| Bond leverage (1.0x–2.5x) | Destroys Calmar in rising-rate regime |
