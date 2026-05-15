# CLAUDE.md — Development Context

This file provides context for AI-assisted development (Claude Code, Copilot, etc.).

## Environment

```bash
# Run from projects/all-weather/
conda run -n allweather python <script>
conda run -n allweather python -m pytest tests/ -v
```

## Repository layout

```
projects/all-weather/
├── main.py                   entry point — imports from engine/ and live/
├── strategies.json           production strategy registry
├── Makefile                  make test / backtest / compare-allw / rebalance-*
│
├── engine/                   pure backtest math — no live IO
│   ├── config.py             all parameters; loads allocation from strategies.json
│   ├── data.py               yfinance/FMP fetch + quality checks
│   ├── stats.py              CAGR, MDD, Sharpe, Sortino, Calmar, Ulcer, compute_stats
│   ├── leverage.py           RSI ETF overlay engine; research-only, base portfolio unchanged
│   ├── backtest.py           run_backtest (monthly rebalance simulation)
│   ├── optimiser.py          compute_risk_parity_weights (SLSQP) + random search
│   └── plotting.py           dark-theme matplotlib charts
│
├── live/                     Alpaca + holdings
│   ├── portfolio.py          load/save holdings + rebalancing instructions
│   └── alpaca_rebalance.py   paper/live trading (multi-account, preview + execute)
│
├── research/                 analyses that run periodically but are not production
│   ├── compare_allw.py       ALLW ETF head-to-head (charts + Excel + metrics)
│   ├── compare_fmp_rp.py     local FMP SQLite RP-weight comparison
│   ├── build_leverage_comparison_report.py  RSI leverage overlay artifact builder
│   ├── validate_leverage_oos.py  OOS validation for RSI leverage overlays
│   ├── rerun_rp_validation.py  yfinance/FMP RP rerun matrix
│   ├── plot_linkedin.py      two-panel LinkedIn figure
│   ├── validation.py         walk-forward + Pareto frontier
│   └── export.py             Excel master log + results formatting
│
├── failed_strategies/        closed investigations — reproducible but off-prod
│   ├── README.md             summary table of every closed investigation
│   ├── strategies_archive.json  demoted entries removed from strategies.json
│   ├── rolling_rp/           rolling RP vs static RP (converges to same weights)
│   ├── momentum_overlay/     SPY momentum overlay (re-entry timing unsolvable)
│   ├── weekly_rebalance/     threshold sensitivity (no improvement after costs)
│   ├── differential_evolution/  DE optimiser (regime mismatch, all 26 experiments fail)
│   ├── eight_asset_universe/ 8-asset validation (6-asset wins on all Calmar windows)
│   ├── bond_leverage/        leverage sweep (destroys Calmar in rising-rate regime)
│   └── universe_scan/        100-ETF scanner (6-asset confirmed optimal)
│
├── notebooks/
│   └── leverage_comparison.py  marimo notebook backed by exported CSV artifacts
│
├── tests/
│   ├── conftest.py
│   ├── test_stats.py
│   ├── test_data.py
│   └── test_rolling_rp.py    rolling RP + win-counter (imports from failed_strategies/)
│
├── archive/                  historical scripts — curators only
├── learning/                 guided engine rewrite (6-session curriculum)
└── logs/                     untracked private paper/live audit logs
```

## Key Constraints

- **IS/OOS discipline:** Never optimise on data after OOS_START. RP covariance uses `end_date` parameter.
- **Calmar ratio** is the primary evaluation metric (CAGR / |max drawdown|).
- **Risk parity** equalises risk contributions via covariance matrix only — does not optimise for returns.
- Production weights (`6asset_tip_gsg_rpavg`): SPY 13.4%, QQQ 10.3%, TLT 17.5%, TIP 34.8%, GLD 14.2%, GSG 9.8%.
- Data sources are configurable in `engine/config.py`: `DATA_SOURCE="yfinance"` or `"fmp"`; for FMP use `FMP_PRICE_COLUMN="adj_close"` when matching total-return methodology.
- 2026-05-09 rerun: yfinance total-return and FMP `adj_close` are effectively identical (Calmar 2018/2020/2022: 0.487/0.503/0.452 vs 0.488/0.504/0.453). Price-return/close materially understates performance.
- RSI ETF leverage overlay is research-only. Latest reviewed bundle: `results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg`.
- Default RSI overlay rule: ETF's own RSI-14, entry <30, exit >50, +20%, one-day execution lag, one ETF at a time. Default GLD and SPY are strongest; GSG default is rejected.
- Grid search currently covers entry 20-36 by 2, exit 40-70 by 2, leverage 15%-50% by 5%. Best in-sample row is GLD 22/46/50% with 8.07% CAGR, 0.429 Calmar, -18.83% Max DD. This is a hypothesis, not a production claim.
- OOS validation runner: `python -m research.validate_leverage_oos` or `make leverage-oos-validation`. It selects rules on IS-only data for 2018/2020/2022 starts, evaluates OOS, and extends GLD's OOS research leverage grid to 100%.
- Next RSI overlay gate: run/review the OOS bundle and add walk-forward/train-test validation before accepting thresholds.
- `results/` is `.gitignore`'d — all output is regenerated by running scripts.
- `logs/performance_tracking_<mode>_<account>.csv` is `.gitignore`'d and kept locally only.

## Closed Investigations

All closed investigations live in `failed_strategies/` with a `README.md` explaining what was tested and why it failed. Each script still runs end-to-end with `from engine import ...` imports.

| Investigation | Result |
|---|---|
| Differential Evolution | All 26 experiments fail OOS — structural regime mismatch |
| SPY momentum overlay | Re-entry timing not learnable; no OOS Calmar improvement |
| Rolling RP | Converges to same weights as static RP across all splits |
| Weekly/threshold rebalancing | No improvement after transaction costs |
| 100-ETF universe scan | 6-asset universe confirmed optimal |
| 8-asset universe | 6-asset wins on all Calmar windows |
| Bond leverage (1.0x–2.5x) | Destroys Calmar in rising-rate regime |
