# Changelog

## [Unreleased]

## [0.2.0] — 2026-04-02

### Added
- Live refreshing dashboard in `run_paper.py` — clears and redraws every scan with z-score bars, open positions, and running P&L
- `analyse.py` — offline log analyser with per-pair breakdown, profit factor, Sharpe, max drawdown, exit reason stats, trade log, and ASCII equity curve
- `_last_states` tracking in `LivePairsTrader` for dashboard z-score display

### Changed
- `entry_z_score` lowered from 2.0 → 1.6 (more signals, confirmed better OOS performance)
- `zscore_window` raised from 60 → 120 candles (better long-run mean estimate)
- `exit_z_score` tightened from 0.5 → 0.3 (let winners run slightly longer)
- `stop_loss_z_score` tightened from 3.5 → 3.0 (cut losers sooner)
- `coint_pvalue_threshold` raised from 0.05 → 0.10 (catches near-cointegrated pairs)
- `lookback_candles` raised from 720 → 2000 (~83 days, more statistical power)
- Logging in `run_paper.py` moved to file-only — dashboard owns stdout cleanly

### Pairs screened and dropped
Tested 24+ pairs across AI, RWA, meme, payment, DeFi, L2, gaming, storage, and privacy sectors.

Confirmed keepers:
- `RENDERUSDT/FETUSDT` — AI compute tokens, p=0.024, +11.62% OOS, Sharpe=3.36
- `ONDOUSDT/POLYXUSDT` — RWA tokenisation, p=0.035, +14.04% OOS, Sharpe=4.49

Dropped (failed cointegration or negative OOS P&L):
- ETH/BTC, LTC/BTC, BCH/BTC, LINK/ETH — cross-sector majors
- SOL/AVAX, ADA/DOT, MATIC/SOL — competing L1s
- DOGE/SHIB, XRP/XLM — cointegrated but -6% OOS (regime drift)
- AAVE/COMP — cointegrated but -8% OOS
- UNI/SUSHI, ARB/OP, APT/SUI — failed cointegration
- TAO/RENDER, RENDER/IOTA — cointegrated but large OOS losses
- STRK/ZK — very strong cointegration (p=0.0002) but -3.8% OOS

## [0.1.0] — 2026-04-01

### Added
- Initial project structure
- `data_fetcher.py` — Binance public OHLCV API with disk caching
- `cointegration.py` — Engle-Granger test, OLS hedge ratio, half-life estimation
- `spread_tracker.py` — rolling z-score and signal classification
- `backtester.py` — walk-forward in-sample/out-of-sample simulation
- `paper_trader.py` — live paper trading engine with JSONL trade logging
- `run_backtest.py` — pair screener with KEEP/WATCH/DROP verdicts
- `run_paper.py` — live paper trading loop
- `models.py` — data classes for candles, signals, positions, results
- `config.py` — centralised strategy parameters
