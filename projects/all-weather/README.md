# All Weather Portfolio Engine

A Python backtesting and validation engine for risk-balanced portfolio strategies, inspired by Ray Dalio's All Weather approach.

Built for investors who prioritise capital preservation over return maximisation.

> **Disclaimer:** This is an educational and research tool, not financial advice.
> Past performance does not guarantee future results.

---

## Head-to-Head: DIY vs Bridgewater's ALLW ETF

Bridgewater launched the ALLW ETF in March 2025 (~$1B+ AUM, 0.85% expense ratio, ~2x leverage on bonds). Here's how our risk parity strategy compares over the same period (March 2025 – March 2026, monthly rebalanced, fee-adjusted):

| Metric | DIY Risk Parity | ALLW (Bridgewater) | Advantage |
|--------|----------------|-------------------|-----------|
| CAGR | 17.4% | 19.1% | ALLW +1.7% |
| Max Drawdown | **-5.7%** | -8.8% | **35% shallower** |
| Calmar Ratio | **3.03** | 2.18 | **39% better** |
| Annual Cost (on $100k) | ~$120 | ~$850 | **85% cheaper** |

ALLW earns ~1.7% more in raw return because of its ~2x bond leverage. But it pays for that leverage with 35% deeper drawdowns and worse risk-adjusted metrics. Different product, different investor.

---

## How It Works

### Risk Parity

Instead of optimising for returns (which overfits to historical regimes), we optimise for **equal risk contribution**: every asset contributes the same amount of portfolio variance.

The objective function:

```
Minimise: Var(RC)   where RC_i = w_i × (Σw)_i / (wᵀΣw)
Subject to: Σ_i w_i = 1, w_i ≥ 0.02
```

Solved via scipy's SLSQP (Sequential Least Squares Programming). The covariance matrix Σ is estimated from 5 years of daily log returns. Production weights are averaged across the 2018, 2020, and 2022 stress-window derivations. These windows overlap, so they are useful robustness checks, not fully independent samples.

### Production Strategy

Machine alias: `6_asset_rp_baseline`
Display name: 6 Asset RP Baseline

The legacy registry key `6asset_tip_gsg_rpavg` is kept for reproducibility
because historical results, reports, and logs reference it.

### Production Allocation

| Asset | ETF | Weight | Role |
|-------|-----|--------|------|
| US Broad Equity | SPY | 13.4% | Core equity exposure |
| US Tech/Growth | QQQ | 10.3% | Growth engine |
| Long-Term Bonds | TLT | 17.5% | Deflation hedge |
| Inflation Bonds | TIP | 34.8% | Rate-shock buffer |
| Gold | GLD | 14.2% | Crisis hedge |
| Commodities | GSG | 9.8% | Stagflation hedge |

### IS/OOS Validation

All optimisation uses data before the applicable stress-window boundary. Results are validated on held-out data the model did not see during that derivation:

| OOS Window | Manual Calmar | RP Calmar | Improvement |
|-----------|--------------|----------|-------------|
| 2020–2026 | 0.406 | **0.480** | +18% |
| 2018–2026 | 0.417 | **0.462** | +11% |
| 2022–2026 | 0.345 | **0.385** | +12% |

RP beats manual allocation on all three stress windows.

### Data-source rerun (2026-05-10)

After fixing the rebalancing code and adding local FMP `adj_close` support, the RP averaging workflow was rerun across the same 2018, 2020, and 2022 OOS windows. The important comparison is `yfinance_total_return` vs `fmp_adj_close`; they are effectively identical, confirming the production conclusion is not a yfinance artifact.

| Data Basis | 2018 OOS Calmar | 2020 OOS Calmar | 2022 OOS Calmar | Interpretation |
|-----------|----------------:|----------------:|----------------:|----------------|
| yfinance total return | 0.487 | 0.503 | 0.452 | Canonical production basis |
| FMP adjusted close | 0.488 | 0.504 | 0.453 | Confirms yfinance total-return result |
| yfinance price return | 0.334 | 0.343 | 0.284 | Missing distributions materially understates results |
| FMP close | 0.334 | 0.343 | 0.284 | Matches yfinance price-return diagnostic |

The corrected FMP adjusted-close averaged RP weights are close to production: SPY 13.34%, QQQ 10.76%, TLT 18.65%, TIP 33.48%, GLD 13.59%, GSG 10.19%. The drift is modest, so production weights remain unchanged for now.

---

## RSI Leverage Overlay Research

This is research-only and is not part of the production allocation. The base
portfolio remains unchanged; each overlay test adds one tactical ETF exposure on
top of the base portfolio, using that ETF's own RSI-14 signal.

Latest bundle reviewed:
`results/leverage_comparison/2026-05-11_12-15-40_6asset_tip_gsg_rpavg`

Default rule: RSI-14 entry below 30, exit above 50, +20% overlay, one-day
execution lag, one ETF overlay at a time.

| Strategy | CAGR | Sharpe | Calmar | Max DD | Active Days | Research read |
|---|---:|---:|---:|---:|---:|---|
| Base | 6.96% | 0.471 | 0.306 | -22.74% | 0.00% | Production reference |
| GLD RSI overlay | 7.27% | 0.489 | 0.324 | -22.45% | 10.92% | Best default risk-adjusted result |
| SPY RSI overlay | 7.64% | 0.503 | 0.323 | -23.62% | 6.46% | Best default return boost, more equity crash risk |
| TLT RSI overlay | 7.02% | 0.463 | 0.309 | -22.74% | 10.38% | Small benefit |
| TIP RSI overlay | 6.94% | 0.459 | 0.291 | -23.86% | 7.97% | Worse than base |
| QQQ RSI overlay | 7.34% | 0.461 | 0.245 | -29.93% | 7.35% | Return up, drawdown too high |
| GSG RSI overlay | 6.06% | 0.323 | 0.225 | -26.93% | 16.68% | Reject default rule |

The expanded grid tests entry RSI 20-36 in steps of 2, exit RSI 40-70 in steps
of 2, and overlay weights from 15% to 50% in 5% steps. Best in-sample candidates:

| Purpose | ETF | Entry | Exit | Overlay | CAGR | Calmar | Max DD | Caveat |
|---|---|---:|---:|---:|---:|---:|---:|---|
| Best risk-adjusted | GLD | 22 | 46 | 50% | 8.07% | 0.429 | -18.83% | Needs OOS validation |
| Best drawdown preservation | GLD | 22 | 64 | 40% | 7.73% | 0.419 | -18.46% | Supports further gold research |
| Best selective equity overlay | SPY | 22 | 42 | 50% | 8.20% | 0.426 | -19.27% | Active only 0.56% of days |
| Reasonable bond candidate | TLT | 36 | 40 | 50% | 7.97% | 0.355 | -22.45% | Rate-regime sensitive |
| Aggressive return candidate | QQQ | 36 | 40 | 25% | 8.66% | 0.337 | -25.71% | Changes portfolio risk profile |

Do not treat these grid winners as production conclusions. The grid contains
6,912 in-sample tests, so threshold and leverage overfitting is the central
risk. Next validation should use the same 2018, 2020, and 2022 stress-window
discipline used for RP, plus walk-forward or train/test splits.

OOS validation is implemented as a separate research runner:

```bash
make leverage-oos-validation
```

It selects RSI overlay rules only on pre-split data, then evaluates them on the
2018, 2020, and 2022 OOS windows. GLD receives an extended leverage sweep up to
100%; the other ETFs retain the 15%-50% research grid.

---

## Installation

```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/projects/all-weather

conda create -n allweather python=3.11
conda activate allweather
pip install -r requirements.txt
```

For reproducible review, prefer the pinned environment:

```bash
conda env create -f environment.yml
conda run -n allweather python -m pytest
```

---

## Quick Start

All commands must be run from `projects/all-weather/`. Use `conda run -n allweather` to stay inside the environment without activating it interactively.

### Run a backtest

```bash
conda run -n allweather python main.py
```

Runs the full backtest with the production `6_asset_rp_baseline` RP weights. Output goes to `results/<timestamp>/`, including `annual_metrics.csv` for year-by-year PnL, returns, and drawdowns.

### Run tests

```bash
conda run -n allweather python -m pytest tests/ -v
# or via Make:
make test
```

The default pytest configuration excludes network-dependent integration tests.
Run vendor/data tests explicitly when online:

```bash
conda run -n allweather python -m pytest -m integration
```

### Compare against Bridgewater's ALLW ETF

```bash
conda run -n allweather python -m research.compare_allw
# or:
make compare-allw
```

### Recompute RP weights from local FMP SQLite data

```bash
conda run -n allweather python -m research.compare_fmp_rp
# or:
make compare-fmp-rp
```

Writes `results/fmp_rp_boundary_weights.csv` and `results/fmp_rp_weight_comparison.csv`.

### Rerun RP validation across yfinance and FMP

```bash
conda run -n allweather python -m research.rerun_rp_validation
```

Runs the full 4-scenario matrix (`yfinance_total_return`, `yfinance_price_return`, `fmp_close`, `fmp_adj_close`) across the 2018/2020/2022 OOS windows. Each split writes the standard results folder and appends to `results/master_log.xlsx`; batch summaries are saved under `results/rp_rerun_<timestamp>/`.

### Build the bank/customer validation bundle

```bash
conda run -n allweather python -m research.production_validation
conda run -n allweather python -m research.sensitivity
```

The customer and bank-facing docs live in `docs/customer_pack.md`,
`docs/bank_pack.md`, and `docs/claim_register.md`.

### Generate LinkedIn comparison plot

```bash
conda run -n allweather python -m research.plot_linkedin
```

### Review validation and data quality in marimo

```bash
make notebook-comparison
make notebook-data
```

The active notebooks live in `notebooks/strategy_comparison.py` and
`notebooks/data_explorer.py`. The legacy annual metrics browser is archived
under `archive/notebooks/` because the production validation bundle now covers
that review surface.

### Live trading — broker-agnostic pipeline

The production rebalancer is `live/rebalance.py`. It talks to a `Broker`
protocol so the same code runs against Alpaca or Tastytrade. See
[`docs/BROKER_SETUP.md`](docs/BROKER_SETUP.md) for credentials, budget cap,
31-day hold gate, notifications, and macOS launchd scheduling.

API keys are loaded automatically from
`/Users/franciscosimao/Documents/QuantFinance/api_keys.env`. Keep broker,
FMP, notification, and optional data-provider secrets there rather than in
shell profiles or launchd plists.

```bash
# 1. Pre-flight health check (no network calls against the broker)
conda run -n allweather python -m live.healthcheck --broker alpaca

# 2. Dry-execute — simulates fills from current price, writes a full RunSummary
#    to logs/runs/, logs/run_summary.jsonl, and logs/monthly_runs.csv.
#    No real orders are placed and cadence/lots/budget state is NOT advanced.
make rebalance-dry-execute BROKER=alpaca ACCOUNT=default MODE=--paper

# 3. Preview (real account data, no orders)
make rebalance-new-preview BROKER=alpaca ACCOUNT=default MODE=--paper

# 4. Execute — places real orders.  Enforces:
#    • ≥31-day minimum interval since last execute (`--min-rebalance-interval-days`)
#    • 31-day per-lot holding period (FIFO ledger in logs/lots_*.json)
#    • Optional budget cap (`--budget AMOUNT` + `--initialize-budget`)
make rebalance-new-execute BROKER=alpaca ACCOUNT=default MODE=--paper
```

Tastytrade is supported via the pinned community SDK (`tastytrade==12.4.1`).
Set `TASTYTRADE_PROVIDER_SECRET` / `TASTYTRADE_REFRESH_TOKEN` for the default
account, or `BROKER_TASTYTRADE_<LABEL>_PROVIDER_SECRET` /
`BROKER_TASTYTRADE_<LABEL>_REFRESH_TOKEN` for named accounts, then validate
OAuth credentials with
`python -m live.brokers.tastytrade login --account default`.

Every run — preview, dry-execute, or execute — writes a structured
`RunSummary` to `logs/run_summary.jsonl`, a per-run JSON archive to
`logs/runs/`, and an aggregate row to `logs/monthly_runs.csv`.  Slack +
SMTP notifications are sent automatically when `ALLW_SLACK_WEBHOOK_URL`
or `ALLW_NOTIFY_EMAIL` are set.

#### Legacy Alpaca rebalancer (kept for backward compatibility)

```bash
# Old Alpaca-only entry point (unchanged, all tests still pass)
make rebalance-preview ACCOUNT=PAPER
make rebalance-live-preview ACCOUNT=LIVE
make rebalance-execute ACCOUNT=PAPER
```

The live rebalancer defaults to strategy live tickers (`GLD -> GLDM`) and
refuses non-production strategies unless explicitly overridden. Live execution
requires explicit `--live` and `--execute`. Rejected, canceled, expired, or
timed-out orders fail the run, and final positions are checked against target
weights before the run is marked successful.

### JEPQ comparison & live-vs-backtest reconciliation

```bash
make compare-jepq        # AW 6-asset RP vs JEPQ since 2022-05-03 inception
make backtest-shadow     # Live performance CSV vs engine-simulated returns
```

---

## Project Structure

```
projects/all-weather/
├── main.py                   entry point — orchestrates a single backtest run
├── strategies.json           production strategy registry (allocations + OOS results)
├── Makefile                  shortcuts: make test / backtest / compare-allw / rebalance-*
│
├── engine/                   pure backtest math — no live IO
│   ├── config.py             all parameters; loads allocation from strategies.json
│   ├── data.py               yfinance/FMP fetch + data quality checks
│   ├── stats.py              CAGR, MDD, Sharpe, Sortino, Calmar, Ulcer, compute_stats
│   ├── backtest.py           run_backtest (monthly rebalance simulation)
│   ├── optimiser.py          compute_risk_parity_weights (SLSQP) + random search
│   └── plotting.py           dark-theme matplotlib charts
│
├── live/                     broker-agnostic live execution
│   ├── portfolio.py          load/save holdings + rebalancing instructions
│   ├── _legacy/              legacy modules preserved for back-compat
│   │   └── alpaca_rebalance.py   pre-broker-agnostic Alpaca-only rebalancer
│   ├── rebalance.py          broker-agnostic rebalancer — see docs/BROKER_SETUP.md
│   ├── budget.py             virtual sub-portfolio cap (--budget AMOUNT)
│   ├── lots.py               FIFO lot ledger + 31-day hold enforcement
│   ├── runlog.py             RunSummary → JSONL + monthly_runs.csv + per-run JSON
│   ├── notify.py             Slack webhook + SMTP email (never raises)
│   ├── healthcheck.py        pre-flight: env, creds, strategies.json, cadence
│   ├── brokers/              Broker Protocol + concrete implementations
│   │   ├── base.py           Broker, PositionSnapshot, OrderResult, ActivityEvent
│   │   ├── factory.py        make_broker(broker_name, trading_mode, account_label)
│   │   ├── alpaca.py         AlpacaBroker (alpaca-py)
│   │   └── tastytrade.py     TastytradeBroker (community SDK, qty-only)
│   └── scheduler/            launchd plist templates (macOS)
│
├── research/                 analyses that run periodically but are not production
│   ├── compare_allw.py       ALLW ETF head-to-head — JEPQ added as 4th benchmark
│   ├── compare_jepq.py       JEPQ vs AW since 2022-05-03 inception
│   ├── backtest_shadow.py    live performance CSV vs engine simulation
│   ├── compare_fmp_rp.py     local FMP SQLite RP-weight comparison
│   ├── rerun_rp_validation.py  4-scenario yfinance/FMP RP rerun matrix
│   ├── plot_linkedin.py      two-panel LinkedIn figure
│   ├── validation.py         walk-forward + Pareto frontier analysis
│   └── export.py             Excel master log + results formatting
│
├── notebooks/                marimo notebooks
│   ├── strategy_comparison.py review production validation artifacts
│   └── data_explorer.py      inspect FMP ETF data and return distributions
│
├── failed_strategies/        closed investigations — reproducible but off-prod
│   ├── README.md             one-line summary + conclusion for every closed experiment
│   ├── strategies_archive.json  demoted strategies removed from strategies.json
│   ├── rolling_rp/           quarterly RP recompute (converges to same weights)
│   ├── momentum_overlay/     SPY exit/re-entry overlay (re-entry timing not learnable)
│   ├── weekly_rebalance/     threshold rebalancing (no improvement after costs)
│   ├── differential_evolution/  return-based global optimiser (regime mismatch)
│   ├── eight_asset_universe/ 8-asset validation (6-asset wins)
│   ├── bond_leverage/        TLT/TIP leverage sweep (destroys Calmar post-2022)
│   └── universe_scan/        100-ETF random scan (6-asset confirmed optimal)
│
├── tests/
│   ├── test_stats.py         unit tests for all performance metrics
│   ├── test_data.py          data fetch + quality check tests
│   └── test_rolling_rp.py    rolling RP backtest + weight-history tests
│
├── archive/                  historical scripts — for reference only
├── docs/                     customer pack, bank pack, claim register
├── learning/                 guided engine rewrite (6-session curriculum)
└── logs/                     untracked — private paper/live audit logs
```

---

## Configuration

All parameters live in `engine/config.py`. The default strategy is loaded from `strategies.json`:

```python
DEFAULT_STRATEGY = "6_asset_rp_baseline"  # alias for production RP-averaged weights
```

Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BACKTEST_START` | 2006-01-01 | Limited by GSG inception |
| `OOS_START` | 2022-01-01 | IS/OOS boundary |
| `DATA_SOURCE` | "yfinance" | `"yfinance"` or `"fmp"` |
| `FMP_PRICE_COLUMN` | "close" | Use `"adj_close"` for FMP total-return-equivalent reruns |
| `DATA_FREQUENCY` | "ME" | Monthly rebalancing |
| `TRANSACTION_COST_PCT` | 0.001 | 0.1% per trade |
| `RISK_FREE_RATE` | 0.035 | Fed funds rate as of March 2026 |

`main.py` also accepts runtime overrides such as `--run-mode`,
`--strategy-id`, `--data-source`, `--fmp-price-column`, `--backtest-start`,
`--backtest-end`, `--transaction-cost-pct`, and `--tax-drag-pct`.

---

## Live ETF Mapping

One substitution is clearly worth making; the others are not supported by the backtest:

| Backtest ETF | Live ETF | ER Saving | Notes |
|-------------|---------|-----------|-------|
| GLD → | **GLDM** | **0.30%** | Same LBMA physical gold. 0.998 correlation over 7.8 years. Substitute. |
| SPY | SPY | — | IVV saves 6.45bp at 13.4% weight = <1bp portfolio impact. Not material. |
| QQQ | QQQ | — | QQQM saves 5bp at 10.3% weight = <1bp portfolio impact. Not material. |
| GSG | GSG | — | PDBC tracks a different index (active roll, 35% sector cap vs ~60% energy). Live ETF portfolio backtest (Oct 2020–today) underperformed by 35bp/yr — 5× the total fee saving. Do not substitute without re-running RP optimisation on PDBC's covariance. |

Portfolio-weighted saving from the GLD→GLDM substitution alone: ~4.3 bp/yr (~$43/yr on $100k).

---

## What We Tried and Rejected

Full code and reproduction steps in `failed_strategies/`. Short summary:

| Approach | Experiments | Result |
|----------|------------|--------|
| Differential Evolution | 26 | All failed OOS — IS is a single falling-rate regime |
| SPY momentum overlay | 126 parameter combos | +1.3% on 2/3 splits, -5.3% on the hardest split |
| Rolling RP (quarterly recompute) | 3 OOS splits | Converges to same weights as static RP |
| Weekly/threshold rebalancing | 3 OOS splits with costs | No improvement after transaction costs |
| 100-ETF universe scan | 50k random subsets | 6-asset universe confirmed optimal |
| 8-asset universe (CPER, DBA, IEF, IJR) | 3 OOS splits | 6-asset beats on all Calmar windows |
| Bond leverage (1.0x–2.5x on TLT+TIP) | 7 levels × 3 splits | Every 0.25x adds ~3% drawdown |

---

## Known Limitations

- ALLW comparison covers a short history only (March 2025 launch)
- The 2018/2020/2022 OOS windows overlap; call them stress windows, not independent samples
- No currency adjustment for non-US investors (GBP, EUR)
- Paper trading started April 2026 via Alpaca (two accounts: backtest ETFs and live ETFs)
- Sortino uses downside std, not standard semi-deviation
- Max drawdown on 20-year backtest computed from monthly data (daily MDD available for ALLW comparison period)
