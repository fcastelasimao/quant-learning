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

Solved via scipy's SLSQP (Sequential Least Squares Programming). The covariance matrix Σ is estimated from 5 years of daily log returns. Weights are computed independently for three OOS windows ending at 2018, 2020, 2022, and averaged.

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

All optimisation uses data from 2006–2020 only. Results are validated on held-out data the model never saw during training:

| OOS Window | Manual Calmar | RP Calmar | Improvement |
|-----------|--------------|----------|-------------|
| 2020–2026 | 0.406 | **0.480** | +18% |
| 2018–2026 | 0.417 | **0.462** | +11% |
| 2022–2026 | 0.345 | **0.385** | +12% |

RP beats manual allocation on all three independent windows.

---

## Installation

```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/projects/all-weather

conda create -n allweather python=3.11
conda activate allweather
pip install -r requirements.txt
```

---

## Quick Start

All commands must be run from `projects/all-weather/`. Use `conda run -n allweather` to stay inside the environment without activating it interactively.

### Run a backtest

```bash
conda run -n allweather python3 main.py
```

Runs the full backtest with the production `6asset_tip_gsg_rpavg` RP weights. Output goes to `results/<timestamp>/`.

### Run tests

```bash
conda run -n allweather python3 -m pytest tests/ -v
# or via Make:
make test
```

### Compare against Bridgewater's ALLW ETF

```bash
conda run -n allweather python3 -m research.compare_allw
# or:
make compare-allw
```

### Generate LinkedIn comparison plot

```bash
conda run -n allweather python3 -m research.plot_linkedin
```

### Paper trading via Alpaca

```bash
# Preview what trades would be made (no orders placed)
conda run -n allweather python3 -m live.alpaca_rebalance --account PAPER --preview

# Execute the rebalance
conda run -n allweather python3 -m live.alpaca_rebalance --account PAPER --execute

# or via Make:
make rebalance-preview ACCOUNT=PAPER
make rebalance-execute ACCOUNT=PAPER
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
│   ├── data.py               yfinance fetch + data quality checks
│   ├── stats.py              CAGR, MDD, Sharpe, Sortino, Calmar, Ulcer, compute_stats
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
│   ├── plot_linkedin.py      two-panel LinkedIn figure
│   ├── validation.py         walk-forward + Pareto frontier analysis
│   └── export.py             Excel master log + results formatting
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
├── learning/                 guided engine rewrite (6-session curriculum)
└── logs/                     untracked — performance_tracking.csv lives here
```

---

## Configuration

All parameters live in `engine/config.py`. The default strategy is loaded from `strategies.json`:

```python
DEFAULT_STRATEGY = "6asset_tip_gsg_rpavg"  # production RP-averaged weights
```

Key settings:

| Parameter | Default | Notes |
|-----------|---------|-------|
| `BACKTEST_START` | 2006-01-01 | Limited by GSG inception |
| `OOS_START` | 2022-01-01 | IS/OOS boundary |
| `DATA_FREQUENCY` | "ME" | Monthly rebalancing |
| `TRANSACTION_COST_PCT` | 0.001 | 0.1% per trade |
| `RISK_FREE_RATE` | 0.035 | Fed funds rate as of March 2026 |

---

## Live ETF Mapping

For actual implementation, use lower-cost ETF equivalents:

| Backtest ETF | Live ETF | Annual Saving |
|-------------|---------|---------------|
| SPY → | IVV | 0.03% |
| GLD → | GLDM | 0.30% |
| GSG → | PDBC | 0.16% |
| QQQ → | QQQM | 0.05% |

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

- ALLW comparison covers ~1 year only (March 2025 launch)
- No currency adjustment for non-US investors (GBP, EUR)
- Paper trading started April 2026 via Alpaca (two accounts: backtest ETFs and live ETFs)
- Sortino uses downside std, not standard semi-deviation
- Max drawdown on 20-year backtest computed from monthly data (daily MDD available for ALLW comparison period)
