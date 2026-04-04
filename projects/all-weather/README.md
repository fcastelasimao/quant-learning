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

## What We Tried and Rejected

| Approach | Experiments | Result | Status |
|----------|------------|--------|--------|
| Differential Evolution (return-based optimisation) | 26 | All failed OOS — IS period is a single falling-rate regime | Closed |
| SPY momentum overlay (derivative-based exit/re-entry) | 126 parameter combos | +1.3% on 2/3 splits, -5.3% on hardest | Closed |
| Rolling RP (quarterly recompute) | 3 OOS splits | Converges to same weights as static | Closed |
| Weekly rebalancing | 3 OOS splits with costs | No improvement after transaction costs | Closed |
| 100-ETF universe scan (8–12 asset subsets) | 50k random subsets | 6-asset universe confirmed optimal | Closed |
| 8-asset universe (CPER, DBA, IEF, IJR added) | 3 OOS splits | 6-asset beats on all Calmar windows | Closed |
| Bond leverage (1.0x–2.5x on TLT+TIP) | 7 levels × 3 splits | Every 0.25x adds ~3% drawdown, destroys Calmar | Closed |

---

## Installation

```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/projects/all-weather

conda create -n allweather python=3.11
conda activate allweather
pip install -r requirements.txt
```

## Quick Start

```bash
# Run a full backtest with production RP weights
python3 main.py

# Compare against Bridgewater's ALLW ETF
python3 compare_allw.py

# Generate LinkedIn comparison plot
python3 plot_linkedin.py

# Paper trade via Alpaca
python3 alpaca_monthly_rebalance.py --preview
python3 alpaca_monthly_rebalance.py --execute

# Run tests
python3 -m pytest tests/ -v
```

## Project Structure

```
├── main.py                      Entry point — orchestrates a single backtest run
├── config.py                    All parameters — loads allocation from strategies.json
├── strategies.json              Validated strategy registry
├── data.py                      Price fetching via yfinance with data quality checks
├── backtest.py                  Simulation engine, performance metrics, rolling RP
├── optimiser.py                 Risk parity (SLSQP), random search, weight projection
├── portfolio.py                 Live holdings management and rebalancing
├── validation.py                Walk-forward and Pareto frontier analysis
├── plotting.py                  Dark-theme backtest charts
├── export.py                    Excel master log, CSV export, terminal formatting
├── compare_allw.py              Head-to-head ALLW ETF comparison
├── plot_linkedin.py             Two-panel LinkedIn comparison figure
├── alpaca_monthly_rebalance.py  Paper trading via Alpaca (multi-account)
├── run_rolling_rp.py            Rolling RP vs static RP experiment
├── tests/
│   ├── conftest.py
│   ├── test_stats.py
│   ├── test_data.py
│   └── test_rolling_rp.py
├── archive/                     Completed experiments and dead code
└── results/                     Generated output (.gitignore'd)
```

## Configuration

All parameters live in `config.py`. The default strategy is loaded from `strategies.json`:

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

## Live ETF Mapping

For actual implementation, use lower-cost ETF equivalents:

| Backtest ETF | Live ETF | Annual Saving |
|-------------|---------|---------------|
| SPY → | IVV | 0.03% |
| GLD → | GLDM | 0.30% |
| GSG → | PDBC | 0.16% |
| QQQ → | QQQM | 0.05% |

## Known Limitations

- ALLW comparison covers ~1 year only (March 2025 launch)
- No currency adjustment for non-US investors (GBP, EUR)
- Paper trading started April 2026 via Alpaca (two accounts: backtest ETFs and live ETFs)
- Sortino uses downside std, not standard semi-deviation
- Max drawdown on 20-year backtest computed from monthly data (daily MDD available for ALLW comparison period)