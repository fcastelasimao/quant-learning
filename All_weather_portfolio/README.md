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
| CAGR | 15.91% | 17.20% | ALLW +1.29% |
| Max Drawdown | **-5.74%** | -8.79% | **35% shallower** |
| Calmar Ratio | **2.775** | 1.959 | **42% better** |
| Ulcer Index | **1.184** | 1.949 | **39% lower** |
| Sortino Ratio | **2.299** | 1.549 | **48% better** |
| Annual Cost (on $100k) | ~$120 | ~$850 | **85% cheaper** |

ALLW earns ~1% more in raw return because of its 2x bond leverage. But it pays for that leverage with 35% deeper drawdowns and meaningfully worse risk-adjusted metrics.

### What This Means for Your Decision

**If you're choosing between the two:**

- **Choose ALLW if** you want maximum returns and can tolerate deeper drawdowns (e.g., long-term wealth building, high risk tolerance)
- **Choose us if** you prioritise capital preservation, stable drawdowns, and transparency (e.g., life savings, sleep-at-night investing, institutional mandate for modest volatility)

**The key trade-off:**
- **ALLW**: 1.2% higher return, but -8.79% max drawdown, 0.85% annual fee
- **Our strategy**: 1.2% lower return, but -5.74% max drawdown, 0.12% annual fee, fully transparent

Put another way: *You sacrifice 1.2% annual return to eliminate 3% of downside risk and save 0.73% in fees—a sensible trade if capital preservation matters more than maximum growth.*

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
| US Broad Equity | SPY | 13% | Core equity exposure |
| US Tech/Growth | QQQ | 11% | Growth engine |
| Long-Term Bonds | TLT | 19% | Deflation hedge |
| Inflation Bonds | TIP | 33% | Rate-shock buffer |
| Gold | GLD | 14% | Crisis hedge |
| Commodities | GSG | 10% | Stagflation hedge |

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
| Rolling RP (quarterly recompute) | 3 OOS splits | Wins 2/3 but differences are small; converges to same weights | Optional |
| Weekly rebalancing | 3 OOS splits with costs | No improvement after transaction costs | Closed |

---

## Installation

```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/All_weather_portfolio

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

# Validate RP weights across 3 OOS windows
python3 run_rp_validation.py

# Run rolling RP vs static RP experiment
python3 run_rolling_rp.py

# Scan ETF universe for optimal subsets
python3 scan_universes.py

# Run tests
python3 -m pytest tests/ -v
```

## Project Structure

```
├── main.py                 Entry point — orchestrates a single backtest run
├── config.py               All parameters — loads allocation from strategies.json
├── strategies.json          Validated strategy registry
├── data.py                  Price fetching via yfinance with data quality checks
├── backtest.py              Simulation engine, performance metrics, rolling RP
├── optimiser.py             Risk parity (SLSQP), random search, weight projection
├── portfolio.py             Live holdings management and rebalancing
├── validation.py            Walk-forward and Pareto frontier analysis
├── plotting.py              Dark-theme backtest charts
├── export.py                Excel master log, CSV export, terminal formatting
├── compare_allw.py          Head-to-head ALLW ETF comparison
├── run_rp_validation.py     3-split RP vs manual validation
├── run_rolling_rp.py        Rolling RP vs static RP experiment
├── scan_universes.py        ETF universe scan (diversification ratio scoring)
├── run_overlay_grid.py      SPY overlay grid search (concluded: no value)
├── tests/
│   ├── conftest.py
│   ├── test_stats.py
│   ├── test_data.py
│   └── test_rolling_rp.py
├── archive/                 Dead code kept for reference
└── results/                 Generated output (.gitignore'd)
```

## Configuration

All parameters live in `config.py`. The default strategy is loaded from `strategies.json`:

```python
DEFAULT_STRATEGY = "6asset_tip_gsg_rp"  # change to load a different strategy
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
- No paper trading track record yet
- Sortino uses downside std, not standard semi-deviation
- Max drawdown on 20-year backtest computed from monthly data (daily MDD available for ALLW comparison period)