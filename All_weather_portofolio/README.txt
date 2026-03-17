# All Weather Portfolio Tracker

A Python tool for backtesting, optimising, and tracking an All Weather-style portfolio. Implements monthly rebalancing, Differential Evolution weight optimisation, Pareto frontier analysis, and walk-forward validation to distinguish genuine robustness from overfitting.

The final validated allocation beats the S&P 500 on every risk-adjusted metric over a 21-year backtest (2004-2026) while experiencing less than half the maximum drawdown.

---

## Table of Contents

- [Strategy Background](#strategy-background)
- [Final Validated Allocation](#final-validated-allocation)
- [Performance Results](#performance-results)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Optimisation Methods](#optimisation-methods)
- [Walk-Forward Validation](#walk-forward-validation)
- [Understanding the Output](#understanding-the-output)
- [Running the Tests](#running-the-tests)
- [Important Caveats](#important-caveats)
- [Known Limitations](#known-limitations)

---

## Strategy Background

Ray Dalio's All Weather Portfolio targets four economic environments: rising growth, falling growth, rising inflation, and falling inflation. The goal is not to maximise returns — it is to preserve and grow wealth steadily across all environments without experiencing catastrophic losses that cause investors to abandon the strategy.

The core insight is behavioural: a portfolio that drops 50% requires a 100% gain just to break even, and most investors cannot psychologically hold through that drawdown. They sell at the bottom, miss the recovery, and end up worse than if they had earned 2% less per year in a smoother portfolio.

This implementation departs from the original Dalio weights in two important ways:

**1. LQD replaced with TIP (Treasury Inflation-Protected Securities)**
Corporate bonds (LQD) and government bonds (TLT) move almost identically in rate shock environments — holding both doubles bond exposure without adding diversification. TIPS specifically protect bond value during inflation, which is precisely what failed in the 2022 rate crash.

**2. Weights optimised using Differential Evolution**
Rather than using the classic 30/40/15/7.5/7.5 split, weights are optimised to maximise the Calmar ratio (CAGR divided by maximum drawdown) over the full backtest period. This approach is validated using walk-forward analysis to test whether the optimised weights genuinely generalise to unseen data.

---

## Final Validated Allocation

| Asset | ETF | Weight | Role |
|-------|-----|--------|------|
| Gold | GLD | 42.7% | Primary return driver and inflation hedge |
| US Tech | QQQ | 30.3% | Growth equity exposure |
| Long Bonds | TLT | 14.6% | Government bond diversification |
| US Broad | SPY | 6.2% | Broad equity stability |
| TIPS | TIP | 6.2% | Residual inflation protection |

This allocation was arrived at through the following workflow:

1. Walk-forward validation on starting weights to confirm optimisation is trustworthy
2. Pareto frontier analysis to understand the CAGR vs drawdown tradeoff space
3. Differential Evolution optimisation on the full 2004-2026 dataset
4. Walk-forward validation on the DE-optimised weights to confirm robustness

---

## Performance Results

Backtest period: January 2004 to January 2026 (21.1 years), $10,000 starting value.

| Metric | All Weather (Rebalanced) | S&P 500 Buy & Hold | Edge |
|--------|-------------------------|--------------------|------|
| CAGR | 11.26% | 10.72% | +0.54%/yr |
| Max Drawdown | -20.78% | -50.78% | 2.4× less severe |
| Sharpe Ratio | 1.078 | 0.766 | +41% better |
| Calmar Ratio | 0.542 | 0.211 | +157% better |
| Final Value | $94,924 | $85,674 | +$9,250 more |

The strategy beats the S&P 500 on raw returns while experiencing less than half the maximum drawdown. A Sharpe above 1.0 means more than 1% of return is earned per 1% of risk accepted — the S&P 500 achieves 0.766 over the same period.

**Key stress test results:**
- 2008 financial crisis: portfolio down ~12% vs S&P 500 down 37%
- 2022 rate shock: portfolio down ~15% — the strategy's known weak spot
- 2020 Covid crash: portfolio largely unaffected, recovered quickly

---

## Project Structure

```
All_weather_portfolio/
│
├── main.py          # Entry point -- orchestration only, no logic
├── config.py        # ALL user parameters -- the only file you need to edit
├── data.py          # fetch_prices() via yfinance
├── backtest.py      # Simulation engine, stat helpers, StrategyStats dataclass
├── portfolio.py     # Real holdings management (load/save/rebalance)
├── optimiser.py     # Four optimisation methods, shared scoring function
├── validation.py    # run_walk_forward() and run_pareto_frontier()
├── plotting.py      # All matplotlib visualisation
├── export.py        # File I/O, master log, terminal printing
│
├── requirements.txt
├── README.md
│
├── portfolio_holdings.json     # auto-generated: your current share counts
│                               # delete this when changing tickers
│
└── tests/
    ├── conftest.py             # shared pytest fixtures
    └── test_stats.py           # 24 unit tests for stat helper functions
```

### Module dependency graph

```
main.py
  ├── config.py         (no project imports)
  ├── data.py           (imports config)
  ├── portfolio.py      (imports config)
  ├── backtest.py       (imports config)
  ├── optimiser.py      (imports config, backtest)
  ├── validation.py     (imports config, backtest, optimiser)
  ├── plotting.py       (imports config, backtest)
  └── export.py         (imports config, backtest)
```

### Results folder structure

```
results/
├── master_log.csv          # one row per run -- compare all runs side by side
└── YYYY-MM-DD_HH-MM-SS_<label>/
    ├── backtest.png
    ├── backtest_history.csv
    ├── stats.csv
    ├── allocation.csv
    ├── run_config.json          # all parameters -- copy to reproduce this run
    ├── walk_forward.png         # only if RUN_WALK_FORWARD = True
    ├── walk_forward.csv         # only if RUN_WALK_FORWARD = True
    ├── pareto_frontier.png      # only if RUN_PARETO = True
    └── pareto_frontier.csv      # only if RUN_PARETO = True
```

---

## Installation

```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/All_weather_portofolio

conda create -n allweather python=3.12
conda activate allweather
pip install -r requirements.txt
```

**Minimum versions:**
- Python >= 3.10
- pandas >= 2.2 (required for `"ME"` resample frequency)
- scipy >= 1.9

---

## Quick Start

**Run a basic backtest with the current allocation:**
```bash
conda activate allweather
python main.py
```

**Run walk-forward validation:**
```python
# In config.py:
RUN_WALK_FORWARD = True
RUN_OPTIMISER    = False
RUN_PARETO       = False
```

**Run the DE optimiser:**
```python
RUN_OPTIMISER = True
OPT_METHOD    = "differential_evolution"
RUN_LABEL     = "my_optimisation_run"
```

You only ever run `main.py`. All other files are modules — save them after editing and run `main.py` again. Changes take effect on the next run.

---

## Configuration

**`config.py` is the only file you need to edit for routine use.**

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_PORTFOLIO_VALUE` | `10_000` | Starting value in USD |
| `BACKTEST_START` | `"2004-01-01"` | Earliest date with all 5 ETFs available |
| `BACKTEST_END` | `"2026-01-01"` | End date |
| `REBALANCE_THRESHOLD` | `0.05` | Minimum drift to trigger rebalancing |
| `DATA_FREQUENCY` | `"ME"` | `"ME"` monthly or `"W"` weekly |
| `SHARPE_ANNUALISATION` | `12` | Must be 12 for monthly, 52 for weekly |
| `RUN_LABEL` | `"TIP_5asset_final_v2"` | Names the results folder — change each run |

### ETF availability

| ETF | Inception | Role |
|-----|-----------|------|
| QQQ | March 1999 | Growth equity |
| SPY | January 1993 | Broad equity |
| TLT | July 2002 | Long-term bonds |
| TIP | December 2003 | Inflation-protected bonds |
| GLD | November 2004 | Gold — limits backtest start to 2004 |

### Switching between monthly and weekly

Always change both parameters together. `validate_config()` will raise an error if they are mismatched:

```python
DATA_FREQUENCY       = "W"    # weekly
SHARPE_ANNUALISATION = 52
```

### Optimiser parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_OPTIMISER` | `False` | Set to `True` to run |
| `OPT_METHOD` | `"differential_evolution"` | See Optimisation Methods below |
| `OPT_MIN_WEIGHT` | `0.05` | Minimum weight per asset |
| `OPT_MAX_WEIGHT` | `0.40` | Maximum weight per asset |
| `OPT_N_TRIALS` | `10_000` | Trials for random/calmar methods |
| `OPT_RANDOM_SEED` | `42` | Set to `None` for different results each run |

### Walk-forward parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_WALK_FORWARD` | `False` | Set to `True` to run |
| `WF_TRAIN_YEARS` | `4` | Training window length |
| `WF_TEST_YEARS` | `2` | Test window length |
| `WF_STEP_YEARS` | `4` | Slide distance per window |
| `WF_OPT_METHOD` | `"calmar"` | `"calmar"` for speed, `"differential_evolution"` for rigour |

---

## Optimisation Methods

| Method | Algorithm | Objective | Best for |
|--------|-----------|-----------|----------|
| `"random"` | Random search | Calmar | Baseline |
| `"calmar"` | Random search | Calmar | Recommended starting point |
| `"differential_evolution"` | scipy DE | Calmar | Best results, ~30-60 min |
| `"sharpe_slsqp"` | Gradient SLSQP | Sharpe | Fast, smooth objective only |

**Calmar ratio** = CAGR / max drawdown. Maximising Calmar finds the best balance between return and downside risk without requiring you to manually weight them.

**Why not SLSQP for drawdown?** Max drawdown depends on a single worst moment — it is discontinuous and gradients are near-zero. SLSQP gets stuck. DE works because it does not need gradients.

**Recommended workflow:**
1. `WF_OPT_METHOD = "calmar"`, `RUN_WALK_FORWARD = True` — validate first
2. `OPT_METHOD = "differential_evolution"`, `RUN_OPTIMISER = True` — optimise
3. Update `TARGET_ALLOCATION` with DE results, run walk-forward again
4. `RUN_OPTIMISER = False`, clean final backtest

---

## Walk-Forward Validation

Walk-forward validation tests whether optimised weights are genuinely robust or simply overfitted to historical data.

**How it works:**
1. Splits data into sliding windows with non-overlapping train and test periods
2. Optimises weights on training data only
3. Evaluates on unseen test data
4. Compares test vs training performance (overfitting) and optimised vs original (value-add)

**Interpreting the overfit ratio** (test Calmar / train Calmar):

| Ratio | Interpretation |
|-------|----------------|
| ≥ 1.0 | Out-of-sample beat in-sample — very robust |
| 0.6 – 1.0 | Acceptable — some degradation but strategy holds |
| < 0.6 | Concerning — high overfitting, treat results with caution |

**Walk-forward results for the final allocation** (4yr train / 2yr test / 4yr step):

| Window | Test Period | Original Calmar | Verdict |
|--------|------------|-----------------|---------|
| 1 | 2008-2010 | 0.293 | Positive through 2008 crash |
| 2 | 2012-2014 | -0.158 | Persistent weak period for bonds |
| 3 | 2016-2018 | 1.531 | Strong |
| 4 | 2020-2022 | 2.691 | Excellent through Covid and early 2022 |

Mean overfit ratio: 0.750 — above the 0.6 warning threshold.

---

## Understanding the Output

### Terminal output

Each run prints: config validation, rebalancing instructions, performance statistics, and file save confirmations.

### Key metrics

| Metric | What it means |
|--------|---------------|
| **CAGR** | Average annual growth rate across the full period |
| **Max Drawdown** | Worst peak-to-trough loss at any point. Negative — closer to 0 is better |
| **Sharpe Ratio** | Return per unit of volatility. Above 1.0 is considered excellent |
| **Calmar Ratio** | Return per unit of drawdown. The primary optimisation objective |

### Output files

| File | Contents |
|------|----------|
| `backtest.png` | Portfolio value over time + annual returns bar chart |
| `backtest_history.csv` | Monthly portfolio values for all three strategies |
| `stats.csv` | All metrics per strategy |
| `allocation.csv` | Weights used in this run |
| `run_config.json` | All parameters — copy into `config.py` to reproduce exactly |
| `walk_forward.png/csv` | Overfitting analysis per validation window |
| `pareto_frontier.png/csv` | Risk-return tradeoff curve |
| `master_log.csv` | One row per run for cross-run comparison |

---

## Running the Tests

```bash
conda activate allweather
pytest tests/test_stats.py -v
```

24 tests covering `compute_cagr`, `compute_max_drawdown`, `compute_sharpe`, and `compute_calmar`. All should pass. Re-run after any changes to `backtest.py`.

**Expected output:**
```
tests/test_stats.py::test_cagr_parametrised[...]    PASSED  (5 parametrised cases)
tests/test_stats.py::test_cagr_negative_growth_exact PASSED
tests/test_stats.py::test_cagr_short_period          PASSED
tests/test_stats.py::test_max_drawdown_*             PASSED  (7 cases)
tests/test_stats.py::test_sharpe_*                   PASSED  (6 cases)
tests/test_stats.py::test_calmar_*                   PASSED  (6 cases)

24 passed
```

---

## Important Caveats

**Overfitting risk** — the final allocation was optimised on the same 2004-2026 period used for the backtest. Walk-forward validation reduces but does not eliminate this concern. The weights reflect what worked historically, not a guarantee of future performance.

**Gold concentration** — 42.7% in a single commodity is a concentrated bet. Gold had an exceptional 2004-2026 run driven by the 2008 crisis, 2020 pandemic, and 2022 inflation. This level of concentration in any single asset carries significant concentration risk if gold enters a prolonged bear market.

**2022 is the known weak spot** — any allocation combining bonds and growth equities struggled in 2022 when rates rose at the fastest pace in 40 years. The strategy lost approximately 15% that year. This is a structural weakness that cannot be optimised away with the current asset universe. TIP reduces but does not eliminate this exposure.

**Rebalancing costs not modelled** — every monthly rebalancing trade is assumed free. Real transaction costs (bid-ask spread, brokerage commissions, tax events) would reduce returns, particularly on a small portfolio.

**Survivorship bias** — the ETFs used all exist today with long track records. Selecting them in hindsight biases results upward versus what would have been achievable in real time.

---

## Known Limitations

- Backtest cannot start before November 2004 (GLD ETF inception date)
- No fractional shares in rebalancing instructions
- Prices from Yahoo Finance via `yfinance` — occasional data gaps may affect results
- Pareto frontier analysis produces a flat curve with this asset universe — the unconstrained Calmar optimum already exceeds all tested CAGR floors
- Walk-forward always uses `WF_OPT_METHOD` regardless of `OPT_METHOD` setting
- GLD weight may slightly exceed `OPT_MAX_WEIGHT` after DE clip-and-renormalise due to minimum weight constraints on other assets

---

## Disclaimer

This tool is for educational and research purposes only. Nothing in this project constitutes financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.