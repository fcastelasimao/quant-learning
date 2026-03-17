# All Weather Portfolio Tracker

A Python tool for backtesting, optimising, and tracking the Ray Dalio All Weather Portfolio. Compares three strategies — monthly rebalanced, buy & hold, and S&P 500 — and includes portfolio optimisation, Pareto frontier analysis, and walk-forward validation to test for overfitting.

---

## Table of Contents

- [Background](#background)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Optimisation Methods](#optimisation-methods)
- [Understanding the Output](#understanding-the-output)
- [Walk-Forward Validation](#walk-forward-validation)
- [Running the Tests](#running-the-tests)
- [Important Caveats](#important-caveats)
- [Known Limitations](#known-limitations)

---

## Background

The All Weather Portfolio was designed by Ray Dalio at Bridgewater Associates to perform well across all four economic environments: rising growth, falling growth, rising inflation, and falling inflation. The simplified public version uses five asset classes with fixed weights:

| Asset | ETF | Weight |
|-------|-----|--------|
| US Stocks | VTI | 30% |
| Long-Term Bonds | TLT | 40% |
| Intermediate Bonds | IEF | 15% |
| Gold | GLD | 7.5% |
| Commodities | DJP | 7.5% |

This tool allows you to backtest this strategy, modify the allocation, optimise the weights, and track your actual portfolio month to month.

---

## Features

- **Backtesting** — simulates monthly rebalancing from any start date and compares against buy & hold and S&P 500
- **Four optimisation methods** — random search, Calmar-maximising random search, Differential Evolution, and Sharpe-maximising SLSQP
- **Pareto frontier** — maps the full risk-return tradeoff curve so you can pick the allocation that matches your risk tolerance
- **Walk-forward validation** — tests whether optimised weights are genuinely robust or just overfitted to historical data
- **Monthly rebalancing instructions** — tells you exactly what to buy and sell each month
- **Versioned results** — every run saves to a timestamped folder, nothing is ever overwritten
- **Master log** — a single CSV comparing all runs side by side
- **Run config** — saves all parameters as JSON so any result can be reproduced exactly
- **Unit tests** — stat helper functions are covered by pytest tests

---

## Project Structure

```
All_weather_portfolio/
│
├── main.py          # Entry point. Orchestration only -- no logic lives here.
│                    # Read this file to understand what the program does at a
│                    # high level. Read the individual modules to understand how.
│
├── config.py        # ALL user parameters and TARGET_ALLOCATION live here.
│                    # This is the only file you need to edit for routine use.
│                    # Also contains validate_config() which checks parameters
│                    # at startup before any work is done.
│
├── data.py          # fetch_prices() -- downloads and cleans price data from
│                    # Yahoo Finance via yfinance. No other logic.
│
├── portfolio.py     # Real holdings management: load/save JSON, initialise
│                    # holdings, compute current weights, generate rebalancing
│                    # instructions, apply a rebalance. Deals with shares you
│                    # actually own today -- separate from the simulation.
│
├── backtest.py      # Simulation engine. Contains StrategyStats dataclass,
│                    # four shared stat helpers (compute_cagr, compute_max_drawdown,
│                    # compute_sharpe, compute_calmar), run_backtest(), and
│                    # compute_stats(). Pure simulation -- no file I/O.
│
├── optimiser.py     # All four optimisation methods. Shared _score_allocation()
│                    # function ensures the objective is computed consistently
│                    # regardless of which method is chosen. optimise_allocation()
│                    # dispatches to the correct method based on config.OPT_METHOD.
│
├── validation.py    # run_walk_forward() and run_pareto_frontier(). Both sit
│                    # above the optimiser in the dependency hierarchy -- they
│                    # call the optimiser internally but the optimiser knows
│                    # nothing about validation.
│
├── plotting.py      # All matplotlib visualisation. plot_backtest() produces
│                    # the two-panel dark-theme chart. style_ax() helper applies
│                    # consistent dark theme to any axes object.
│
├── export.py        # Everything that writes files to disk: make_results_dir(),
│                    # export_results(), save_run_config(), append_to_master_log().
│                    # Also contains the terminal print functions (print_stats,
│                    # print_rebalancing, print_header) since they are output
│                    # formatting closely related to export.
│
├── requirements.txt
├── README.md
│
├── portfolio_holdings.json     # auto-generated on first run: your current share
│                               # holdings. Delete this if you change your
│                               # allocation tickers.
│
└── tests/
    └── test_stats.py           # pytest unit tests for the four stat helper
                                # functions in backtest.py. Run with:
                                # pytest tests/test_stats.py -v
```

### Module dependency graph

Arrows mean "imports from". No circular dependencies.

```
main.py
  ├── config.py         (no project dependencies -- imports only stdlib + numpy)
  ├── data.py           (imports config)
  ├── portfolio.py      (imports config)
  ├── backtest.py       (imports config)
  ├── optimiser.py      (imports config, backtest)
  ├── validation.py     (imports config, backtest, optimiser)
  ├── plotting.py       (imports config, backtest)
  └── export.py         (imports config, backtest)

tests/
  └── test_stats.py     (imports backtest)
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
    ├── run_config.json          # all parameters -- use to reproduce this run
    ├── pareto_frontier.png      # only if RUN_PARETO = True
    ├── pareto_frontier.csv      # only if RUN_PARETO = True
    ├── walk_forward.png         # only if RUN_WALK_FORWARD = True
    └── walk_forward.csv         # only if RUN_WALK_FORWARD = True
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/All_weather_portfolio
```

**2. Create a dedicated Python environment**
```bash
conda create -n allweather python=3.12
conda activate allweather
```

**3. Install dependencies**
```bash
pip install -r requirements.txt
```

**Minimum versions required:**
- Python >= 3.10
- pandas >= 2.2 (required for `"ME"` resample frequency)
- scipy >= 1.9

---

## Quick Start

**Run a basic backtest with no optimisation:**
```bash
conda activate allweather
python main.py
```

**Run with Calmar optimisation:**

Edit `config.py`:
```python
RUN_OPTIMISER = True
OPT_METHOD    = "calmar"
```

**Run full analysis (optimise + Pareto frontier + walk-forward validation):**
```python
RUN_OPTIMISER    = True
RUN_PARETO       = True
RUN_WALK_FORWARD = True
```

> **Note:** Running all three together can take 30–60 minutes depending on `OPT_N_TRIALS` and your hardware, since each optimisation step runs thousands of backtests internally.

---

## Configuration

**All parameters live in `config.py`. It is the only file you need to edit for routine use.**

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_PORTFOLIO_VALUE` | `10_000` | Starting value in USD |
| `BACKTEST_START` | `"2006-01-01"` | Start date (YYYY-MM-DD). Cannot go before ~2006 due to DJP ETF inception |
| `BACKTEST_END` | `"2026-01-01"` | End date (YYYY-MM-DD) |
| `REBALANCE_THRESHOLD` | `0.05` | Minimum drift (as a fraction) before rebalancing is triggered |
| `RUN_LABEL` | `"original_allweather"` | Name for this run's results folder — change before each run |

### Target allocation

Edit `TARGET_ALLOCATION` in `config.py` to use any ETFs you want. Weights must sum to 1.0 — an assert at the bottom of `config.py` will raise an error immediately if they don't.

```python
TARGET_ALLOCATION = {
    "VTI":  0.30,
    "TLT":  0.40,
    "IEF":  0.15,
    "GLD":  0.075,
    "DJP":  0.075,
}
```

> **Important:** If you change the tickers, delete `portfolio_holdings.json` before running, or the script will detect the mismatch and reset automatically.

### ETF availability

The earliest available start dates for the default ETFs:

| ETF | Inception |
|-----|-----------|
| VTI | May 2001 |
| TLT | July 2002 |
| IEF | July 2002 |
| GLD | November 2004 |
| DJP | February 2006 |

**DJP is the limiting factor** — the earliest reliable backtest start is `"2006-06-01"`.

### Optimiser parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_OPTIMISER` | `False` | Set to `True` to run optimisation |
| `OPT_METHOD` | `"calmar"` | See [Optimisation Methods](#optimisation-methods) |
| `OPT_MIN_WEIGHT` | `0.05` | Minimum weight per asset (0.0–1.0) |
| `OPT_MAX_WEIGHT` | `0.60` | Maximum weight per asset (0.0–1.0) |
| `OPT_MIN_CAGR` | `0.0` | Minimum acceptable CAGR — prevents finding portfolios that barely grow |
| `OPT_N_TRIALS` | `2000` | Number of random trials (random/calmar only) |
| `OPT_RANDOM_SEED` | `42` | Seed for reproducibility — set to `None` for different results each run |

---

## Optimisation Methods

| Method | Algorithm | Objective | Best for |
|--------|-----------|-----------|----------|
| `"random"` | Random search | Maximise Calmar ratio | Baseline, simple to understand |
| `"calmar"` | Random search | Maximise Calmar ratio | Balanced risk-return, recommended starting point |
| `"differential_evolution"` | scipy DE | Maximise Calmar ratio | Better results than random in fewer trials |
| `"sharpe_slsqp"` | Gradient-based | Maximise Sharpe ratio | Fast, deterministic, optimises volatility not drawdown |

**Calmar ratio** = CAGR / |max drawdown|. A Calmar of 0.5 means you earn 0.5% of annual return per 1% of maximum drawdown accepted. Maximising Calmar finds the best balance between return and downside risk without requiring you to manually choose how to weight them.

**Why not SLSQP for drawdown?** Max drawdown is not a smooth function — it depends on a single worst moment in the backtest. Gradient-based methods get stuck because small weight changes produce near-zero gradients. SLSQP works correctly for Sharpe ratio because Sharpe (mean/std of returns) is smooth and differentiable.

All four methods share the same `_score_allocation()` function in `optimiser.py` so the objective is computed consistently regardless of which method is chosen.

---

## Understanding the Output

### Terminal output

Each run prints:
1. Config validation confirmation
2. **Rebalancing instructions** — which assets to BUY or SELL and by how much in dollars, triggered only when drift exceeds `REBALANCE_THRESHOLD`
3. **Performance statistics** — CAGR, max drawdown, Sharpe, Calmar, and final value for all three strategies
4. File save confirmations

### Key metrics

| Metric | What it means |
|--------|---------------|
| **CAGR** | Average annual growth rate, smoothed across the full period |
| **Max Drawdown** | Worst peak-to-trough loss at any point. Negative number — closer to 0 is better |
| **Sharpe Ratio** | Return per unit of volatility. Above 1.0 is considered excellent |
| **Calmar Ratio** | Return per unit of drawdown. Higher is better. Balances CAGR and risk in one number |

### Output files

| File | Contents |
|------|----------|
| `backtest.png` | Portfolio value over time + annual returns bar chart |
| `backtest_history.csv` | Monthly portfolio values for all three strategies |
| `stats.csv` | All metrics per strategy including Calmar ratio |
| `allocation.csv` | Weights used in this run |
| `run_config.json` | All parameters — copy into `config.py` to reproduce the run exactly |
| `pareto_frontier.png/csv` | Risk-return tradeoff curve across CAGR targets |
| `walk_forward.png/csv` | Overfitting analysis per validation window |
| `master_log.csv` | One row per run — compare all runs side by side |

---

## Walk-Forward Validation

The walk-forward analysis tests whether optimised weights are genuinely robust or simply overfitted to the historical period.

**How it works:**
1. Splits the data into sliding windows, each with a training and test period
2. Optimises weights using only the training data
3. Evaluates those weights on the unseen test data
4. Compares test performance to both the in-sample training performance and the original unoptimised allocation

**Key output — the overfit ratio:**

```
Overfit ratio = test Calmar / train Calmar
```

| Overfit ratio | Interpretation |
|---------------|----------------|
| 0.8 – 1.0+ | Low overfitting — allocation is robust |
| 0.6 – 0.8 | Moderate — treat results with caution |
| Below 0.6 | High overfitting — do not use for live trading |

**Walk-forward parameters in `config.py`:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `RUN_WALK_FORWARD` | `False` | Set to `True` to run |
| `WF_TRAIN_YEARS` | `8` | Years used to optimise weights |
| `WF_TEST_YEARS` | `4` | Years used to evaluate out-of-sample |
| `WF_STEP_YEARS` | `4` | How far to slide the window each step |

---

## Running the Tests

Unit tests cover the four stat helper functions in `backtest.py` — `compute_cagr`, `compute_max_drawdown`, `compute_sharpe`, and `compute_calmar`. Each test uses a known input with a mathematically verifiable expected output.

**Install pytest** (included in `requirements.txt`):
```bash
pip install pytest
```

**Run all tests:**
```bash
pytest tests/test_stats.py -v
```

**Expected output:**
```
tests/test_stats.py::test_cagr_known_doubling              PASSED
tests/test_stats.py::test_cagr_no_growth                   PASSED
tests/test_stats.py::test_cagr_quadrupling                 PASSED
tests/test_stats.py::test_cagr_negative_growth             PASSED
tests/test_stats.py::test_max_drawdown_monotonically_increasing  PASSED
tests/test_stats.py::test_max_drawdown_known_crash         PASSED
tests/test_stats.py::test_max_drawdown_end_crash           PASSED
tests/test_stats.py::test_max_drawdown_multiple_crashes    PASSED
tests/test_stats.py::test_max_drawdown_returns_negative_number   PASSED
tests/test_stats.py::test_sharpe_positive_consistent_returns     PASSED
tests/test_stats.py::test_sharpe_negative_consistent_returns     PASSED
tests/test_stats.py::test_sharpe_zero_volatility           PASSED
tests/test_stats.py::test_sharpe_empty_after_dropna        PASSED
tests/test_stats.py::test_calmar_basic                     PASSED
tests/test_stats.py::test_calmar_high_value                PASSED
tests/test_stats.py::test_calmar_zero_drawdown_returns_zero      PASSED
tests/test_stats.py::test_calmar_negative_cagr             PASSED
tests/test_stats.py::test_calmar_proportional              PASSED

18 passed in 0.XX s
```

Re-run the tests after any change to `backtest.py` to confirm nothing is broken.

---

## Important Caveats

**Overfitting risk** — If you optimise weights on the same period you then backtest, the results will look artificially good. The optimiser has seen the data and found weights that work well historically with no guarantee they work in the future. Always run walk-forward validation before trusting optimised weights.

**Survivorship bias** — The ETFs used all exist today and have long track records. This slightly biases results upward versus what you would have experienced choosing ETFs in real time in 2006.

**Transaction costs not modelled** — Every rebalancing trade is assumed to be free. In reality you pay bid-ask spread and potentially brokerage commissions. This is small for large portfolios but meaningful for small ones.

**Tax implications not modelled** — Every rebalancing trade in a taxable account triggers a capital gains event. Monthly rebalancing may be significantly less tax-efficient than quarterly or annual rebalancing.

**2022 bond crash caveat** — The 2006–2026 period includes the fastest interest rate hiking cycle in 40 years, which caused bonds and stocks to fall simultaneously in 2022. This is the worst-case scenario for All Weather and is historically unusual. Results from this period are not representative of typical All Weather performance.

---

## Known Limitations

- Backtest cannot start before February 2006 (DJP ETF inception date)
- No support for fractional shares in the rebalancing instructions
- Prices sourced from Yahoo Finance via `yfinance` — occasional data gaps or adjustments may affect results
- Walk-forward validation always uses random search internally regardless of `OPT_METHOD` setting
- The Pareto frontier is approximate — each point is the best found in `OPT_N_TRIALS` random trials, not a mathematically guaranteed optimum
- No transaction costs or taxes modelled

---

## Disclaimer

This tool is for educational and research purposes only. Nothing in this project constitutes financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions.