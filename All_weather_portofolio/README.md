# All Weather Portfolio Tracker

A Python backtesting, optimisation, and validation tool for an All Weather-style
portfolio. Implements monthly rebalancing, Differential Evolution weight
optimisation, walk-forward validation, and a strict in-sample / out-of-sample
methodology to produce honest, non-overfitted performance results.

The primary metric throughout is the Calmar ratio (CAGR / max drawdown).
The goal is capital preservation and competitive risk-adjusted returns --
not return maximisation.

---

## Table of Contents

- [Strategy Background](#strategy-background)
- [Current Validated Allocation](#current-validated-allocation)
- [Performance Results](#performance-results)
- [IS/OOS Methodology](#isoos-methodology)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Run Modes](#run-modes)
- [Optimisation Methods](#optimisation-methods)
- [Walk-Forward Validation](#walk-forward-validation)
- [Understanding the Output](#understanding-the-output)
- [Running the Tests](#running-the-tests)
- [Known Limitations](#known-limitations)
- [Important Caveats](#important-caveats)

---

## Strategy Background

Ray Dalio's All Weather Portfolio targets four economic environments: rising
growth, falling growth, rising inflation, and falling inflation. The goal is
not to maximise returns -- it is to preserve and grow wealth steadily across
all environments without experiencing catastrophic losses.

The core insight is behavioural: a portfolio that drops 50% requires a 100%
gain just to break even. Most investors sell at the bottom and miss the
recovery, ending up worse than if they had earned 2% less per year in a
smoother portfolio.

This implementation uses total return prices throughout (dividends and interest
reinvested). The yfinance library returns total return data by default via
auto_adjust=True, which is what the project uses.

---

## Current Validated Allocation

| Asset | ETF | Weight | Role |
|-------|-----|--------|------|
| US Tech | QQQ | 15% | Growth equity |
| Long bonds | TLT | 25% | Deflation hedge / duration |
| Gold | GLD | 15% | Inflation hedge / crisis |
| Broad US equity | SPY | 10% | Equity diversification |
| US value equity | IWD | 10% | Low-beta equity alternative |
| Intermediate bonds | IEF | 10% | Duration buffer |
| Commodities | GSG | 10% | Stagflation / supply shock hedge |
| Short-term bonds | SHY | 5% | Stability anchor |

Asset class caps enforced during optimisation:

| Group | Assets | Cap |
|-------|--------|-----|
| Stocks | SPY, QQQ, IWD | 40% |
| Long bonds | TLT | 40% |
| Intermediate bonds | IEF, SHY | 25% |
| Gold | GLD | 20% |
| Commodities | GSG | 20% |

GSG inception July 2006 sets the backtest start date.

---

## Performance Results

All figures use total return prices (dividends reinvested). Starting value $10,000.
Note: all historical runs in this project used total return data by default
(yfinance auto_adjust=True is the default). Results labelled "price_return"
in older master log rows should be interpreted as total return.

### In-sample 2006-2020 (training period, never used for OOS evaluation)

| Metric | AW Rebalanced | 60/40 | SPY |
|--------|--------------|-------|-----|
| CAGR | 7.13% | 9.18% | 9.34% |
| Max Drawdown | -21.23% | -26.06% | -50.78% |
| Calmar | 0.336 | 0.352 | 0.184 |
| Ulcer Index | 4.48 | 5.34 | 13.92 |
| Max DD Duration | 22 months | 34 months | 52 months |

### Out-of-sample 2020-2026 (held-out, honest evaluation)

| Metric | AW Rebalanced | 60/40 | SPY |
|--------|--------------|-------|-----|
| CAGR | 8.11% | 6.94% | 15.23% |
| Max Drawdown | -18.38% | -26.37% | -23.93% |
| Calmar | 0.441 | 0.263 | 0.636 |
| Ulcer Index | 6.24 | 10.58 | 7.79 |
| Max DD Duration | 26 months | 32 months | 23 months |

### Full period 2006-2026

| Metric | AW Rebalanced | 60/40 | SPY |
|--------|--------------|-------|-----|
| CAGR | 7.51% | 8.61% | 11.06% |
| Max Drawdown | -21.23% | -26.37% | -50.78% |
| Calmar | 0.354 | 0.327 | 0.218 |
| Ulcer Index | 5.09 | 7.36 | 12.36 |
| Final Value ($10k) | $40,795 | $49,754 | $76,620 |

Key stress tests:
- 2008 financial crisis: AW down ~10% vs SPY down ~37%
- 2022 rate shock: AW down ~15%, 60/40 down ~20% -- AW held up best
- SPY wins on raw returns over 20 years -- the AW advantage is entirely
  on risk-adjusted terms (Calmar, Ulcer, drawdown duration)

---

## IS/OOS Methodology

Three dates define the entire research boundary:

  |-------- In-sample (train) --------|---- Out-of-sample (test) ----|
  2006-01-01                     2020-01-01                    2026-01-01
  BACKTEST_START                  OOS_START                    BACKTEST_END

Run modes respect this boundary automatically:

| Mode | Data window | Purpose |
|------|-------------|---------|
| backtest | IS only | IS baseline with current TARGET_ALLOCATION |
| optimise | IS only | DE weight search, never sees OOS data |
| walk_forward | IS only | Sliding window robustness validation |
| pareto | IS only | CAGR vs drawdown frontier |
| oos_evaluate | OOS only | Single honest held-out test |
| full_backtest | Full period | Final reporting chart |

Important: run oos_evaluate as few times as possible. Every time you view
the OOS result and adjust something, you leak information from the test set.

---

## Project Structure

  All_weather_portfolio/
  |
  +-- main.py               Entry point, orchestration only
  +-- config.py             ALL user parameters -- only file you need to edit
  +-- data.py               fetch_prices() via yfinance (total return by default)
  +-- backtest.py           Simulation engine, 9 stat helpers, StrategyStats
  +-- portfolio.py          Real holdings management (load/save/rebalance)
  +-- optimiser.py          Four optimisation methods with asset class caps
  +-- validation.py         run_walk_forward() and run_pareto_frontier()
  +-- plotting.py           Three-panel dark-theme backtest chart
  +-- export.py             Excel master log, stats CSV, terminal printing
  |
  +-- requirements.txt
  +-- README.txt
  +-- ToDo.txt
  +-- strategies.json       Curated registry of validated strategies (planned)
  +-- merge_master_logs.py  One-off utility to merge master log eras
  |
  +-- portfolio_holdings.json   auto-generated: current share counts
  |                             delete this when changing tickers
  |
  +-- conftest.py               shared pytest fixtures
  +-- test_stats.py             unit tests for stat helper functions
  +-- test_data.py              unit and integration tests for data and costs
  |
  +-- results/
      +-- master_log.xlsx       one row per run, all metrics, grouped headers
      +-- YYYY-MM-DD_HH-MM-SS_label/
          +-- backtest.png
          +-- backtest_history.csv
          +-- stats.csv
          +-- allocation.csv
          +-- run_config.json
          +-- run_log.txt
          +-- walk_forward.png / .csv / _weights.csv  (walk_forward mode)
          +-- pareto_frontier.png / .csv               (pareto mode)

Module dependency graph:

  main.py
    +-- config.py         (no project imports)
    +-- data.py           (imports config)
    +-- portfolio.py      (imports config)
    +-- backtest.py       (imports config)
    +-- optimiser.py      (imports config, backtest)
    +-- validation.py     (imports config, backtest, optimiser)
    +-- plotting.py       (imports config, backtest)
    +-- export.py         (imports config, backtest)

---

## Installation

  git clone https://github.com/fcastelasimao/quant-learning.git
  cd quant-learning/All_weather_portofolio

  conda create -n allweather python=3.12
  conda activate allweather
  pip install -r requirements.txt

Minimum versions: Python >= 3.10, pandas >= 2.2, scipy >= 1.9, openpyxl >= 3.1

---

## Quick Start

Run a baseline in-sample backtest:
  conda activate allweather
  python main.py
  (RUN_MODE = "backtest" in config.py)

Full validated workflow:
  Step 1: RUN_MODE = "backtest"        -- IS baseline
  Step 2: RUN_MODE = "walk_forward"    -- validate robustness (30-60 min)
  Step 3: RUN_MODE = "optimise"        -- find best IS weights
  Step 4: update TARGET_ALLOCATION with optimised weights
  Step 5: RUN_MODE = "oos_evaluate"    -- single honest OOS test
  Step 6: RUN_MODE = "full_backtest"   -- final reporting chart

You only ever run main.py. Change config.py between runs.

---

## Configuration

config.py is the only file you need to edit for routine use.

### Date parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| BACKTEST_START | "2006-01-01" | Start of IS period |
| OOS_START | "2020-01-01" | IS ends, OOS begins |
| BACKTEST_END | "2026-01-01" | End of OOS period |

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| INITIAL_PORTFOLIO_VALUE | 10_000 | Starting value in USD |
| REBALANCE_THRESHOLD | 0.05 | Minimum drift to trigger rebalancing |
| DATA_FREQUENCY | "ME" | "ME" monthly or "W" weekly |
| SHARPE_ANNUALISATION | 12 | Must match: 12 for ME, 52 for W |
| PRICING_MODEL | "total_return" | "total_return" or "price_return" |
| TRANSACTION_COST_PCT | 0.0 | Cost per trade (0.001 = 0.1%) |
| TAX_DRAG_PCT | 0.0 | Annual drag (0.0 for ISA/SIPP) |

### ETF availability

| ETF | Inception | Role |
|-----|-----------|------|
| SPY | Jan 1993 | Broad US equity |
| QQQ | Mar 1999 | US tech equity |
| IWD | Jan 2000 | US value equity |
| TLT | Jul 2002 | Long-term bonds |
| IEF | Jul 2002 | Intermediate bonds |
| SHY | Jul 2002 | Short-term bonds |
| GLD | Nov 2004 | Gold |
| GSG | Jul 2006 | Commodities (limits backtest start) |

### Optimiser parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| OPT_METHOD | "differential_evolution" | See Optimisation Methods |
| OPT_MIN_WEIGHT | 0.05 | Minimum weight per asset |
| OPT_MAX_WEIGHT | 0.25 | Maximum weight per asset |
| OPT_N_TRIALS | 10_000 | Trials for random/calmar methods |
| OPT_RANDOM_SEED | 42 | Set None for different results each run |

### Walk-forward parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| WF_TRAIN_YEARS | 5 | Training window length |
| WF_TEST_YEARS | 2 | Test window length |
| WF_STEP_YEARS | 2 | Slide distance per window |
| WF_OPT_METHOD | "differential_evolution" | Use "calmar" for speed |

---

## Run Modes

| Mode | Data window | What it does |
|------|-------------|-------------|
| backtest | IS only | Backtest with current TARGET_ALLOCATION |
| optimise | IS only | DE weight search, then backtest optimised weights |
| walk_forward | IS only | Sliding window validation, saves walk_forward.* |
| pareto | IS only | CAGR vs drawdown frontier sweep |
| oos_evaluate | OOS only | Backtest on held-out 2020-2026 data |
| full_backtest | Full period | 2006-2026 reporting chart |

walk_forward and pareto modes do not run the generic backtest pipeline.
They save their own outputs and exit cleanly.

---

## Optimisation Methods

| Method | Algorithm | Objective | Notes |
|--------|-----------|-----------|-------|
| "random" | Random search | Calmar | Simple baseline |
| "calmar" | Random search | Calmar | Recommended for walk-forward speed |
| "differential_evolution" | scipy DE | Calmar | Best results, 30-60 min |
| "sharpe_slsqp" | Gradient SLSQP | Sharpe | Fast, smooth objective only |

Note on DE value-add: walk-forward analysis showed the DE optimiser beats
the manual allocation in only 1-3 out of 4 windows across multiple cap
configurations. The manual allocation is recommended for live use. The
optimiser is most useful as a diagnostic tool.

---

## Walk-Forward Validation

Walk-forward tests whether optimised weights genuinely generalise or are
overfitted to the training period.

How it works:
1. Slides a window across IS period (BACKTEST_START to OOS_START)
2. Each window: optimise on train, evaluate on test
3. Compute overfit ratio = test Calmar / train Calmar per window
4. Low ratio = IS result does not generalise

Interpreting the overfit ratio:
  >= 1.0  -- test beat train, robust
  0.6-1.0 -- acceptable generalisation
  < 0.6   -- concerning overfitting

Note: the mean overfit ratio can be distorted by windows where the test
period was unusually benign (ratios of 5-17x). The median is more
representative. This is a known limitation under active improvement.

Output files:
  walk_forward.png          -- Calmar per window + overfit ratios
  walk_forward.csv          -- per-window metrics
  walk_forward_weights.csv  -- per-window DE weights

---

## Understanding the Output

### Performance metrics

| Metric | What it means |
|--------|---------------|
| CAGR | Compound Annual Growth Rate |
| Max Drawdown | Worst peak-to-trough loss (negative, closer to 0 is better) |
| Avg Drawdown | Mean of all drawdown values -- what a typical bad period looks like |
| Max DD Duration | Longest consecutive months below a previous peak |
| Avg Recovery | Average months to recover from each drawdown episode |
| Ulcer Index | RMS of all drawdown values -- penalises long time underwater |
| Sharpe Ratio | Return per unit of total volatility |
| Sortino Ratio | Return per unit of downside volatility only |
| Calmar Ratio | CAGR / max drawdown -- primary optimisation objective |

### Master log

results/master_log.xlsx -- one row per run, all metrics grouped by strategy.
Key META columns: Timestamp, Label, Run Mode, Backtest Start, Backtest End,
OOS Start, Pricing Model, Tx Cost %, Tax Drag %, Data Freq, Tickers.
Then 10 metrics x 4 strategies, then Results Folder.

Old rows (pre-2026-03-20) have blank Pricing Model / Tx Cost % / Tax Drag %
cells but their metric data is intact and correct (was already total return).

---

## Running the Tests

  conda activate allweather
  pytest test_stats.py test_data.py -v

  Skip integration tests (require network):
  pytest test_stats.py test_data.py -v -m "not integration"

test_stats.py covers the four original stat helpers.
test_data.py covers total return verification and cost modelling.
The five newer stat helpers (AvgDD, MaxDDDuration, AvgRecovery, Ulcer,
Sortino) do not yet have tests -- see ToDo.txt.

---

## Known Limitations

Total return prices only
  The project uses auto_adjust=True (yfinance default) which returns
  dividend-reinvested total return prices. There is no lookahead bias here --
  yfinance adjusts prices retroactively, which is standard for backtesting.
  The PRICING_MODEL = "price_return" option uses auto_adjust=False.

No transaction costs by default
  TRANSACTION_COST_PCT = 0.0 by default. Real bid-ask spreads (~0.01-0.05%
  for liquid ETFs) and FX conversion costs (for UK investors buying USD ETFs)
  are not modelled unless you set this parameter explicitly.

No tax modelling by default
  TAX_DRAG_PCT = 0.0 by default. The current tax model is a blunt annual
  drag on portfolio value -- not a proper CGT calculation on realised gains.
  See the Important Caveats section for guidance on setting this parameter.

Backtest cannot start before July 2006
  GSG (commodity ETF) inception date limits history.

No fractional shares
  Rebalancing instructions show dollar amounts but not share counts.

Walk-forward mean overfit ratio can be misleading
  Individual window ratios above 5x inflate the mean significantly.
  Median overfit ratio is more informative. Improvement planned.

MiFID II restrictions (UK investors)
  Some UK brokers restrict US-listed ETFs for retail clients. UK-listed
  equivalents (CSPX, EQQQ, IDTL, IGLN) exist but may have different
  tracking and expense ratios.

---

## Important Caveats

Tax wrapper recommendation
  This strategy is best implemented inside a tax-sheltered account -- an ISA
  or SIPP in the UK. Monthly rebalancing is a taxable event in a general
  investment account (GIA). Each sell trade above your annual CGT allowance
  (£3,000 in 2026) is subject to Capital Gains Tax at 18-24% depending on
  your marginal income tax rate. Set TAX_DRAG_PCT = 0.0 for ISA/SIPP.
  For a taxable account, set TAX_DRAG_PCT = 0.03-0.05 as a conservative
  estimate. Consult a tax adviser for your specific situation.

2022 is the structural weakness
  The simultaneous bond/equity/gold drawdown from the fastest rate rise in
  40 years cannot be optimised away with this asset universe. The strategy
  lost approximately 15% in 2022. GSG (commodities) provided a partial hedge
  but could not offset the bond losses.

Rebalanced vs Buy-and-Hold
  The rebalanced version consistently trails buy-and-hold on Calmar in
  trending markets. Monthly rebalancing sells winners in a bull market.
  The rebalanced version's advantage is a stable, explainable allocation that
  does not drift arbitrarily -- after 15 years of a gold rally, a B&H
  portfolio may be 40% gold, which no rational investor would choose.

SPY wins on raw returns
  SPY outperformed every All Weather version on raw returns over 20 years.
  The All Weather advantage is entirely on risk-adjusted terms: lower Calmar
  drawdown, higher Ulcer Index score, shorter recovery periods. The 2008
  stress test is the most compelling argument -- SPY fell ~37%, AW fell ~10%.

Overfitting risk
  Walk-forward analysis shows moderate overfitting risk. The optimiser does
  not reliably beat the manual allocation. All published results should be
  accompanied by the OOS Calmar, not just the IS Calmar.

This tool is for educational and research purposes only. Nothing in this
project constitutes financial advice. Past performance does not guarantee
future results. Always consult a qualified financial adviser before making
investment decisions. In the UK, providing financial advice without FCA
authorisation is a criminal offence.