# All Weather Portfolio Tracker

A Python tool for backtesting, optimising, and validating an All Weather-style portfolio. Implements monthly rebalancing, Differential Evolution weight optimisation, walk-forward validation, Pareto frontier analysis, and a strict in-sample / out-of-sample methodology to produce honest, non-overfitted performance results.

The primary metric throughout is the Calmar ratio (CAGR divided by maximum drawdown). The goal is capital preservation, not return maximisation.

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

Ray Dalio's All Weather Portfolio targets four economic environments: rising growth, falling growth, rising inflation, and falling inflation. The goal is not to maximise returns — it is to preserve and grow wealth steadily across all environments without experiencing catastrophic losses that cause investors to abandon the strategy.

The core insight is behavioural: a portfolio that drops 50% requires a 100% gain just to break even. Most investors cannot hold through that drawdown. They sell at the bottom, miss the recovery, and end up worse than if they had earned 2% less per year in a smoother portfolio.

This implementation extends the original Dalio framework in three ways:

**1. Expanded asset universe**
The original 5-asset Dalio allocation (stocks, long bonds, intermediate bonds, gold, commodities) is extended to 8 assets by splitting equities into three distinct ETFs (broad, tech, value) and splitting intermediate bonds into two duration buckets. This provides more granular exposure to each economic regime.

**2. Strict IS/OOS split**
All optimisation and walk-forward validation is confined to the in-sample period (2006-2020). The out-of-sample period (2020-2026) is held out and only evaluated once, after all IS work is complete. This prevents the common failure mode of optimising on the full period and reporting an inflated OOS number.

**3. Calmar-optimised weights (optional)**
Weights can be optimised using Differential Evolution to maximise Calmar ratio on the IS period. Walk-forward validation tests whether optimised weights genuinely generalise. In practice, the DE optimiser does not reliably beat the manual allocation on this dataset — the manual allocation is the current recommended configuration.

---

## Current Validated Allocation

| Asset | ETF | Weight | Role |
|-------|-----|--------|------|
| US Tech | QQQ | 15% | Growth equity |
| Long bonds | TLT | 25% | Government bond / deflation hedge |
| Gold | GLD | 15% | Inflation hedge / crisis protection |
| Broad US equity | SPY | 10% | Equity diversification |
| US value equity | IWD | 10% | Low-beta equity alternative |
| Intermediate bonds | IEF | 10% | Duration buffer |
| Commodities | GSG | 10% | Stagflation / supply shock hedge |
| Short-term bonds | SHY | 5% | Stability anchor |

**Asset class groups and caps:**

| Group | Assets | Cap |
|-------|--------|-----|
| Stocks | SPY, QQQ, IWD | 40% |
| Long bonds | TLT | 40% |
| Intermediate bonds | IEF, SHY | 25% |
| Gold | GLD | 20% |
| Commodities | GSG | 20% |

GSG (commodity ETF) inception July 2006 sets the backtest start date at 2006-01-01.

---

## Performance Results

All figures use price returns only (dividends and interest not yet modelled — see Known Limitations). Starting value $10,000.

### In-sample (2006-2020, training period)

| Metric | AW Rebalanced | 60/40 | SPY |
|--------|--------------|-------|-----|
| CAGR | 7.13% | 9.18% | 9.34% |
| Max Drawdown | -21.23% | -26.06% | -50.78% |
| Calmar | 0.336 | 0.352 | 0.184 |
| Ulcer Index | 4.48 | 5.34 | 13.92 |
| Max DD Duration | 22 months | 34 months | 52 months |

### Out-of-sample (2020-2026, held-out test)

| Metric | AW Rebalanced | 60/40 | SPY |
|--------|--------------|-------|-----|
| CAGR | 8.11% | 6.94% | 15.23% |
| Max Drawdown | -18.38% | -26.37% | -23.93% |
| Calmar | 0.441 | 0.263 | 0.636 |
| Ulcer Index | 6.24 | 10.58 | 7.79 |
| Max DD Duration | 26 months | 32 months | 23 months |

### Full period (2006-2026)

| Metric | AW Rebalanced | 60/40 | SPY |
|--------|--------------|-------|-----|
| CAGR | 7.51% | 8.61% | 11.06% |
| Max Drawdown | -21.23% | -26.37% | -50.78% |
| Calmar | 0.354 | 0.327 | 0.218 |
| Ulcer Index | 5.09 | 7.36 | 12.36 |
| Final Value ($10k) | $40,795 | $49,754 | $76,620 |

**Key stress tests:**
- 2008 financial crisis: AW down ~10% vs SPY down ~37% — the strategy's strongest argument
- 2022 rate shock: AW down ~15%, 60/40 down ~20%, SPY down ~18% — everyone fell, AW fell least
- SPY beats everything on raw returns over 20 years — the All Weather advantage is entirely on risk-adjusted terms

---

## IS/OOS Methodology

The three date parameters in `config.py` define the entire research boundary:

```
|-------- In-sample (train) --------|---- Out-of-sample (test) ----|
2006-01-01                     2020-01-01                    2026-01-01
BACKTEST_START                  OOS_START                    BACKTEST_END
```

**Run modes respect this boundary automatically:**

| Mode | Data used | Purpose |
|------|-----------|---------|
| `backtest` | BACKTEST_START → OOS_START | IS baseline, reference point |
| `optimise` | BACKTEST_START → OOS_START | DE weight search on IS data only |
| `walk_forward` | BACKTEST_START → OOS_START | Test optimisation robustness |
| `pareto` | BACKTEST_START → OOS_START | CAGR vs drawdown frontier |
| `oos_evaluate` | OOS_START → BACKTEST_END | Honest held-out evaluation (run sparingly) |
| `full_backtest` | BACKTEST_START → BACKTEST_END | Final reporting chart |

**Important:** `oos_evaluate` should be run as few times as possible. Every time you view the OOS result, adjust something, and re-run, you leak information from the test set back into your decisions. Run it once when all IS work is complete.

---

## Project Structure

```
All_weather_portfolio/
│
├── main.py          # Entry point -- orchestration only, no logic
├── config.py        # ALL user parameters -- the only file you need to edit
├── data.py          # fetch_prices() via yfinance
├── backtest.py      # Simulation engine, 9 stat helpers, StrategyStats dataclass
├── portfolio.py     # Real holdings management (load/save/rebalance)
├── optimiser.py     # Four optimisation methods with asset class cap enforcement
├── validation.py    # run_walk_forward() and run_pareto_frontier()
├── plotting.py      # Three-panel backtest chart (value, annual returns, drift)
├── export.py        # Excel master log, stats CSV, terminal printing
│
├── requirements.txt
├── README.txt
├── ToDo.txt
│
├── portfolio_holdings.json     # auto-generated: current share counts
│                               # delete this when changing tickers
│
├── conftest.py                 # shared pytest fixtures
├── test_stats.py               # 21 unit tests for stat helper functions
│
└── results/
    ├── master_log.xlsx         # one row per run, all metrics, grouped headers
    └── YYYY-MM-DD_HH-MM-SS_<label>/
        ├── backtest.png               # three-panel chart
        ├── backtest_history.csv       # monthly portfolio values
        ├── stats.csv                  # all 10 metrics per strategy
        ├── allocation.csv             # weights used in this run
        ├── run_config.json            # all parameters for exact reproduction
        ├── run_log.txt                # full terminal output
        ├── walk_forward.png           # walk-forward chart (walk_forward mode only)
        ├── walk_forward.csv           # per-window metrics (walk_forward mode only)
        ├── walk_forward_weights.csv   # per-window weights (walk_forward mode only)
        ├── pareto_frontier.png        # Pareto chart (pareto mode only)
        └── pareto_frontier.csv        # frontier points (pareto mode only)
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
- openpyxl >= 3.1 (required for Excel master log)

---

## Quick Start

**Run a baseline in-sample backtest:**
```bash
conda activate allweather
python main.py
# RUN_MODE = "backtest" in config.py
```

**Run the full validated workflow:**
```python
# Step 1: IS baseline
RUN_MODE = "backtest"

# Step 2: Walk-forward (takes 30-60 mins with DE)
RUN_MODE = "walk_forward"

# Step 3: IS optimisation
RUN_MODE = "optimise"

# Step 4: OOS evaluation (run once, after IS work is complete)
RUN_MODE = "oos_evaluate"
# Update TARGET_ALLOCATION with optimised weights first if using DE result

# Step 5: Full period reporting chart
RUN_MODE = "full_backtest"
```

You only ever run `main.py`. All other files are modules. Change `config.py` between runs.

---

## Configuration

**`config.py` is the only file you need to edit for routine use.**

### Date parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BACKTEST_START` | `"2006-01-01"` | Start of IS period. GSG inception limits to 2006. |
| `OOS_START` | `"2020-01-01"` | IS ends, OOS begins. All optimisation stays left of this. |
| `BACKTEST_END` | `"2026-01-01"` | End of OOS period. |

### Core parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `INITIAL_PORTFOLIO_VALUE` | `10_000` | Starting value in USD |
| `REBALANCE_THRESHOLD` | `0.05` | Minimum drift to trigger rebalancing |
| `DATA_FREQUENCY` | `"ME"` | `"ME"` monthly or `"W"` weekly |
| `SHARPE_ANNUALISATION` | `12` | Must match frequency: 12 for ME, 52 for W |
| `RUN_MODE` | `"backtest"` | See Run Modes section |

### ETF availability

| ETF | Inception | Role |
|-----|-----------|------|
| SPY | January 1993 | Broad US equity |
| QQQ | March 1999 | US tech equity |
| IWD | January 2000 | US value equity |
| TLT | July 2002 | Long-term government bonds |
| IEF | July 2002 | Intermediate government bonds |
| SHY | July 2002 | Short-term bonds |
| GLD | November 2004 | Gold |
| GSG | July 2006 | Broad commodities ← limits backtest start |

### Optimiser parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `OPT_METHOD` | `"differential_evolution"` | See Optimisation Methods |
| `OPT_MIN_WEIGHT` | `0.05` | Minimum weight per asset |
| `OPT_MAX_WEIGHT` | `0.30` | Maximum weight per asset |
| `OPT_MIN_CAGR` | `0.0` | Minimum acceptable CAGR |
| `OPT_N_TRIALS` | `10_000` | Trials for random/calmar methods |
| `OPT_RANDOM_SEED` | `42` | Set `None` for different results each run |

### Walk-forward parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `WF_TRAIN_YEARS` | `5` | Training window length |
| `WF_TEST_YEARS` | `2` | Test window length |
| `WF_STEP_YEARS` | `2` | Slide distance per window |
| `WF_OPT_METHOD` | `"differential_evolution"` | Use `"calmar"` for speed |

### Asset class caps

The optimiser enforces group-level caps defined in `ASSET_CLASS_GROUPS` and
`ASSET_CLASS_MAX_WEIGHT`. Removing these dicts from `config.py` disables caps
gracefully. Caps apply to both `optimise` and `walk_forward` modes.

---

## Run Modes

Set `RUN_MODE` in `config.py`:

| Mode | Data window | What it does |
|------|-------------|-------------|
| `backtest` | IS only | Backtest with current TARGET_ALLOCATION |
| `optimise` | IS only | DE weight search, then backtest with optimised weights |
| `walk_forward` | IS only | Sliding window validation, saves walk_forward.* files |
| `pareto` | IS only | CAGR vs drawdown frontier sweep |
| `oos_evaluate` | OOS only | Backtest on held-out 2020-2026 data |
| `full_backtest` | Full period | 2006-2026 reporting chart |

`walk_forward` and `pareto` modes do not run the generic backtest pipeline — they save their own outputs and exit cleanly.

---

## Optimisation Methods

| Method | Algorithm | Objective | Notes |
|--------|-----------|-----------|-------|
| `"random"` | Random search | Calmar | Simple baseline |
| `"calmar"` | Random search | Calmar | Recommended for walk-forward speed |
| `"differential_evolution"` | scipy DE | Calmar | Best results, ~30-60 min |
| `"sharpe_slsqp"` | Gradient SLSQP | Sharpe | Fast but cannot handle drawdown objective |

**Why DE for drawdown?** Max drawdown is discontinuous — gradients are near-zero or undefined. SLSQP gets stuck at local minima. DE evolves a population without needing gradients and finds better solutions.

**Note on optimiser value-add:** Walk-forward analysis across multiple cap configurations showed the DE optimiser beats the manual allocation in only 1-3 out of 4 windows. The manual allocation is recommended for live use. The optimiser is most useful as a diagnostic tool to understand which parts of the allocation the data supports.

---

## Walk-Forward Validation

Walk-forward validation tests whether optimised weights genuinely generalise or are overfitted to the training period.

**How it works:**
1. Slides a window across the IS period (BACKTEST_START to OOS_START)
2. Each window: optimise weights on training data, evaluate on test data
3. Compute overfit ratio = test Calmar / train Calmar per window
4. A low ratio means the IS result does not generalise

**Interpreting the overfit ratio:**

| Ratio | Interpretation |
|-------|----------------|
| ≥ 1.0 | Test beat train — robust (but check if test period was unusually benign) |
| 0.6 – 1.0 | Acceptable generalisation |
| < 0.6 | High overfitting — treat IS results with caution |

**Key finding from 8-asset experiments:** The 2010-2015 window is structurally difficult — a low-volatility equity bull run causes the optimiser to find a tech-heavy allocation that fails in the subsequent 2015-2017 period regardless of cap configuration. This is a regime problem, not a code problem.

**Walk-forward output files:**
- `walk_forward.png` — two-panel chart: Calmar per window + overfit ratios
- `walk_forward.csv` — per-window metrics including win rate, worst/best month
- `walk_forward_weights.csv` — weights found by DE in each window

---

## Understanding the Output

### Performance metrics

| Metric | What it means |
|--------|---------------|
| CAGR | Compound Annual Growth Rate — average annual return |
| Max Drawdown | Worst peak-to-trough loss (negative, closer to 0 is better) |
| Avg Drawdown | Mean of all drawdown values — what a typical bad period looks like |
| Max DD Duration | Longest consecutive months below a previous peak |
| Avg Recovery | Average months to recover from each drawdown episode |
| Ulcer Index | RMS of all drawdown values — penalises long time underwater |
| Sharpe Ratio | Return per unit of total volatility (annualised) |
| Sortino Ratio | Return per unit of downside volatility only |
| Calmar Ratio | CAGR / max drawdown — primary optimisation objective |

### Output files per run

| File | Contents |
|------|----------|
| `backtest.png` | Three-panel chart: value over time, annual returns, B&H drift |
| `backtest_history.csv` | Monthly portfolio values for all four strategies |
| `stats.csv` | All 10 metrics per strategy |
| `allocation.csv` | Weights used in this run |
| `run_config.json` | All parameters — copy into `config.py` to reproduce exactly |
| `run_log.txt` | Full terminal output including rebalancing instructions |

### Master log

`results/master_log.xlsx` contains one row per run with all metrics grouped by strategy. Key columns: Timestamp, Label, Run Mode, Backtest Start, Backtest End, OOS Start, then 10 metrics × 4 strategies, then Results Folder.

Old rows (pre-new-metrics) have blank cells in the Avg_DD, Max_DD_Dur, Avg_Rec, Ulcer, and Sortino columns — this is expected and correct.

---

## Running the Tests

```bash
conda activate allweather
pytest test_stats.py -v
```

21 tests covering `compute_cagr`, `compute_max_drawdown`, `compute_sharpe`, and `compute_calmar`. The five new stat helpers (`compute_avg_drawdown`, `compute_max_drawdown_duration`, `compute_avg_recovery_time`, `compute_ulcer_index`, `compute_sortino`) are not yet covered — see ToDo.txt.

---

## Known Limitations

**Price returns only (most important)**
Dividends and interest income are not modelled. This understates performance materially — particularly for TLT (historically ~3% coupon) and SHY (currently ~4-5% yield). Switching to total return prices via yfinance `auto_adjust=True` is a planned improvement. All results should be treated as conservative lower bounds on total return performance.

**No transaction costs**
Every rebalancing trade is assumed free. Real bid-ask spreads (~0.01-0.05% for liquid ETFs), brokerage commissions, and FX conversion costs (for UK investors buying USD ETFs) are not modelled.

**No tax modelling**
Monthly rebalancing is a taxable event in most jurisdictions. Results assume a tax-sheltered account (ISA or SIPP equivalent). In a taxable account, CGT drag would reduce returns meaningfully.

**Backtest cannot start before July 2006**
GSG (commodity ETF) inception date limits history. Dropping GSG allows a 2004 start using the original 7-asset configuration.

**No fractional shares**
Rebalancing instructions show dollar amounts but not share counts. For small portfolios (<£5,000), some positions may require fractional shares which not all brokers support.

**Walk-forward mean overfit ratio is distorted**
The mean overfit ratio is inflated by windows where the test period was unusually benign (individual ratios of 5-17x). The median is more representative. This is a known code limitation — see ToDo.txt.

**MiFID II restrictions (UK investors)**
Some UK brokers restrict US-listed ETFs for retail clients. UK-listed equivalents (CSPX, EQQQ, IDTL, IGLN) exist but may have slightly different tracking and expense ratios.

---

## Important Caveats

**The central finding is robust but conditional.** The 8-asset All Weather allocation beats 60/40 on all risk-adjusted metrics out-of-sample (OOS Calmar 0.441 vs 0.263). This holds on genuinely held-out data including the 2022 rate shock. However, the result is conditional on total return not being modelled — adding dividends and interest will change the absolute numbers.

**2022 is the honest weakness.** The simultaneous bond/equity/gold drawdown caused by the fastest rate rise in 40 years cannot be optimised away with the current asset universe. GSG (commodities) provided a partial hedge but could not offset the bond losses entirely. This is a structural feature of the macro environment, not a fixable bug.

**The rebalanced version consistently trails B&H on Calmar in trending markets.** Monthly rebalancing systematically sells winners in a bull market. The B&H version has a higher Calmar in almost every period tested. The rebalanced version's advantage is a stable, explainable allocation that does not drift arbitrarily. For an investor who needs to understand and explain their portfolio, rebalancing is worth the Calmar cost.

**This tool is for educational and research purposes only.** Nothing in this project constitutes financial advice. Past performance does not guarantee future results. Always consult a qualified financial advisor before making investment decisions. In the UK, providing financial advice without FCA authorisation is a criminal offence.