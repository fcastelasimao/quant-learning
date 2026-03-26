# All Weather Portfolio Engine

A Python backtesting, optimisation, and validation engine for risk-balanced
portfolio strategies inspired by Ray Dalio's All Weather approach. Built for
investors who prioritise capital preservation over return maximisation.

Primary metric: **Calmar ratio** (CAGR / |max drawdown|).

Validated across 23 asset universe experiments, 229 backtest runs, and
multiple independent out-of-sample windows from 2006 to 2026.

> **Disclaimer:** This tool is for educational and research purposes only.
> Nothing here constitutes financial advice. Past performance does not
> guarantee future results. In the UK, providing personalised financial
> advice without FCA authorisation is a criminal offence.

---

## Table of Contents

- [Why This Exists](#why-this-exists)
- [How This Differs from ALLW](#how-this-differs-from-allw)
- [Validated Strategies](#validated-strategies)
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
- [Key Research Findings](#key-research-findings)

---

## Why This Exists

Ray Dalio's All Weather strategy targets four economic environments: rising
growth, falling growth, rising inflation, falling inflation. The goal is not
to beat the S&P 500 — it is to preserve and grow wealth steadily through
all environments without catastrophic losses.

The behavioural insight: a portfolio that drops 50% requires a 100% gain
just to recover. Most investors sell at the bottom and miss the recovery,
ending up worse than if they had earned 2% less per year in a smoother
portfolio. Risk-adjusted returns — not raw returns — predict real investor
outcomes.

This project goes beyond simple backtesting by implementing:
- Strict in-sample / out-of-sample methodology to prevent overfitting
- Walk-forward validation to test whether optimised weights generalise
- Multi-window stress testing (including the 2022 rate shock)
- Differential Evolution optimisation with asset class constraints
- Total return pricing (dividends and interest reinvested)

---

## How This Differs from ALLW

In March 2025, Bridgewater Associates and State Street launched the
SPDR Bridgewater All Weather ETF (ticker: ALLW). It is a single-ticker,
actively managed implementation of Bridgewater's institutional All Weather
strategy. As of March 2026, ALLW has accumulated over $1 billion in AUM.

This project is **not** a clone of ALLW. It is a transparent, unlevered,
customisable alternative. The key differences:

| Feature | ALLW | This project |
|---------|------|-------------|
| Transparency | Opaque — Bridgewater provides a daily model, no methodology disclosed | Fully transparent — all weights, logic, and validation published |
| Leverage | ~2x on bonds (amplifies gains AND losses) | None — all strategies are unlevered |
| Customisation | One portfolio, take it or leave it | Four risk tiers: Growth, Balanced, Conservative |
| Cost | 0.85% expense ratio ($850/yr on $100k) | DIY ETF cost ~0.10-0.15% ($100-150/yr on $100k) |
| Validation | No public walk-forward or IS/OOS proof | Walk-forward validated with published overfit ratios |
| Tax efficiency | 99% annual portfolio turnover | Monthly rebalancing (lower in threshold-based mode) |
| Risk tiers | Single allocation | Growth, Balanced, Conservative options |

ALLW is the right choice for investors who want single-ticker simplicity
and trust Bridgewater's brand. This project is for investors who want to
understand exactly what they own, control their allocation, avoid leverage,
and pay less in fees.

**Note:** "All Weather" is a trademark of Bridgewater Associates. This
project uses the term descriptively to reference the strategy philosophy,
not as a product or brand name.

---

## Validated Strategies

Four strategies have passed IS + OOS validation across multiple test windows.
Each maps to a different risk tolerance.

### Tier 1 — Growth

**6asset_tip_gsg** — Primary production allocation

| Asset | ETF | Weight | Role |
|-------|-----|--------|------|
| US broad equity | SPY | 15% | Core equity |
| US tech/growth | QQQ | 15% | Growth engine |
| Long-term bonds | TLT | 30% | Deflation hedge / duration |
| Inflation bonds | TIP | 15% | Rate-shock buffer |
| Gold | GLD | 15% | Crisis hedge |
| Commodities | GSG | 10% | Stagflation / supply shock hedge |

Validated across 3 independent OOS windows (2020-2026, 2018-2026, 2022-2026).
All three beat 60/40 decisively. MaxDD structurally stable at ~-20% across
all windows. Simplest production candidate with fewest assets to manage.

**7asset_tip_gsg_vnq** — Most regime-stable strategy

Same as above plus VNQ (REITs) at 16%, with other weights scaled down:
SPY 12%, QQQ 12%, TLT 25%, TIP 12%, GLD 13%, GSG 10%, VNQ 16%.

Calmar range across 3 windows: only 0.042 — the most stable of any strategy
tested. VNQ works because TIP + GSG provide an inflation anchor that
neutralises REIT rate sensitivity. This was the key surprise finding.

### Tier 2 — Balanced

**7asset_tip_djp**
SPY 12%, QQQ 13%, IWD 8%, TLT 27%, TIP 13%, GLD 15%, DJP 12%.

DJP (35% energy / 35% metals / 30% agriculture) replaces energy-heavy GSG.
Walk-forward median 0.970 — highest robustness of any multi-asset strategy.
Most reliable generaliser. Lower peak Calmar but more consistent.

### Tier 3 — Conservative

**5asset_dalio**
SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, GSG 7.5%.

Classic Dalio allocation. Simplest implementation. Lowest MaxDD in 2022
stress test (-15.83%). Best for investors who want the original philosophy
without modification.

### Live ETF mapping

| Backtest ETF | Live ETF | Annual fee saving | Status |
|-------------|---------|-------------------|--------|
| SPY → | IVV | 0.03% | ✅ Confirmed identical |
| GLD → | GLDM | 0.30% | ✅ Confirmed identical |
| GSG → | PDBC | 0.16% | ✅ Better contango management |
| QQQ → | QQQM | 0.05% | ✅ Same index |

UK investors (MiFID II): CSPX, EQQQ, IDTL, IGLN, CMOD are potential
alternatives. Run spot-check backtests to confirm tracking error < 0.5%.

---

## Performance Results

All figures use total return prices (dividends reinvested). Starting $10,000.

### OOS Calmar comparison — all candidates

| Strategy | 2020-2026 | 2018-2026 | 2022-2026 | Range |
|----------|----------|----------|----------|-------|
| 6asset_tip_gsg | **0.476** | 0.473 | 0.405 | 0.071 |
| 7asset_tip_gsg_vnq | 0.465 | 0.471 | **0.429** | **0.042** |
| 7asset_tip_djp | 0.432 | — | 0.334 | 0.098 |
| 5asset_dalio | 0.383 | — | 0.337 | 0.046 |
| 8asset_manual (demoted) | 0.441 | — | 0.313 | 0.128 |
| 60/40 benchmark | 0.263 | 0.293 | 0.191 | 0.102 |

All candidates beat 60/40 in every window including the hardest (2022-2026).
8asset_manual showed the worst regime sensitivity (-29% Calmar drop) and
was demoted despite passing initial OOS.

ALLW comparison (March 2025+) is pending — see experiment plan B0. ALLW
uses ~2x leverage on bonds, which is a headwind in the current rising-rate
environment. Head-to-head comparison is the highest-priority experiment.

### 2022 stress test — the definitive window

| Strategy | Calmar | CAGR | MaxDD | vs 60/40 |
|----------|-------|------|-------|---------|
| 7asset_tip_gsg_vnq | **0.429** | 8.00% | -18.65% | +124% |
| 6asset_tip_gsg | 0.405 | 7.53% | -18.57% | +112% |
| 5asset_dalio | 0.337 | 5.34% | -15.83% | +76% |
| 7asset_tip_djp | 0.334 | 5.90% | -17.69% | +75% |
| 60/40 | 0.191 | 4.32% | -22.64% | — |

### Full period 2006-2026

| Metric | 6asset_tip_gsg | 7asset_tip_gsg_vnq | 60/40 | SPY |
|--------|---------------|-------------------|-------|-----|
| CAGR | 8.50% | 8.34% | 8.61% | 11.06% |
| Max Drawdown | -22.36% | -21.25% | -26.37% | -50.78% |
| Calmar | 0.380 | 0.392 | 0.327 | 0.218 |
| Ulcer Index | 5.54 | 5.57 | 7.36 | 12.36 |
| $10k → | $48,796 | $47,419 | $49,754 | $76,620 |

SPY wins on raw returns — always has, likely always will. The All Weather
advantage is entirely risk-adjusted: lower drawdowns, shorter recovery,
lower Ulcer Index. For investors who must stay invested through bear
markets without panic-selling, the smoother curve has genuine value.

---

## IS/OOS Methodology

Three dates define the research boundary:

```
|-------- In-sample (train) --------|---- Out-of-sample (test) ----|
2006-01-01                     2020-01-01                    2026-01-01
BACKTEST_START                  OOS_START                    BACKTEST_END
```

All optimisation is confined to the IS period. The OOS period is never seen
during training. Multiple OOS windows are tested by varying OOS_START, but
each window is a separate independent experiment — not a tuning loop.

| Mode | Data window | Purpose |
|------|-------------|---------|
| backtest | IS only | Baseline with TARGET_ALLOCATION |
| optimise | IS only | DE weight search (never sees OOS) |
| walk_forward | IS only | Sliding window robustness check |
| oos_evaluate | OOS only | Single honest held-out test |
| full_backtest | Full period | Final reporting chart |

**Rule:** Run oos_evaluate as few times as possible. Every time you see
an OOS result and adjust, you leak information from the test set.

---

## Project Structure

```
All_weather_portfolio/
│
├── main.py               Entry point — orchestration only
├── config.py             ALL user parameters — only file to edit
├── strategies.json       Validated strategy registry (source of truth)
├── data.py               fetch_prices() via yfinance
├── backtest.py           Simulation engine, stat helpers, StrategyStats
├── portfolio.py          Real holdings management (load/save/rebalance)
├── optimiser.py          Differential Evolution with asset class caps
├── validation.py         Walk-forward and Pareto frontier analysis
├── plotting.py           Dark-theme backtest visualisation
├── export.py             Excel master log, CSV export, terminal output
│
├── run_experiment.py     Batch experiment runner (full IS→OOS pipeline)
├── curate_master_log.py  Curated + Archive tabs from master_log.xlsx
├── results_dashboard.py  Interactive HTML dashboard + scatter PNG
├── merge_master_logs.py  One-off utility (legacy log migration)
├── requirements.txt
│
├── tests/
│   ├── conftest.py
│   ├── test_stats.py
│   └── test_data.py
│
└── results/
    ├── master_log.xlsx
    ├── master_log_curated.xlsx
    └── YYYY-MM-DD_HH-MM-SS_label/
        ├── backtest.png
        ├── backtest_history.csv
        ├── stats.csv
        ├── allocation.csv
        ├── run_config.json
        ├── run_log.txt
        └── walk_forward.png / .csv / _weights.csv
```

---

## Installation

```bash
git clone https://github.com/fcastelasimao/quant-learning.git
cd quant-learning/All_weather_portofolio
conda create -n allweather python=3.11
conda activate allweather
pip install -r requirements.txt
```

---

## Quick Start

```bash
conda activate allweather

# Run full IS → optimise → walk-forward → OOS → full pipeline
# for a single named experiment:
python3 run_experiment.py --experiments 6asset_tip_gsg

# Run all experiments (long — use overnight):
python3 run_experiment.py --auto-yes

# Dry run to preview without executing:
python3 run_experiment.py --dry-run

# Regenerate curated log after experiments:
python3 curate_master_log.py

# Generate results dashboard:
python3 results_dashboard.py
```

---

## Configuration

All parameters live in `config.py`. Key settings:

### Date parameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| BACKTEST_START | "2006-01-01" | GSG inception limits this floor |
| OOS_START | "2020-01-01" | IS/OOS boundary |
| BACKTEST_END | "2026-01-01" | Extend when updating study |

### Transaction and tax costs

| Parameter | Default | Notes |
|-----------|---------|-------|
| TRANSACTION_COST_PCT | 0.001 | 0.1% per trade (realistic retail) |
| TAX_DRAG_PCT | 0.0 | 0.0 for ISA/SIPP/IRA; 0.03-0.05 for taxable |
| PRICING_MODEL | "total_return" | Always use for backtesting |

### ETF universe

| ETF | Inception | Role |
|-----|-----------|------|
| SPY | Jan 1993 | Broad US equity |
| QQQ | Mar 1999 | US tech/growth |
| IWD | Jan 2000 | US value equity |
| TLT | Jul 2002 | Long-term bonds (15+ yr duration) |
| IEF | Jul 2002 | Intermediate bonds (7-10 yr) |
| SHY | Jul 2002 | Short-term bonds (1-3 yr) |
| TIP | Dec 2003 | Inflation-linked bonds |
| GLD | Nov 2004 | Gold |
| VNQ | Sep 2004 | US REITs |
| GSG | Jul 2006 | Broad commodities (70% energy) |
| DJP | Jun 2006 | Bloomberg commodity index (balanced) |
| ALLW | Mar 2025 | Bridgewater All Weather ETF (benchmark only) |

---

## Optimisation Methods

| Method | Algorithm | Best for |
|--------|-----------|----------|
| differential_evolution | scipy DE | Best overall results, slow |
| calmar | Random search | Walk-forward speed |
| random | Random search | Simple baseline |
| sharpe_slsqp | Gradient SLSQP | Smooth objectives only |

**Key finding (Phase 9, confirmed):** After fixing all four optimiser bugs,
DE still fails to beat manual allocation. 26 experiments, zero beat manual
OOS Calmar 0.403. Root cause: IS period 2006-2020 is a single falling-rates
regime; DE finds TLT-heavy allocations that collapse in the 2022 rate shock.
Gate 1 closed — do not run further DE experiments.

**Current framework: Risk Parity (Phase 10C).** RP 5yr weights achieve OOS
Calmar 0.512 vs manual 0.403 (+27%). See research_log.md Phase 10C.

---

## Walk-Forward Validation

Tests whether optimised weights genuinely generalise or are overfitted.

1. Slide a training window across the IS period
2. For each window: optimise on training data, evaluate on unseen test data
3. Compute overfit ratio = test Calmar / train Calmar
4. Aggregate: use **median** (not mean — outlier windows distort mean)

| WF median | Classification |
|-----------|---------------|
| ≥ 0.6 | HIGH reliability |
| 0.3 – 0.6 | MODERATE — treat with caution |
| < 0.3 | LOW — do not use for live trading |

**Finding (retracted):** ~~Including 2022 in training data improves WF
reliability.~~ Split2022 experiments produced the worst OOS results in the
entire table (0.129–0.323). The OOS window 2022-2026 is a recovery/Iran war
period; weights tuned to survive the shock fail the recovery. Retracted Phase 9.

**WF median threshold retracted:** ~~WF median > 0.6 = HIGH reliability.~~
The highest WF medians (e.g. 2.000) predicted OOS failure, not success.
WF median is not a reliable predictor of OOS performance. Retracted Phase 9.

**Note:** ALLW does not publish walk-forward validation results. This is
one of the key differentiators of this project — every validated strategy
has published overfit ratios across multiple training windows.

---

## Understanding the Output

### Performance metrics

| Metric | What it measures | Better |
|--------|-----------------|--------|
| CAGR | Compound annual growth rate | Higher |
| Max Drawdown | Worst peak-to-trough loss | Less negative |
| Avg Drawdown | Mean of all drawdown periods | Less negative |
| Max DD Duration | Longest months below previous peak | Lower |
| Avg Recovery | Mean months to recover from drawdown | Lower |
| Ulcer Index | RMS of drawdowns (penalises time underwater) | Lower |
| Sharpe Ratio | Return per unit of total volatility | Higher |
| Sortino Ratio | Return per unit of downside volatility | Higher |
| Calmar Ratio | CAGR / |max drawdown| — primary metric | Higher |

### Master log

`results/master_log.xlsx` — one row per run, 10 metrics per strategy,
grouped headers. 229 rows as of 2026-03-21. Run `curate_master_log.py`
after experiment batches to generate the clean decision-making view.

---

## Running the Tests

```bash
conda activate allweather
pytest tests/ -v

# Skip integration tests (require network):
pytest tests/ -v -m "not integration"
```

---

## Known Limitations

**Monthly max drawdown understates true drawdowns (partially fixed).**
Monthly MDD from month-end values is reported. Daily max drawdown is also
computed and stored (`max_drawdown_daily` in StrategyStats, added Phase 10A).
Use the daily figure for investor-facing reporting.

**Backtest rebalances unconditionally; live portfolio uses 5% threshold.**
`run_backtest()` rebalances every period. `portfolio.py` only rebalances
when drift exceeds 5%. This creates tracking error between model and reality.
Comparison experiment planned (threshold=0.0 vs threshold=0.05).

**No currency adjustment.**
All returns are in USD. Non-US investors face currency risk that can easily
exceed the portfolio's annual return. GBP and EUR-adjusted backtests are
planned.

**Sharpe/Sortino risk-free rate (fixed Phase 10A).**
Rf = 0.035 (US Fed funds March 2026). Pre-Phase 10A runs used Rf = 0 and
are not directly comparable.

**No direct comparison to ALLW yet.**
ALLW launched in March 2025, providing roughly one year of data. A head-to-
head comparison is the highest-priority pending experiment. ALLW's use of
~2x leverage on bonds makes direct comparison nuanced — in rising-rate
environments the leverage is a headwind, in falling-rate environments
it amplifies returns.

**US-only equity exposure.**
All equity (SPY, QQQ, IWD) is US-listed. International equity (EFA) was
tested and eliminated (lowest CAGR at 5.43%), but this conclusion is drawn
from a period of US outperformance. ALLW includes global equities, which
is a structural difference.

**Calmar is biased against short OOS windows.**
Short windows where the stress event occurs early give the recovery CAGR
less time to compound. Compare strategies on the same window length only.

**GSG inception (July 2006) limits backtest start.**
All strategies containing GSG or DJP cannot be tested before mid-2006.

**Sortino uses downside standard deviation, not semi-deviation.**
Consistent within this project but not directly comparable to external
Sortino calculations.

---

## Key Research Findings

These principles emerged from 33 experiments across Phases 1–10 (Phase 9: 26 DE experiments, Phase 10: RP framework).

1. **Commodities (GSG/DJP) are non-negotiable.** Removing them costs -0.118
   OOS Calmar. Every validated strategy includes commodity exposure.

2. **TIP inflation bonds are essential.** The only experiment to fall below
   60/40 was missing both QQQ and TIP.

3. **QQQ (growth equity) is non-negotiable.** Removing QQQ for bonds produced
   the highest IS Calmar (0.626) but lowest OOS Calmar (0.198) — pure overfitting.

4. **Return-based DE optimisation does not beat manual allocation. Gate 1 closed.**
   26 experiments with the fully fixed DE optimiser: zero beat manual baseline (0.403).
   Root cause: IS period (2006-2020) is a single falling-rates regime. DE over-learns it.
   The fix (risk parity) is now confirmed — see finding 14.

5. **Three independent OOS windows required** for production promotion.

6. ~~**WF median > 0.6 = HIGH reliability.**~~ **RETRACTED (Phase 9).**
   WF median is not predictive of OOS performance. The highest WF median experiments
   (e.g. 5asset_dalio_split2022: 2.000) produced the worst OOS results (0.140).

7. ~~**Including 2022 in IS training improves robustness.**~~ **RETRACTED (Phase 9).**
   Split2022 results are the worst in the entire table (best: 0.323, worst: 0.129).

8. **2022-style shocks recur ~once per decade.** Stress testing is not academic.
   Current active risk (March 2026): Iran war, oil shock, sticky inflation.

9. **Tax shelter is critical.** Monthly rebalancing in a taxable account
   generates capital gains events. Tax-sheltered accounts strongly recommended.

10. **SPY always wins on raw returns.** The edge is behavioural — lower
    drawdowns keep investors invested through bear markets.

11. **ALLW is the primary competitive benchmark.** Not 60/40, not Portfolio
    Visualizer. Every experiment must include ALLW. The value proposition is
    transparency, no leverage, customisation, and lower cost.

12. **Use Martin ratio for optimisation, Calmar for reporting.** Calmar
    depends on a single worst data point — discontinuous, overfits to one event.
    Martin ratio (CAGR / Ulcer Index) uses all drawdown data.

13. **Use per-asset bounds, not uniform bounds.** TLT needs [0.20, 0.45].
    GSG needs [0.05, 0.15]. Uniform bounds exclude the best allocations.

14. **Risk parity is the correct optimisation foundation.**
    RP 5yr weights achieve OOS Calmar 0.512 on 6asset vs manual 0.403 (+27%).
    RP hurts full-period Calmar (0.292 vs 0.382) due to IS falling-rates regime — expected.
    RP gain depends on degree of over-concentration corrected: +27% on 6asset (TLT dominant),
    flat on 7asset (multiple over-concentrated assets cancel). Static RP is the current
    approach; rolling RP and hybrid (manual + RP blend) are the next experiments.

15. **High IS optimised Calmar is a red flag, not a green light.**
    IS optimised Calmar > 0.7 consistently predicts OOS collapse across 26 experiments.