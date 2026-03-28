# CLAUDE.md — All Weather Portfolio Engine

## Role

Operate as **quantitative finance researcher**, **software engineer**, and
**product strategist**. Challenge methodology, enforce IS/OOS discipline,
ground all recommendations in data.

## Environment

```bash
conda run -n allweather python3 <script>
conda run -n allweather python3 -m pytest tests/
```

## Module map

| File | Purpose |
|---|---|
| `config.py` | Single source of truth for all parameters |
| `main.py` | Entry point for single runs |
| `run_rp_validation.py` | Automated 3-split RP vs manual validation |
| `run_rolling_rp.py` | Rolling RP vs Static RP — IS sanity + 3 OOS splits |
| `scan_universes.py` | ETF universe scan by diversification ratio |
| `run_overlay_grid.py` | SPY overlay grid search (conclusion: no value) |
| `compare_allw.py` | ALLW ETF head-to-head comparison |
| `backtest.py` | Core engine, `compute_stats()`, overlay signal |
| `optimiser.py` | RP weights, random/SLSQP search |
| `data.py` | yfinance price fetching |
| `export.py` | Results folders, master log |
| `plotting.py` | Charts |
| `portfolio.py` | Live portfolio state and rebalancing |

## IS/OOS discipline — never violate

- IS = 2006-01-01 to OOS_START. Optimise and tune here only.
- OOS = OOS_START to today. Evaluate only, never tune.
- RP covariance must use IS data only (`end_date` parameter).

## Production strategy: 6asset_tip_gsg with averaged RP weights

| Asset | Weight | Role |
|---|---|---|
| SPY | 13% | US broad equity |
| QQQ | 11% | US tech/growth |
| TLT | 19% | Long bonds / deflation hedge |
| TIP | 33% | Inflation-linked bonds |
| GLD | 14% | Gold / crisis hedge |
| GSG | 10% | Commodities / stagflation |

Weights averaged across 3 independent RP computations (2020/2018/2022 splits).

## Validation summary

**RP vs manual (3 OOS windows):**

| Split | Manual | RP | Improvement |
|---|---|---|---|
| 2020 | 0.406 | 0.480 | +18% |
| 2018 | 0.417 | 0.462 | +11% |
| 2022 | 0.345 | 0.385 | +12% |

**ALLW comparison (March 2025-2026, live ETFs, fee-adjusted):**

| Metric | 6asset RP | ALLW |
|---|---|---|
| Calmar | 2.782 | 1.779 |
| Max DD | -5.66% | -8.79% |
| CAGR | 15.75% | 15.64% |

## Closed investigations

| Investigation | Result | Status |
|---|---|---|
| DE optimiser (26 experiments) | All fail vs manual | Gate CLOSED |
| WF median as OOS predictor | Does not predict | Retracted |
| Split2022 improves robustness | Makes OOS worse | Retracted |
| SPY overlay (126 param combos) | +1.3% on 2/3 splits, -5.3% on hardest | CLOSED — no value |
| Universe scan (16k subsets) | Confirms 6-asset is near-optimal | Complete |
| Rolling RP (quarterly recompute) | TBD — run `run_rolling_rp.py` | In progress |

## Metrics

Calmar (primary), Max DD daily, CAGR, Ulcer, Sortino, Martin.

## Files excluded from git

`results/`, `strategies.json`, `research_log.md`, `session_handoff.md`, `portfolio_holdings.json`