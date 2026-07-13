---
verdict: production
summary: "DIY RP portfolio beats ALLW on Calmar (2.797 vs 1.961) with lower max drawdown; JEPQ added as income benchmark"
promoted: "research/production_validation/ — benchmark lines in strategy bundle"
---

# ALLW & Benchmark Comparisons

Comparison of the DIY risk-parity portfolio against Bridgewater's All Weather ETF (ALLW), S&P 500, JEPQ (JPMorgan Nasdaq Equity Premium Income), and a 60/40 benchmark.

## Key results (monthly rebalanced, fee-adjusted, Mar 2025-Mar 2026)

| Metric | DIY RP | ALLW |
|---|---|---|
| CAGR | 16.05% | 17.23% |
| Max DD | -5.74% | -8.79% |
| Calmar | 2.797 | 1.961 |
| Ulcer | 1.134 | 1.845 |

## Scripts

- `compare_allw.py` — ALLW ETF head-to-head (includes JEPQ benchmark)
- `compare_jepq.py` — JEPQ vs AW since 2022-05-03 inception
- `compare_live_etfs.py` — backtest vs live ETF performance pairwise
- `plot_linkedin.py` — two-panel LinkedIn figure

## Run
```bash
conda run -n allweather python3 research/allw_benchmark/compare_allw.py
conda run -n allweather python3 research/allw_benchmark/compare_jepq.py
```
