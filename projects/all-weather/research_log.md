# Research Log

---

## Project goal

Backtesting and validation engine for risk-balanced portfolios.
Primary metric: Calmar ratio. Positioned against Bridgewater's ALLW ETF.

## Phase 1-8 (2026-03-18 to 2026-03-20)
Explored 8 universes. TIP replaces LQD. 6asset_tip_gsg as best universe.

## Phase 9 — DE optimiser (2026-03-21 to 2026-03-23)
26 experiments, all fail vs manual. Gate 1 closed.

## Phase 10 — RP foundation (2026-03-23 to 2026-03-25)
Rf=0.035. Daily MDD. RP diagnostic: TLT 2x overweighted, TIP 2x underweighted.
RP 5yr OOS Calmar 0.512 vs manual 0.403.

## Phase 11 — Validation (2026-03-25 to 2026-03-26)

### Data integrity: OOS results in strategies.json corrected.

### RP multi-window: Gate 2 PASSED
| Split | Manual | RP | Improvement |
|---|---|---|---|
| 2020 | 0.406 | 0.480 | +18% |
| 2018 | 0.417 | 0.462 | +11% |
| 2022 | 0.345 | 0.385 | +12% |

### Universe scan: 15 ETFs, 16k subsets. Confirms 6-asset universe.

### Overlay grid: 126 combos, 3-split OOS. Does not add value. Closed.

### ALLW comparison (monthly rebalanced, fee-adjusted, Mar 2025-Mar 2026)
| Metric | rpavg | ALLW |
|---|---|---|
| CAGR | 16.05% | 17.23% |
| Max DD | -5.74% | -8.79% |
| Calmar | 2.797 | 1.961 |
| Ulcer | 1.134 | 1.845 |

## Phase 12 — Expanded experiments (2026-04-03 to 2026-04-04)

### Rolling RP
Recomputes RP weights quarterly from trailing 5-year covariance.
Result: converges to same weights as static. Closed.

### Weekly rebalancing
No improvement after transaction costs. Closed.

### 100-ETF universe scan
Researched ~100 candidates across 7 macro buckets, added to scan_universes.py.
Random sampling (50k subsets) from 50 post-dedup ETFs. Top ETFs: TLT, CPER, DBA, GLD.
Result: 6-asset universe confirmed optimal.

### 8-asset validation
Tested {SPY, QQQ, IJR, TLT, IEF, GLD, CPER, DBA} and {SPY, QQQ, IJR, TLT, TIP, GLD, CPER, DBA}.
Used proper RP-averaged weights (3 IS windows, averaged, OOS evaluated).
Result: 6-asset production beats both on all Calmar windows. Closed.

### Bond leverage (1.0x–2.5x on TLT+TIP)
Every 0.25x adds ~0.5% CAGR but ~3% deeper drawdowns.
At 2x leverage, 2022 OOS Calmar drops from 0.355 to 0.079.
Result: leverage destroys risk-adjusted returns in rising-rate regime. Closed.

---

## Phase 13 — Paper trading + code cleanup (2026-04-03)

### Paper trading launched
Two Alpaca accounts:
- Backtest ETFs: SPY, QQQ, TLT, TIP, GLD, GSG
- Live ETFs: IVV, QQQM, TLT, TIP, GLDM, PDBC (lower-cost equivalents)

Multi-account support added to `alpaca_monthly_rebalance.py` via `--account` CLI arg.

### compare_allw.py refactored
- Strategy registry with StrategyDef dataclass, reads allocations from strategies.json
- `enabled` flag per strategy for toggling without deletion
- Excel export cleaned: grey headers, no background colors, spacer rows between groups
- Results path changed to `__file__`-relative (no longer depends on CWD)

### Repository reorganised
Moved to `projects/` layout:
- `All_weather_portfolio/` → `projects/all-weather/`
- `wave_rider/` → `projects/wave-rider/`
- Dead projects to `archive/`
- Dead all-weather code to `projects/all-weather/archive/`

### ALLW comparison updated (fee-adjusted, Mar 2025–Apr 2026)
| Metric | rpavg | ALLW |
|---|---|---|
| CAGR | 17.4% | 19.1% |
| Max DD | -5.7% | -8.8% |
| Calmar | 3.03 | 2.18 |

### LinkedIn post + plot
Two-panel figure: equity curves with metrics table inset + worst drawdown zoom.
Post drafted, gated code sharing ("DM me for methodology").

### Dead code archived
Moved completed experiment scripts to archive/:
run_8asset_experiments.py, run_leverage_experiment.py, scan_universes.py, run_rp_validation.py.
Fixed config.py DEFAULT_STRATEGY (was pointing to 8-asset, now 6asset_tip_gsg_rpavg).
Disabled 8-asset entry in compare_allw.py.

---

## Open questions
- Brand name, FCA compliance, GBP/EUR adjustment
- Live vs backtest ETF performance divergence over time
- Optimal rebalancing trigger (threshold-based vs calendar-based)