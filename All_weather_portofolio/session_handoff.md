# Session Handoff — 2026-03-23

This document captures full context from the 2026-03-22 session.

---

## What happened in this session

### 2026-03-23 — Phase 10A foundation fixes (Claude Code)
1. Full project review conducted: code, MD files, all findings independently verified
2. CLAUDE.md created with project rules, IS/OOS discipline, rejected experiments table
3. Claude Code designated as primary tool for ALL work (strategy + engineering)
4. Phase 10A implemented: 5 targeted code fixes, all 55 tests passing

### 2026-03-22 — Phase 9 full run
1. Phase 9 full re-run: 33 experiments with fixed DE optimiser
2. ALLW comparison completed (compare_allw.py)
3. Backtest vs live ETF comparison — PDBC identified as divergence source
4. DE failure re-diagnosed as regime mismatch, not setup bugs
5. Multiple code improvements via Claude Code
6. Master log archived and rebuilt with new column order + Martin ratio
7. Crisis analysis script briefed (not yet implemented)
8. MD files updated to reflect Phase 9 findings

---

## ⚠️ CRITICAL: DE still fails 2020-split OOS

After fixing all four optimiser bugs, DE-optimised weights still fail OOS:
- Best optimised OOS Calmar: 0.349 (7asset_no_iwd)
- Manual allocation OOS Calmar: 0.403 (6asset_tip_gsg)
- Manual beats every optimised result

Root cause: IS period 2006-2020 is a single falling-rates regime. DE
correctly finds TLT-heavy allocations that maximise Martin ratio on that
data. 2022 rate shock is in OOS. The optimiser was never penalised for
bond concentration.

This is NOT an optimiser bug. It is a training data coverage problem.

**The fix: split2022 experiments.** OOS_START = 2022-01-01 forces the
rate shock into IS training. The split2022 OOS result is the most important
number still pending. If split2022 DE beats 0.403, optimised allocation
becomes a product feature.

---

## ⚠️ CRITICAL: Phase 9 complete — DE confirmed failed definitively

33 experiments run. 26 with DE-optimised weights. Zero beat manual (0.403).

**Split2022 result: 0.129 — the worst result in the entire table.**

The split2022 fix made things worse, not better. The OOS window 2022-2026
is a 4-year recovery/Iran war period. Weights optimised to survive the
2022 rate shock are wrong for the recovery that follows.

**WF median is not a reliable signal. Retracted.**
High WF median (even 2.000) predicts OOS failure as often as OOS success.
Do not use WF median to assess strategy quality going forward.

**Principle 7 retracted.** Including 2022 in IS training does not improve
OOS reliability. Confirmed across 6 split2022 experiments.

**Gate 1 closed: DE does not add value. Pivot to risk parity.**

Do not run further DE experiments. The next experiment phase requires
the risk parity framework to be designed first (Opus session).

## ⚠️ CRITICAL: Opus session is now the highest priority action

Before any further engineering work on optimisation, an Opus session is
needed to design the risk parity framework. Without this, all further
experiments are directionless.

Bring to Opus:
- Phase 9 full results table (26 experiments, all fail vs manual 0.403)
- WF median retraction evidence
- Split2022 retraction evidence
- Risk parity math (already in research_log.md Phase 10)
- DJP finding (better OOS stability than GSG in optimised context)
- Key design questions: objective function, covariance window, leverage constraint

The Phase 9 run was at experiment 17/33 when this session ended.
split2022 IS optimise completed (IS Calmar 0.639) but OOS not yet run.
Wait for the run to complete before drawing conclusions about DE value.

---

## ⚠️ IMPORTANT: Stale data warning

yfinance returns total return data with ~45 day lag for some tickers.
Warning threshold updated to 45 days. If lag exceeds 60 days, verify
manually. Current data ends approximately 2025-12-31 in some tickers.

---

## Current validated strategies (manual weights)

| Strategy | Allocation | OOS Calmar (2020-split) |
|---|---|---|
| 6asset_tip_gsg | SPY 15%, QQQ 15%, TLT 30%, TIP 15%, GLD 15%, GSG 10% | 0.403 (manual) |
| 7asset_tip_gsg_vnq | SPY 12%, QQQ 12%, TLT 25%, TIP 12%, GLD 13%, GSG 10%, VNQ 16% | TBD split2022 |
| 7asset_tip_djp | SPY 12%, QQQ 13%, IWD 8%, TLT 27%, TIP 13%, GLD 15%, DJP 12% | TBD split2022 |
| 5asset_dalio | SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, GSG 7.5% | 0.188 (optimised, bad) |

Use manual weights for all production claims and content.

---

## ALLW comparison results (March 2025 to March 2026)

6asset_tip_gsg (fee-adj) vs ALLW (fee-adj):
- CAGR: 17.33% vs 15.64%
- Max DD: -6.70% vs -8.79%
- Calmar: 2.586 vs 1.779
- Worst month: -3.97% vs -7.01%
- Volatility: 10.07% vs 12.60%

Strong result. Caveat: 12 months in macro environment that specifically
favours commodity-weighted unlevered strategies. Not a general claim.

---

## ETF decisions

| Purpose | Backtest | Live recommendation |
|---|---|---|
| US large cap | SPY | IVV |
| US tech/growth | QQQ | QQQM |
| Long bonds | TLT | TLT |
| Inflation bonds | TIP | TIP |
| Gold | GLD | GLDM |
| Commodities | GSG | GSG (or PDBC with disclosure) |
| REITs | VNQ | VNQ |
| Value equity | IWD | IWD |
| Diversified commodities | DJP | DJP |
| Intermediate bonds | IEF | IEF |

PDBC diverges from GSG in energy-shock environments. Use GSG for all
backtesting and comparison charts. Disclose the difference to customers.

---

## Code changes made this session (all via Claude Code)

1. config.py: TARGET_ALLOCATION = 6asset_tip_gsg weights
2. config.py: ASSET_CLASS_GROUPS and ASSET_CLASS_MAX_WEIGHT updated
3. config.py: ASSET_BOUNDS added (per-asset, all four strategies covered)
4. config.py: BACKTEST_END = date.today() (dynamic)
5. config.py: load_strategy() function added
6. export.py: _Tee captures stderr
7. export.py: METRIC_COLS reordered (Calmar/Martin first)
8. export.py: META_COLS reordered (config detail last)
9. export.py: Martin ratio added to build_log_row()
10. export.py: auto-archive logic on first run
11. backtest.py: martin field added to StrategyStats
12. backtest.py: compute_stats() computes Martin = CAGR / Ulcer
13. data.py: data quality checks added
14. data.py: stale data threshold updated to 45 days
15. optimiser.py: DE bugs fixed (bounds, normalisation, objective, projection)
16. optimiser.py: optimise_random label reflects actual method
17. optimiser.py: fallback warning names missing tickers explicitly
18. compare_allw.py: new script (ALLW head-to-head comparison)
19. compare_allw.py: 5asset_dalio added
20. compare_allw.py: plot_growth_chart() replaces plot_drawdown_chart()
21. compare_allw.py: Excel export (allw_comparison_YYYYMMDD.xlsx)
22. compare_allw.py: additional metrics (Vol, Worst Month, Best Month)
23. crisis_analysis.py: briefed, not yet implemented

---

## Immediate next actions (priority order)

1. **Run risk parity diagnostic on 6asset_tip_gsg** — compare RP weights vs manual
   `python3 -c "from optimiser import compute_risk_parity_weights; from data import fetch_prices; import config; p = fetch_prices(list(config.TARGET_ALLOCATION.keys()), config.BACKTEST_START, config.BACKTEST_END); compute_risk_parity_weights(p, list(config.TARGET_ALLOCATION.keys()))"`
2. **Design risk parity experiment methodology** (covariance window sensitivity: 1yr/3yr/5yr)
3. **Implement risk parity as DE objective** in optimiser.py (replace Martin ratio)
4. **Re-run Phase 9 universe** with risk parity weights — compare OOS vs manual 0.403
5. **Implement crisis_analysis.py**
6. Choose brand name
7. Write ALLW comparison blog post (Rf/MDD now fixed — blog post unblocked on metrics side)
8. FCA compliance research

~~Fix Sharpe/Sortino Rf = 0~~ **DONE (2026-03-23)**
~~Fix daily MDD~~ **DONE (2026-03-23)**

## Decision gates (updated)

- **Gate 0 (Opus session):** Risk parity framework design. What is the
  objective function? What is the covariance window? Hybrid or pure?
  This gates all further optimisation work.
- **Gate 1: CLOSED.** DE does not add value. Confirmed.
- **Gate 2 (blog post response):** Organic demand? >100 signups = proceed.
- **Gate 3 (paper trading):** <1% tracking error = proceed to live capital.
- **Gate 4 (FCA review):** Can we offer rebalancing instructions legally?

---

## How Francisco works

- Critical analysis valued over confirmation
- **Everything in Claude Code** — strategy, analysis, code, planning. No context split.
- CLAUDE.md is the persistent project rulebook — update it when decisions change
- session_handoff.md + research_log.md are the persistent state — keep them current
- Iterative: run experiment → share output/file → analyse → refine
- GitHub: https://github.com/fcastelasimao/quant-learning