# Session Handoff — 2026-03-21

This document captures the full context from the Opus 4.6 review session
so that subsequent sessions (Sonnet or Opus) can continue without loss.

---

## What happened in this session

Francisco started a fresh chat asking for a comprehensive critical review
of the entire All Weather Portfolio project. The review covered:

1. **Full code review** of all 9 modules + infrastructure scripts
2. **Bug identification** (6 bugs found, documented in TODO)
3. **Strategy critique** (survivorship bias, US-only equity, TLT concentration)
4. **Macro assessment** (Iran war, oil shock, Fed pause — live stress test)
5. **Gap analysis** (no FX adjustment, no daily MDD, no regime analysis, etc.)
6. **Market validation** with competitive research
7. **Optimiser root-cause analysis** — discovered why DE "fails" to beat manual

The market validation produced the most important business finding.
The optimiser analysis produced the most important engineering finding.

## ⚠️ CRITICAL: ALLW ETF exists

Bridgewater and State Street launched the SPDR Bridgewater All Weather ETF
(ticker: ALLW) in March 2025. Key facts:
- AUM: $1.17 billion (in ~1 year)
- Expense ratio: 0.85%
- Uses ~2x leverage on bonds
- 22.7% 1-year return, 4.4% yield
- 99% portfolio turnover (tax-inefficient)
- Allocations: ~43% global stocks, ~73% nominal bonds, ~41% TIPS,
  ~24% commodities (percentages exceed 100% due to leverage)

This changes the competitive positioning fundamentally:
- Cannot position as "access to the All Weather strategy" (ALLW owns that)
- Cannot use "All Weather" in the product name (Bridgewater trademark)
- Must position as: "validated, transparent, unlevered, customisable
  alternative to ALLW at lower cost"
- ALLW is now the primary benchmark, replacing 60/40 in all comparisons

## ⚠️ CRITICAL: Optimiser "failure" is a setup bug, not a real finding

The previous conclusion that "manual allocation beats DE, use manual for
production" was based on flawed experimental design. Four problems were
identified:

**Problem 1: Search space excludes the good allocations.**
OPT_MAX_WEIGHT = 0.25, but manual TLT is 30-40% in every validated
strategy. The optimiser literally cannot find the allocations it's being
compared to. It's restricted to the wrong region of weight space.

**Problem 2: Normalisation distorts DE's search.**
`de_objective()` normalises `w / w.sum()` — this warps the landscape so
DE's internal model doesn't match reality. Two nearby raw-weight points
can map to very different normalised weights.

**Problem 3: Calmar is a bad optimisation objective.**
Calmar = CAGR / |MaxDD| depends on a single worst data point. The
landscape is discontinuous (small weight changes can shift which month is
worst), causing the optimiser to overfit to one event that won't recur
in the test period.

**Problem 4: Post-convergence projection distorts the result.**
After DE converges, `_project_weights()` clips and renormalises, moving the
result away from what DE actually optimised. DE never explored the
neighbourhood of the projected weights.

**Fixes (all in optimiser.py):**
1. Per-asset bounds instead of uniform `[0.05, 0.25]` — let TLT reach 0.45
2. Remove normalisation from DE objective (use Dirichlet or N-1 parameterisation)
3. Switch objective from Calmar to Martin ratio (CAGR / Ulcer Index) — smooth, uses all data
4. Move projection inside the DE evaluation loop, not after convergence

**Impact:** If the optimiser works properly, "optimised allocation" becomes
a Pro tier product feature instead of a diagnostic-only tool. This reopens
a major product opportunity.

**Status:** Fixes not yet implemented. High priority engineering task.

## Current project state

- 23 universe experiments completed, 229 master log rows
- Four validated strategies ready for paper trading
- config.py still has the DEMOTED 8-asset allocation (needs fixing)
- Optimiser is broken by design (four bugs identified, fixes specified)
- No paper trading has started yet
- No product/web app exists yet — still in research/strategy phase
- Product targets global customers, not UK-only

## Documents produced in this session

All are in the project knowledge files. Use the _v3 versions where they
exist — they supersede earlier versions.

| File | Purpose | ALLW-aware? | Optimiser-aware? |
|------|---------|------------|-----------------|
| ToDo_v3.md | Master action plan | ✅ | ✅ |
| experiment_plan_v3.md | All experiments | ✅ | ✅ |
| research_log_v3.md | Research history | ✅ | ✅ |
| README_v3.md | Project README | ✅ | ✅ |
| visualisation_strategy_v2.md | Chart specs | ✅ | N/A (no change) |
| market_validation.md | Competitive analysis | ✅ | N/A (no change) |
| strategies.json | Strategy registry | N/A | N/A (no change) |
| session_handoff (this file) | Context bridge | ✅ | ✅ |

## What to work on next (priority order)

### Immediate (same day):
1. A0.1: Update config.py to 6asset_tip_gsg (10 min)
2. A0.5: Fix _Tee stderr capture (5 min)
3. A0.4: Add data validation to data.py (30 min)
4. A0.6: Drop strategies.json into project root (already created)

### This week (dual track):
5. **A0.7: Fix optimiser per-asset bounds** (15 min) ← new, high impact
6. **A0.8: Switch DE objective to Martin ratio** (30 min) ← new, high impact
7. **A0.9: Remove normalisation from DE objective** (1 hour) ← new
8. B0.1/E1: ALLW head-to-head comparison (3 hours)
9. E2: Iran war stress test WITH ALLW (1 hour)
10. A0.2: Fix rebalancing mismatch in backtest engine (2 hours)
11. A0.3: Add daily max drawdown computation (1.5 hours)
12. **E5: Re-run walk-forward with fixed optimiser** (2 hours) ← new, validates fixes

### Next week:
13. E3: GBP-adjusted backtest (2 hours)
14. B0.3: Choose brand name (decision)
15. B0.2: Write ALLW comparison blog post (4 hours)
16. B0.4: Build landing page (3 hours)

## Key bugs to fix (reference)

1. Backtest rebalances unconditionally; live uses 5% threshold
2. Max drawdown uses monthly data (understates true drawdowns)
3. Sharpe/Sortino assume Rf = 0 (Fed at 3.5-3.75%)
4. No GBP/FX adjustment
5. _Tee doesn't capture stderr
6. config.py defaults to demoted allocation
7. **Optimiser bounds exclude the best allocations** ← new
8. **DE normalisation distorts search space** ← new
9. **Calmar objective is discontinuous / overfits to single point** ← new
10. **Post-convergence projection distorts DE result** ← new

## Decision gates

- **Gate 1** (after ALLW comparison): Can we compete on risk-adjusted terms?
- **Gate 2** (after content launch): Is there organic demand? (>100 email signups)
- **Gate 3** (after 3 months paper trading): Does model match reality? (<1% tracking error)
- **Gate 4** (after optimiser fix + WF re-run): Does DE now beat/match manual? If yes → Pro tier feature.

## Validated strategies (for quick reference)

| Strategy | Allocation | Best Calmar | Worst Calmar |
|----------|-----------|-------------|-------------|
| 6asset_tip_gsg | SPY 15%, QQQ 15%, TLT 30%, TIP 15%, GLD 15%, GSG 10% | 0.476 | 0.405 |
| 7asset_tip_gsg_vnq | SPY 12%, QQQ 12%, TLT 25%, TIP 12%, GLD 13%, GSG 10%, VNQ 16% | 0.471 | 0.429 |
| 7asset_tip_djp | SPY 12%, QQQ 13%, IWD 8%, TLT 27%, TIP 13%, GLD 15%, DJP 12% | 0.432 | 0.334 |
| 5asset_dalio | SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, GSG 7.5% | 0.383 | 0.337 |

## How Francisco works

- Critical analysis valued over confirmation
- Strategy discussion in Claude Projects; code changes via Claude Code
- Iterative: run experiment → share output → analyse → refine
- Configuration-driven: all params in config.py
- GitHub repo: https://github.com/fcastelasimao/quant-learning