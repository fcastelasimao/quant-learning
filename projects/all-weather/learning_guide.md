# All Weather Portfolio — Learning Guide

A plain-English explanation of everything this project does, with the maths
shown step by step and grounded in our actual results.

---

## Table of Contents

1. [The Big Idea](#1-the-big-idea)
2. [The Assets and What They Do](#2-the-assets-and-what-they-do)
3. [The Four Economic Regimes](#3-the-four-economic-regimes)
4. [What a Backtest Is](#4-what-a-backtest-is)
5. [The Metrics That Matter](#5-the-metrics-that-matter)
6. [IS/OOS Discipline — Why It Matters](#6-isoos-discipline--why-it-matters)
7. [The Optimiser — What We Tried and Why It Failed](#7-the-optimiser--what-we-tried-and-why-it-failed)
8. [Risk Parity — What It Is and What We Found](#8-risk-parity--what-it-is-and-what-we-found)
9. [The Data Integrity Lesson](#9-the-data-integrity-lesson)
10. [RP-Based Asset Selection — The Next Frontier](#10-rp-based-asset-selection--the-next-frontier)
11. [The Research Map — Where We Are](#11-the-research-map--where-we-are)

---

## 1. The Big Idea

The All Weather portfolio was designed by Ray Dalio at Bridgewater Associates.
The core insight:

> **No one knows what the economy will do next. So build a portfolio that
> survives all conditions.**

Traditional portfolios are heavily weighted in stocks. When the economy grows,
they do well. When it contracts, they get destroyed.

All Weather spreads risk across four economic regimes: growth up, growth down,
inflation up, inflation down. You always have something doing well, so the
total portfolio never collapses.

**What makes our version different from Dalio's original:**
- Dalio uses leverage (~2x on bonds) to equalise risk. We do not.
- We replaced pure bonds with a mix including TIP (inflation-linked) and
  added GSG (commodities).
- We use Risk Parity maths to derive weights from first principles, rather
  than choosing them by hand.

---

## 2. The Assets and What They Do

Each ETF is a basket of securities you can buy like a stock.

| ETF | What it holds | What environment it likes |
|-----|---------------|--------------------------|
| **SPY** | 500 largest US companies | Strong economy, rising profits |
| **QQQ** | Top 100 tech/growth companies | Strong economy, falling rates |
| **TLT** | Long US government bonds (15+ yr) | Recession, falling rates |
| **TIP** | Bonds that adjust for inflation | Inflation, mild rate rises |
| **GLD** | Physical gold | Inflation, crisis, USD weakness |
| **GSG** | Oil, metals, agriculture | Rising inflation, supply shocks |

**Why TLT and TIP together?** TLT bets rates will fall. TIP bets inflation
will rise. In 2022, rates rose sharply: TLT fell -31%, TIP only -9%. Having
both means you're not fully exposed to one side of the rate question.

---

## 3. The Four Economic Regimes

| Regime | What wins | What loses |
|--------|-----------|------------|
| Rising growth + falling inflation | SPY, QQQ | GLD, GSG |
| Rising growth + rising inflation | GSG, TIP | TLT |
| Falling growth + falling inflation | TLT, GLD | SPY, QQQ |
| Falling growth + rising inflation (stagflation) | GSG, GLD, TIP | SPY, TLT |

Stagflation (2022, arguably 2026 Iran war) is where most portfolios break.
Commodities and inflation-linked bonds are the only assets that reliably
work in stagflation — this is why they're in the portfolio.

---

## 4. What a Backtest Is

A backtest simulates "what would have happened if I'd held this portfolio
in the past?" using real historical prices.

Our backtest:
1. Starts with $10,000 on the first trading day
2. Buys each ETF according to target weights
3. At month-end, rebalances back to target weights
4. Records portfolio value at each step
5. Computes performance metrics

**Total return** includes dividends and interest. **Price return** ignores
them. We always use total return (TLT price return: -5%, total return: +79%).

---

## 5. The Metrics That Matter

We track six metrics. Here's what each means for our goal: "sleep at night,
don't lose your savings."

### 5.1 CAGR — Compound Annual Growth Rate

"What's my average annual return?"

    CAGR = (Final Value / Starting Value)^(1/Years) - 1

Example: $10,000 → $41,000 over 20 years = 7.3%

### 5.2 Max Drawdown (daily)

"What's the worst peak-to-trough loss?" The scariest number for any investor.
$100,000 → $78,000 = max drawdown of -22%. Computed on daily prices because
monthly readings can mask worse intra-month drops.

### 5.3 Calmar Ratio — primary metric

"How much return per unit of worst-case pain?"

    Calmar = CAGR / |Max Drawdown|

Example: 7.5% / 19.6% = 0.38. Higher is better. SPY ≈ 0.21 — double the
drawdown per unit of return.

### 5.4 Ulcer Index — sustained pain

"How bad was it overall to be underwater?" Captures both depth and duration
of all drawdown periods.

    For each step: d_t = (Value_t - Peak) / Peak × 100
    Ulcer = √(mean(d_t²))

A -5% drawdown lasting 36 months scores worse than -20% for 1 month.
The sustained grind is what makes people sell at the bottom.

### 5.5 Martin Ratio — smooth Calmar

    Martin = CAGR / Ulcer Index

Used for optimisation because Ulcer is smooth (good for numerical search)
while Max DD jumps discontinuously (bad for optimisers). Calmar is the
reporting metric; Martin drives the maths.

### 5.6 Sortino Ratio — downside risk only

    Sortino = (Return - Rf) / Downside Deviation

Like Sharpe but only penalises downside volatility. A +20% month is a
feature, not a bug.

**Why we dropped Sharpe:** Sharpe penalises all volatility equally. For a
defensive portfolio, upside volatility is good. Sharpe says otherwise.

---

## 6. IS/OOS Discipline — Why It Matters

The most important concept in the project.

### The three-date boundary

    2006 ──────────── 2020 ──────────── Today
         IN SAMPLE          OUT-OF-SAMPLE
      Optimise here.       Evaluate once.
      Look freely.         Never tune.

### Why it matters

Try 1000 random weight combinations on 20 years. Some look amazing by luck,
not skill. IS/OOS prevents this: develop on one period, test on another.

### Multiple OOS windows

One test can be lucky. We use three splits:

| Split | IS period | OOS period | OOS captures |
|---|---|---|---|
| 2020-split | 2006–2020 | 2020–today | COVID, rate shock, Iran war |
| 2018-split | 2006–2018 | 2018–today | Late bull, COVID, rate shock |
| 2022-split | 2006–2022 | 2022–today | Post-crash recovery, Iran war |

A strategy must hold across all three.

---

## 7. The Optimiser — What We Tried and Why It Failed

### What DE does

Differential Evolution searches for the best weights by evolving a population
of candidates. It maximises Martin ratio on IS data.

### Why it failed

26 experiments with fully fixed DE. Zero beat manual's OOS Calmar of 0.403.

**Root cause: regime mismatch.** IS (2006–2020) = falling rates. DE learns
"lots of TLT." OOS (2020–2026) = rate spike. TLT collapses. No matter where
you draw the line, IS and OOS contain different regimes.

### The four bugs we fixed first

1. Bounds: uniform bounds prevented reaching good allocations
2. Normalisation: distorted DE search space
3. Objective sign: DE minimises; needed negation
4. Projection: post-convergence correction moved the result

After all fixes, DE still failed → structural, not a coding error.

---

## 8. Risk Parity — What It Is and What We Found

### The concept

Standard optimisation: "what weights maximise return for a risk level?"
Risk Parity: **"what weights make each asset contribute equally to risk?"**

### The maths

**Step 1:** Compute covariance matrix Σ from daily returns.

**Step 2:** Risk contribution of asset i with weights w:

    RC_i = w_i × (Σw)_i / (wᵀΣw)

**Step 3:** Solve for equal contributions:

    Minimise: Σᵢ Σⱼ (RC_i - RC_j)²
    Subject to: w_i ≥ 0, Σw_i = 1

Convex and smooth — SLSQP handles it well.

### What we found

| Asset | Manual | RP 5yr | Difference |
|---|---|---|---|
| TLT | 30% | 17% | -13% |
| TIP | 15% | 33% | +18% |
| QQQ | 15% | 10% | -5% |
| SPY | 15% | 13% | -2% |
| GLD | 15% | 14% | -1% |
| GSG | 10% | 13% | +3% |

RP OOS Calmar: 0.512 vs manual 0.403 (+27%). But one window only.

### Why RP beats DE

- DE optimises returns over a specific period → regime-dependent
- RP optimises risk balance from covariance → regime-independent
- Covariance changes slowly; return regimes change abruptly
- RP is principled ("equal risk") vs DE is data-mined

---

## 9. The Data Integrity Lesson

### What went wrong

During Phase 9, we ran experiments with DE-optimised weights and recorded OOS
results. Later, we chose different manual weights as the production standard.
But we accidentally attributed the DE results to the manual allocations.

Example: 6asset manual weights (SPY 15%, TLT 30%) give OOS Calmar 0.403.
But strategies.json claimed 0.476, which came from DE weights (QQQ 25%,
TLT 25%) — a completely different portfolio. The 7asset_vnq manual allocation
was never OOS-tested at all.

### The lesson

This is exactly what kills credibility. If we published "Calmar 0.476" and
someone reproduced it, they'd find 0.403 and dismiss everything.

**Rule:** every OOS number in strategies.json must trace to a master_log row
using the exact same weights. No exceptions.

---

## 10. RP-Based Asset Selection — The Next Frontier

### Current limitation

We hand-pick assets (macro intuition), then use RP for weights. Asset
selection itself is still subjective.

### The idea: let covariance guide selection

Start with 15-20 ETFs across all macro quadrants:

| Role | Candidates |
|---|---|
| US equity | SPY, QQQ, IWD |
| International | EFA, EEM |
| Long bonds | TLT |
| Intermediate bonds | IEF, SHY |
| Inflation bonds | TIP |
| Corporate bonds | LQD, AGG |
| Gold | GLD |
| Commodities | GSG, DJP, PDBC, COMT |
| REITs | VNQ |
| Alternatives | BTAL, TAIL |

Compute full covariance matrix. Select the 6-8 asset subset that, combined
with RP weights, gives the best risk diversification.

### Selection criteria

**Maximum diversification ratio:** ratio of weighted-average asset volatility
to portfolio volatility. Higher = assets cancel each other's risk more
effectively. Uses only covariance, no returns → regime-resistant.

**Minimum Ulcer:** backtest each subset with RP weights, pick lowest Ulcer
on IS data. Uses returns, but RP constrains weights so less overfit-prone.

**Correlation filter:** for each macro role, pick the ETF with lowest average
correlation to the others. Avoids redundancy (LQD ≈ TLT → don't include both).

### What this means for the product

If it works, the full methodology becomes systematic:
1. Define candidate universe
2. Compute covariance matrix
3. Select optimal subset (diversification ratio)
4. Compute RP weights
5. Rebalance monthly

Every step is repeatable, transparent, updatable. That's the defensible
methodology that competes with Bridgewater.

### The numbers that matter

    OOS Calmar 0.403 — manual 6asset, 2020-split (verified)
    OOS Calmar 0.512 — RP 5yr 6asset, 2020-split (one window, promising)
    Everything else needs re-validation.

    # Learning Guide — Addendum (2026-03-26)

Add these sections to the end of learning_guide.md, before the Research Map.

---

## 11. The Overlay Experiment — Why Trend-Following Didn't Help

### The idea

Use SPY as a crash detector. When SPY drops more than X% from its peak,
AND momentum is negative (D1 < 0), AND momentum is accelerating downward
(D2 < 0) — exit the SPY position. Re-enter when momentum turns positive
or price recovers to the exit level.

The logic is sound: don't try to predict the top, but once a crash is
clearly underway (confirmed by three independent signals), get out.

### What we tested

126 parameter combinations:
- D_window: 5, 10, 15, 20, 30, 40, 60 trading days
- Threshold: 5%, 8%, 10%, 12%, 15%, 20% drawdown
- Reduce_pct: 50%, 75%, 100% of SPY position

110 out of 126 beat the baseline on IS data. Best: d=15, threshold=12%,
reduce=100% (full exit when SPY is down 12%+ with 3-week negative momentum).

### What happened on OOS

| Split | Baseline | Overlay | Result |
|---|---|---|---|
| 2020 | 0.452 | 0.458 | +1.3% |
| 2018 | 0.450 | 0.456 | +1.3% |
| 2022 | 0.359 | 0.340 | -5.3% |

The overlay wins marginally on two windows but loses meaningfully on the
hardest one — the 2022 split, where the crash happens right at the start
of OOS.

### Why it fails

The exit works fine — the overlay correctly identifies crashes and reduces
exposure. The problem is re-entry. The conditions for re-entering (D1 > 0
AND D2 > 0) require sustained positive momentum, which only appears after
the market has already recovered 5-10% from the bottom. By the time you're
back in, you've missed the sharpest part of the rebound.

In the 2022-split, the crash happens early. The overlay exits SPY during
the drawdown (saving perhaps 3-5%), but then misses the 2023-2024 recovery
rally. Net result: worse than just staying invested.

### The lesson

For a buy-and-hold defensive portfolio, the RP allocation already limits
max drawdown to ~5-6% (vs ALLW's 8.8%). An overlay that saves another
percentage point in some windows but costs you in others adds complexity
without reliable improvement. The risk-adjusted cost of the overlay
(implementation risk, additional trades, missed recovery) exceeds its
risk-adjusted benefit.

This doesn't mean overlays never work — it means they don't add value
on top of a well-diversified RP portfolio for our asset universe and
rebalancing frequency.

---

## 12. The Research Map — Final State

```
Phase 1-8:   Universe exploration → 6asset_tip_gsg
Phase 9:     DE optimiser → 26 experiments, all fail → CLOSED
Phase 10A:   Fixes (Rf, daily MDD, RP function)
Phase 10B-D: Risk parity → TLT overweighted, RP OOS +27%
Phase 11:    Validation sweep:
             → RP multi-window: beats manual on all 3 splits ✓
             → Universe scan: confirms 6-asset choice ✓
             → Overlay grid: does not add value ✗ → CLOSED
             → ALLW comparison: Calmar 2.78 vs 1.78 ✓
             → Data integrity: all claims verified ✓

RESEARCH COMPLETE.

Next: Blog post → Landing page → Paper trading → Demand signal
```

### The production numbers

    Allocation: SPY 13%, QQQ 11%, TLT 19%, TIP 33%, GLD 14%, GSG 10%
    OOS Calmar: 0.480 / 0.462 / 0.385 (across 3 splits)
    vs ALLW:    Calmar 2.78 vs 1.78, MDD -5.7% vs -8.8%, fees 0.12% vs 0.85%

Every number above is traceable to a specific master_log row using the
exact weights listed. No borrowed results, no wrong allocations.