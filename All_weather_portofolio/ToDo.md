# All Weather Portfolio — TODO & Action Plan

Last updated: 2026-03-22 (post Phase 9 run)

Two parallel tracks run simultaneously:
- **Track A (Engineering):** Code fixes, experiments, analysis scripts
- **Track B (Market):** Demand validation, content, positioning

Priority key:
🔴 = Blocking (this week). 🟡 = Before live capital. 🟢 = Before product launch. 🔵 = Product roadmap.

---

## ✅ COMPLETED

### Engineering
- [x] config.py defaults to 6asset_tip_gsg (was demoted 8-asset)
- [x] _Tee captures stderr
- [x] Data validation after yfinance download
- [x] Four DE optimiser bugs fixed (bounds, normalisation, objective, projection)
- [x] Martin ratio added to StrategyStats, master log, optimiser output
- [x] Master log column order: Calmar/Martin first, config detail last
- [x] Master log auto-archives on first run
- [x] ASSET_BOUNDS extended for all four validated strategies (VNQ, IWD, DJP, IEF)
- [x] Optimise_random label fixed (Martin ratio vs Calmar)
- [x] BACKTEST_END = date.today() (dynamic)
- [x] compare_allw.py: $10k growth chart, fee drag chart, Excel export
- [x] Single chart output (1200×675) — dual reddit/twitter format removed
- [x] Spot-check experiments separated into VALIDATION_CHECKS list
- [x] 5asset_dalio added to ALLW comparison
- [x] Backtest ETFs vs live ETFs comparison — PDBC identified as divergence source

### Research
- [x] 33-universe Phase 9 run completed with fixed DE
- [x] ALLW live comparison: 6asset beats ALLW on all risk metrics
- [x] DE failure diagnosed as regime mismatch, not setup bugs
- [x] Manual allocation confirmed as production standard until split2022 validates

---

## 🔴 TRACK A — Immediate (this week)

### A1: ✅ Phase 9 complete — Gate 1 closed
Split2022 OOS result: 0.129 — worst result in the entire dataset.
DE confirmed to fail across all 26 experiments. Do not run further
DE experiments. The return-based optimisation chapter is closed.

### A2: Opus session — risk parity optimisation design (HIGHEST PRIORITY)
The most important next action. Do not implement anything without this.
Agenda for the session:
- Present Phase 9 complete results (all 26 experiments fail)
- Present WF median retraction (not a reliability signal)
- Present split2022 retraction (makes things worse)
- Discuss risk contribution equalisation as the replacement framework
- Question 1: implement risk parity objective directly in optimiser.py?
  (minimise Σᵢ Σⱼ (TRC_i - TRC_j)²)
- Question 2: use risk parity as structural baseline, constrained
  tactical layer on top?
- Question 3: DJP vs GSG in risk parity context (DJP showed better
  OOS stability in Phase 9)
- Question 4: covariance matrix estimation window sensitivity
- Question 5: how to achieve target CAGR without leverage?
Output required: a clear design spec before any implementation begins.

### A3: Fix rebalancing mismatch ⏱ 2 hours
Backtest rebalances monthly unconditionally. Live portfolio uses 5% drift
threshold. These are different strategies. Run comparison: threshold=0.0
vs threshold=0.05 for 6asset_tip_gsg. Quantify the tracking error.

### A3: Fix max drawdown to use daily data ⏱ 1.5 hours
Monthly MDD understates true intraday drawdowns. Add daily MDD computation.
Report both. All current OOS Calmar numbers are slightly overstated.

### A4: Complete crisis_analysis.py ⏱ 3 hours
Script briefed (see session notes). Runs buy-and-hold across 6 historical
crisis windows. Includes ALLW only for iran_war window. Produces Excel
summary and per-crisis charts. This is both research and content.

### A5: Move spot-check experiments to VALIDATION_CHECKS in run_experiment.py
Brief Claude Code. Already designed — just needs implementation.

---

## 🔴 TRACK B — Market validation (this week)

### B1: Choose a brand name ⏱ decision
Cannot use "All Weather" (Bridgewater trademark). Requirements: memorable,
available as .com domain, not financial-jargon-heavy, global appeal.
Candidates to consider: StormProof, FourSeasons, WeatherVane, Equinox,
AllTide, Steadfast. Check domain + trademark before deciding.

### B2: Write the ALLW comparison blog post ⏱ 4 hours
**Wait until:** split2022 OOS known, Rf fixed, daily MDD fixed, brand name
decided, landing page live. Target: 2-3 weeks from now.

Title: "ALLW vs DIY All Weather: Is the 0.85% worth it?"
Key data points:
- $10k growth chart (March 2025 to March 2026)
- Fee drag projection ($100k over 30 years)
- Worst month: -3.97% vs -6.97% (most legible risk metric for retail)
- Calmar: 2.61 vs 1.89
- Iran war as real-time stress test

Legal requirements for Bridgewater comparison:
- Add disclaimer: "ALLW is a registered trademark of Bridgewater Associates /
  State Street. Independent analysis, no affiliation."
- Past performance disclaimer required
- No forward-looking claims
- Frame as alternative for different investor type, not as attack
- Let numbers speak — never characterise ALLW as a bad product

LinkedIn strategy:
- Post methodology content first (IS/OOS discipline, walk-forward) —
  builds credibility with technical audience
- Then results content ($10k chart, worst month) — drives engagement
- Then product content (landing page, email signup) — drives conversions
- Technical readers (quant researchers, PMs) care about methodology
- Retail readers care about $10k chart and worst-month number

GitHub for job applications:
- Exclude: results/ folder, strategies.json, research_log.md,
  session_handoff.md
- Include: all engine code (backtest.py, optimiser.py, validation.py, etc.)
- Describe as: "quantitative portfolio research tool with strict
  out-of-sample validation methodology"
- Do not mention ALLW or commercial intent in job context

### B3: Build a landing page ⏱ 3 hours
Do not build before brand name is decided. Single page: $10k growth chart,
three-sentence value prop, email signup. Vercel or Netlify.
Success metric: 100 signups in 4 weeks.

### B4: FCA compliance review ⏱ research task
Monthly rebalancing instructions on subscription may constitute regulated
investment advice under FSMA 2000. Talk to a UK fintech lawyer before
any paid product launch. Research "execution-only" and "guidance vs advice"
distinction. This is a business-ending risk if ignored.

---

## 🟡 Before live capital (next 2-4 weeks)

### A6: GBP-adjusted backtest ⏱ 2 hours
Non-US customers hold USD-denominated ETFs. GBP/USD movements affect real
returns. Run full backtest with GBP conversion. Required for honest UK product
claims. Expect ~1-2% CAGR impact depending on FX trends.

### A7: Add risk-free rate to Sharpe and Sortino ⏱ 30 min
Rf = 0 assumption inflates both ratios significantly with Fed at 3.5-3.75%.
Use 3-month T-bill rate as Rf. All published Sharpe/Sortino numbers will
change — do not publish current numbers externally.

### A8: Bootstrap confidence intervals ⏱ 3 hours
Point estimates without error bars are misleading. Add bootstrap CI (95%)
for Calmar and CAGR. This is also a product differentiator — most tools
don't show uncertainty.

### A9: User research interviews ⏱ ongoing
10-15 conversations with DIY All Weather implementors. Questions:
- What allocation do you currently use?
- What tools do you use to track it?
- Do you know about ALLW? Would you buy it instead of DIY?
- Would you pay for monthly rebalancing instructions?
Find via Reddit r/Bogleheads, r/portfolios, Twitter/X #portfoliomanagement.

---

## 🟡 Analysis and robustness (next month)

### A10: Regime-conditional performance analysis ⏱ 4 hours
Break full backtest into macro regimes (rising/falling rates, high/low
inflation). Show Calmar per regime. This is the analytical foundation of
the All Weather thesis and currently unquantified.

### A11: Drawdown decomposition ⏱ 2 hours
Which asset caused each major drawdown? Which asset rescued it?
Visual decomposition chart. High content value.

### A12: Complete 3-window validation for Tier 2/3 strategies ⏱ 2 hours
7asset_tip_djp and 5asset_dalio each have only 2 OOS windows.
Three windows required for production promotion per Principle 5.

### A13: Inflation-adjusted returns ⏱ 1 hour
Real returns matter. Show CAGR net of CPI for each strategy.

---

## 🟢 Before product launch

### P1: Visualisation suite
Full chart library per visualisation_strategy.md. Regime heatmap,
drawdown decomposition, weight stability chart, rolling Calmar.

### P2: Trademark and brand name finalised

### P3: FCA/regulatory compliance confirmed

### P4: Minimum portfolio size analysis
Below what portfolio size do transaction costs make monthly rebalancing
uneconomic? Calculate the threshold. Important for customer segmentation.

### P5: Competitive positioning document (centred on ALLW)

---

## 🔵 Product roadmap

### P6: Web interface (FastAPI + React)
### P7: User accounts and data persistence
### P8: Monthly rebalancing reminders with drift alerts
### P9: Trend-following overlay (Pro tier)
### P10: Referral mechanism
### P11: Advisor tier (compliance-grade walk-forward documentation)
### P12: Optimised allocation Pro tier
If split2022 validates DE, offer DE-tuned weights updated quarterly.
Transparent, inspectable optimisation vs ALLW's opaque Bridgewater model.

---

## Rejected experiments (do not repeat)

### Universe failures — structural, not fixable by optimisation
| Experiment | OOS Calmar | Reason |
|---|---|---|
| 8asset_agg_replaces_qqq | 0.169 | Removes growth equity anchor. Universe failure. |
| 8asset_lqd_replaces_shy | 0.287 | LQD fails in rate shocks. |
| 7asset_vnq_reits (no TIP+GSG) | 0.151 | REITs without inflation anchor. |
| 7asset_6tip_plus_ief | 0.172 | Duration overlap, no new information. |
| 6asset_duration_ladder | 0.253 | Extreme overfit. No commodities. |
| 7asset_tip_gsg_dual | 0.162 | IEF+TIP overlap, unstable. |
| 8asset_tip_replaces_shy | 0.161 | Worst Phase 9 result. |
| 7asset_vnq_reits | 0.151 | Worst in group. |

### Split2022 experiments — do not repeat
All split2022 results are the worst in the table. The OOS window
2022-2026 is too short and too specific. Including 2022 in IS training
produces weights that fail the specific 4-year recovery/Iran war OOS.
| Experiment | OOS Calmar | Reason |
|---|---|---|
| 6asset_tip_gsg_split2022 | 0.129 | Worst result overall |
| 7asset_tip_gsg_vnq_split2022 | 0.140 | |
| 5asset_dalio_split2022 | 0.140 | WF median 2.000, OOS 0.140 — extreme overfit |
| 7asset_tip_djp_split2022 | 0.165 | |

### DE optimisation in general — do not repeat
Return-based DE optimisation (any objective) confirmed to fail across
26 experiments. Do not run further DE experiments until risk parity
framework is designed and implemented (Opus session required).