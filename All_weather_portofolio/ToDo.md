# All Weather Portfolio -- TODO List

Last updated: 2026-03-20

---

## 🔴 Phase 1: Code Audit Before Any New Experiments

Before running any new experiments, Claude Code should do a full audit
and debug pass on the codebase. This is the professional approach -- you
do not build on a foundation you have not verified. A clean, correct codebase
also makes results trustworthy and reproducible.

Prompt for Claude Code:

  Perform a full code audit of this quantitative finance backtesting project.
  Read every .py file. For each file:

  1. BUGS: Identify any logic errors, edge cases that could crash or silently
     produce wrong results, off-by-one errors, incorrect financial calculations,
     or inconsistencies between modules.

  2. CODE QUALITY: Identify dead code, unused imports, duplicated logic,
     functions that do too much, inconsistent naming, missing edge case handling.

  3. FINANCIAL CORRECTNESS: Specifically check:
     - Does the backtest engine correctly simulate monthly rebalancing?
     - Is the 60/40 benchmark correctly rebalanced annually?
     - Are transaction costs applied correctly and only to trades?
     - Is the tax drag applied at the correct time (start of new year)?
     - Does the walk-forward correctly prevent data leakage between windows?
     - Are there any lookahead bias issues in the backtest loop?

  4. PERFORMANCE: Identify bottlenecks given that walk-forward runs 4+
     DE optimisations and takes 30-60 minutes.

  5. KNOWN ISSUES TO FIX:
     - plotting.py line 102: s_aw, s_bh, s_spy, s_6040 = stats_list
       will crash if 60/40 is absent. Add a guard.
     - export.py still imports load_workbook but never uses it. Remove it.
     - export_copy.py is a leftover backup file. Remove it.
     - Walk-forward mean overfit ratio is distorted by outlier windows
       (individual ratios of 5-17x). Clamp to max 2.0 before averaging
       and report median alongside mean.
     - migrate_log.py is superseded by merge_master_logs.py.
       Archive or remove it.

  6. TESTS: The five new stat helpers have no tests:
     compute_avg_drawdown, compute_max_drawdown_duration,
     compute_avg_recovery_time, compute_ulcer_index, compute_sortino.
     Add at least 3 tests per function.

  Produce a summary of all findings before making any changes.
  Wait for confirmation before editing any files.

---

## 🔴 Phase 2: Remaining Pre-Paper-Trading Items

- [ ] Transaction costs sensitivity analysis
  Run three full_backtest runs with the 8-asset allocation:
    TRANSACTION_COST_PCT = 0.0    (baseline)
    TRANSACTION_COST_PCT = 0.001  (0.1% realistic UK retail)
    TRANSACTION_COST_PCT = 0.005  (0.5% traditional broker)
  Key question: does the strategy still beat 60/40 at 0.5% costs?

- [ ] Add README note on tax wrapper
  Add to Important Caveats: strategy is best inside ISA or SIPP.
  Monthly rebalancing triggers CGT in a taxable account.
  Guidance: TAX_DRAG_PCT = 0.0 for ISA/SIPP, 0.03-0.05 for taxable.

- [ ] Fractional share support in rebalancing instructions
  Add share count calculation to portfolio.py.
  Flag positions requiring fractional shares (< 1.0 shares) with a warning.

- [ ] Minimum portfolio size analysis
  Calculate the portfolio size below which monthly rebalancing costs
  exceed the benefit over buy-and-hold. Document this in the README.

---

## 🟡 Phase 3: Asset Universe Experiments

The right way to test new ETFs:
  One change at a time, always against the same baseline, IS/OOS split enforced.
  For each new universe:
    1. RUN_MODE = "backtest" (IS only, 2006-2020) -- compare IS Calmar vs 0.336
    2. If IS improves: RUN_MODE = "walk_forward" -- check overfit ratio
    3. If walk-forward acceptable: RUN_MODE = "oos_evaluate" -- single honest test
    4. Compare OOS Calmar vs current benchmark of 0.441
  Never add more than one new ETF at a time.

5-ASSET CONFIGURATIONS

- [ ] Dalio original (baseline reference -- run this first)
  SPY 30%, TLT 40%, IEF 15%, GLD 7.5%, GSG 7.5%
  Rationale: pure reference. How much does all the engineering add?
  Run full_backtest 2006-2026. No optimisation.

- [ ] TIP-based previous best (updated with total-return data)
  SPY 14.2%, QQQ 20.3%, TLT 30%, TIP 14.2%, GLD 21.3%
  Rationale: re-establish the old benchmark on the new data baseline.

6-ASSET CONFIGURATIONS

- [ ] 6-asset: TIP model + GSG (bridge between 5 and 8 asset)
  SPY 15%, QQQ 20%, TLT 30%, TIP 10%, GLD 15%, GSG 10%
  Rationale: does adding commodities alone improve the 2022 stress test
  without the complexity of the full 8-asset model?

- [ ] 6-asset: duration ladder without inflation protection
  SPY 15%, QQQ 20%, TLT 25%, IEF 15%, GLD 15%, GSG 10%
  Rationale: two-rung bond ladder. Cleaner than TIP which mixes
  duration and inflation protection in one instrument.

- [ ] 6-asset: add international equity
  SPY 15%, QQQ 10%, EFA 10%, TLT 30%, TIP 10%, GLD 25%
  Note: EFA inception June 2001, backtest from 2006 is fine.
  Rationale: tests geographic diversification vs US concentration.

7-ASSET CONFIGURATIONS

- [ ] 7-asset: current 8-asset minus GSG (test commodity contribution)
  SPY 10%, QQQ 15%, IWD 12%, TLT 28%, IEF 12%, SHY 5%, GLD 18%
  Rationale: if removing GSG improves results, it should not be in the
  portfolio regardless of the stagflation thesis.

- [ ] 7-asset: current 8-asset minus IWD (test value tilt contribution)
  SPY 15%, QQQ 15%, TLT 25%, IEF 10%, SHY 5%, GLD 15%, GSG 15%
  Rationale: IWD was consistently minimised by DE. Does it contribute
  anything beyond what SPY provides?

- [ ] 7-asset: TIP + GSG dual inflation hedge
  SPY 12%, QQQ 15%, TLT 25%, TIP 12%, SHY 5%, GLD 15%, GSG 16%
  Rationale: TIP (inflation-linked bonds) + GSG (commodities) as
  complementary inflation hedges. Tests whether the combination
  provides superior 2022 protection.

8-ASSET CONFIGURATIONS

- [ ] 8-asset: replace IWD with EFA (geographic diversification)
  SPY 10%, QQQ 15%, EFA 10%, TLT 25%, IEF 10%, SHY 5%, GLD 15%, GSG 10%
  Rationale: geographic diversification vs domestic value tilt.

- [ ] 8-asset: replace SHY with TIP (dual inflation hedge)
  SPY 10%, QQQ 15%, IWD 8%, TLT 25%, IEF 10%, TIP 8%, GLD 14%, GSG 10%
  Rationale: SHY is largely redundant when IEF covers intermediate
  duration. TIP adds direct inflation protection.

---

## 🟡 Phase 4: Winning Strategies Registry

The professional approach: validated strategies are stored in a curated
registry file separate from the master log. The master log is a raw run
history. The registry contains only strategies that have passed IS + OOS
validation and are safe to implement live.

- [ ] Create strategies.json in the project root

  Structure per entry:
    id, description, status, validated_date, universe, pricing_model,
    backtest_start, oos_start, backtest_end, allocation,
    asset_class_groups, asset_class_max_weight,
    is_metrics (cagr, max_drawdown, calmar, ulcer, sharpe, sortino),
    oos_metrics (same),
    walk_forward (mean_overfit_ratio, windows_opt_beat_original, verdict),
    notes, results_folder

  Status values:
    "candidate"  -- IS only, not yet OOS tested
    "validated"  -- IS + OOS passed, safe to implement
    "rejected"   -- OOS failed, do not use
    "superseded" -- beaten by a later strategy, archived for reference

  Only "validated" entries should be considered for live implementation.

- [ ] Add load_strategy() helper to config.py
  Loads a validated strategy from strategies.json into config globals.
  This eliminates copy-pasting weights between experiments.
  Usage: set strategy_id = "8asset_manual_v1" and all allocation,
  caps, and dates load automatically.

- [ ] Populate strategies.json with the two currently validated strategies:
    "5asset_tip_v1" -- IS Calmar 0.560, OOS Calmar 0.378
    "8asset_manual_v1" -- IS Calmar 0.336, OOS Calmar 0.441

---

## 🟡 Phase 5: Paper Trading Setup

- [ ] Choose broker and open paper trading account
  Requirements: supports all 8 ETFs or UK equivalents, paper trading
  with real market prices. Recommended: Interactive Brokers (IBKR).
  Check MiFID II restrictions first.

- [ ] Resolve MiFID II: identify UK-listed equivalents if needed
  SPY -> CSPX, QQQ -> EQQQ, TLT -> IDTL, IEF -> IBTM,
  SHY -> IBTS, GLD -> IGLN, IWD -> IWVL (approximate), GSG -> CMOD (approximate)
  Run spot-check backtest US vs UK equivalents to confirm tracking error < 0.5%.

- [ ] Define paper trading protocol
  At month-end: run oos_evaluate to get instructions, execute within 2 days.
  Log actual fill prices vs model prices to measure slippage.
  Go/no-go for live money: 3+ months paper trading, < 1% cumulative tracking error.

---

## 🔵 Phase 6: Product

- [ ] Web interface (FastAPI + React) -- ~80% of remaining work before sellable
- [ ] User accounts and data persistence (SQLite -> PostgreSQL)
- [ ] Automated monthly reminders
- [ ] Legal disclaimers and FCA compliance review

---

## ✅ Completed

- [x] Modular 9-file architecture
- [x] StrategyStats dataclass with 11 metrics
- [x] 9 stat helpers including Ulcer, Sortino, AvgDD, MaxDDDuration, AvgRecovery
- [x] Four optimisation methods with asset class group caps
- [x] Walk-forward validation with overfit ratio chart
- [x] Pareto frontier analysis
- [x] IS/OOS boundary enforcement via RUN_MODE + OOS_START
- [x] 60/40 as fourth strategy throughout
- [x] Three-panel backtest chart (value, annual returns, B&H allocation drift)
- [x] Excel master log with 10 metrics per strategy, grouped headers
- [x] Transaction cost modelling (TRANSACTION_COST_PCT)
- [x] Tax drag modelling (TAX_DRAG_PCT, blunt annual drag)
- [x] Total return prices via auto_adjust (PRICING_MODEL)
- [x] Verified: all historical runs were already total return (yfinance default)
- [x] PRICING_MODEL, Tx Cost %, Tax Drag % logged to master log and run_config.json
- [x] Bug fixed: double-rebalance in backtest.py
- [x] merge_master_logs.py: merged 66 old + 11 new rows
- [x] test_data.py: integration and unit tests for total return and cost modelling
- [x] GitHub repository
- [x] 8-asset experiment sequence complete: IS -> optimise -> walk-forward -> OOS -> full
- [x] 8-asset OOS Calmar 0.441 vs 60/40 OOS Calmar 0.263 (validated)
- [x] Confirmed DE optimiser does not reliably beat manual allocation