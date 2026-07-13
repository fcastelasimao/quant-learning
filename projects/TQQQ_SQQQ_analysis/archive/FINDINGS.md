# Project Findings — Lab Notebook

> **⚠ This file is the lab notebook for the ORIGINAL `archive/full_history_feature_scan.py` pipeline runs.**
> **For the newer exploratory research pass (items 01–17), read `research/SYNTHESIS.md`.**
> **For session-resumption state (what's done, what's next, decisions made), read `SESSION_HANDOFF.md` at project root.**
>
> Historical note: the 2026-05-29 research-pass claim that item 17 was a two-symbol enriched-sizing Pareto improvement is superseded. On 2026-06-09, item 06 was corrected to forbid same-day daily context (`context_date < entry_date`), and item 17 weakened materially. The strict-prior `p_severe` scorer is research/validation tooling, not a final production sizing rule.

Living document of substantive findings as the full-history pattern discovery project progresses. Loaded into context for any future session. **Not a plan or spec** — those live in `PROJECT_PLAN.md`. Findings are dated. Stale or superseded findings are deleted, not buried.

Legacy notebook from the paused RSI leverage-overlay project lives at `archive/LEGACY_FINDINGS.md`. None of its specific numbers transfer to the new 2013–2026 dataset.

---

## Headline takeaways (read first)

- **SQQQ's median trade loses money** (median pnl_pct = −0.74 %, mean = +0.84 %). The strategy is a right-skewed, few-big-winners profile — classic short-bias on a 3× inverse ETF. Loser-filter rules matter much more for SQQQ than for TQQQ.
- **TQQQ is closer to symmetric** (median +0.59 %, mean +0.86 %, loser rate 44 %).
- **Exactly one pre-trade rule survives OOS**: SQQQ `rsi_x_atr_cell_3_1` — fires when `RSI_entry ∈ [56.4, 59.85]` and `atr_pct ∈ [0.39, 0.47]`. OOS precision 87.5 % on 8 trades, 95 % bootstrap CI on net pnl impact = [+2.43, +8.45]. Small absolute edge (~1 pp/year over the OOS window) but statistically credible.
- **TQQQ has no surviving rule.** Every candidate flagged a net-negative impact when applied OOS across 2021–2026.
- **18 of 19 SQQQ candidate rules fail OOS.** Classic over-fitting of bucket discovery on IS.
- **Absolute baseline Sharpe / CAGR numbers are inflated** by a data quality issue (see data finding #2). Use them only for *relative* comparisons within the same run, not as headline strategy performance.

## Data semantics (load-bearing, applies to the new 2013–2026 dataset)

1. **`pnl_pct` is in percentage points** (×100). `2.5` means +2.5 %. Divide by 100 in any return formula. Universal CSV convention; do not double-scale.
2. **The six CSVs in `full_history_canonical/trades_backtest/` are separate backtest runs concatenated.** Each starts capital at $10k. `capital_before` resets at year boundaries when CSVs were joined, so:
    - Per-trade `pnl / capital_before` is correct *within* a CSV but is meaningless to chain across the boundaries. The `total_return_chain` column in `traditional_metrics_baseline.csv` (~$1.29 × 10⁸ for TQQQ over 13 years) is artificially inflated by these resets.
    - The daily-equity Sharpe / CAGR / MaxDD numbers from `performance_metrics` (Sharpe 4.0 / CAGR 285 % for TQQQ, Sharpe 2.17 / CAGR 180 % for SQQQ) are less affected but still inherit the per-trade scaling, so treat absolute values with skepticism.
    - **Fix candidate:** track each row's `source_file`, chain returns within each CSV separately, or normalize all trades to a constant notional notional. Listed as an open question below.
3. **TQQQ and SQQQ are always analyzed separately.** Do not pool.
4. **In-sample window** is hardcoded as `entry_time <= 2020-12-31` (`IS_END`). Out-of-sample is 2021–2026.
5. **Daily context for intraday entries must be strict-prior**: `context_date < entry_date`. Same-day daily bars are look-ahead leakage because the daily bar is not complete at entry time. This correction supersedes the original item-06/item-17 enriched-context headline.

## Full-history feature scan — first run (2026-05-28, run `20260528_1208_feature_scan`)

5. **Per-symbol baseline metrics:**
    | Symbol | n | mean pnl | median pnl | loser % | severe-loss % | Sharpe (daily) | CAGR (daily) | MaxDD |
    |---|---|---|---|---|---|---|---|---|
    | SQQQ | 1930 | +0.84 | −0.74 | 54.7 | 39.0 | 2.17 | 180 % | −20.4 % |
    | TQQQ | 2343 | +0.86 | +0.59 | 44.0 | 29.8 | 4.01 | 285 % | −16.2 % |
    Caveat: absolute Sharpe/CAGR inflated per data finding #2.
6. **Surviving rule: SQQQ `rsi_x_atr_cell_3_1_high_loser_rate`.** RSI ∈ [56.4, 59.85] × atr_pct ∈ [0.39, 0.47]. IS precision 78 %, OOS precision 87.5 %, OOS trigger rate 1.1 % (8 trades), OOS net pnl impact +6.35 pp, bootstrap 95 % CI [+2.43, +8.45]. Per-year OOS: 2022/2023 had zero triggers, 2024–2026 had 100 % precision on the 6 flagged trades. See `full_history_research/20260528_1208_feature_scan/05_validation/focus_SQQQ_rsi_x_atr_cell_3_1_high_loser_rate.md`.
7. **TQQQ has zero surviving rules.** Closest is `rsi_x_atr_cell_3_0` but it gives back −8.3 pp in 2021 alone, swamping the 2024/2025 positives.
8. **Top features by combined-score rank are ATR-centric**, not RSI-centric:
    - TQQQ: `atr_pct` (0.21), `atr_pct_roll_pctile_252` (0.19), `atr` (0.12), then MA slope features.
    - SQQQ: `atr_pct` (0.24), `atr_pct_roll_pctile_252` (0.19), `dist_to_MA20` (0.12).
    Notable: Spearman is positive for TQQQ `atr_pct` but negative for SQQQ `atr_pct` — the two symbols read volatility in opposite directions.
9. **`regime_entry` is populated for both symbols** in the new data (unlike the old TQQQ canonical). SQQQ has a `sideways_lowvol` regime that fires 32 % of the time, but using it as a loser filter loses heavily OOS (−161 pp).
10. **Engine fixes that landed alongside this run** (1208):
    - Random-baseline filter is now reproducible (single `np.random.default_rng(RANDOM_SEED)` shared across rules).
    - Edge-bucket OOS coverage extended to ±∞ via `_bucket_mask`.
    - `performance_metrics` rewritten to compound returns onto a business-day calendar before annualizing — replaces the prior per-trade-as-daily annualization which was inflating Sharpe.
    - `walk_forward_yearly_rule_checks.csv` now contains real per-year rows instead of an IS-vs-OOS drift summary.
    - Focus rule validation module added (period breakdown, bootstrap CI, flagged-trades dump, MD summary) for the one surviving rule.

## Open questions

- **Fix the multi-CSV `capital_before` reset** so headline baseline metrics are trustworthy. Until then, all absolute Sharpe / CAGR / total-return numbers in the project should be read as upper bounds.
- **Is the focus rule's narrow feature region stable**, or is it a 2024-2026 artifact? Bootstrap CI is positive on 8 trades, but six of those are clustered in the last 18 months of the window. A block-by-year bootstrap or a forward-looking probe (when fresh data arrives) would settle it.
- **Why does TQQQ have no surviving loser rule** but SQQQ has one? Hypothesis: TQQQ loser trades happen on bullish gaps that flip the trailing stop — those are not predictable from pre-trade indicators. SQQQ loser trades cluster around the entry-stop band for moderate-vol regimes, which *is* predictable.
- **Should regime-conditional analysis be done before or after the leakage filter?** The current pipeline includes `regime_entry` as a pre-trade label but flags causality as unconfirmed. Worth tracing back to the source strategy to verify.
