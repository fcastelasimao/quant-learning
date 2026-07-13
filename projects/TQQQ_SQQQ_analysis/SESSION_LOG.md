# Session log

Append-only "what got done when". One entry per substantive session, two lines each.

---

## 2026-05-29

- Built items 01–17 of the exploratory research pass; SYNTHESIS.md consolidated; README.md created; FEATURE_DICTIONARY.md extended; GLOSSARY.md created.
- Completed bug-review + tests + session-continuity pass. Created SESSION_HANDOFF.md + SESSION_LOG.md + `tests/` (33 tests, 1 failure caught a real overstatement in the SYNTHESIS GBM claim, now fixed) + `research/CODE_REVIEW.md`. Applied 4 will-fix items (item 11 bounded split, item 13 NaN check idiom, item 16 FOMC count comment, item 04 sklearn-version docstring). Added Depends-on docstring blocks to items 05, 06, 11, 14, 15, 16, 17. CLAUDE.md updated to point at SESSION_HANDOFF first on resumption.
- Active deployment target unchanged: item 17 `linear_skip_enriched_1pct`. Next priority: productionize into `archive/full_history_feature_scan.py` (Phase C4).

## 2026-06-09

- Fixed item-06 daily-context causality: same-day daily bars are now disallowed and enriched rows require `context_date < entry_date`. Rebuilt item 06 and item 17 under strict-prior context, added calibration diagnostics, and superseded the old same-day-context deployment headline.
- Added `p_severe_scorer/` with documented feature contract, annual JSON artifacts, sklearn-free artifact scoring, in-memory yearly checkpoint training/scoring, both-symbol support, examples, and tests. Current status: research/validation tooling, not final production sizing.
- Expanded docs and tests for the hardening pass. Current test suite: 46 passing tests in the `quant` conda env.

## 2026-06-10

- Added `> Period:` date-range headers to all 13 findings files missing them (items 01–06, 08–09, 11, 13–16).
- Added walk-forward permutation importance (`permutation_importance_oos()`) and pruning (`prune_and_refit()`) to `build_06_context_enrichment.py`. Key results: TQQQ top features `atr_pct` (+0.021), regime dummies, `hour_of_entry`, `RSI_entry`; SQQQ top `RSI_entry` (+0.020), `VIX_pctile_252d`. Pruned to 13 (TQQQ) and 19 (SQQQ) features. `HYG_LQD_ratio` negative delta for both — corrected. Updated findings_06 with full ranked importance tables and pruned feature sets.
- Created `deployable_strategies/` package: `metrics.py` (constant_notional_metrics), `continuous_sizing/sizing.py` (5 sizing functions + SIZING_FUNCTIONS dict), `focus_rules/sqqq_rsi_atr_skip.py`, `regime_rules/tqqq_sideways_skip.py`, `README.md` (colleague-facing integration guide). Moved `p_severe_scorer/` into `deployable_strategies/continuous_sizing/`.
- Built items 18 (combined strategy), 19 (SQQQ target grid), 20 (cross-symbol signal) with build scripts, CSVs, plots, and findings. Item 18 headline: TQQQ combined Sharpe 4.27 MaxDD −10.0%, SQQQ combined Sharpe 2.63 MaxDD −15.2%; zero rule/p_severe overlap — components genuinely complementary. Item 19: best Calmar −1% sqrt_skip (6.60), best Sharpe −2% sqrt_skip (2.62). Item 20: own-symbol AUC (0.593/0.581) beats cross (0.576/0.548) — keep TQQQ/SQQQ separate.
- Created `research/_walkforward.py` shared helpers; refactored items 12/17 to import from it. Added `tests/test_deployable_strategies.py` (14 tests) and post-run headline tests for items 18–20. Full test suite: 65 passing tests.
- Updated all project-level MDs: SYNTHESIS.md (full rewrite with 7 sections + actual results), README.md, CLAUDE.md, SESSION_HANDOFF.md (full rewrite), FEATURE_DICTIONARY.md (pruned feature sets + new outcome columns section).
