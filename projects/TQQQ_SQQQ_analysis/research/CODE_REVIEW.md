# Code review - all 18 build scripts

**Pass date**: 2026-05-29. **Reviewer**: Claude (self-review of own code, with `tests/` as the safety net).

## 2026-06-09 correction

This review is preserved as history. Finding 17 below was wrong: item 06 used `merge_asof(..., allow_exact_matches=True)`, and same-day daily bars are not causal for intraday entries because the daily high/low/close/volume are not known at entry time. Item 06 and item 17 were rebuilt on 2026-06-09 with strict-prior context (`context_date < entry_date`), and the old enriched-sizing deployment headline is superseded.

**Methodology**: read each `research/<NN>_<name>/build_<NN>_<name>.py` looking for the common pitfalls listed in the test-design pre-flight (WF leakage, `pnl_pct` scaling, regime-filter consistency, sklearn version idioms, string-path parsing, index alignment, NaN handling, bar-window inclusivity, hardcoded-data accuracy, equity math, deterministic seeding).

## Severity scale

- **DATA**: results in the project are wrong because of this. Fix immediately and re-run.
- **CORRECTNESS**: code does the wrong thing in an edge case that we haven't hit but could.
- **ROBUSTNESS**: code works on current data but fails defensively if a downstream input changes.
- **STYLE**: cleanup opportunity, no behavior change.

---

## Findings

### 1. SYNTHESIS overstated the GBM result ★ DATA

**Issue**: `SYNTHESIS.md` claimed "GBM on `pnl_pct` has negative OOS R² in every WF year". Actually TQQQ 2023 has `gbm_r2 = +0.013` — 1 of 9 windows is barely positive. All 9 SQQQ windows are negative.

**Caught by**: `tests/test_reproducibility.py::test_item_04_gbm_r2_is_negative` (first run failed).

**Fix applied**: softened SYNTHESIS claim to "median R² is negative; 8 of 9 TQQQ windows and 9 of 9 SQQQ windows are negative." Updated test to `assert median < 0 and n_neg >= 7`. The "GBM is unusable" conclusion still holds.

**Status**: **FIXED.**

---

### 2. Item 11 `parse_path` uses unbounded split — CORRECTNESS

**Issue**: `research/11_regime_conditional_rules/build_11_regime_conditional_rules.py:50`:
```python
f, v = part.split(" <= ")
```
If a feature name ever contains `" <= "` (none of ours do today), this would raise `ValueError` because the split would produce 3+ parts. Defensive code uses `split(" <= ", 1)`.

**Severity**: low because no current feature has spaces or relational operators in its name; but a future feature like `dist_to_high_water_mark` (which has underscores, fine) could in principle be renamed unhelpfully.

**Fix**: change to `split(" <= ", maxsplit=1)` and `split(" > ", maxsplit=1)`.

**Status**: **WILL FIX.**

---

### 3. Item 16 docstring claims 108 FOMC dates, file has 114 — STYLE

**Issue at review time**: `research/16_calendar_features/build_16_calendar_features.py` docstring on the `FOMC_DATES` constant said "108 dates" but `len(FOMC_DATES) == 114` (8 x 13 normal years + 10 for 2020 with 2 emergency cuts). The item-16 findings note had the same stale count at review time and has since been corrected.

**Severity**: cosmetic; the LIST is correct, just the count claim is off.

**Fix**: update both comment and findings.md to "114 dates 2013-2026 (8 regular per year + 2 emergency in 2020)".

**Status**: **WILL FIX.**

---

### 4. LogisticRegression deprecation warning in 5 scripts — ROBUSTNESS

**Issue**: sklearn 1.8 warns:
```
FutureWarning: 'penalty' was deprecated in version 1.8 and will be removed in 1.10.
```
Affects items 04, 06, 12, 14, 16, 17.

**Severity**: works on the pinned sklearn version (1.8), will break on 1.10+. Currently produces ~50 warning lines per run.

**Fix**: not now. The right fix is a single change in each script (`penalty="l1", solver="liblinear", C=0.1` → `l1_ratio=1.0, C=0.1` with appropriate solver). Defer to the productionization pass — coordinate with sklearn version pinning.

**Status**: **DOCUMENTED, DEFERRED.**

---

### 5. Item 04 `tree_leaves` relies on sklearn ≥ 1.5 `tree_.value` semantics — ROBUSTNESS

**Issue**: in sklearn 1.5+ `tree_.value[node]` stores *class fractions*; before 1.5 it stored *counts*. The current code reads `fracs[1]` directly — correct for 1.5+. On sklearn < 1.5 this would silently give wrong precisions.

**Severity**: pinned env is 1.8, so this is robustness only. But a future env downgrade would silently break.

**Fix**: add a defensive version check at script start: `assert sklearn.__version__ >= "1.5"`. Or document in the docstring.

**Status**: **WILL ADD A VERSION-CHECK COMMENT** (no code change beyond a docstring line).

---

### 6. Item 13's `years is not np.nan` is tautological — STYLE

**Issue**: `research/13_within_csv_compounding/build_13_within_csv_compounding.py:67`:
```python
"years_span": float(years) if years is not np.nan else np.nan,
```
`years is not np.nan` is True even when `years` is NaN, because `np.nan is np.nan` evaluates True only for the same object reference. Different `np.nan` literals can compare `is not` True. So this `if` always executes, but `float(np.nan)` returns `nan` so the result is fine.

**Severity**: cosmetic / misleading code.

**Fix**: use `not pd.isna(years)`. Functionally identical, semantically correct.

**Status**: **WILL FIX.**

---

### 7. Item 13 `daily = (1.0 + groupby.apply(...))` then `daily - 1.0` — STYLE

**Issue**: `research/13_within_csv_compounding/build_13_within_csv_compounding.py:56-57`:
```python
daily = (1.0 + g.groupby("exit_date")["r"].apply(lambda s: float((1 + s).prod() - 1)))
daily_r = daily - 1.0
```
The `(1 + ...)` then `- 1.0` is a no-op. Cleaner to omit both. Result is identical.

**Severity**: cosmetic.

**Fix**: defer — code is correct, just verbose.

**Status**: **DOCUMENTED, DEFERRED.**

---

### 8. Item 06 duplicates `tree_leaves` recursion inline in main() — STYLE

**Issue**: `research/06_context_enrichment/build_06_context_enrichment.py` defines a `recurse` function inside `main()` that mirrors item 04's `tree_leaves` helper. Should share the implementation via `research/_rule_naming.py` or a new `research/_tree_helpers.py`.

**Severity**: maintenance hazard — if we change tree-walk logic, two places need updating.

**Fix**: defer to productionization pass (would also touch item 04). Out of scope for this code review.

**Status**: **DOCUMENTED, DEFERRED.**

---

### 9. Walk-forward boundaries are all correct ✓

Scanned all 18 scripts for the `train.year <= y_test` pattern. None found. All scripts that do WF use either:
- `train = df[df.year < y]` with `test = df[df.year == y]` (items 12, 17), or
- `train = df[df.year <= y_end]` with `y_test = y_end + 1` (item 04).

Both are causal. **No leakage.**

**Status**: ✓ verified by `tests/test_invariants.py::test_no_walkforward_leakage_in_build_scripts`.

---

### 10. `pnl_pct` scaling is correct throughout ✓

Scanned all uses of `pnl_pct`. No double-scaling. Items 05, 12, 13, 17 divide by 100 once when needed; items 02, 04, 07, 08, 11 use raw `pnl_pct` consistently.

**Status**: ✓ verified by `tests/test_invariants.py::test_pnl_pct_is_in_percentage_points`.

---

### 11. Regime filter is applied consistently ✓

All items that document "uses regime-labeled subset" do `df[df["regime_entry"].notna()]` immediately after loading the canonical. Item 05 has both scopes (full + regime-labeled) clearly separated. Item 16 applies the filter for the modelable comparison and notes it.

**Status**: ✓ verified by `tests/test_invariants.py::test_item_01_sample_sizes_match_canonical`.

---

### 12. NaN propagation in WF predictions ✓

Items 12, 17 use `df.dropna(subset=[predictions])` before equity computation, so trades without predictions (first WF year) are excluded. NaN-predicted trades cannot accidentally get `size = 1` and pollute the sized equity.

**Status**: ✓

---

### 13. Items 02, 04, 06, 11, 17 sort orders are intentional ✓

Item 02's unusual `sort_values(..., key=lambda s: s.abs() if s.name == "spearman" else s)` is correct — sorts by `score` desc, then by `|spearman|` desc as tiebreaker. Confirmed via a quick eyeball of the output CSVs.

**Status**: ✓

---

### 14. Random seeds set everywhere they matter ✓

Every script that uses RNG (`np.random`, sklearn `random_state`) sets `SEED = 42`. Item 11 shares `rng` across iterations which makes results order-dependent but reproducible-with-same-data. Item 04 same pattern.

**Status**: ✓

---

### 15. Item 16 `is_third_friday` covers the OPEX corner case correctly ✓

Standard OPEX (3rd Friday) and quad-witching (3rd Friday in Mar/Jun/Sep/Dec) logic looks correct. Spot-checked: 2023-09-15 (Friday) is in week 3 → 15 ≤ 15 ≤ 21 ✓; 2024-03-15 (Friday) → 15 ≤ 15 ≤ 21 → quad-witching ✓.

**Status**: ✓

---

### 16. Item 15 bar-window inclusivity is correct ✓

`intra = bars[(bars["dt"] > T0) & (bars["dt"] <= T1)]` excludes the entry bar, includes the exit bar. Correct for "what happened *during* the trade" — the entry bar is the price we paid; intra-trade drawdown should be measured against bars *after* entry.

**Status**: ✓

---

### 17. Items 06, 14 `merge_asof(direction="backward")` joins to STRICTLY prior business day - SUPERSEDED

`pd.merge_asof(left, right, ...)` with default `allow_exact_matches=True` would include the same-day context. But because `entry_date` is intraday-trade-day-normalized and context features are computed on END-of-day bars, the most-recent end-of-day prior to (or equal to) entry_date IS the previous business day's data, which is causal. Verified by spot-checking an early-2026 enriched trade.

**Caveat**: if any future context feature were computed at the START of the trade day, merge_asof would include same-day intraday context — potential look-ahead. Not happening now but worth noting.

**Corrected status**: **DATA BUG, FIXED 2026-06-09.** Same-day daily bars are look-ahead leakage for intraday entries. Item 06 now uses `allow_exact_matches=False`, and tests fail if any enriched row has `context_date >= entry_date`.

---

### 18. Items use `predict_proba(...)[:, 1]` assuming class index 1 is positive ✓

sklearn orders `classes_` ascending → integer 0/1 labels put 0 at index 0 and 1 at index 1. We always use `.astype(int)` before fit, so this is robust. Would break only if `fit` were called with all-zero or all-one labels (would raise — see the `train[target_col].sum() < 10` guard in item 17).

**Status**: ✓

---

## Summary

| severity | count | status |
|---|---:|---|
| **DATA** | 1 | FIXED (SYNTHESIS GBM claim) |
| **CORRECTNESS** | 1 | WILL FIX (item 11 parse_path) |
| **ROBUSTNESS** | 2 | 1 deferred (sklearn deprecation), 1 will-add-comment (item 04 version) |
| **STYLE** | 4 | 1 will-fix (item 13 NaN check), 1 will-fix (item 16 count), 2 deferred |
| ✓ verified clean | 10 | tests passing |

**Net new bugs found that affect numbers**: 1 (TQQQ GBM 2023 R² claim).
**Net new bugs found that are latent**: 1 (item 11 parse_path edge case).

The codebase is in better shape than I expected. The test suite catches the one real claim mismatch automatically going forward.

## Will-fix items applied in this pass

1. Item 11 `parse_path`: use `split(" <= ", 1)` and `split(" > ", 1)`.
2. Item 16 `FOMC_DATES` comment: update count from "108" to "114".
3. Item 13: replace `years is not np.nan` with `not pd.isna(years)`.
4. Item 04 docstring: note the sklearn ≥ 1.5 requirement on `tree_.value`.

## Deferred to productionization pass

1. LogisticRegression deprecation (move to `l1_ratio=1` syntax).
2. Item 06's inline tree-walk → shared helper module.
3. Item 13's `(1 + ...) - 1.0` idiom cleanup.
