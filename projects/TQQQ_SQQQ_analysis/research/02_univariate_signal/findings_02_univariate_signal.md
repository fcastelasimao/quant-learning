# 02 — Univariate signal

> Period: IS 2015–2020, regime-labeled subset (1,782 TQQQ / 1,511 SQQQ).

**Scope.** Regime-labeled subset (1,782 TQQQ / 1,511 SQQQ). All 28 numeric features × all 3 targets × both symbols → 6 ranked CSVs.

For binary targets (`is_loser`, `is_severe_loss`): AUC, directional-AUC (max(AUC, 1−AUC)), Spearman / Pearson, KS, Mann-Whitney, mutual information. Sorted by `(AUC_directional − 0.5) + mutual_info`.

For continuous `pnl_pct`: Spearman, Pearson, mutual information. Sorted by `|Spearman| + mutual_info`.

`is_cluster_representative` flag carries over the 13-feature curated set from item 1.

## Headlines

1. **Single-variable signal is weak across the board.** Best AUC anywhere is **0.626 (SQQQ atr_pct → is_severe_loss)**. Most features sit in 0.50–0.55. There is no single feature with which we can confidently classify losers. Multivariate combinations (item 04) need to do the heavy lifting.

2. **`atr_pct` is the strongest single signal — but for severity, not direction.** Across both symbols `atr_pct` is the top feature for `is_severe_loss` (AUC 0.62), top by MI for `pnl_pct`, and only middling for `is_loser`. Spearman with `pnl_pct` is tiny (~0.03–0.08); MI is large. Interpretation: **atr_pct controls the *variance* of pnl_pct, not its sign or mean**. High volatility → larger swings in both directions, but the tail of severe losses fattens more than the tail of severe wins. Matches the existing finding in `FINDINGS.md`.

3. **`hour_of_entry` is the top signal for SQQQ `is_loser`** (AUC 0.60, Spearman +0.17). Later in the session → more likely a loser. This is novel relative to the existing pipeline's bucket scans and worth a closer look in item 04 — likely a microstructure effect (e.g. SQQQ trades entered near close don't have time to mean-revert).

4. **`bars_since_last_stop` is the second-strongest signal for TQQQ `is_severe_loss`** (Spearman +0.107). Longer since the last stop → larger severe-loss rate. Interpretable: the strategy may be more aggressive after a quiet stretch.

5. **Rolling-percentile features add nothing.** Across all 12 rankings, the `_roll_pctile_252` version of a feature sits next to its raw parent with near-identical AUC / Spearman / MI. This re-confirms the item-01 multicollinearity finding from a different angle: redundancy holds for *signal* and not just *correlation*.

6. **Raw price-level features (`MA20`, `MA50`, `MA100`, `high_water_mark_entry`) show small but non-zero MI on `pnl_pct`** for SQQQ. This is a spurious "good years vs bad years" effect — these features encode time-since-2015, not a per-trade state. They must not enter predictive models.

## Per-symbol per-target winners (the top three of each)

| symbol | target | top 1 | top 2 | top 3 |
|---|---|---|---|---|
| TQQQ | is_loser | atr_pct_roll_pctile_252 (AUC 0.58) | atr_pct (0.58) | volume_cur (0.55) |
| TQQQ | is_severe_loss | atr_pct (AUC 0.62) | atr_pct_roll_pctile_252 (0.61) | atr (0.59) |
| TQQQ | pnl_pct | atr_pct (MI 1.85) | atr_pct_roll_pctile_252 (MI 0.60) | atr (MI 0.26) |
| SQQQ | is_loser | hour_of_entry (AUC 0.60) | atr_pct_roll_pctile_252 (0.57) | atr_pct (0.57) |
| SQQQ | is_severe_loss | atr_pct (AUC 0.63) | atr_pct_roll_pctile_252 (0.60) | MA50 spurious (0.55) |
| SQQQ | pnl_pct | atr_pct (MI 2.37) | atr_pct_roll_pctile_252 (MI 0.52) | atr (MI 0.23) |

The MI magnitudes on `pnl_pct` look very large (1.8 and 2.4 nats for atr_pct), but Spearman is < 0.1. That gap is exactly the variance-vs-mean distinction in point 2 above. Treat the MI ranking on `pnl_pct` as a *variance-predictor* ranking, not a *mean-predictor* ranking.

## What carries no signal at all

Across both symbols and all three targets, features that never appear in any top-10 and have AUC ∈ [0.50, 0.52]:

- `is_bullish_c1c2` (last-2-candle direction) — completely flat. The candle-state info isn't differentiating outcomes.
- `MA100_D1` (longest MA's slope) — marginal.
- The rolling-percentile pair where the raw version is also weak (e.g. `BBP_entry_roll_pctile_252` on TQQQ).

`is_bullish_c1c2` can probably be dropped from item 04's modeling pool with no loss.

## Implications for item 04 (models)

- **Use the curated 13-feature set from item 01**, except drop `is_bullish_c1c2` (no signal). Reduces to 12 features.
- **Run two parallel target families:** `is_severe_loss` (where atr_pct gives a clear handhold, AUC 0.62) and `is_loser` (the original "skip bad trades" target). They will likely surface different rules.
- **Hour-of-day matters for SQQQ** — make sure the tree models can split on it without being penalized by the smaller `is_loser` AUC.
- **Don't expect single-rule magic.** Even the best individual feature only moves AUC ~0.12 above chance. Rules will likely need 3–4 conditions to get useful precision.

## How to read the plots

**`top_features_<sym>_<target>.png`** — horizontal bar chart of the top 12 features for that (symbol, target) combination. Six files total.

- **For binary targets** (`is_loser`, `is_severe_loss`): x-axis is **directional AUC** (`max(AUC, 1−AUC)` — see `GLOSSARY.md` § 2). The dashed vertical line at 0.50 is the random-baseline. Bars to the right of it = real signal; bars near 0.50 = noise.
- **For continuous target** (`pnl_pct`): x-axis is **|Spearman ρ|** with the target. Dashed line at 0.
- **Bar color**: **red** = the feature is a representative of one of the 13 curated clusters from item 01. **Blue** = the feature is in a cluster but isn't the representative (so it's redundant with a red bar elsewhere).
- **What to look for**: how far the top bar extends past the dashed line tells you the strongest single-feature signal. For `is_severe_loss`, `atr_pct` sits at AUC ≈ 0.62 (~12 pp above random) — the strongest handhold in the dataset.
- **Why bar color matters**: if the top bars are mostly **red**, your curated set captures the signal. If a top bar is **blue**, you might be cutting an important feature out of the curated set.

## Artifacts

| file | content |
|---|---|
| `build.py` | the analysis script |
| `univariate_<sym>_<target>.csv` | one ranked table per (symbol, target) — 6 files |
| `top_features_<sym>_<target>.png` | horizontal AUC-or-|Spearman| bar chart per (sym, target) — 6 files |
| `findings.md` | this note |
