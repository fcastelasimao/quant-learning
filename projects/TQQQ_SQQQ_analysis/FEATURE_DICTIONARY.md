# Feature Dictionary

This document explains the variable names used in the full-history TQQQ/SQQQ
analysis. The goal is to make the research outputs readable without needing to
inspect the Python code.

## Naming Convention

- `*_pct` means the value is expressed as a percentage.
- `dist_to_*` means distance from the entry price to another reference level.
- `*_roll_pctile_252` means rolling percentile rank over the last 252 trades for
  that symbol.
- `bucket_0` to `bucket_4` means quintile buckets:
  - `bucket_0` = lowest 20 percent of observed values.
  - `bucket_4` = highest 20 percent of observed values.
- `cell_X_Y` means a two-variable heatmap cell:
  - `X` is the bucket for the first variable.
  - `Y` is the bucket for the second variable.

## Core Outcome Variables

| Variable | Meaning |
|---|---|
| `pnl` | Dollar profit/loss for the trade, from the source backtest. |
| `pnl_pct` | Trade return in percentage points. Example: `2.5` means +2.5 percent. |
| `is_loser` | `True` when `pnl_pct < 0`. This is the main loser-classification target. |
| `is_severe_loss` | `True` when `pnl_pct <= -1.0`. This marks larger losing trades. |
| `abs_pnl_pct` | Absolute value of `pnl_pct`; useful for measuring trade magnitude. |

## Main Pre-Trade Features

| Variable | Meaning |
|---|---|
| `RSI_entry` | RSI value at trade entry from the source strategy. |
| `atr` | ATR value at trade entry. Raw ATR is price-scale dependent. |
| `atr_pct` | Normalized ATR: `atr / avg_order_price * 100`. This is ATR as a percent of entry price and is usually more comparable through time than raw ATR. |
| `BBP_entry` | Bollinger Band Percent / position-style indicator at entry, from the source strategy. |
| `volume_cur` | Current volume field from the source strategy. |
| `volume_ratio` | Current volume relative to a source-defined reference volume. |
| `log_volume_ratio` | Natural log of `volume_ratio`; reduces the effect of extreme volume spikes. |
| `hour_of_entry` | Hour of day when the trade was entered. |
| `bars_since_last_stop` | Number of bars since the previous stop event, from the source strategy. |
| `is_bullish_c1c2` | Binary flag from the source strategy. Almost certainly "both of the last two completed candles were bullish (close > open)", but the exact definition is in the strategy code (not in this repo). Confirm against the producing source before relying on it. 56 % / 44 % split in the canonical CSVs, no NaN. |
| `high_water_mark_entry` | **Equals `avg_order_price` on every row in the current source data** (verified TQQQ 2,343/2,343 and SQQQ 1,930/1,930). The intended semantic appears to be a running-max reference, but as shipped it is an alias for the entry price and carries no independent signal. Do not use as a predictor. See "Data caveats" below. |
| `regime_entry` | Strategy-provided regime label at entry. Treat as pre-trade only if its construction is confirmed causal. |

## Moving-Average Features

The source files include moving-average levels and slope/change fields.

| Variable Pattern | Meaning |
|---|---|
| `MA20`, `MA50`, `MA100` | Source moving-average levels. Raw levels are price-scale dependent. |
| `EMA20` | Source exponential moving-average level. |
| `dist_to_MA20` | `avg_order_price / MA20 - 1`. Positive means entry price is above MA20. |
| `dist_to_MA50` | `avg_order_price / MA50 - 1`. Positive means entry price is above MA50. |
| `dist_to_MA100` | `avg_order_price / MA100 - 1`. Positive means entry price is above MA100. |
| `dist_to_high_water_mark` | `avg_order_price / high_water_mark_entry - 1`. **Identically zero on every row in the current source data**, because `high_water_mark_entry == avg_order_price` always. See "Data caveats" below. |
| `MA20_D1`, `MA20_D3`, `MA20_D5` | Source-provided MA20 change/slope fields over 1, 3, or 5 bars/days, depending on the strategy definition. |
| `MA50_D1`, `MA50_D3`, `MA50_D5` | Source-provided MA50 change/slope fields. |
| `MA100_D1` | Source-provided MA100 change/slope field. |

## Rolling Percentile Features

| Variable | Meaning |
|---|---|
| `RSI_entry_roll_pctile_252` | Percentile rank of current `RSI_entry` versus the recent 252-trade history for that symbol. |
| `atr_pct_roll_pctile_252` | Percentile rank of current `atr_pct` versus the recent 252-trade history for that symbol. |
| `BBP_entry_roll_pctile_252` | Percentile rank of current `BBP_entry` versus the recent 252-trade history for that symbol. |
| `volume_ratio_roll_pctile_252` | Percentile rank of current `volume_ratio` versus the recent 252-trade history for that symbol. |

These rolling percentile features are designed to reduce non-stationarity. For
example, `atr_pct = 1.0` may mean something different in 2013 than in 2022, but
`atr_pct_roll_pctile_252 = 0.90` always means ATR was high relative to the
recent history of that symbol.

## Candidate Rule Names

Candidate rules are named mechanically so they can be traced back to the tables.

| Example | Meaning |
|---|---|
| `atr_pct_bucket_0_high_loser_rate` | Trades where `atr_pct` was in the lowest quintile during in-sample discovery. |
| `rsi_x_atr_cell_3_1_high_loser_rate` | Trades in RSI bucket 3 and ATR bucket 1. Buckets are quintiles. |
| `regime_sideways_lowvol_high_loser_rate` | Trades where `regime_entry` was `sideways_lowvol`. |

Important: a rule name saying `high_loser_rate` means it was selected because it
looked risky in the in-sample discovery period. It does not mean the rule worked
out of sample.

## Review Guidance

When reviewing results, prioritize:

1. `atr_pct` over raw `atr`.
2. Bucket tables over raw correlations.
3. Out-of-sample validation over in-sample rule discovery.
4. Net PnL, Calmar, Ulcer, and max drawdown impact over classification accuracy.

## Columns introduced during research (not in the source CSVs)

These are emitted by analysis scripts under `research/` and by
`archive/full_history_feature_scan.py`. They are *not* trade features in the source data
sense — they are derived from features or are output-schema columns. Listed here
so future you can see what is pipeline-original vs research-pass-added.

### One-hot regime dummies (used in items 03–04 models)

| Variable | Meaning |
|---|---|
| `regime_bull` | 1 if `regime_entry == "bull"`, else 0. **Reference category** — dropped from model inputs to avoid collinearity. |
| `regime_chop_highvol` | 1 if `regime_entry == "chop_highvol"`, else 0. |
| `regime_sideways_lowvol` | 1 if `regime_entry == "sideways_lowvol"`, else 0. |

### Feature-family labels (from `research/01_data_diagnostics/feature_clusters.csv`)

Used to group the 28 numeric features into 10 interpretation buckets. Not predictors. Categories:

| Label | Members |
|---|---|
| `ma_raw_level` | `MA20`, `MA50`, `MA100` — raw price-level features (drift with the underlying). |
| `price_level_ref` | `high_water_mark_entry` — see Data caveats. |
| `ma_slope` | `MA20_D1/D3/D5`, `MA50_D1/D3/D5`, `MA100_D1`. |
| `ma_dist` | `dist_to_MA20`, `dist_to_MA50`, `dist_to_MA100`. |
| `atr` | `atr`, `atr_pct`, `atr_pct_roll_pctile_252`. |
| `rsi` | `RSI_entry`, `RSI_entry_roll_pctile_252`. |
| `bbp` | `BBP_entry`, `BBP_entry_roll_pctile_252`. |
| `volume` | `volume_cur`, `volume_ratio`, `log_volume_ratio`, `volume_ratio_roll_pctile_252`. |
| `stop_history` | `bars_since_last_stop`. |
| `session_clock` | `hour_of_entry`. |
| `candle_state` | `is_bullish_c1c2`. |

### Output schema columns commonly used in research CSVs

| Column | Meaning |
|---|---|
| `score` | A combined ranking score; the exact formula is documented at the top of the producing script (item 01 uses one definition for clustering ranking, item 02 uses another for univariate signal — they are not interchangeable). |
| `is_cluster_representative` | In `feature_clusters.csv`, true if this feature was picked as the cluster's representative (highest `|Spearman(feature, pnl_pct)|` in the auto-pick; overridden by the interpretability-first set in `01/findings.md`). |
| `cluster_id`, `cluster_size`, `members` | Multicollinearity-cluster metadata. |
| `auc_directional` | `max(AUC, 1 − AUC)`. Used so a feature that is anti-correlated with the target (AUC 0.40) is treated symmetrically to one that is positively correlated (AUC 0.60). |
| `precision_minus_random` | A candidate rule's measured precision minus the median precision of a random filter triggering at the same rate. The honest "is this rule better than random" metric. |
| `net_pnl_pct_impact` | `−sum(flagged trades' pnl_pct)`. Positive = skipping these trades would have *avoided* net negative PnL = good rule. Negative = skipping would have lost positive PnL = bad rule. |
| `tree_pos_frac` | The decision tree's stored positive-class fraction at a leaf (sklearn 1.5+ stores fractions, not counts, in `tree_.value`). Used only for cross-checking against the realized IS precision. |

### Constant-notional equity columns (from item 05)

| Column | Meaning |
|---|---|
| `r` | Per-trade return on a fixed notional: `pnl_pct / 100`. |
| `daily_pnl` | Sum of `r` over trades exiting on a given business day. |
| `equity` | `1 + cumulative sum of daily_pnl`. **Additive**, not compounded — see `GLOSSARY.md` § Performance metrics. |
| `annualized_arith_return` | `total_return / years`. Replaces compounded CAGR under the constant-notional convention. |

## Pruned feature sets (item 06 permutation importance)

Walk-forward permutation importance on the enriched L1-logit (35 features) found that 13/19 features have positive mean AUC delta for TQQQ/SQQQ respectively. Use these instead of all 35 for downstream sizing models.

### TQQQ-13 (use for TQQQ enriched sizing)

`atr_pct`, `regime_sideways_lowvol`, `regime_chop_highvol`, `hour_of_entry`, `RSI_entry`, `QQQ_gap_overnight`, `dist_to_MA20`, `bars_since_last_stop`, `QQQ_realized_vol_20d`, `QQQ_dist_MA200`, `QQQ_50d_return_pctile_252`, `HYG_5d_change`, `TNX_5d_change`

Full set: `research/06_context_enrichment/pruned_feature_set_TQQQ.csv`. Pruned AUC 0.605 vs full 0.611 (−0.006).

### SQQQ-19 (use for SQQQ enriched sizing)

`RSI_entry`, `VIX_pctile_252d`, `QQQ_dist_MA200`, `QQQ_50d_return_pctile_252`, `QQQ_dist_high_20d`, `dist_to_MA20`, `QQQ_realized_vol_20d`, `QQQ_50d_return`, `atr_pct`, `VIX_level`, `MA20_D5`, `yield_curve_slope`, `QQQ_drawdown_5d`, `SPY_RSI_14`, `hour_of_entry`, `dist_to_MA50`, `bars_since_last_stop`, `regime_sideways_lowvol`, `BBP_entry`

Full set: `research/06_context_enrichment/pruned_feature_set_SQQQ.csv`. Pruned AUC 0.535 vs full 0.534 (+0.001).

**Important:** `HYG_LQD_ratio` has **negative** permutation importance for both symbols. Do not include it in any enriched model.

## New outcome columns (items 12, 17, 18, 19)

| Column | Meaning |
|---|---|
| `is_severe_loss_1p5pct` | `pnl_pct <= -1.5`. Intermediate severity target explored in items 12/17. |
| `is_severe_loss_2pct` | `pnl_pct <= -2.0`. Higher-severity target; preferred for SQQQ per items 17/19. |
| `p_severe` | Walk-forward L1-logit predicted probability of severe loss for the chosen target. |
| `rule_fires` | Binary: 1 if the crisp skip rule (TQQQ regime rule or SQQQ focus rule) applies to this trade. |
| `cagr` | Annualized arithmetic return under constant-notional convention. Renamed from `annualized_return`. |

## Data caveats / known issues

### `high_water_mark_entry` and `dist_to_high_water_mark` are broken

In the current source CSVs, `high_water_mark_entry == avg_order_price` on every
single row (TQQQ 2,343/2,343; SQQQ 1,930/1,930). Consequently
`dist_to_high_water_mark` is identically zero everywhere. The intended semantic
was almost certainly a running-max price reference, but as shipped:

- `high_water_mark_entry` carries **no independent signal** — it is an alias for the entry price.
- `dist_to_high_water_mark` carries **no signal at all** — it is the zero feature.
- Both must be excluded from predictive models. Item 01's "raw price-level features that drift with time" conclusion incorrectly attributed `high_water_mark_entry`'s high correlation with MA20/MA50/MA100 to "tracking the price level". The real cause is that it *equals* `avg_order_price`, which itself drifts with time.

**Fix path:** correct the producing backtest pipeline so the HWM is a true running maximum, then re-export the CSVs.

### `is_bullish_c1c2` definition is unconfirmed

The name strongly suggests "both of the last two candles were bullish (close > open)" — c1 and c2 = candle 1 and candle 2 going back from the entry. But the producing logic is in the strategy code, which is not part of this repo. Item 02 found it carries no univariate signal under either binary target; that finding is symmetric to either interpretation. **Confirm against the strategy source before using it in any reframe** (e.g., as an input to a sizing rule).

### `regime_entry` causality is unconfirmed

The label is treated as pre-trade throughout this project, but item 02 of the
existing `FINDINGS.md` open questions calls this out: the regime classifier
might use information from bar boundaries that won't be available at the actual
trade entry instant. Treat regime-based predictions as conditional on this
assumption holding.
