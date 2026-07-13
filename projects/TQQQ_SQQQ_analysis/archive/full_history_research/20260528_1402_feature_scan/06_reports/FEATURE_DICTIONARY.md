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
