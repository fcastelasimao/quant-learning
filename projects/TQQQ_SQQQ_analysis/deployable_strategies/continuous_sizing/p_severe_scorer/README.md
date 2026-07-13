# `p_severe` Walk-Forward Sizing Guide

Status: **research / validation tooling**. Strict-prior daily context is now
enforced, and the original item-17 deployment headline weakened under this
correction. Review the regenerated item-06 and item-17 outputs before treating
this as production sizing.

This package is **not** a full backtest engine and it is not a drop-in
replacement for the other team's simulator. It provides the scoring pieces that
their backtest must call at candidate-trade time.

Implementation sentence to keep in mind:

```text
Compute p_severe before each candidate trade is placed, using only pre-entry
strategy fields and strict-prior daily context, then set
size_multiplier = 1 - p_severe.
```

## 1. What The Scorer Does

For a candidate TQQQ or SQQQ trade, the scorer returns:

```text
p_severe
size_multiplier = 1 - p_severe
```

Use `normal_position_size * size_multiplier`.

## 2. Data Required Before Entry

Each candidate trade row must contain only information available before entry:

```text
symbol
entry_time                          # datetime; hour_of_entry is derived from this
decision_price                      # signal price at entry — NOT avg_order_price (post-fill)
atr or atr_pct                      # atr_pct = atr / decision_price * 100 if not pre-computed
RSI_entry
BBP_entry
volume_ratio or log_volume_ratio
bars_since_last_stop
MA20, MA50, MA100, or precomputed dist_to_MA20/dist_to_MA50/dist_to_MA100
MA20_D5
MA50_D5
MA100_D1
regime_entry
```

The scorer cannot reconstruct `RSI_entry`, `BBP_entry`,
`bars_since_last_stop`, or `regime_entry`; the calling backtest must provide
them.

Daily context must be strictly prior to the trade date. Same-day daily bars are
not allowed because they are not complete at intraday entry time.

Any row with NaN in a required feature column will produce a NaN `p_severe`.
The scorer does not impute missing values. If you want an explicit error instead,
pass `validate_features=True` (the default for single-symbol scoring).

## 3. What This Repo Provides

This repo provides:

```text
compute_required_features(...)
score_trades_with_models_by_symbol(...)
train_models_from_history(...)
JSON artifact loading/scoring helpers
example candidate rows
example yearly checkpoint script
```

The external backtest must provide:

```text
run_backtest(...)
candidate-trade generation
pre-entry strategy fields
strict-prior market context
completed trade outcomes after each year closes
```

## 4. Recommended Yearly Loop

Do not generate all future trades first. Run one year at a time:

```python
from p_severe_scorer import (
    compute_required_features,
    score_trades_with_models_by_symbol,
    train_models_from_history,
)

# 1. Baseline history with completed outcomes.
# --- your backtest engine ---
completed_history = run_backtest(start_year=A, end_year=B, use_p_severe=False)

# 2. Train only on completed history through B.
models = train_models_from_history(completed_history)

# 3. Run future years one at a time.
for year in range(B + 1, final_year + 1):

    def sizing_for_candidate(candidate_trade):
        features = compute_required_features(
            candidate_trade,
            daily_context=strict_prior_daily_context,
        )
        scored = score_trades_with_models_by_symbol(features, models)
        return float(scored.iloc[0]["size_multiplier"])

    # --- your backtest engine ---
    completed_year = run_backtest(
        start_year=year,
        end_year=year,
        sizing_fn=sizing_for_candidate,
    )

    # --- your backtest engine ---
    completed_history = concat_trades(completed_history, completed_year)
    models = train_models_from_history(completed_history)
```

During year `Y`, the model has only seen completed trades from years `< Y`.

The `for year in ...` loop is chronological. It is not permission to train on
all future years. At each iteration, `completed_history` must contain only
trades whose outcomes are already known.

Do **not** do this:

```python
all_trades = run_backtest(start_year=A, end_year=final_year)
models = train_models_from_history(all_trades)
scored = score_trades_with_models_by_symbol(all_trades, models)
```

That trains on future outcomes and then scores the past.

## 5. Two-Pass Backtest (Train From Your Own Trade Log)

If you want to train the L1 model entirely from trades your backtest generates
rather than using the pre-built JSON artifacts, use the two-pass approach:

**Pass 1 — run the backtest without sizing** to produce a labeled trade log.
The log must include `pnl_pct` (so `is_severe_loss` can be derived) and all
strategy fields listed in section 2.

```python
from p_severe_scorer.export_annual_models import export_annual_models
from p_severe_scorer.score import score_trades

# Pass 1: unsized backtest → labeled trade log
trades = run_backtest(start_year=2013, end_year=2026, use_p_severe=False)

# Train walk-forward annual models from your trades.
# export_annual_models handles the year < Y split internally — no leakage.
export_annual_models(trades[trades.symbol == "TQQQ"], symbol="TQQQ", out_dir="./my_artifacts")
export_annual_models(trades[trades.symbol == "SQQQ"], symbol="SQQQ", out_dir="./my_artifacts")

# Pass 2: score all trades — each year uses only the artifact trained through Y-1
scored_tqqq = score_trades(trades[trades.symbol == "TQQQ"], symbol="TQQQ", artifact_dir="./my_artifacts")
scored_sqqq = score_trades(trades[trades.symbol == "SQQQ"], symbol="SQQQ", artifact_dir="./my_artifacts")
# Each row now has p_severe and size_multiplier.
```

**Important:** `score_trades` enforces the walk-forward rule — for each trade
in year Y it loads the artifact trained through Y-1. Rows in years with no
prior artifact receive NaN (controlled by `on_missing_model`).

### Burn-in recommendation

The model needs enough labeled trades before its predictions are meaningful.
The minimum thresholds are 100 training rows and 10 severe-loss positives
(`MIN_TRAIN_ROWS`, `MIN_POSITIVES` in `training.py`). In practice, the first
reliable model artifact is the one trained through 2017 (predicting 2018),
because 2013–2017 provides ~300+ TQQQ trades and a stable positive rate.

**Recommendation:** apply `size_multiplier` only from 2018 onwards. For
2013–2017, treat `size_multiplier = 1.0` (full size). This matches the
original item-17 validation window (OOS 2018–2026) and avoids sizing decisions
made by an under-trained model.

```python
sized_trades = scored.copy()
sized_trades.loc[sized_trades["entry_time"].dt.year < 2018, "size_multiplier"] = 1.0
```

## 6. Getting `p_severe`

```python
features = compute_required_features(candidate_trade, daily_context=strict_prior_daily_context)
scored = score_trades_with_models_by_symbol(features, models)

p_severe = float(scored.iloc[0]["p_severe"])
size_multiplier = float(scored.iloc[0]["size_multiplier"])  # 1 - p_severe
```

## 7. Current Evidence

The strict-prior result is mixed:

```text
TQQQ: promising; enriched 1% sizing modestly improves Sharpe and drawdown.
SQQQ: unresolved; enriched 1% sizing improves drawdown but not Sharpe.
```

Treat `p_severe` as a sizing/ranking signal first. Review calibration diagnostics
before treating it as an absolute probability.

## 8. Example

Run the included example:

```bash
~/opt/anaconda3/envs/quant/bin/python \
  deployable_strategies/continuous_sizing/p_severe_scorer/examples/yearly_checkpoint_example.py
```

The example trains on completed strict-prior enriched trades through 2025 and
scores one 2026 TQQQ candidate plus one 2026 SQQQ candidate.

### 8b. Raw daily context example

If your backtest provides raw daily OHLC bars instead of precomputed context
columns, pass them as a dict to `compute_required_features`:

```bash
~/opt/anaconda3/envs/quant/bin/python \
  deployable_strategies/continuous_sizing/p_severe_scorer/examples/raw_daily_context_example.py
```

This example builds synthetic daily bars, creates a candidate trade with only
strategy-internal fields, and shows the 21 daily context columns being joined
from raw bars via `compute_required_features(trade, daily_context=raw_bars)`.

## 9. Optional CLI

The Python API is the intended backtest integration. The CLI is only for batch
inspection.

Score rows using each row's own `symbol`:

```bash
~/opt/anaconda3/envs/quant/bin/python -m p_severe_scorer.score \
  --symbol AUTO \
  --input candidate_trades.csv \
  --output scored_trades.csv
```

`--symbol BOTH` is diagnostic only: it applies both models to the same rows.
For production backtests, use `AUTO` or `score_trades_with_models_by_symbol`.

## 10. Rebuilding JSON Artifacts

JSON artifacts are optional for batch scoring. Rebuild them from completed,
labeled, strict-prior enriched trades:

```bash
~/opt/anaconda3/envs/quant/bin/python -m p_severe_scorer.export_annual_models \
  --symbol TQQQ \
  --input enriched_completed_trades_TQQQ.csv
```

Training requires `pnl_pct` or `is_severe_loss`. Candidate-trade scoring does
not.
