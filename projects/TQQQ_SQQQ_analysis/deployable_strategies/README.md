# Deployable Strategies

Self-contained package of production-ready strategy components for TQQQ / SQQQ
intraday trading. Copy this folder into your project; no dependency on the
`research/` or `archive/` directories.

---

## Contents

```
deployable_strategies/
  metrics.py                        constant_notional_metrics()
  continuous_sizing/
    sizing.py                       linear_skip, sqrt_skip, SIZING_FUNCTIONS dict
    p_severe_scorer/                walk-forward severe-loss probability scorer
      score.py                      score_trades() — main entry point
      training.py                   train_model_from_history(), _fit_model()
      export_annual_models.py       export_annual_models() — train and save JSON artifacts
      features.py                   compute_required_features()
      artifacts.py                  load_models(), DEFAULT_ARTIFACT_DIR
      constants.py                  MODEL_FEATURES, CURATED_NUMERIC, DAILY_CONTEXT
      artifacts/TQQQ/*.json         pre-trained yearly checkpoints (2016–2026)
      artifacts/SQQQ/*.json         pre-trained yearly checkpoints (2016–2026)
      examples/
        yearly_checkpoint_example.py   score live trades from saved JSON models
        raw_daily_context_example.py   compute features from raw daily bars
  focus_rules/
    sqqq_rsi_atr_skip.py            sqqq_focus_rule_mask() — skip SQQQ danger cell
  regime_rules/
    tqqq_sideways_skip.py           tqqq_regime_rule_mask() — skip TQQQ sideways zone
```

---

## How to combine the components

For each incoming trade, compute `final_size`. All functions in this package
operate on **`pandas.DataFrame`** objects — not dicts, not CSV paths. If you
have a single trade as a dict, wrap it with `pd.DataFrame([trade])`. If you
are reading from a CSV, load it first with `pd.read_csv(..., parse_dates=["entry_time"])`.

```python
import pandas as pd
from deployable_strategies.continuous_sizing.p_severe_scorer.score import score_trades
from deployable_strategies.continuous_sizing.sizing import linear_skip
from deployable_strategies.focus_rules.sqqq_rsi_atr_skip import sqqq_focus_rule_mask
from deployable_strategies.regime_rules.tqqq_sideways_skip import tqqq_regime_rule_mask

# 1. Continuous p_severe sizing (both symbols)
scored = score_trades(pd.DataFrame([trade]), symbol=trade["symbol"])
p = float(scored.iloc[0]["p_severe"])
size = float(linear_skip(p))         # graceful scale-down: 1 - p_severe

# 2. Focus rule override (SQQQ only)
if trade["symbol"] == "SQQQ":
    if sqqq_focus_rule_mask(pd.DataFrame([trade])).iloc[0]:
        size = 0.0                   # full skip

# 3. Regime rule override (TQQQ only)
if trade["symbol"] == "TQQQ":
    if tqqq_regime_rule_mask(pd.DataFrame([trade])).iloc[0]:
        size = 0.0                   # full skip

final_size = size                    # multiply trade notional by this factor
```

Rule skips take precedence over continuous sizing. The two overrides never
conflict (focus rule is SQQQ-only, regime rule is TQQQ-only).

---

## p_severe_scorer: required inputs

`score_trades()` expects a DataFrame with the following columns available
before the trade is placed:

```text
symbol, entry_time (datetime)
decision_price                      # signal price — NOT avg_order_price (post-fill)
atr or atr_pct                      # atr_pct = atr / decision_price * 100 if not pre-computed
RSI_entry, BBP_entry
volume_ratio or log_volume_ratio
bars_since_last_stop                # strategy must track this; scorer cannot derive it
MA20, MA50, MA100                   # or pre-computed dist_to_MA20/50/100
MA20_D5, MA50_D5, MA100_D1
regime_entry                        # "bull" | "chop_highvol" | "sideways_lowvol"
```

`hour_of_entry` is derived automatically from `entry_time`.

### Daily context features (21 total)

These must reflect the **prior business day** (strict: `context_date < entry_date`).
Pass them pre-computed, or pass raw daily OHLC bars and let
`compute_required_features()` build them automatically.

**Required raw symbols (OHLC bars):** `QQQ`, `SPY`, `^VIX`, `^VIX3M`, `HYG`, `LQD`, `^TNX`, `^IRX`

| Feature | Formula |
|---|---|
| `QQQ_RSI_14` | 14-period RSI of QQQ close |
| `QQQ_dist_MA20` | `QQQ / MA(QQQ, 20) − 1` |
| `QQQ_dist_MA50` | `QQQ / MA(QQQ, 50) − 1` |
| `QQQ_dist_MA200` | `QQQ / MA(QQQ, 200) − 1` |
| `QQQ_realized_vol_20d` | 20-day realised vol, annualised (`std(log-returns) × √252`) |
| `QQQ_dist_high_20d` | `QQQ / max(QQQ, 20d) − 1` (≤ 0) |
| `QQQ_50d_return` | `QQQ.pct_change(50)` |
| `QQQ_50d_return_pctile_252` | Rolling 252-day percentile rank of `QQQ_50d_return` |
| `QQQ_drawdown_5d` | `QQQ / max(QQQ, 5d) − 1` (≤ 0) |
| `QQQ_drawdown_60d` | `QQQ / max(QQQ, 60d) − 1` (≤ 0) |
| `QQQ_gap_overnight` | `QQQ_open / QQQ_prev_close − 1` |
| `SPY_RSI_14` | 14-period RSI of SPY close |
| `SPY_dist_MA50` | `SPY / MA(SPY, 50) − 1` |
| `VIX_level` | Raw VIX close (`^VIX`) |
| `VIX_5d_change` | `VIX.pct_change(5)` |
| `VIX_pctile_252d` | Rolling 252-day percentile rank of VIX level |
| `VIX_term_structure` | `^VIX / ^VIX3M` (< 1 = contango, > 1 = backwardation) |
| `HYG_LQD_ratio` | `HYG_close / LQD_close` (high-yield vs investment-grade; rising = credit stress) |
| `HYG_5d_change` | `HYG.pct_change(5)` |
| `yield_curve_slope` | `^TNX − ^IRX` (10-year minus 3-month yield, in percentage points) |
| `TNX_5d_change` | `^TNX.diff(5)` (level change, not pct) |

#### How to compute the 21 columns from raw OHLC bars

Call `build_daily_context()` once before scoring. It expects a dict of
DataFrames — one per symbol — each with columns `date`, `open`, `high`,
`low`, `close` (and optionally `adj_close`). The dates must span the full
backtest range plus at least 252 prior calendar days so the rolling windows
warm up correctly.

```python
import pandas as pd
from deployable_strategies.continuous_sizing.p_severe_scorer.features import build_daily_context

# Each DataFrame has columns: date, open, high, low, close (adj_close optional)
# date must be a business-day series; no gaps allowed.
daily_bars = {
    "QQQ":   pd.read_csv("QQQ_daily.csv",   parse_dates=["date"]),
    "SPY":   pd.read_csv("SPY_daily.csv",   parse_dates=["date"]),
    "^VIX":  pd.read_csv("VIX_daily.csv",   parse_dates=["date"]),
    "^VIX3M": pd.read_csv("VIX3M_daily.csv", parse_dates=["date"]),
    "HYG":   pd.read_csv("HYG_daily.csv",   parse_dates=["date"]),
    "LQD":   pd.read_csv("LQD_daily.csv",   parse_dates=["date"]),
    "^TNX":  pd.read_csv("TNX_daily.csv",   parse_dates=["date"]),
    "^IRX":  pd.read_csv("IRX_daily.csv",   parse_dates=["date"]),
}

# Returns a DataFrame with columns: date + the 21 context features above.
# One row per calendar day. Rolling windows produce NaN for early rows
# (up to 252 days from the start of the series) — this is expected.
ctx = build_daily_context(daily_bars)
```

Then pass `ctx` to the scorer. The strict-prior join (`context_date < entry_date`)
is applied automatically — you do not need to shift dates manually:

```python
from deployable_strategies.continuous_sizing.p_severe_scorer.score import score_trades

scored = score_trades(trades, symbol="TQQQ", daily_context=ctx)
```

**Alternatively**, pre-compute and cache `ctx` to avoid rebuilding it on every
run:

```python
ctx.to_csv("daily_context_cache.csv", index=False)

# On subsequent runs:
ctx = pd.read_csv("daily_context_cache.csv", parse_dates=["date"])
scored = score_trades(trades, symbol="TQQQ", daily_context=ctx)
```

Notes on the input DataFrames:
- `^VIX` and `^VIX3M` use `close`, not `adj_close` (index levels, not prices).
- `^TNX` and `^IRX` are yield levels in percentage points (e.g. `4.25` means 4.25%). Their `diff(5)` and subtraction are in the same unit — no conversion needed.
- For `QQQ`, `SPY`, `HYG`, `LQD`: use `adj_close` if available (preferred for return-based features); falls back to `close` automatically.
- All 8 series must cover the same date range. Missing dates cause NaN context rows, which propagate to NaN `p_severe`.

### Strict-prior constraint (mandatory)

Daily context features must come from the **business day before** the trade
entry date:

```python
context_date < entry_date           # strict inequality, no same-day bars
```

Same-day daily bars are look-ahead leakage for intraday entries (item 06,
2026-06-09 correction). The feature helper enforces this automatically.

---

## p_severe_scorer: training the model

### Option A — use the pre-built JSON artifacts

The `artifacts/` directory contains models trained through 2025. For trades
in year Y the scorer automatically loads the artifact trained through Y-1.
No training step required.

### Option B — train from your own backtest trade log (two-pass)

Run your backtest once without sizing to produce a labeled trade log, then
train and score:

```python
from deployable_strategies.continuous_sizing.p_severe_scorer.export_annual_models import export_annual_models
from deployable_strategies.continuous_sizing.p_severe_scorer.score import score_trades

# Pass 1: unsized backtest → labeled trade log as a DataFrame (needs pnl_pct column)
# If your backtest writes a CSV, load it first:
#   trades = pd.read_csv("trades.csv", parse_dates=["entry_time", "exit_time"])
trades = run_backtest(start_year=2013, end_year=2026, use_p_severe=False)

# Train walk-forward annual models — year < Y split is handled internally
export_annual_models(trades[trades.symbol == "TQQQ"], symbol="TQQQ", out_dir="./my_artifacts")
export_annual_models(trades[trades.symbol == "SQQQ"], symbol="SQQQ", out_dir="./my_artifacts")

# Pass 2: score — each year Y uses only the artifact trained through Y-1
scored = score_trades(trades[trades.symbol == "TQQQ"], symbol="TQQQ", artifact_dir="./my_artifacts")
# scored has p_severe and size_multiplier columns
```

**Burn-in:** apply `size_multiplier` from 2018 onwards only. The model needs
~5 years of trades (2013–2017) before its predictions are reliable. For
2013–2017 set `size_multiplier = 1.0`. This matches the item-17 OOS window
(2018–2026).

```python
scored.loc[scored["entry_time"].dt.year < 2018, "size_multiplier"] = 1.0
```

See `continuous_sizing/p_severe_scorer/README.md` section 5 for full details.

---

## Evidence summary

| Component | Research item | OOS metric | Period |
|---|---|---|---|
| p_severe continuous sizing | Items 12, 17 | TQQQ Sharpe 4.22, MaxDD −10.0% | 2018–2026 |
| SQQQ focus rule | Item 08 | 7 flagged, 71% severe precision, +5.26 pp | 2021–2026 |
| TQQQ regime rule | Item 11 | 10 flagged, 100% precision, +10.6 pp | 2021–2025 |
| **Combined (p_severe + rules)** | **Item 18** | **TQQQ Sharpe 4.27 / MaxDD −10.0%; SQQQ Sharpe 2.63 / MaxDD −15.2%** | 2018–2026 |

Sizing recommendation per item 17:
- **TQQQ**: `linear_skip` with `is_severe_loss @ −1%` target (Calmar 10.89, MaxDD −10.0%)
- **SQQQ**: `sqrt_skip` with `is_severe_loss @ −2%` target (Sharpe 2.62, best OOS)

These are validated research outputs. Re-validate on each annual checkpoint before deploying.
