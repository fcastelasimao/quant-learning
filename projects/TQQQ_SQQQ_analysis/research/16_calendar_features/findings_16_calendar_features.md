# 16 — Calendar / FOMC / seasonality features

> Period: IS 2015–2020, OOS 2021–2026 (pre-strict-prior rebuild; see status note).

**2026-06-09 status note.** This item compared calendar features against the original item-06 daily-context artifacts, which later proved to allow same-day daily-bar leakage. The calendar conclusion is still low priority because its incremental signal was tiny, but the exact AUC values below are historical, not strict-prior deployment evidence.

**Scope.** Self-contained pass that adds 16 calendar features and runs (a) the item-02 univariate methodology against all three targets and (b) the item-06 AUC-comparison methodology on `is_severe_loss`. **All prior research directions are untouched.**

## Features added

| feature | definition |
|---|---|
| `dow_of_entry` | 0=Mon..4=Fri |
| `month_of_entry` | 1..12 |
| `is_monday`, `is_friday` | day-of-week dummies |
| `is_summer` | June–August |
| `is_santa_rally_window` | Dec 20–Jan 5 |
| `is_first_session_of_month` / `is_last_session_of_month` | day-of-month ≤3 / ≥27 |
| `is_monthly_opex` | 3rd Friday of any month |
| `is_quad_witching` | 3rd Friday of Mar/Jun/Sep/Dec |
| `is_half_day` | Jul 3, Christmas Eve, Black Friday (NYSE early close) |
| `fomc_signed_days_to_nearest` | signed days to the nearest FOMC announcement (negative = before, positive = after) |
| `fomc_abs_days_to_nearest` | absolute version |
| `is_fomc_day`, `is_within_3d_of_fomc`, `is_week_after_fomc` | FOMC proximity flags |

FOMC dates hardcoded from the Fed published calendar 2013-2026 (114 dates: 8 regular meetings per normal year plus 2 emergency 2020 cuts). Verify against `https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm` before relying on out-year accuracy.

## Headline at the model level — calendar features add nothing

OOS AUC on `is_severe_loss`:

| symbol | feature set | n_features | tree AUC | L1-logit AUC |
|---|---|---:|---:|---:|
| TQQQ | curated_12 only | 14 | 0.670 | 0.593 |
| TQQQ | + 21 daily ctx (item 06) | 35 | **0.716** | **0.716** |
| TQQQ | + daily + calendar (item 16) | 51 | **0.716** | **0.708** |
| SQQQ | curated_12 only | 14 | 0.616 | 0.508 |
| SQQQ | + 21 daily ctx (item 06) | 35 | **0.699** | **0.555** |
| SQQQ | + daily + calendar (item 16) | 51 | **0.699** | **0.553** |

**Tree AUC identical to two decimal places. L1-logit slightly worse** (calendar features add coefficient noise that L1 doesn't fully zero out). The model-level verdict is: calendar features carry no incremental predictive signal over what daily context (item 06) already captures.

## But the univariate picture is more interesting

Strongest calendar features per (symbol, target), univariate:

| symbol | target | top feature | directional AUC | Spearman |
|---|---|---|---:|---:|
| TQQQ | is_severe_loss | `dow_of_entry` | 0.525 | +0.042 |
| TQQQ | is_severe_loss | `month_of_entry` | 0.520 | −0.032 |
| TQQQ | is_severe_loss | `is_within_3d_of_fomc` | 0.514 | −0.035 |
| TQQQ | is_severe_loss | `is_friday` | 0.516 | +0.037 |
| **SQQQ** | **is_severe_loss** | **`fomc_signed_days_to_nearest`** | **0.540** | **−0.069** |
| SQQQ | is_severe_loss | `month_of_entry` | 0.514 | −0.025 |
| SQQQ | is_severe_loss | `is_friday` | 0.517 | −0.043 |
| SQQQ | is_loser | `month_of_entry` | 0.506 | +0.011 |

**Real but small effects**:
- **Day-of-week matters for TQQQ severe losses**: positive Spearman of `dow_of_entry` with severe loss = later in the week → more severe-loss-prone. **Friday TQQQ severe losses are slightly more common.**
- **For SQQQ severe losses, FOMC distance matters**: `fomc_signed_days_to_nearest` Spearman −0.069. **Trades just BEFORE FOMC have higher severe-loss rate**, declining as we move past the announcement. AUC 0.540 is the strongest single calendar signal.
- **Month-of-year carries small signal** for both symbols on `is_severe_loss` — some months systematically worse.

## Why don't these add to the model?

The univariate signal is real but **redundant with what daily context already captures**:
- `fomc_signed_days_to_nearest` correlates with `VIX_level` and `yield_curve_slope` (markets price in FOMC expectations into VIX and rates curve).
- `month_of_entry` partially captures the same `QQQ_50d_return` cycle that the daily features measure.
- `dow_of_entry` and `is_friday` overlap with `hour_of_entry` (late-Friday entries cluster).

L1-logit regularization confirms this: it could have given these features positive coefficients if they were informationally distinct; it kept them at near-zero and the AUC drifted slightly down (because the model now has 16 more dimensions to overfit on).

## What this means

1. **For the production sizing rule (item 12 / 06)**: don't add calendar features. They don't move OOS AUC.
2. **For interpretability / monitoring**: the univariate FOMC effect on SQQQ severe losses is real (AUC 0.54). Worth being **aware** when reviewing live performance — a string of SQQQ severe losses concentrated around FOMC announcements is consistent with normal behavior, not strategy decay.
3. **For future hypothesis-testing**: the `fomc_signed_days_to_nearest` × `regime_entry` interaction would be the next thing to check — does the FOMC effect concentrate in chop_highvol vs sideways_lowvol? Not run here.

## Caveats

- **FOMC date list is hardcoded** through 2026. After Dec 2026 the script will silently use stale data; refresh before then.
- **Half-day list is incomplete** — only Jul 3, Christmas Eve, Black Friday are flagged. Other early-close events (per-year Fed-announced) are not. ~0.5 % of trades.
- **NYSE official closures** are not used as features. Trades can't happen on NYSE-closed days, so a "day of holiday" feature would always be 0 — not informative.
- **The FOMC announcement always lands at 2pm ET**. Trades entered before 2pm ET on FOMC day have a different microstructure than after. Not modeled here.

## How to read the plots

**`dow_outcomes.png`** — four-panel grid: mean pnl_pct (top) and loser-rate / severe-loss-rate (bottom) by day of week, per symbol.

- **Top row, blue bars**: mean pnl per trade for trades entered each weekday. Annotated with sample size n.
- **Bottom row, red+purple bars**: loser_rate (red, full width) and severe_loss_rate (purple, narrow, alpha=0.7) per weekday. The dashed black horizontal line is the symbol's baseline loser_rate.
- **What to look for**: any weekday where the red bar clearly rises above the dashed line, or where the blue (mean pnl) bar dips clearly below 0. For TQQQ, a Friday red bar above baseline + a Friday blue bar dipping low would corroborate the univariate finding. For SQQQ, look for the same pattern.

**`fomc_proximity_pnl.png`** — two side-by-side panels (TQQQ left, SQQQ right) showing mean pnl bucketed by signed days to nearest FOMC.

- **X axis**: bucket of `fomc_signed_days_to_nearest`. Negative = trade is N days BEFORE FOMC; positive = AFTER.
- **Y axis**: mean pnl_pct of trades in each bucket. Annotated with bucket sample size n.
- **What to look for**: a dip near 0 (FOMC day) followed by recovery; or a build-up of negative pnl in the −3:−1 buckets matching the SQQQ univariate signal. The "+1:0" bucket (FOMC day + immediate day after) is the announcement window — this is where the volatility shock happens.

## Artifacts

| file | content |
|---|---|
| `build_16_calendar_features.py` | analysis script |
| `enriched_calendar_<sym>.csv` | per-trade enriched canonical incl. calendar features |
| `univariate_calendar.csv` | per (sym, feature, target) Spearman / AUC / MI |
| `headline_auc_calendar.csv` | AUC under 3 feature sets per symbol |
| `dow_outcomes.png` | mean pnl + loser-rate by day of week |
| `fomc_proximity_pnl.png` | mean pnl by FOMC-day distance |
| `findings_16_calendar_features.md` | this note |
