# 14 — Intraday context enrichment

> Period: IS 2015–2020, OOS 2021–2026 (pre-strict-prior rebuild; see status note).

**2026-06-09 status note.** This item compared intraday features against the original item-06 daily-context artifacts, which later proved to allow same-day daily-bar leakage. The main qualitative conclusion - intraday features did not add enough to justify production use - remains a closed branch, but the exact AUC numbers below are historical and should not be used as current strict-prior benchmarks. Re-run item 14 from the strict-prior item-06 outputs before making any modeling decision based on these rows.

**Scope.** For each trade entry timestamp `T`, compute QQQ and own-ticker (TQQQ/SQQQ) intraday features from the 15-min bars **strictly before** `T` on the same session. Six features modelable (full coverage), two descriptive-only (VIX intraday, ~25 % coverage).

The fetch (`python data_manager.py --intervals 15min --symbols QQQ ^VIX TQQQ SQQQ`) provided:

| symbol | coverage | bars |
|---|---|---:|
| QQQ 15-min | 2011-03-23 → 2026-05-28 | 98,739 |
| TQQQ 15-min | 2010-02-11 → 2026-05-28 | 105,920 |
| SQQQ 15-min | 2010-02-11 → 2026-05-28 | 105,757 |
| ^VIX 15-min | **2023-09-25** → 2026-05-28 | 17,303 |

**The VIX intraday coverage limit is structural** — FMP doesn't have older ^VIX intraday history. With IS = 2015-2020, **zero IS rows have VIX intraday** so it can't enter the modeling pool. We compute it descriptively for the 2.5-year window where it's available, then exclude from features.

## Six modelable intraday features

| feature | definition | coverage |
|---|---|---:|
| `QQQ_intraday_return_since_open` | (last bar before T close) / (day open of QQQ) − 1 | 99.8 % |
| `QQQ_intraday_range_position` | (last close − low so far) / (high so far − low so far) | 99.8 % |
| `QQQ_intraday_realized_vol_13bar` | annualized stdev of last ~13 bar returns | 78–80 % |
| `QQQ_intraday_volume_vs_5d_avg` | bar volume / 5-session avg at same time of day | 99.8 % |
| `SELF_intraday_return_since_open` | own-ticker return since session open | 100 % |
| `SELF_intraday_dist_to_prior_close` | own-ticker last bar close / prior daily close − 1 | 100 % |

## Headline — intraday adds essentially nothing over daily context

OOS AUC on `is_severe_loss`:

| symbol | feature set | n_is | n_oos | tree AUC | L1-logit AUC |
|---|---|---:|---:|---:|---:|
| TQQQ | curated_12 only | 923 | 850 | 0.670 | 0.593 |
| TQQQ | + 21 daily ctx (item 06) | 920 | 850 | 0.716 | 0.716 |
| TQQQ | + intraday (this item) | 709 | 674 | **0.715** | **0.720** |
| SQQQ | curated_12 only | 777 | 725 | 0.616 | 0.508 |
| SQQQ | + 21 daily ctx (item 06) | 772 | 725 | 0.699 | 0.555 |
| SQQQ | + intraday (this item) | 624 | 573 | **0.707** | **0.540** |

TQQQ moves by ≤ 0.005 in either direction. SQQQ tree gains +0.008. **All within noise of the daily-context model.**

## Why intraday flatlines

1. **Sample loss.** Adding the intraday columns forces NaN-drop for the ~22 % of trades without complete intraday context (mostly the QQQ_realized_vol_13bar requirement). Training set shrinks 920 → 709 (TQQQ). Smaller training set partly offsets any feature gain.
2. **Information overlap.** Daily features already encode `QQQ_drawdown_5d` and `QQQ_realized_vol_20d`. The intraday return-since-open mostly recapitulates that signal at lower resolution than daily.
3. **VIX intraday is excluded.** The single most plausibly-informative intraday feature (VIX intraday change since open) can't enter the model because of FMP's 2023-09-25 start. With more years of ^VIX 15-min history, this could move.

## Verdict

- **Modelable intraday features: not worth the complexity.** Daily context (item 06) is where the signal is.
- **VIX intraday: deferred.** Re-run this analysis when FMP has 5+ years of ^VIX 15-min history (or pull intraday VIX from a different source).
- **Intraday is still useful for tighter-stop simulation (item 15)** — that uses *intra-trade* price paths, not entry-context features. Different application.

## Side observations

- `QQQ_intraday_realized_vol_13bar` requires 14 prior 15-min bars on the entry day. Early-session entries (first 3.5 hours) get NaN. ~20 % of trades are early-session.
- `SELF_intraday_return_since_open` is essentially redundant with `QQQ_intraday_return_since_open` for TQQQ (3× leveraged QQQ → 3× the QQQ intraday return + small tracking error). For SQQQ it's −3× of QQQ. Could drop one of them as a follow-up.

## How to read the plots

**`intraday_auc_compare.png`** — grouped bar chart of OOS AUC on `is_severe_loss` across symbols × models × feature sets.

- **Two x-positions**: TQQQ left, SQQQ right.
- **For each symbol, six bars total**: three feature sets × two models.
  - **Color**: C0 = curated only, C1 = + daily context, C2 = + intraday.
  - **Pattern**: solid = tree, hatched (`//`) = L1-logit.
- **Dashed line at 0.5** = random.
- **What to look for**: in pass-1 (item 06) we saw the C1 hatched bar jump way over C0 hatched. Here in item 14 the C2 bar should be **the same height or barely taller than C1** — which is exactly what the chart shows. That visual flatness IS the headline: intraday doesn't move the needle.
- **Why it matters**: if you were tempted to think "more features must help," this plot is the empirical disproof. AUC saturated around 0.72 with the right daily features, and added intraday columns just shrank the sample.

## Artifacts

| file | content |
|---|---|
| `build_14_intraday_context.py` | analysis script |
| `enriched_intraday_<sym>.csv` | per-trade enriched canonical incl. intraday + VIX-intraday (descriptive) |
| `headline_auc_intraday.csv` | the headline table above |
| `intraday_auc_compare.png` | grouped bar chart of AUC under 3 feature sets |
| `findings_14_intraday_context.md` | this note |
