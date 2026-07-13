# 13 — Within-CSV compounding view

> Period: Per-CSV segments 2013–2026, geometric mean across six source CSVs.

**Scope.** Compute compounded equity within each source CSV (where `capital_before` is continuous), then aggregate via geometric mean across CSVs. Reports as an *auxiliary* view alongside constant-notional from item 05.

## Headline

| symbol | n_csvs | constant-notional CAGR (item 05) | **within-CSV geo-mean CAGR** | OLD pipeline CAGR (broken) | median within-CSV Sharpe |
|---|---:|---:|---:|---:|---:|
| TQQQ | 3 | 158 % | **59 %** | 285 % | 5.88 |
| SQQQ | 3 | 113 % | **43 %** | 180 % | 3.89 |

Three different numbers for "CAGR":

1. **Old (broken)**: 285 % / 180 %. Compounded across all six source CSVs as if it were one chain. Inflated because each CSV resets capital to $10k.
2. **Constant-notional (item 05)**: 158 % / 113 %. Each trade contributes pnl_pct/100 to a fixed base. Linear arithmetic, no compounding. The fair number for *cross-rule comparison*.
3. **Within-CSV compounded (this item)**: 59 % / 43 %. Compound within each CSV separately, geo-mean the per-CSV final equities, annualize over the union timespan. The fair number for "what would a single trader compounding capital have achieved (within one continuous backtest)."

## Why the within-CSV CAGR is *lower* than the constant-notional annualized return

Constant-notional sums pnl_pct/100 across years and divides by the span. The result is in "linear % per year" units.

Within-CSV compounded grows multiplicatively, so the same total wealth maps to a lower CAGR over a long horizon — `504^(1/13.4) − 1 = 59 %` per year compounded, but linear would be `(504 − 1)/13.4 = 38× / year` if you tried to interpret it linearly.

Both are correct under different assumptions. **The compounded number is what a trader actually experiences with reinvestment**; the linear constant-notional number is what you'd see if you keep a fixed allocation forever and skim profits.

## How to read these together

- For **cross-rule / cross-feature comparison**: use constant-notional (item 05). It's self-consistent and the scale is the same regardless of starting capital.
- For **headline "what would this strategy have done" claims**: use within-CSV compounded (this item). 59 % CAGR for TQQQ over 13.4 years is the more defensible figure for a presentation.
- **Never use the old 285 % / 180 %** — it's a per-CSV-reset artifact.

## Caveat — the within-CSV CAGR is still a workaround

The three CSVs are still arbitrary backtest segments. A trader running the strategy continuously from 2013 wouldn't have a $10k reset at each CSV boundary. The "true" compounded CAGR would be obtained by re-running the strategy's backtest with a single continuous capital chain — which we don't have access to. The geo-mean of per-CSV CAGRs is the best available approximation given the data shape.

If the data team can re-export the backtest with a continuous capital chain (no resets), that result would supersede both the constant-notional and within-CSV-compounded views as the single authoritative CAGR.

## Sharpe holds up

Within-CSV daily Sharpe medians of 5.88 (TQQQ) and 3.89 (SQQQ) are *higher* than the cross-CSV Sharpes reported in item 05 (4.00 and 2.28). This is because individual CSV segments have less regime-shift noise than the chained equity. Reading: the strategy's risk-adjusted-return within a clean segment is even better than the chained metric suggested.

## Artifacts

| file | content |
|---|---|
| `build_13_within_csv_compounding.py` | analysis script |
| `per_csv_compounded.csv` | per (symbol, source_file) compounded equity + Sharpe |
| `compounding_compare.csv` | the headline row per symbol |
| `findings_13_within_csv_compounding.md` | this note |
