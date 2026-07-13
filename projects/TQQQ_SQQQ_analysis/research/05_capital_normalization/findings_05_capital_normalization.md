# 05 — Capital normalization

> Period: Full history 2013–2026, constant-notional equity convention.

**Scope.** Recompute baseline equity-curve metrics under the **constant-notional** convention, replacing the existing pipeline's `pnl / capital_before` chain that compounds across the six concatenated source CSVs and inflates `total_return_chain` and CAGR. Compare side-by-side with the old run's headline `traditional_metrics_baseline.csv` from `full_history_research/20260528_1402_feature_scan/`.

Constant notional means: each trade contributes `pnl_pct / 100` to a fixed-base daily P&L stream. Equity is additive (`equity_t = 1 + cumsum(daily_pnl_t)`), no compounding. Daily series indexed on a business-day calendar with zero-fill for no-trade days. Sharpe / Sortino / max-DD use that daily series; CAGR is replaced by **annualized arithmetic return** = `total_return / years`, which is the right framing under "how much is made in percentage, not by the whole trade".

## Headline comparison

| metric | TQQQ old | TQQQ new (full) | TQQQ new (regime) | SQQQ old | SQQQ new (full) | SQQQ new (regime) |
|---|---:|---:|---:|---:|---:|---:|
| CAGR (annualized return) | 285 % | **146 %** | 152 % | 180 % | **117 %** | 120 % |
| Sharpe (daily) | 4.01 | **4.00** | 4.09 | 2.17 | **2.28** | 2.24 |
| Sortino (daily) | 9.48 | 9.32 | 9.48 | 5.98 | 5.96 | 5.70 |
| Calmar | 17.6 | **8.0** | 8.3 | 8.8 | **5.1** | 5.2 |
| Ulcer index | 4.16 | 4.50 | 4.74 | 5.80 | 6.19 | 6.11 |
| Max drawdown | −16.2 % | **−18.3 %** | −18.3 % | −20.4 % | **−23.0 %** | −23.0 % |

## What changed

1. **CAGR roughly halved.** The old pipeline's compounded total return (`total_return_chain` ≈ 1.29 × 10⁸ for TQQQ) was an artifact of compounding (1 + r) across six CSV resets, each of which starts with $10k capital. Stripping the compounding gives an arithmetic annualized return of **146 % (TQQQ)** and **117 % (SQQQ)**. These are still very high — the strategy averages ~0.86 pp per trade and runs 150–250 trades/year — but they're now interpretable as "% of fixed notional per year" instead of a misleading compounded multiplier.

2. **Sharpe is essentially unchanged.** Mean-to-std ratio is scale-invariant; the compounding artifact didn't move it materially (4.01 → 4.00 TQQQ, 2.17 → 2.28 SQQQ). **The old Sharpe was already approximately right — use it as-is across past runs.**

3. **Sortino is essentially unchanged**, same reason.

4. **Calmar dropped by ~50 %** because Calmar = annualized return / |max DD| and the numerator just dropped by ~50 %. Cross-rule Calmar comparisons inside the same old run are still valid (relative ranking unchanged); absolute Calmar across runs needs the constant-notional version.

5. **Max DD got slightly worse** (−16 % → −18 % TQQQ, −20 % → −23 % SQQQ). Compounded equity hides drawdowns when equity is low; additive equity exposes them at their true depth.

6. **Ulcer index got slightly worse** (same mechanism).

7. **Regime-labeled subset is metrically comparable to full data** (Sharpe 4.00 → 4.09 TQQQ, 2.28 → 2.24 SQQQ). Filtering to the 76 % with regime labels doesn't materially shift the equity-curve characteristics. The regime filter is safe to apply for the rest of this pass without worrying that we're carving out a biased subset.

## How to read this going forward

- **Use the "new (full)" column** as the corrected baseline whenever absolute equity metrics are reported.
- **Sharpe / Sortino can be quoted from old runs** without correction.
- **CAGR, Calmar, total return, Ulcer, max DD must come from the constant-notional series**; the old values are upper bounds, not realities.
- The arithmetic-annualized 146 % / 117 % figures assume each trade gets the full notional. In practice with sized positions and capital reserved, the realized number would be lower. They serve as a comparability anchor for *relative* analyses (rule A vs rule B), not as an investable headline.

## How to read the plots

**`equity_curve_compare.png`** — two-panel chart (TQQQ top, SQQQ bottom) showing both equity conventions overlaid.

- **Blue line, left y-axis (linear scale)**: constant-notional equity = `1 + Σ pnl_pct/100`. Grows roughly linearly because each trade contributes ~1 % of $1 notional regardless of running equity.
- **Red line, right y-axis (LOG scale)**: compounded equity = `Π (1 + per-trade r)` chained across CSV resets — i.e. the old pipeline's `total_return_chain` view.
- **Why two y-axes**: the compounded line ends up at ~1.29 × 10⁸ for TQQQ (the inflated old number); plotting both on the same linear axis would make the blue line invisible. Log scale on the right makes the red curve readable.
- **What to look for**: the red curve has visible step-jumps where the next source CSV starts (because capital reset to $10k there compounded multiplicatively in the chain). Those steps are the artifact. The blue curve has no such artifact — it grows steadily.
- **Why it matters**: visualizing the discrepancy is the cleanest way to see why the old 285 % CAGR is wrong and the constant-notional 158 % is the right cross-rule comparison metric.

## Artifacts

| file | content |
|---|---|
| `build.py` | analysis script |
| `metrics_constant_notional.csv` | constant-notional metrics for full and regime-labeled scopes per symbol |
| `metrics_compare.csv` | side-by-side comparison vs the old inflated baseline |
| `findings.md` | this note |
