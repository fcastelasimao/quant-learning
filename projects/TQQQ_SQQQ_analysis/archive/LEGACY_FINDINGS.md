# LEGACY FINDINGS — RSI Leverage Overlay (paused)

Frozen lab notebook from the original RSI leverage-overlay project (TQQQ/SQQQ canonical 2020–2026 datasets). **Paused** as of 2026-05-28 in favor of the full-history pattern discovery work. All findings below refer to the old canonical CSVs (`TRADES_TQQQ_canonical.csv`, `TRADES_SQQQ_canonical.csv`) which are no longer present in the active project. Numbers do not transfer to the new 2013–2026 dataset.

For the active project's findings, see `FINDINGS.md`.

---

## Headline takeaways (read this section first)

- **The strategy outperforms B&H TQQQ on risk-adjusted basis by a wide margin** (baseline Sharpe 1.54 vs B&H 0.80; baseline MaxDD 30.5 % vs B&H 81.6 %). The leverage overlay is amplification on top of an already-good underlying strategy.
- **RSI does NOT predict mean trade returns** in a linear sense, **but it powerfully predicts the *shape* of the return distribution** — particularly the downside tail.
- **Heavy-lift / dead-zone structure is the key insight.** The `[55, 60)` RSI bin contributes 26 % of total P/L from 14.6 % of trades. The `[60, 65)` bin contributes 2 % of total P/L from 15.2 % of trades. Drop [55, 60) → CAGR falls from 72.5 % to 48.9 %; drop [60, 65) → CAGR falls only to 71.5 %.
- **Window-cell results in Step 2 are largely explained by whether the window straddles or skips the dead zone.** They're not telling us about RSI signal strength so much as about how the grid carves around heavy-lift bins.
- **Across 22 windowed scenarios, Sharpe is essentially flat (1.49–1.57)** — leverage acts as an exposure dial. CAGR varies (72 % → 97 %) and MaxDD varies (30 % → 39 %) but Sharpe doesn't move with the window choice.
- **At low trigger rates, position matters; at high trigger rates, position washes out.** Counterfactual: target p ≈ 0.30, RSI-windowed final $506k vs random-trigger median $425k (+$81k from position). At p ≈ 0.90, windowed essentially matches random.

---

## Data findings (2026-05-19)

1. **TQQQ canonical CSV has 186 chain gaps** where `capital_end[i] ≠ capital_before[i+1]`. These are external cash flows in the original backtest. **Use `pnl / capital_before` for per-trade portfolio return, not `capital_end / capital_before`.** Walking the chain literally produces a meaningless ~$3.6 × 10³⁷ final equity.
2. **`pnl_pct` is stored in percentage form (×100), not fraction.** Use `pnl_pct / 100` in any return formula. Confirmed in Step 1 EDA: residuals collapse to ≤ 6 bps after %-scaling.
3. **`RSI_entry` only spans `[35, 72]`** in the canonical data — the strategy never enters when RSI is outside that range. Step 1's original threshold sweep grid `{20, …, 80}` would have been mostly inactive.
4. **TQQQ canonical starts with $10,000** (not $5,000 as PROJECT_PLAN.md originally assumed). SQQQ canonical starts with ~$5,000.
5. **`regime_entry` is all-NaN in TQQQ canonical.** Regime-conditional analysis on TQQQ is not possible from canonical data.
6. **76 % of TQQQ trades overlap an SQQQ trade in time.** Important for the combined-portfolio mode that's currently deferred.
7. **TQQQ strategy deploys ~95 % of capital per trade**; SQQQ deploys ~99.8 %. Each was designed assuming sole use of its wallet.
8. **99.94 % of trades exit via `TRAIL_STOP`** (1626/1627 for TQQQ; 1571/1572 for SQQQ). One `FINAL_LIQUIDATION` per file at end of backtest.

## Statistical findings (2026-05-19)

9. **Linear regression `pnl_pct ~ RSI_entry`**: slope = −0.0045, p = 0.397, R² = 0.0004. No monotone linear signal.
10. **Quadratic regression**: peak at RSI ≈ 52, R² = 0.0016, t-stat on quadratic term = −1.36 (p ≈ 0.18). Marginal but suggestive of inverted-U.
11. **Kruskal-Wallis across 5-pt RSI bins** (any cross-bin difference in central tendency): H = 6.86, p = 0.444. Bins are not significantly different in *mean*.
12. **The mean-return story is null; the distribution story is real.** Tail risk, variance, and contribution-density vary strongly by RSI bin. See finding 15.

## Risk-asymmetry findings (2026-05-19)

13. **Hard loss floor at ~−2.1 % when `RSI_entry < 45`**. Worst losses in `[35, 40)`: −2.12 %, −2.11 %, −2.08 %. All `TRAIL_STOP`. Likely tighter stop logic for low-RSI entries in the original strategy.
14. **Loss-side std varies ~2.6x across RSI bins**: 0.36 at `[35, 45)` vs 0.95 at `[45, 50)`. Low-RSI losses are tightly clustered; mid-RSI losses are dispersed.
15. **Low-RSI trades have asymmetric risk** — capped downside, unbounded upside. This makes them paradoxically attractive for a leverage overlay despite low mean returns.

## Contribution decomposition (2026-05-19)

16. **`[55, 60)` is the workhorse**: 14.6 % of trades, **25.9 % of total P/L**. Sharpe-per-trade 0.20 (highest).
17. **`[60, 65)` is the dead zone**: 15.2 % of trades, **2.0 % of total P/L**. Sharpe-per-trade 0.02. Fine bin `[62.5, 65)` has **negative** mean pnl_pct (−0.20 %) — actively losing.
18. **Fine bin `[55, 57.5)` alone**: 6.6 % of trades, **20.5 % of total P/L**. Mean 0.77 % per trade — 3x the unconditional average.
19. **Counterfactual: drop each 5-pt bin, recompute baseline CAGR:**
    - Drop `[55, 60)` → CAGR 48.9 % (Δ −23.6 pp) — biggest hit
    - Drop `[50, 55)` → CAGR 52.9 % (Δ −19.7 pp)
    - Drop `[45, 50)` → CAGR 58.3 % (Δ −14.3 pp)
    - Drop `[40, 45)` → CAGR 59.9 % (Δ −12.7 pp)
    - Drop `[65, 70)` → CAGR 59.1 % (Δ −13.4 pp)
    - **Drop `[60, 65)` → CAGR 71.5 % (Δ −1.0 pp)** — barely matters
    - Drop edges `[35, 40)` or `[70, 75)` → CAGR ~70 % (Δ ≈ −2.5 pp)
20. **The "alpha" lives in two zones**: `[50, 60)` (heavy lifter) and `[65, 70)` (secondary), separated by the dead zone at `[60, 65)`.

## Step 2 engine / metric findings (2026-05-19)

21. **Baseline final equity = $318,723** (CAGR 72.5 %, Sharpe 1.54, MaxDD 30.5 %) over 2020-01-02 → 2026-05-08. Computed via `1 + pnl/capital_before` per trade — see finding 1 for why.
22. **B&H TQQQ over the same window**: $73,205 (CAGR 36.8 %, Sharpe 0.80, MaxDD 81.6 %). Strategy ~doubles B&H CAGR with ~1/3 the drawdown.
23. **Window-cell results, headline numbers:**
    - Best Sharpe: `low50_high60` (1.569) — captures both heavy-lift bins, skips dead zone.
    - Best CAGR: `low35_high70` (97.4 %, $752k final) — maximum trigger rate (92 %).
    - Worst at matched-trigger: `low60_high70` (CAGR 75.7 %) — leads with dead zone.
24. **`vs_bh_beta` is biased low (0.17–0.22)** in current metrics output. Cause: daily-resampled equity series only ticks on trade-close days, producing sparse returns that mute correlation with B&H's full-density returns. Real economic beta is ~0.5. `vs_bh_alpha_ann` and `vs_bh_info_ratio_ann` inherit this bias. Fixable only with intraday marks (v2).
25. **Win rate, max losing streak, avg hold days invariant across all 22 scenarios** (0.5452, 9, 0.72 d respectively). Mathematically correct: sleeve P/L is linear in `pnl_pct` so it preserves the sign of every trade's net P/L; borrow cost is too small (~3 bps of gross at typical holds) to flip signs.
26. **Deflated Sharpe across the 22-cell grid is 0.999** — best Sharpe survives multiple-comparison correction comfortably. But best ≈ average — all cells are within bootstrap CI of each other.
27. **Borrow cost is negligible at observed hold periods.** Median hold 0.20 days × ~9.5 % annual → ~0.005 % per trade. Borrow doesn't materially affect any conclusion at v1 hold lengths.

## Strategy character (2026-05-19)

28. **Mostly intraday**: median hold 0.20 days (~5 h), p90 2.76 days, max 4.11 days.
29. **0 self-overlaps within TQQQ** — trades are sequential. Step 2 per-symbol portfolio is plain sequential walk.
30. **Sharpe ~1.5 across all configurations** — the strategy is high quality, but the RSI gating doesn't sharpen it. Adding leverage scales the curve, doesn't improve it.
31. **Returns are positively skewed** (skew 1.2–1.4 across scenarios). Right tail dominates left tail. B&H TQQQ has slightly negative skew (−0.16) — typical equity-fund profile.

## Open questions / TODO (frozen)

- **Step 1 EDA missed the non-linearity.** Linear regression can only see monotonic relationships; the strategy has a non-monotone risk-asymmetric structure. Step 2.5 should add quadratic + Kruskal-Wallis + variance-equality tests and conditional-distribution plots.
- **Targeted-bin sleeves not yet tested.** Hypothesis: a sleeve that fires only when RSI ∈ `[55, 60)` has the highest Sharpe-per-fire of any rule in this data, at the cost of small absolute CAGR contribution.
- **Skip-the-dead-zone sleeve not yet tested.** Hypothesis: a sleeve that fires when RSI ∈ `[40, 60) ∪ [65, 70)` is the best "wide-coverage" leverage rule, dodging the unproductive zone.
- **Always-on baselines at multiple sleeve sizes not yet computed.** Needed to distinguish "RSI selection has value" from "the strategy is just being leveraged."
