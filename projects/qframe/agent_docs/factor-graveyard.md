# Factor Graveyard

*Append-only ledger of retired factors. One entry per retired factor, ~100 words each.*
*Purpose: prevent the LLM (or human) from re-proposing factors that were already found to be invalid.*

**When to add an entry:** immediately after a factor is retired via `kb.update_hypothesis_status(id, "retired")` with a confirmed reason (look-ahead bias, data error, structural duplicate of an existing factor).

**Format:** add a new `###` section at the top (newest first). Include: impl_id, factor name, retirement date, reason, and concrete evidence.

---

## Retired Factors (newest first)

### impl_82 / impl_92 — `trend_quality_calmar_ratio` / `calmar_proxy_252`
- **Retirement date:** 2026-04-19
- **impl_id:** 82, 92
- **Reported IC (before retirement):** IC=0.0646, t=10.74 (OOS 2018–2024, S&P 500)
- **Reason:** Look-ahead bias — `pct_change(251).shift(-1)` uses tomorrow's closing price
  at computation time. The Calmar ratio numerator (1-year forward return) was computed
  with a negative shift, meaning each day's factor value incorporated the *next day's*
  return. The new truncated-panel look-ahead guard (boundary check) flags 100% of stocks.
- **Evidence:**
  1. Boundary check: full-panel factor has finite values at the truncated-panel cutoff date,
     but the truncated-panel factor is NaN at the same date — the classic `shift(-1)` signature.
  2. Crypto cross-market replication (25 Binance USDT pairs, 2020–2024): after fixing the shift,
     IC = −0.0048 (t = −0.75). Zero alpha on an independent market.
  3. Fixed version (remove `.shift(-1)`): OOS IC = 0.0138, t = 2.34 — below BHY threshold of
     t ≥ 4.0 at m = 84 positive-IC factors.
- **Gate 3 consequence:** Phase 2.5 Sharpe 4.27 was driven by this factor. Gate 3 REVOKED.
- **Do not re-propose:** any variation of `pct_change(N).shift(-M)` where M ≥ 1 for the
  numerator of a Calmar ratio, momentum ratio, or any return-based signal.

### impl_53 — `mean_reversion_factor`
- **Retirement date:** 2026-04-18 (statistical grounds — not look-ahead)
- **impl_id:** 53
- **Reported IC (before correction):** IC=0.0490, t=8.15 (initial, using fast-signal formula)
- **Reason:** The factor is a genuine slow signal (h=63 days IC >> h=1 day IC). The
  fast-signal t-stat formula (`ICIR × √N_daily_obs`) inflated significance by ~63× relative
  to the correct slow-signal formula. Correct t-stat: `slow_icir_63 × √(N/63)` =
  1.31 — not significant at the BHY threshold of t ≥ 4.0 (m = 84).
- **Evidence:** `slow_icir_63 = 0.247`, N_OOS = 1762 days → t = 0.247 × √(1762/63) = 1.31.
- **Status:** kept in KB for reference (not look-ahead; the math was wrong, not the code).
  May be worth revisiting if universe expands (more tickers → larger cross-section → higher IC).
- **Do not re-propose** the same formula and expect a different t-stat without more data or
  a larger cross-section. The signal exists but is too weak to survive multiple-testing correction.
