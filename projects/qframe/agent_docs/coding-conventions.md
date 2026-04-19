# qframe — Coding Conventions

*Reference file for the qframe-research skill.  Loaded when writing or reviewing code.*

---

## Sharpe Ratio Conventions

Two distinct Sharpe ratio quantities exist in this codebase.  **Never mix them up.**

| Symbol | Formula | Where used |
|--------|---------|-----------|
| `sharpe_annual` | `ICIR × √252` | Leaderboard display, Gate 3 threshold (> 0.5), DB column `backtest_results.sharpe` |
| `sharpe_per_period` | `ICIR` = `mean_IC / std_IC` | DSR computation (`deflated_sharpe_ratio(sharpe_obs=...)`), IC Sharpe denominator |

### Rules

1. **DB column `sharpe` stores the annualised figure** (= ICIR × √252).  The column is sometimes called `ic_sharpe` in older notebooks — same quantity.

2. **`deflated_sharpe_ratio(sharpe_obs=...)` expects the per-period SR = ICIR**.  Pass `icir`, NOT `sharpe`.  The function multiplies internally by `√T` (number of OOS days), so passing the annualised figure would inflate DSR by `√252 ≈ 15.87`.

3. **`compute_t_stat(ic, sharpe, ...)` expects the annualised SR** (what the DB stores).  It internally divides by `√252` to recover ICIR.  Do not pre-divide.

4. **Leaderboard annotations** must label the axis: `"IC Sharpe (annualised, = ICIR × √252)"`.

5. **Variable naming** in notebooks and source:
   - Annualised: `sharpe`, `sharpe_annual`, `ic_sharpe_annual`
   - Per-period: `icir`, `sharpe_per_period`, `ic_sharpe_daily`
   - **Never** use bare `sr` without a suffix that disambiguates.

### Common Mistake

```python
# ❌ WRONG — passes annualised SR to DSR:
dsr = deflated_sharpe_ratio(sharpe_obs=result['sharpe'], ...)  # sharpe ≈ 4.0

# ✅ CORRECT — pass ICIR (per-period SR):
dsr = deflated_sharpe_ratio(sharpe_obs=result['icir'], ...)    # icir ≈ 0.38
```

---

## IC / ICIR / t-stat Formulas

```
Fast signal (h = 1 day):
  t = icir × √(N_OOS / 252)
  where N_OOS = number of OOS trading days (default 1762)

Slow signal (h = 63 days, non-overlapping windows):
  t = slow_icir_63 × √(N_OOS / 63)
  where N_OOS / 63 is the number of non-overlapping windows

Annualised Sharpe (leaderboard):
  sharpe_annual = icir × √252

Per-period Sharpe (DSR input):
  sharpe_per_period = icir
```

**Slow signal rule:** use the slow formula whenever `|IC@63d| > |IC@1d|` AND `IC@63d` and `IC@1d` have the same sign AND `slow_icir_63` is available in the DB.  The daily t-stat formula overcounts observations by up to `√63 ≈ 8×` for slow signals.

---

## Factor Function Contract

Factor functions must satisfy:
- Signature: `def factor(prices: pd.DataFrame) -> pd.DataFrame`
- Available names in sandbox: `pd`, `np`, `stats` (scipy.stats)
- Input `prices`: close-price panel, shape `(T, N)`, no NaN after warm-up
- Output: factor score panel, same shape as input
- **No `shift(-1)`** on returns or prices — this is look-ahead bias
- **No `expanding()` on full history** — use `rolling(window)` with explicit lookback
- **No external data** — the function receives only `prices`; all computation must be from that alone
- Return `pd.DataFrame` with the same index and columns as `prices` (or a subset of columns)

---

## Look-Ahead Bias Checklist

Before committing a new factor implementation, verify:

1. No `shift(-1)` applied to returns or forward-looking prices
2. No `pct_change(n).shift(-1)` — classic look-ahead pattern
3. No `rolling(…).apply(…)` where the applied function references future rows
4. `check_lookahead_bias()` passes in CI (`tests/test_pipeline.py::TestCheckLookaheadBias`)

**Post-discovery note (2026-04-19):** `impl_82` (`trend_quality_calmar_ratio`) used `pct_change(251).shift(-1)`, causing the OOS IC to include tomorrow's return.  This inflated IC from ~0 to 0.065.  Crypto replication confirmed the true IC = −0.005.  All six new pipeline guards were added as a result.

---

## Multiple Testing Conventions

- Always apply BHY (not Bonferroni) — factor signals are correlated
- Threshold at m=84 positive-IC factors: **t ≥ ~4.0** (α=0.05 FDR)
- For slow signals: use `slow_icir_63 × √(N/63)`, not the daily formula
- DSR check: always call `deflated_sharpe_ratio(icir, n_trials, n_oos_days)`
- **Never cite a t-stat without specifying which formula was used** (fast vs slow)

---

## Naming Conventions

| Concept | Variable / column name |
|---------|----------------------|
| Mean OOS IC at h=1 | `ic` |
| ICIR (mean IC / std IC, daily) | `icir` |
| Annualised IC Sharpe | `sharpe` |
| Non-overlapping 63-day ICIR | `slow_icir_63` |
| OOS t-stat (fast) | `t_stat` |
| Maximum OOS drawdown | `max_drawdown` (negative float, e.g. −0.12) |
| One-way daily turnover | `turnover` (fraction, e.g. 0.10 = 10%/day) |
| Regime lift ratio | `lift` (e.g. 1.5 = 50% improvement) |
| Factor implementation ID | `impl_id` (integer) or `implementation_id` in DB |
| Short snake_case factor name | `factor_name` (e.g. `trend_quality_calmar_ratio`) |

**Always use both ID and name on first mention:** `impl_82 (trend_quality_calmar_ratio)`.  Subsequent mentions: `impl_82`.

---

## Cost Model Parameters

Default cost params (`DEFAULT_COST_PARAMS` in `costs.py`):
- Spread: 10 bps round-trip
- Market impact: Almgren-Chriss, 10% ADV fraction, γ=30
- Short borrow: 50 bps/year

Aggressive params (`AGGRESSIVE_COST_PARAMS`):
- Spread: 20 bps, impact: 50 bps, borrow: 150 bps, funding: 550 bps

Always use `DEFAULT_COST_PARAMS` for initial factor screening.  Use `AGGRESSIVE_COST_PARAMS` only for stress-testing high-turnover factors before Phase 3 advancement.
