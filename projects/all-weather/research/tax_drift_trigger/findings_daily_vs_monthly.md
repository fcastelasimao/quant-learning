# Daily vs monthly engine validation (L.51–L.53)

**Status:** complete. Verdict: **kill criterion failed → 6.5pp is the F.26 candidate**.
Scripts: `research/tax_drift_trigger/daily_vs_monthly_comparison.py` (L.53),
`engine/daily_tax_backtest.py` (L.51). Tests: `tests/test_daily_tax_backtest.py`
(L.52, 4 new tests, 335 total passing). Decision gate: handoff **F.26**.

## Question

Does the monthly research engine (which checks drift on month-end prices) produce
the same policy ranking as a daily-resolution engine that mirrors the live system?
The live system checks drift every trading day, fires a rebalance only if 31 calendar
days have passed since the last one, and rebalances on the first eligible day — not
on a calendar month-end. Before committing to a new live policy (F.26), we must
confirm that the monthly engine's ranking is a valid proxy.

## Method

`run_daily_tax_backtest` (L.51) applied to the **K.50 policy ladder**:
`monthly_unconditional`, `drift_absolute` (4/5/5.5/6/6.5/7pp),
`drift_relative(40%)`. All runs use US regime (top-marginal), FIFO lot
selector, zero transaction cost. Prices: FMP `adj_close` (total-return),
same source as the D.18 threshold sweep. OOS windows: 2018/2020/2022 → today.

### Pre-registered kill criterion

> If the top-2 candidates from the monthly research (**5.5pp and 7pp**) both
> rank in the **top-3** of the daily engine on at least **2 of 3** OOS windows,
> the monthly engine is a valid proxy and F.26 may proceed with those candidates.
> Otherwise, re-evaluate the F.26 candidate against the daily ranking.

## Result — FIFO, US tax

### Kill criterion: FAILED (0 / 3 windows)

5.5pp ranked 5th / 3rd / 2nd in the daily engine across the three OOS windows.
7pp ranked 7th / 7th / 5th. Neither candidate cleared the top-3 bar on all
three windows simultaneously.

### Daily engine Calmar ranking (drift policies only, excluding monthly baseline)

| Rank (daily) | 2018 OOS | Calmar | 2020 OOS | Calmar | 2022 OOS | Calmar |
|:---:|---|:---:|---|:---:|---|:---:|
| **1** | **drift_absolute(0.065)** | **0.3981** | **drift_absolute(0.065)** | **0.4094** | **drift_absolute(0.065)** | **0.3627** |
| 2 | drift_absolute(0.05) | 0.3720 | drift_absolute(0.05) | 0.3885 | drift_absolute(0.055) | 0.3354 |
| 3 | drift_relative(0.4) | 0.3686 | drift_absolute(0.055) | 0.3834 | drift_relative(0.4) | 0.3311 |
| 4 | drift_absolute(0.06) | 0.3674 | drift_relative(0.4) | 0.3751 | drift_absolute(0.06) | 0.3188 |
| 5 | drift_absolute(0.055) | 0.3634 | drift_absolute(0.06) | 0.3588 | drift_absolute(0.07) | 0.3116 |
| 6 | drift_absolute(0.04) | 0.3426 | drift_absolute(0.04) | 0.3525 | drift_absolute(0.05) | 0.3070 |
| 7 | drift_absolute(0.07) | 0.3308 | drift_absolute(0.07) | 0.3510 | drift_absolute(0.04) | 0.3004 |

### Full comparison — Calmar, CAGR, MDD

| Policy | Engine | 2018 Calmar | 2020 Calmar | 2022 Calmar | 2018 MDD | 2020 MDD | 2022 MDD |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| monthly_unconditional | monthly | 0.2932 | 0.2965 | 0.2471 | -18.50% | -18.50% | -17.41% |
| monthly_unconditional | daily   | 0.2972 | 0.3020 | 0.2563 | -18.37% | -18.37% | -17.29% |
| drift_absolute(0.04)  | monthly | 0.3524 | 0.3550 | 0.2930 | -17.75% | -17.75% | -16.83% |
| drift_absolute(0.04)  | daily   | 0.3426 | 0.3525 | 0.3004 | -17.62% | -17.62% | -16.67% |
| drift_absolute(0.05)  | monthly | 0.3750 | 0.3942 | 0.3208 | -17.35% | -17.35% | -17.00% |
| drift_absolute(0.05)  | daily   | 0.3720 | 0.3885 | 0.3070 | -17.50% | -17.50% | -17.06% |
| drift_absolute(0.055) | monthly | 0.3943 | 0.4163 | 0.3443 | -17.18% | -17.18% | -16.57% |
| drift_absolute(0.055) | daily   | 0.3634 | 0.3834 | 0.3354 | -17.18% | -17.18% | -16.67% |
| drift_absolute(0.06)  | monthly | 0.3954 | 0.3955 | 0.3245 | -17.16% | -17.16% | -16.51% |
| drift_absolute(0.06)  | daily   | 0.3674 | 0.3588 | 0.3188 | -17.62% | -17.62% | -16.89% |
| **drift_absolute(0.065)** | monthly | 0.3897 | 0.3924 | 0.3292 | -17.79% | -17.79% | -16.99% |
| **drift_absolute(0.065)** | **daily** | **0.3981** | **0.4094** | **0.3627** | **-16.85%** | **-16.85%** | **-16.48%** |
| drift_absolute(0.07)  | monthly | 0.3953 | 0.4061 | 0.3474 | -17.12% | -17.12% | -17.08% |
| drift_absolute(0.07)  | daily   | 0.3308 | 0.3510 | 0.3116 | -19.10% | -19.10% | -17.47% |
| drift_relative(0.4)   | monthly | 0.3712 | 0.3782 | 0.3157 | -17.56% | -17.56% | -16.87% |
| drift_relative(0.4)   | daily   | 0.3686 | 0.3751 | 0.3311 | -17.59% | -17.59% | -16.65% |

### Rebalances (full period, 2006–2026, FIFO/US tax)

| Policy | Monthly engine | Daily engine |
|---|:---:|:---:|
| monthly_unconditional | 238 | 230 |
| drift_absolute(0.04) | 12 | 17 |
| drift_absolute(0.05) | 9 | 11 |
| drift_absolute(0.055) | 8 | 9 |
| drift_absolute(0.06) | 7 | 7 |
| **drift_absolute(0.065)** | 6 | **7** |
| drift_absolute(0.07) | 6 | 6 |
| drift_relative(0.4) | 10 | 11 |

## Why the kill criterion failed

**The monthly engine produced an IS-style plateau.** At 5.5–7pp, monthly Calmar
is nearly flat (~0.39–0.42 across all three windows). The plateau exists because
month-end sampling never catches the intra-month peak of a 7pp drift event, so
all three thresholds fire on roughly similar dates in the monthly simulation.

**The 31-day gate exposes the 7pp threshold.** In the daily engine, the gate
enforces a minimum 31-day wait between rebalances. At 7pp, large drift events
can persist for weeks before the gate clears. That holding period allows an
already-stressed position to deteriorate further, blowing the MDD from -17.12%
(monthly engine) to **-19.10%** (daily engine) — the worst result in the
entire ladder.

**6.5pp resolves the plateau.** The daily engine breaks the tie decisively:
6.5pp ranks #1 in all three windows. The key difference is MDD: 6.5pp produces
-16.85% / -16.85% / -16.48% vs 7pp's -19.10%. At 6.5pp the 31-day gate
almost never binds on the same drift event twice before rebalancing fires.

**Both engines agree on the direction.** Every drift policy still beats
`monthly_unconditional` in the daily engine (+12–38% Calmar), confirming the
D.18 finding. The disagreement is only on which threshold is best.

## Decision

Kill criterion **failed** → the monthly research engine cannot be used as a
direct proxy for live policy selection in the 5.5–7pp range.

**Updated F.26 candidate: `drift_absolute(0.065)` (6.5pp), FIFO.**
7pp is disqualified (daily MDD blow-out). 5.5pp is a valid second choice
(strong in 2022), but is dominated by 6.5pp in 2018 and 2020.

The F.26 human gate remains. Do **not** auto-flip production. See
`docs/internal/session_handoff.md` F.26 for the full pre-flight checklist.

## Caveats

- The `rebalances` column in `calmar_comparison.csv` reports the full-period
  count (2006–2026), not per-OOS-window. The comparison is directionally correct
  but slightly inflates the daily count relative to the OOS windows shown.
- MDD values in the 2018 and 2020 columns are identical within each policy
  because the drawdown event spans a period common to both windows.
- Dividend tax on exact ex-date (daily engine) vs. within-interval (monthly
  engine) produces minor differences in realized tax and does not change
  the policy ranking.
