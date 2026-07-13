# Findings 09 — `calibrate()` validation (Stage 11)

**Date:** 2026-07-03 · **Tickers:** TQQQ, SQQQ, QQQ · **Method:** run the library's
`slippage.calibrate()` on the *real* daily + 15-min DBs and check it reproduces the numbers the
research chain published from bespoke, inline measurement code.

## Why

`calibrate()` is the library's one data-touching path, but until now it was exercised only by
*synthetic* recovery tests (`test_calibrate.py`). The HANDOFF's top open item was to confirm it
reproduces the **measured** findings — σ, $ADV, half-spread — from actual pulls, so a colleague
who calls `calibrate(daily, intraday)` gets the same inputs the capacity work was built on.

## What reproduces (probe: `build_09_calibrate_validation.py`)

Every *measured* statistic reproduces to well within tolerance:

| metric | TQQQ | SQQQ | QQQ | vs published | max gap |
|---|---|---|---|---|---|
| $ADV | 4.93B | 2.81B | 25.8B | findings_04 | +0.7% |
| thin $ADV (p10) | 2.93B | 1.02B | 12.6B | findings_04 | +2.1% |
| σ stress (p90) | 510 | 510 | 172 | findings_04 | +0.3% |
| σ **median** | 339 | 341 | 113 | findings_04 | +0.4% |
| half-spread (15-min CS) | 0.74 | 1.00 | 0.72 | findings_01 | +0.4% |

So the library, fed the raw DBs, recovers the whole measured input set. The only value that did
**not** match at first was **σ normal** — and that turned out to be a definition choice, not a bug.

## The one discrepancy → the σ mean/median decision

`calibrate()` reported σ normal **~9.2% higher** than the published 339/341/113 (370/372/124). Root
cause: **σ is time-varying**, so we compute a *rolling* 20-day realized vol → a ~504-long series,
then collapse it to one "normal" scalar. That collapse is a **summary-statistic choice**, and the
series is right-skewed:

- The published chain (build_04/05, build_06's hardcoded dict, build_07) used the **median**.
- `calibrate()` used the **mean**.

Both are legitimate; they answer different questions, so we now **report both**:

- **mean σ → expected-cost / lifetime-average moment.** Impact cost per day ≈ `Y·σ_t·√(Q/V)`; the
  average over many trades is `Y·√(Q/V)·mean(σ_t)`. Cost is linear in σ, so the **mean** is the
  moment that makes the average-cost / net-Sharpe math exact. This is the **library default**
  (`sigma_daily_bps`, used by `normal`) and the new **headline**.
- **median σ → typical-day moment.** Cost on a representative single day, with cleaner regime
  separation (the p90 **stress** figure already carries the tail). Exposed as
  `sigma_daily_median_bps`, kept as the labeled **sensitivity**.

The gap is real but **immaterial against the dominant uncertainty**: 9.2% on σ → ~16% on capacity
(TQQQ central ~$17M → ~$14M), which sits *inside* the irreducible 12× **Y-band** ($4M–$48M). No
capacity conclusion changes; both summaries say "low tens of $M per trade, single digits in stress."

Precedent: findings_01 already made the analogous call for **spread**, deliberately using the
signed **mean** (the CS point estimate) over the median. Using the mean for σ is consistent.

## What this validates / doesn't

- **Validates:** the `calibrate()` code path end-to-end on real data — the loaders, the CS
  aggregate-clamp, the σ/ADV percentiles — all reproduce the hand-rolled research numbers. A
  colleague's `calibrate()` call is trustworthy.
- **Does not re-open** anything upstream: σ/ADV were always measured; this just confirms the
  packaged path matches. `Y` is still adopted, not fitted (Stage 4), and the entry drag is still a
  live-fill input, never set by `calibrate` (findings_08).

## Guardrail

`tests/test_calibrate_live.py` codifies the reproduction (ADV ±5%, σ_stress ±5%, σ_median ±3%,
half-spread ±10%, and mean > median within 20%), parametrised over the three tickers. It
**skips cleanly** when the DBs aren't present, so it protects the path without making the suite
depend on the data.

## Reproduce

```
/Users/.../envs/quant/bin/python research/09_calibrate_validation/build_09_calibrate_validation.py
/Users/.../envs/quant/bin/python -m pytest tests/test_calibrate_live.py -v
```
Outputs in `results/` (gitignored): `calibration_check.csv`, `calibration_check.png`.
