# Findings E02 — Interruption risk: hazard curve + a simple cost model

**Date:** 2026-07-08 · **Tickers:** TQQQ, SQQQ · **Question:** measure the mid-fill hazard (the
trailing stop fires, or the signal flips, while an order is still being worked) and give the
scheduler a simple cost model for it. **Answer: not a tail concern — a quarter of trades are gone
within ~2.7h, but interruption cost stays modest because it composes with the already-small
alpha-decay curve (E01); regime barely matters (<5pp spread); the real design payoff is the
event-stream interface left for E04.**

## Headline

The hazard curve confirms the plan's own context note almost exactly: **p25 hold ≈ 2.7–2.8h**
(TQQQ 165 min, SQQQ 150 min) — a genuinely common, not-tail, occurrence. But because E01's `g(h)`
is still small at that horizon (g(165min) ≈ 4%), an interruption at the *typical* early exit
costs little; it's only interruptions **combined with a long wait** (few hours in) that compound
into real cost. **Volatility regime barely moves the hazard** (TQQQ: 31.5–34.1% at h=4h across
calm/normal/stress — a 2.6pp spread) — unlike every cost-magnitude measurement elsewhere in this
project, *when* a trailing stop fires doesn't depend much on the vol regime, only cost *size*
does.

## Strategy and mathematics

**Hazard curve:** `P(exit within h) = mean(hold_min <= h)` across trades, at a grid from 15 min to
~1 week, computed overall, by S03's exact volatility-regime tercile split, and by `exit_reason`.
**Scope caveat, stated prominently per the module docstring:** the TRADES CSVs label
**>99.9% of exits as `TRAIL_STOP`** (TQQQ: 2,341/2,343; SQQQ: 1,929/1,930) — there is no
separately-labeled "signal flip" exit in this dataset. So the hazard curve measures the
**observed exit-timing hazard**, dominated by the trailing stop, as a *proxy* for "any
interruption" — it is not decomposed by cause beyond what `exit_reason` offers.

**Interruption-cost model**, given an order interrupted at time `h` having filled fraction `phi`:

```
mode="cancel":       cost = phi * g(h) + (1 - phi) * 1.0     # residue's edge fully forfeited
mode="complete_now":  cost = g(h)                             # residue rushed, no EXTRA delay
```

`g(h)` is E01's fitted alpha-decay curve. `"cancel"` treats the unfilled residue as a total loss
of its edge (weight 1.0 — you never got that part of the position on at all); `"complete_now"`
treats the whole order as uniformly suffering only the delay-forfeiture already priced by `g(h)`,
explicitly **not** pricing the extra impact cost of rushing an unsliced residual (that's
`expected_slippage_bps`'s job, kept separate). Both modes are labeled **Modeled**, not measured —
this is a design choice for the scheduler to reason about, not an empirical finding.

**Design for E04 (owner requirement):** `interruption_hazard`/`interruption_cost` describe the
**historical-average** hazard — an expected-value input. E04's replay harness is built to accept
an **event stream** (actual stop-fire/signal-flip timestamps) instead, so a real backtest or live
run can bypass these averages entirely in favor of ground truth. This module doesn't need to
change for that; it's simply not called when real events are available.

## Numbers

**Hazard curve, overall** (tier: **Measured**), P(exit within h), %:

| h | 15m | 30m | 1h | 2h | 4h | 6.5h | 13h | 1d | 2d |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TQQQ | 0.9 | 3.6 | 10.2 | 20.1 | 32.7 | 38.5 | 38.5 | 76.1 | 83.1 |
| SQQQ | 0.8 | 2.8 | 9.7 | 20.2 | 36.8 | 41.7 | 41.7 | 81.7 | 86.9 |

p25 hold: TQQQ 165 min (2.7h), SQQQ 150 min (2.5h) — matching the plan's own context note
("p25 hold ≈ 2.8h") closely. The **flat stretch between 6.5h and 13h** (both tickers unchanged)
reflects that `hold_days` is calendar time, not trading time — a position held "overnight" jumps
straight from end-of-session to next-session-open with no trading-hour hazard accruing in
between; the **jump between 13h and 1d** (38.5%→76.1% TQQQ) is where overnight-held positions
cluster.

**By volatility regime, h=4h** (tier: **Measured**) — small spread, no clean monotonic ordering:

| ticker | calm | normal | stress |
|---|--:|--:|--:|
| TQQQ | 34.1% | 31.5% | 32.3% |
| SQQQ | 41.2% | 35.8% | 33.1% |

Unlike cost-*magnitude* measurements elsewhere (which consistently order calm<normal<stress),
interruption *timing* doesn't — a real, if modest, distinction the scheduler should respect: size
the interruption-cost impact by regime (via E01/predict.py's regime-aware pieces), but don't
expect the *hazard* itself to shift much with regime.

**Cost model sanity** (tests, `tests/test_interruption.py`, 13 tests): `cancel` with `phi=1` ==
`g(h)` exactly; `cancel` with `phi=0` == 1.0 (total forfeiture); `cancel` always costs at least as
much as `complete_now` for any partial fill; `complete_now` is `phi`-invariant by construction.

## Caveats

- **No separately-observable "signal flip"** — flagged above; the hazard curve is an
  exit-timing proxy dominated by the trailing stop, not a cause-decomposed hazard.
- **`hold_days` is calendar time** — the flat 6.5h-13h stretch and the following jump are a
  real artifact of how the underlying data defines "hold," not a data error; documented so a
  reader doesn't mistake it for noise.
- **The cost model is a deliberately simple v1**, explicitly labeled Modeled — `"complete_now"`
  doesn't price the rushed-residual's extra impact; a more careful version would compose with
  `predict_slippage`'s impact band for the residual notional. Left as a known extension.
- **Regime-conditioning uses S03's exact tercile split** (reused from E01/P01) — consistent
  across the project, but the same daily-vol proxy limitation applies (a trailing 20-day measure,
  not a live nowcast).

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/scheduler/02_interruption_risk/build_02_interruption_risk.py
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_interruption.py -v
```
Outputs in `results/` (gitignored): `hazard_overall_{SYM}.csv`, `hazard_by_regime_{SYM}.csv`,
`hazard_by_exit_reason_{SYM}.csv`, `interruption_hazard.png`.
