# Findings 08 — Live-fill validation (Stage 9)

**Date:** 2026-06-25 · **Ticker:** TQQQ (SQQQ didn't trade in-window) · **Data:** 572 live Alpaca
cycle logs, 2026-05-22 → 06-24 · **Method:** parse every real fill, compare realized slippage
(decision → fill) to the model.

## Scope (what this can and can't validate)

The live trades are **~$80k–180k** — retail size, where the capacity model says **impact ≈ 0**. So
this validates the **spread + timing** end of the model, **not** the impact/`Y`/capacity curve (you
can't see impact you don't cause). Per CLAUDE.md we do **not** calibrate against these (~6 weeks, too
short); this is a **sanity check** and, more importantly, the way to **resolve W3** (is delay a
symmetric risk or a signed drag for this strategy?).

Realized slippage, signed so **+ = adverse** (worse than the decision price): buys
`(fill−decision)/decision`, sells `(decision−fill)/decision`.

## What the fills show

**30 fills (15 buy, 15 sell) + 1 limit sell that never filled.** Notional $79k–$181k.

| side | n | mean | 1σ | min | max | execution |
|---|---:|---:|---:|---:|---:|---|
| **buy** | 15 | **+14.2** | 35.0 | −45.9 | +93.0 | **limit** (chases the price) |
| **sell** | 15 | −0.3 | 2.1 | −4.0 | +4.2 | **market** (clean) |

(bps.) Model reference: spread floor ~1–2.5 bps one-way; timing ~17 bps 1σ at ~1-min latency.

## Validation verdicts

1. **Magnitude is the timing scale, not the impact scale.** Buy slippage lives in the *tens* of bps
   (1σ 35), not the *hundreds* — and shows **no separable impact signal** (it doesn't grow with the
   small notional spread). Realized retail cost is **spread + timing**, consistent with impact being
   immaterial at $150k. *Caveat:* the √-law, if applied literally to a fast $150k clip, nominally
   returns ~8–18 bps of "impact" — that is **extrapolation below the law's metaorder domain**; the
   fills don't support it, so don't read the model's small-size impact term literally (it's an upper
   bound). The model is for *capacity at scale*, not retail clips.
2. **Effective chase latency ≈ a few minutes.** The buy 1σ (35 bps) sits between the model's **1-min
   (17 bps)** and **5-min (36 bps)** timing levels — i.e. the limit orders take a few minutes to fill
   while price moves, landing on the steeper part of the √t curve. Consistent with the timing model.
3. **Delay is a SIGNED drag here — W3 resolved.** The buy mean is **+14 bps, clearly positive, not
   ≈0.** The engine buys on bullish confirmation (momentum) with a limit order, so price runs away
   while it chases → systematically adverse fills (tails to +93 bps). **This is the momentum case we
   flagged:** for this strategy the timing term is a *signed cost*, not a symmetric risk.
4. **Cost is concentrated on the buy side.** Sells use **market** orders and come in clean (−0.3 bps,
   1σ 2). The asymmetry is an *execution-style* choice (limit entries, market exits), not the market.
5. **Opportunity cost is real and live.** One **limit sell never filled** (`filled_qty=0`) — the
   price-guard / adverse-selection cost (W3) that disappears from the P&L instead of being measured.

## Implications for the cost model

- **Apply the timing term as a signed, adverse drag on entries for this (momentum) strategy** — not
  the symmetric ±σ risk. At current scale this ~14 bps adverse entry is the **dominant** cost, well
  above the ~1–2.5 bps spread and the ≈0 impact. **Now wired in:** `CostModel(..., entry_drag_bps=14)`
  charges this mean once per round trip (entry only), on top of — not instead of — the symmetric
  `timing_bps` variance (they are different moments of the same move). Default 0 keeps the symmetric
  assumption; it is a live-fill input, never set by `calibrate`.
- **The cheap fix is on the execution side**, not the model: limit-chasing entries pay the momentum
  drag; a more aggressive/marketable entry (as the sells already use) would trade some of that
  timing drag for a certain spread cross — the Almgren–Chriss / planner question.
  **Now surfaced:** `plan_execution(..., entry_drag_bps=14)` returns `cross_entry` — a differential
  call (impact cancels between the two styles) comparing the spread you'd cross to the drift you'd
  chase. For narrow-spread TQQQ, 14 bps drift ≫ ~0.7 bp spread → **cross** at every size, matching
  the clean market-order sells above; it only flips to "rest a limit" when the spread exceeds the drift.
- **Suggestion:** entries should cross the spread (marketable order) instead of resting a limit. 
  Crossing costs the half-spread, ~0.7–1 bp, with a certain immediate fill; chasing costs ~14 bps on average. 
  That's a ~13 bps/entry saving at current size, and it's already computed by plan_execution (the cross_entry output). 
  The sells already do this — that's why they're clean. Caveat to carry: the +14 is estimated from 15 fills 
  (standard error ~9 bps), so treat the level as provisional; the decision is robust anyway because you're 
  risking 1 bp to save an estimated 14.
- **Impact stays a future-scale story.** Nothing here contradicts the capacity curve; it confirms its
  premise that today's cost is spread+timing and the impact question only appears at $M+ size.

## Caveats

- **30 fills over 6 weeks — directional, not statistical.** Treat magnitudes as order-of-magnitude
  confirmation, not a calibration. Re-run as fills accumulate.
- **TQQQ only** (SQQQ's signal skipped all in-window entries).
- **Decision price = the engine's logged decision** (15-min bar close proxy); a couple of fast-tape
  fills dominate the buy mean and tails.

## Reproduce

```
/Users/.../envs/quant/bin/python research/08_live_validation/build_08_live_validation.py
```
Outputs in `results/` (gitignored): `live_fills.csv`, `live_validation.png`.
