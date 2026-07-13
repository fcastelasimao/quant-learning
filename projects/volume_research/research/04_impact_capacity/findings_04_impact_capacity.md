# Findings 04 — Market impact (√-law) and capacity

**Date:** 2026-06-24 · **Tickers:** TQQQ, SQQQ, QQQ · **Method:** square-root impact law
`I = Y·σ·(Q/V)^β` with σ and V **measured** from recent (~2y) data and Y **adopted from the
literature** (Almgren 2005 / Bouchaud, Y ∈ [0.3, 1.0], central 0.5). β = ½ anchor.

## The honesty caveat (read first)

Unlike spread (Block 1) and delay (Block 3), **impact is not measurable from our data** — we never
traded, so there is no counterfactual to fit Y against. Y is a literature prior, so **every capacity
number here is a band, not a point.** The band below spans Y = 0.3 (optimistic) to Y = 1.0
(conservative) — roughly a 12× range. This is honest uncertainty, not imprecision to be polished away.

## Strategy, mathematics

### The strategy (the one term we can't measure, only adopt)

Spread (Block 1) and delay (Block 3) are *measurable* from history. **Impact is not** — it is the
price move *our own* trading would cause, and since we never traded at scale there is no
counterfactual in the data to fit. So the strategy here is different: take the **functional form**
and **constant** from the published literature (the square-root law; Almgren 2005 / Bouchaud), plug
in our **measured** σ and volume `V`, and report the result as a **band** over the literature range
of the constant — never a single fitted line. We measure what we can (σ, V) and borrow what we
can't (the impact constant `Y`).

### The mathematics (as implemented in `slippage/impact.py`)

The square-root impact law — impact of trading `Q` against volume `V` over a matching horizon:

```
I = Y · σ · (Q / V)^β            β = ½ anchor (Almgren 2005: temp. impact ≈ 3/5)
```

- `I` in bps; `σ` = volatility over the same horizon (bps); `Q`, `V` in the same units; `Q/V` is the
  **participation rate**. `Y` = O(1) constant, **adopted** as `Y ∈ [0.3, 1.0]`, central 0.5.

Two inversions of the same law give the capacity answer. The largest order whose impact stays
within a budget `I_budget`:

```
Q_max = V · ( I_budget / (Y·σ) )^(1/β)          # capacity()
```

and the participation rate that produces exactly that impact (volume-independent — "10 bps ↔ X% of
volume"):

```
Q/V = ( I_budget / (Y·σ) )^(1/β)                # participation_for_impact()
```

Because `Q_max ∝ 1/Y^(1/β) = 1/Y²` at β=½, the [0.3, 1.0] `Y` band maps to a ~**12× capacity band**
— this is the bracket on every capacity number, not noise to be averaged away.

### Where it bites

`σ` and `V` are **regime-dependent**: both worsen together in stress (σ↑, V↓), so capacity is
reported normal vs **stress** (p90 σ, p10 ADV), not as one ADV. The single-name law also assumes
*our* book is the binding liquidity — true only up to ~$50M for 3× ETFs (above that the underlying
QQQ/futures binds, and the law **underestimates** cost; W6). And this is one order's *temporary*
impact — repeated 15-min trading accumulates permanent footprint not modelled here (W5).

## Measured inputs (recent ~2y)

| ticker | σ mean (bps/day) | σ median | σ stress | $ADV | $ADV thin (p10) | price |
|---|---|---|---|---|---|---|
| TQQQ | **370** | 339 | 510 | $4.9B | $2.9B | $78 |
| SQQQ | **372** | 341 | 510 | $2.8B | $1.0B | $40 |
| QQQ  | **124** | 113 | 172 | $25.8B | $12.6B | $723 |

(σ from rolling 20-day realized vol. The series is right-skewed, so we report **both** central
summaries — the **mean** (expected-cost / time-average moment, the headline) and the **median**
(typical-day sensitivity); the mean is ~9% higher. See findings_09 for the reasoning and the
`calibrate()` validation. stress = 90th-pct vol. $ADV from daily volume × close; thin = 10th-pct
day.) TQQQ's median 339 bps/day matches the plan's 3.5% assumption.

## Capacity at a 10 bps impact budget (band over Y, central 0.5)

Headline uses the **mean** σ (expected-cost); *typical-day* is the **median**-σ sensitivity.

| ticker | normal (mean σ) | typical-day (median σ) | stress |
|---|---|---|---|
| TQQQ | **$14M** [$4M – $40M] | $17M [$4M – $48M] | $4M [$1M – $12M] |
| SQQQ | $8M [$2M – $23M] | $10M [$2M – $27M] | $2M [$0.4M – $4M] |
| QQQ  | $672M [$168M – $1.87B] | $801M [$200M – $2.2B] | $170M [$42M – $471M] |

The TQQQ typical-day $17M reproduces the plan's $18M worked example from measured inputs; the
expected-cost headline is ~$14M (the mean σ is higher → slightly less capacity). Both sit inside
the 12× Y-band, so the verdict is unchanged. **Read this as: the leveraged-ETF strategies can run
on the order of tens of $M per trade before impact crosses 10 bps in normal conditions — and only
single-digit $M in stress** (thin volume + high vol together, exactly when a signal most wants to
fire). QQQ is ~50× more capacious.

Q: What is the notation **$14M** [$4M – $40M]? What does this mean?

A: $14M is the **central** estimate (Y=0.5). The brackets are the **uncertainty band from the Y
range**: $40M at the optimistic Y=0.3 (less impact → more capacity) and $4M at the conservative
Y=1.0 (more impact → less capacity). Read it as "best guess $14M, but plausibly anywhere from $4M to
$40M depending on the true impact constant Y, which we cannot measure from our data." It is a range,
not a symmetric ± error bar. (The Y-band spans ~10× — far wider than the ~16% mean-vs-median σ shift.)

## The full per-trade cost picture (Blocks 1 + 3 + 4)

The three components dominate at different scales (TQQQ, normal):

(impact at the mean-σ headline; ~9% lower on the median-σ typical-day basis)

| order size | spread (½) | impact (Y=0.5) | which dominates |
|---|---|---|---|
| $100k | ~1 bp | ~0.9 bp | spread / delay |
| $1M | ~1 bp | ~2.6 bps | impact ≈ delay |
| $14M | ~1 bp | ~10 bps | **impact** |
| $100M | ~1 bp | ~26 bps | impact (past validity ceiling) |

Plus **delay/timing risk** (Block 3): ~17 bps/min (1σ, mean 0) — the dominant cost for *small* trades
filled over minutes, but a *risk* not a drag. So: **small size → delay-risk-dominated; large size →
impact-dominated; spread is a ~1 bp floor throughout.**

## Caveats (the SLIPPAGE_PLAN weaknesses, made concrete)

- **W1 — Y is adopted, not fitted.** The 12× band is irreducible from OHLC. Bound it later with live
  fills (compare realized `avg_order_price − decision_price` to the model).
- **W6 — leveraged-ETF ceiling.** Above ~$50M (TQQQ/SQQQ) the binding liquidity is the *underlying*
  (QQQ / Nasdaq futures via creation/redemption), not TQQQ's own book; the single-name √-law
  **underestimates** cost there. Curve is annotated valid to ~$50M for the 3× ETFs.
- **W2 — V is regime-dependent.** Handled via the stress column (thin-volume + high-σ), not a single
  ADV. Stress capacity is ~4× lower than normal.
- **β exponent.** ½ anchor; Almgren 2005 found temporary impact ≈ 3/5. `impact_bps(beta=0.6)` is
  available; at small participation 0.6 gives *less* impact, so ½ is the conservative choice.
- **W5 — per-trade, not aggregate.** This is one order's impact. Trading every 15 min, permanent
  impact accumulates; aggregate capacity over a session is lower. Out of scope for this block.

## Next

This is **per-trade** capacity. The headline **net-Sharpe-vs-AUM** curve (Stage 6) converts it to AUM
using the strategy's turnover and gross edge: AUM drag ≈ (round-trips/yr) × cost(Q at that AUM). That
needs the strategy's trade sizes/frequency wired in.

## Reproduce

```
/Users/.../envs/quant/bin/python research/04_impact_capacity/build_04_impact_capacity.py
/Users/.../envs/quant/bin/python -m pytest tests/test_impact.py -v
```
Outputs in `results/` (gitignored): `market_params.csv`, `capacity_table.csv`, `impact_capacity.png`.
