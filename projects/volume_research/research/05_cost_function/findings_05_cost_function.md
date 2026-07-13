# Findings 05 — Size-aware cost function (Stage 5)

**Date:** 2026-06-24 · **Tickers:** TQQQ, SQQQ, QQQ · **Method:** compose Blocks 1+3+4 into
`cost(Q, participation)` with the volume↔time link `t = (Q/ADV)/participation·day`. `slippage/cost.py`.

## Headline

Assembling the three measured pieces surfaces a **correction to Block 4's capacity**: that number
assumed leisurely day-long execution. Our strategy decides every 15 min and must fill within that
window, which **forces fast trading and cuts real per-trade capacity ~26× below the day-execution
figure.**

The same **$14M TQQQ** order (the mean-σ expected-cost capacity; ~$17M on the median-σ typical-day
basis):


| execution                 | participation | impact       | timing (1σ) |
| ------------------------- | ------------- | ------------ | ----------- |
| over a day (Block 4 view) | 0.3%          | **9.8 bps**  | 370 bps     |
| in 15 min (our cadence)   | 7.4%          | **50.2 bps** | 72 bps      |


Block 4's "$14M at 10 bps" is real *only* if you accept ~370 bps of timing risk by working it all day.
Fill it in 15 min and the impact alone is ~50 bps.

## Strategy, mathematics

### The strategy (assemble the three pieces into one knob)

Blocks 1, 3, 4 each measured one cost in isolation. Here we compose them into a single
`cost(Q, participation)` — but the three don't live on the same axis until you notice what links
them: **volume sets the fill time.** Trade a bigger slice of the market per minute (higher
participation) and you fill fast but push the price more (**impact↑**); trade slowly and you bleed
less impact but sit exposed to drift longer (**timing↑**). That opposition — impact vs timing,
governed by how fast you go — is the Almgren–Chriss trade-off, and participation is the dial.

### The mathematics (as implemented in `slippage/cost.py`)

The volume↔time identity that ties everything together:

```
fill_time   t = (Q / ADV) / participation · day        # minutes, given the day length
```

Then the three components, all in bps:

```
spread = half_spread                                   # fixed floor (Block 1)
impact = Y · σ_daily · participation^β                 # ↑ with speed (Block 4 √-law, β=½)
timing = σ_1min · √t,   σ_1min = σ_daily / √(min/day)  # ↑ with exposure time (Block 3)

expected_cost = spread + impact                        # the MEAN drags
risk_adjusted = spread + impact + timing               # + the 1σ timing RISK
```

**Consistency check:** at `participation = Q/ADV` (work the order over a full day, `t = 1 day`) the
impact term collapses back to Block 4's exact √-law `Y·σ·√(Q/ADV)`.

Two derived quantities. The risk-aversion-weighted optimum (the A–C dial):

```
participation* = argmin_p [ expected_cost(p) + λ · timing(p) ]      # optimal_participation()
```

and capacity at a fixed execution window — invert `spread + Y·σ·p^β ≤ budget` for `Q`:

```
p_max = ( (budget − half_spread) / (Y·σ) )^(1/β)
Q_max = p_max · (exec_minutes / day) · ADV                          # capacity_at_horizon()
```

### Where it bites

`expected_cost` (spread+impact) is a **mean drag**; `timing` is a **variance** (mean≈0). They are
different channels and must **not** be summed into one headline cost — capacity-by-horizon
(expected only) is therefore **λ-free**, while any *optimal-execution* bps figure is λ-dependent
(deferred to Stages 6–7). Inherits the adopted-`Y` band and the $50M / W5 caveats from Block 4.

## Capacity vs fill horizon (≤25 bps **expected** cost, spread+impact)


(mean-σ expected-cost headline; median-σ typical-day capacity is ~15% larger)

| ticker | 1 min  | 5 min  | **15 min** | 60 min | day   |
| ------ | ------ | ------ | ---------- | ------ | ----- |
| TQQQ   | $218k  | $1.1M  | **$3.3M**  | $13.1M | $85M  |
| SQQQ   | $120k  | $600k  | **$1.8M**  | $7.2M  | $47M  |
| QQQ    | $10.2M | $50.8M | **$152M**  | $609M  | $3.96B |


(This table uses **expected** cost only — no risk weighting — so it is independent of risk aversion.)
**The leveraged-ETF per-trade capacity, executed on the strategy's 15-min cadence, is single-digit $M
— not the tens of $M the day-execution √-law implied.** QQQ stays large (~$180M).

## The trade-off (Almgren–Chriss)

For TQQQ $10M the cost is U-shaped in participation — impact rises (∝√p), timing falls (∝1/√p) — with
an interior optimum ≈ **9% POV** (~9-min fill), where impact ≈ timing ≈ 51 bps. Optimal execution for
a few sizes (TQQQ, risk-aversion λ=1):


| order | POV  | fill    | impact | timing(1σ) |
| ----- | ---- | ------- | ------ | ---------- |
| $1M   | 2.9% | 2.7 min | 28.8   | 28.3       |
| $5M   | 6.3% | 6.2 min | 42.7   | 42.8       |
| $10M  | 9.0% | 8.7 min | 50.9   | 50.7       |


## Important caveat — timing is a *risk*, and λ matters

Impact is an expected **drag** (lowers mean return); timing is a **risk** (mean ≈ 0, lowers Sharpe
through *variance*). They are different channels and should **not** be summed into one "cost" without
care. The optimal-execution table above weights 1σ timing as a cost (λ=1) — a risk-averse choice; it is
why even a $1M order shows ~29 bps. A risk-neutral trader (λ=0) executes slowly and pays only the small
impact, bearing the timing variance instead. **The capacity-by-horizon table is the robust, λ-free
result; the optimal-execution bps are λ-dependent and illustrative.** Pinning λ is a Stage-7 (cost-aware
sizing) and Stage-6 (Sharpe) question — there, impact enters the return, timing the variance.

Inherited caveats: **Y is adopted, not fitted** (band, Block 4); single-name √-law valid to ~$50M for
the 3× ETFs (W6); **permanent impact / repeated-trade accumulation (W5) not modelled** — this is
single-order temporary impact.

## Next (Stage 6)

The net-Sharpe-vs-AUM curve: at each AUM, the strategy's trade size and turnover imply a per-trade
notional → `cost(Q)` here → impact as a return drag and timing as added variance → net Sharpe. Needs
the TQQQ/SQQQ **turnover, per-trade size vs AUM, and gross (pre-cost) Sharpe** from `TQQQ_SQQQ_analysis`.

## Reproduce

```
/Users/.../envs/quant/bin/python research/05_cost_function/build_05_cost_function.py
/Users/.../envs/quant/bin/python -m pytest tests/test_cost.py -v
```

Outputs in `results/` (gitignored): `capacity_by_horizon.csv`, `cost_function.png`.