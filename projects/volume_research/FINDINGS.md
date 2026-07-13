# Slippage & Capacity — Findings Summary

*2026-06-25 · TQQQ / SQQQ (15-min decision strategy) · detail in `research/0N_*/findings_*.md`*

Map of all stages and current status: see [ROADMAP.md](ROADMAP.md).

## TL;DR

- We replaced the backtest's **flat 20 bps** (size-blind) cost with a **size-aware** model
  calibrated to our instruments — so our Sharpes are now honest about scale.
- **Capacity is small for the 3× ETFs, and the picture holds up under a more careful look.** The
  original (15-min-pinned) estimate: edge largely gone by **~$14M (TQQQ) / ~$9M (SQQQ)**. Refreshed
  (C02) with the scheduler's own chosen horizon and the full alpha-forfeiture + interruption cost
  charged (not just execution): central Sharpe-zero **~$65M (TQQQ) / ~$37M (SQQQ)** (envelope
  ~$13M–$300M+) — a real but moderate extension, not a dramatic one, and still capped well short
  of $1B. See the headline section below for both.
- **Two cost channels behave oppositely:** market **impact** is an expected *drag* (grows with
  size), execution **timing** is a *risk* (grows with delay). Spread is a tiny floor.
- **Actionable now:** live fills show the strategy is **momentum** and pays **~14 bps** chasing
  entries with limit orders — **crossing instead saves ~13 bps/entry** at current size (the model +
  planner now encode this).
- Deliverable: a small, drop-in **`slippage` library** any backtest can use, plus the capacity
  curve and a cost-aware sizing rule.

## The question

Not "what does cost today" (tiny trades → cost is ~constant). The question is **capacity**: how does
cost grow with order size, and **how much capital can the strategy run before slippage eats the
edge?** Execution (how to fill cheaply) is a solved broker service — we build the layer *above* it.

## What slippage is made of

| Component | Size (TQQQ) | Role | Verdict |
|---|---|---|---|
| **Spread** | ~0.7–1 bp half (≤3.5 stress) | fixed floor | small — under the flat 5 bps, **not binding** |
| **Timing / delay** | ~17 bps @1-min, ~59 @15-min (1σ) | **risk** (mean≈0) | dominates *small* trades; inflates variance, not mean |
| **Market impact** | ~10 bps at $14M, grows as √(size) | **drag** (expected) | dominates *large* trades; the capacity driver |

Impact uses the square-root law `I = Y·σ·√(Q/V)`. The constant **`Y` is adopted from the
literature, not fittable from our data** — so every impact/capacity number is a **band**, not a point.

*Plots — spread: `research/01_spread_estimate/results/cs_spread.png`; timing risk + its
distribution: `research/03_delay_cost/results/delay_cost.png`, `delay_distribution.png`; impact &
capacity: `research/04_impact_capacity/results/impact_capacity.png`; the impact-vs-timing trade-off:
`research/05_cost_function/results/cost_function.png`.*

## Headline: net Sharpe vs AUM (the capacity curve)

Gross (pre-cost) Sharpe: **TQQQ 4.9, SQQQ 2.8**.

### Current (C02): scheduler horizon + envelope band, own measured edge

Cost at each AUM = execution (Almgren/√-law envelope, at the scheduler's own chosen horizon h*,
not a fixed 15-min pin) **+ alpha-forfeiture + interruption cost** — the full
`schedule_order` objective, not execution alone (an early draft of this table omitted the
alpha/interruption add-on and overstated capacity — see findings_C02). `edge_bps` is each
ticker's own measured average trade return (TQQQ 107.1 bps, SQQQ 103.9 bps), not an assumption.

| AUM | TQQQ Sharpe (env mid) | [env lo, env hi] | SQQQ Sharpe (env mid) | [env lo, env hi] |
|---|---:|---:|---:|---:|
| $1M | 3.92 | [4.62, 3.07] | 2.07 | [2.58, 1.46] |
| $10M | 2.56 | [4.20, 0.52] | 1.15 | [2.23, −0.20] |
| $30M | 1.40 | [3.79, −1.54] | 0.27 | [1.96, −1.83] |
| $100M | −0.75 | [3.12, −5.05] | −1.30 | [1.46, −4.54] |

*(Numbers refreshed 2026-07-09 after audit fixes #1+#2 — stretched-exp `g(h)` and the removed
objective double-count. Both were conservative biases, so central capacity moved up modestly: the
central zero-crossing roughly doubled vs the pre-fix C02 run.)*

**Capacity verdict:** TQQQ Sharpe-zero (central) sits at **~$65M**, SQQQ **~$37M** (both were ~$14M
/ ~$9M under the 15-min pin) — a **moderate** extension, not dramatic, because the longer horizons
a bigger order needs still cost real alpha-forfeiture and interruption risk (E04: this benefit
shrinks, and can reverse, at large size in a real historical replay — the curve above is a model,
not a replay). The **envelope band is wide** — TQQQ's "gone" AUM ranges from ~$13M (pessimistic
edge) to $300M+ (optimistic edge), reflecting C01's finding that Almgren and the √-law diverge
sharply at these participations; note the central crossing lands right around the **$50M ETF
structural ceiling**, beyond which the single-name model is flagged invalid anyway. 50%-edge
sensitivity roughly **halves** the zero-crossing AUM. Full table (Sortino, maxDD, return-on-AUM,
both tickers): `research/capacity/02_capacity_refresh/findings_02_capacity_refresh.md`.

### Previous (S06): 15-min-pinned, √-law Y-band only — kept for continuity

Headline uses the **mean** σ (expected-cost moment); the *typical-day* median-σ basis is ~9% lower
and is shown in ⟨⟩.

| Per-trade AUM | TQQQ net Sharpe | SQQQ net Sharpe |
|---|---:|---:|
| $1M | 3.7 ⟨3.8⟩ | 1.8 ⟨1.9⟩ |
| $3M | 2.8 ⟨2.9⟩ | 1.1 ⟨1.3⟩ |
| $10M | 1.0 ⟨1.4⟩ | −0.2 ⟨0.0⟩ |
| $30M | −1.8 ⟨−1.2⟩ | −2.4 ⟨−2.0⟩ |

**Capacity verdict (normal conditions):** TQQQ holds half its gross Sharpe to **~$4M**, edge gone
by **~$14M** ⟨~$20M typical-day⟩; SQQQ **~$2M** / **~$9M** ⟨~$2.5M / ~$10M⟩. **Stress (high-vol +
thin volume) roughly halves these.** The numbers carry a wide band from the unmeasurable `Y` (a
~10× range, far wider than the mean-vs-median σ choice) — report the *range*, not a single line.

*σ note: daily volatility is time-varying, so the "normal" σ is a summary of a right-skewed
rolling-vol series. We report both — the **mean** (the expected-cost / lifetime-average moment, the
headline) and the **median** (typical-day). They differ ~9% in σ / ~16% in capacity, both inside
the Y-band. Stress is the separate p90. See findings_09 for the `calibrate()` validation.*

*Plot — `research/06_capacity_curve/results/capacity_curve.png` (net Sharpe vs AUM: Y-band + stress
line, and the λ view).*

## Cost-aware sizing extends the ceiling

Don't trade all-in at large AUM — **size down**. Because impact is convex, the cost-optimal *trade
size* saturates (**~$6.8M TQQQ / ~$3.6M SQQQ**); above that, deploy a shrinking fraction and leave the
rest idle. This **holds the Sharpe flat** instead of going negative — the real cost of scale then
shows up as **return-on-AUM falling** (idle capital), not as a broken strategy. The sizing cap
composes with the existing confidence signal: `final size = min(1 − p_severe, cost-optimal fraction)`.

*Plot — `research/07_cost_aware_sizing/results/cost_aware_sizing.png` (deploy fraction f\* and net
return-on-AUM vs AUM, for a few risk-aversion levels).*

## Reality check: live fills (May–June 2026)

We parsed the live Alpaca logs (30 fills) and compared realized slippage to the model. Two takeaways:

- **It validates the small-size end.** Realized cost is **spread + timing scale** (tens of bps), with
  no impact signal — consistent with impact being immaterial at retail size, as the model says.
- **It resolves how timing enters.** Buy fills average **+14 bps adverse (not ~0)** — the strategy is
  **momentum** (buys breakouts with a limit order, price runs away as it chases), so timing is a
  **signed entry drag**, not a symmetric risk. Market-order *sells* are clean (~0 bps).
- **The library now encodes both.** `CostModel(entry_drag_bps=…)` charges the signed drag as a mean
  cost (on top of, not instead of, the symmetric timing variance), and `plan_execution` returns
  `cross_entry`: for these narrow-spread ETFs it says **cross the entry** (spread ~0.7 bp ≪ 14 bp
  chase drift), turning ~14 bps/entry into ~the spread. **That is the cheap win — an execution-style
  fix, not a model change**, worth ~13 bps per entry at current (retail) size.

*(30 fills / 6 weeks — directional, not statistical; can't yet validate impact — trades too small.)*
*Plot — `research/08_live_validation/results/live_validation.png`.*

## What we deliver

- **`slippage/`** — a dependency-light library (spread + impact + timing, toggleable) that drops
  into a backtest to replace flat cost handling, with a `calibrate()` for any ticker and an optional
  signed **momentum entry drag** (`entry_drag_bps`) for signal-correlated execution. See
  `slippage/README.md`.
- **The capacity curve** (net Sharpe vs AUM, with uncertainty band + stress line). *Assumes efficient
  entry execution — the momentum entry drag is a current-execution P&L item (fix by crossing), not a
  capacity input: it's flat in size, so it dents retail Sharpe, not the impact-bound ceiling.*
- **A cost-aware sizing rule** integrating with `1 − p_severe`.
- **An execution planner** — `plan_execution(order)` → participation rate (how fast), fill horizon,
  slice plan, expected-cost band, and a **cross-vs-rest entry call** (`cross_entry`); the urgency to
  hand a broker's POV/IS algo. (Earns its keep at $M+; at retail the answer is "just cross.")

## Caveats to state out loud

- **`Y` band is irreducible from OHLC** (~12× range) — shrink it later with live fills as they
  accumulate. Always quote capacity as a band.
- **Valid to ~$50M for the 3× ETFs** — above that the binding liquidity is the underlying
  (QQQ/futures), not TQQQ's own book.
- **Per-trade, not aggregate** — repeated 15-min trading accumulates footprint; true capacity is
  somewhat lower.
- **Strategy type declared: momentum** (live fills) — so the timing cost has a **signed entry drag**
  (~14 bps) on top of the symmetric risk; it's in the model (`entry_drag_bps`) and the fix (cross
  entries) is in the planner. Re-measure the drag as live fills accumulate.

## Bottom line

The TQQQ/SQQQ strategies are **genuinely capacity-constrained to the low tens of $M per trade** —
viable for a meaningful book, but not a $100M+ single-name strategy without hitting the underlying.
The backtests are now honest about this, and we have the tooling (cost model + capacity curve +
sizing) to size capital to the edge rather than guess.

---

*Plots live in each stage's `results/` (gitignored); regenerate any of them by running that stage's
`research/0N_*/build_0N_*.py` in the `quant` env.*
