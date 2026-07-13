# Slippage & Market-Impact Research — Conclusions and Plan

> **Historical document (2026-06).** Superseded by [ROADMAP.md](../../ROADMAP.md). Kept for the
> original scoping rationale and Q&A.

*Date: 2026-06-18. Scope: TQQQ/SQQQ strategies that decide every 15 min whether to be in or out.*

This document summarizes what we learned, what is already solved by the market (brokers), what
**we** must build, the equations behind it, and a staged plan. A critical review with weaknesses
and proposed fixes is at the end.

---

## 0. One-paragraph summary

Execution — *how to fill a decided order cheaply* — is a solved, commoditized broker service
(VWAP/TWAP/POV/Implementation-Shortfall algos). We are **not** rebuilding that. Our job is the
layer **upstream** of execution: a **pre-trade cost model calibrated to TQQQ/SQQQ** that makes our
backtests honest, a **capacity curve** (net Sharpe vs. AUM) that tells us how much capital the
strategy can run, and **cost-aware sizing**. We are applying established, published methods to our
specific instruments — solid engineering, not novel science.

### 0.1 What the broker execution algos mean

These are the menu of "how to fill a decided order" that brokers expose. All take a parent order
(buy/sell `Q`) and chop it into child orders; they differ in the *schedule* they follow and the
*benchmark* they are judged against.

| Algo | What it does | Benchmark | When you'd pick it |
|---|---|---|---|
| **TWAP** (Time-Weighted Average Price) | Equal-sized slices at regular time intervals, ignoring volume | Simple average price over the window | Predictable, steady fills; thin or unreliable volume profile |
| **VWAP** (Volume-Weighted Average Price) | Slices to follow the day's historical volume profile (U-shaped — heavy at open/close), so fills track the volume-weighted average | The session's VWAP | Blend into normal flow over a full day; "be invisible" |
| **POV** (Percentage of Volume / participation) | Trades a fixed % of *realized* market volume in real time — speeds up when volume is high, slows when thin | Interval VWAP | Cap your footprint at a bounded share of the market; directly sets the participation rate |
| **IS** (Implementation Shortfall / arrival-price) | Optimises the schedule to minimise cost vs the *decision* price, front-/back-loading by urgency — explicitly trading impact (go slow) against timing risk (price drifts while you wait) | Arrival (decision) price | When the decision price is what you're judged against; this is the Almgren–Chriss objective in product form |

**How this maps to our model:** the broker picks the *schedule*; we only set an **urgency** parameter.
POV is the participation lever that feeds **temporary impact** (the √-law, Stage 4); the IS algo is the
Almgren–Chriss **impact-vs-timing** trade-off operationalised — and the "timing risk" it trades against
is exactly the **delay cost** we measured in Stage 3. Brokers minimise the *avoidable* part of these
costs; the structural floor (spread + impact) remains, and that is what we model upstream.

---

## 1. What slippage actually is (decomposition)

"Slippage" is several distinct costs. The umbrella benchmark is **Implementation Shortfall (IS)**
(Perold 1988): the gap between the *decision price* and the *realized outcome*, including the cost
of shares we wanted but never filled.

| Component | What it is | Scales with | Source for us |
|---|---|---|---|
| **Spread** | Cross bid↔ask at an instant | ~constant (½ spread/side) | Estimate from candles |
| **Temporary impact** | We drain the book by trading *fast*; reverts after we stop | participation rate | Modeled (√-law) |
| **Permanent impact** | Equilibrium price shift; persists | total size | Modeled |
| **Delay / timing** | Price drifts between decision and fill | volatility × time | Measure from 1-min bars |
| **Commission** | Broker/exchange fees | per share/trade | Out of scope (≈0) |

**Spread vs. delay:** spread is the cost of crossing the book *at one instant* (exists in a frozen
market); delay is the cost of the market *moving while we wait* to fill (exists even at zero spread).

---

## 2. What brokers already do (and what they cannot)

Brokers solve **execution**: given a decided order Q, fill it cheaply vs. a benchmark. They
**minimize the avoidable cost** (naive timing, needless spread-crossing, information leakage) but
**cannot remove the structural cost** — the spread you must cross and the impact of moving size.
Impact is a *law*, not an inefficiency.

| Cost | Broker algo can reduce? |
|---|---|
| Avoidable (bad timing, naive market orders, signaling) | **Yes** |
| Structural (spread + `Y·σ·√(Q/V)` impact) | **No** |

A broker answers only *"how do I fill this order now?"* It cannot answer *whether* to trade, *how
big*, *how much capital the strategy can run*, or *whether the backtest Sharpe is honest* — those
are strategy- and capital-specific. **Those are ours.**

---

## 3. The two-layer architecture

- **Signal / alpha layer (every 15 min):** decides the **target position** (all-in / flat, or a
  fraction via `1 − p_severe`).
- **Execution layer (sub-minute, if we ever go live at size):** drives `current → target`. This is
  **one signed mechanism** (buy and sell are just the sign of `target − current`), not two separate
  buy/sell algorithms. In production this is a **broker algo**, parameterized by an urgency setting.

For the research phase we **model** the execution cost; we do not build a live engine.

---

## 4. The core equations

**Square-root impact law** (Bouchaud; Almgren 2005). Impact of an order of size `Q`:

```
I  =  Y · σ · sqrt( Q / V )
```

- `I` = price impact (fraction; ×10⁴ for bps)
- `σ` = volatility over the reference horizon
- `Q` = our order size (shares) — **we choose this**
- `V` = market volume over the same horizon (ADV, or interval volume) — **we observe this**
- `Y` = calibration constant, O(1), typically ≈ 0.3–1

Almgren (2005) splits impact into **permanent** `g(v)=γ·v` and **temporary** `h(v)=η·v^β`, and
empirically found temporary impact closer to a **3/5 power** than ½ — we will test both exponents.

**Total per-trade cost (our Q1 — "what will I pay vs. decision price X?"):**

```
cost(Q)  ≈  ½ · spread  +  Y · σ · sqrt(Q / V)  +  delay
```

**Capacity / sizing (our Q2 — invert the impact term against a budget `I_budget`):**

```
Q_max  =  V · ( I_budget / (Y · σ) )²
```

*Worked example:* `I_budget = 10 bps = 0.001`, `Y = 0.5`, daily `σ = 3.5%`, `V = ADV = 80M sh`
→ `Q_max ≈ 0.0033 · V ≈ 260k shares ≈ $18M` before impact crosses 10 bps. (Use **interval**
volume and matching σ for the 15-min execution view.)

**Spread estimation from OHLC** (no quote data needed): **Corwin–Schultz (2012)** high–low
estimator, or Abdi–Ranaldo. Gives a *time-varying* spread that widens in stress.

---

## 5. Data (status: mostly already in hand)

Shared store: `/Users/.../QuantFinance/data`, one SQLite DB per ticker
(`DB_<TKR>_historical_data.db`), one table per interval (`candles_1d`, `candles_15min`, …).

- **Already have:** TQQQ/SQQQ/QQQ daily + 15-min, full history 2010 → 2026-05-28.
- **To pull:** 1-min (for delay-cost and spread refinement) via
  `quantcore-ingest --symbols TQQQ SQQQ QQQ --intervals 1min`. New `candles_1min` table; **no clash**
  with existing tables/code. ~0.5–1 GB worst case; fits on internal disk.
- **Live fills:** only since May 2026 (~1.5 months) — too short to calibrate; use later as a sanity
  check on the model, not as a calibration set.

Note: current TQQQ/SQQQ backtest logs bake in a **flat, size-independent 5 bps entry / 15 bps exit**
(~20 bps round-trip). This is *conservative at small scale* but **blind to capacity** — it reports
the same cost at \$1k and \$1B. Replacing it is the point.

---

## 6. The plan (staged)

| Stage | Deliverable | Needs |
|---|---|---|
| **0 — done** | Audit: flat 5/15 bps, deterministic, size-blind; engine is FMP-OHLC backtest | — |
| **1** | Confirm fill-timing assumption (decision-bar vs next-bar) | read engine |
| **2** | **Spread estimate** for TQQQ/SQQQ (Corwin–Schultz on existing 15-min) | have data |
| **3** | **Delay-cost distribution** (post-decision 1-min drift) | 1-min pull |
| **4** | **Calibrate impact** (`Y`, exponent) — start from Almgren 2005, refine on our data | data + lit |
| **5** | **Size-aware cost function** `cost(Q)` replacing flat 20 bps | stages 2–4 |
| **6** | **Capacity curve: net Sharpe vs. AUM** (\$100k → \$1B) — *headline result* | stage 5 |
| **7** | **Cost-aware sizing** integrated with `1 − p_severe` | stage 6 |
| **8** | **Reusable cost library** (modular: spread / impact / delay toggleable) | stages 2–5 |

**Literature backbone (priority order):** Almgren, Thum, Hauptmann & Li (2005) *Direct Estimation
of Equity Market Impact* (the numbers); Bouchaud et al. *Trades, Quotes and Prices* (the theory);
Almgren & Chriss (2000) (execution schedule); Kissell *Science of Algorithmic Trading* (TCA/IS,
closest to the library we want); Perold (1988) (IS framework). Forums (Reddit/Wilmott/Quant SE) for
*hypotheses only* — never for constants.

---

## 7. Critical review — strengths, weaknesses, and proposed fixes

### Strengths
- **Right scope.** We build the upstream (decision/capacity) layer and reuse brokers for execution.
  No wasted effort reinventing execution algos.
- **Data is mostly in hand**; the model needs only a small 1-min pull.
- **Conservative starting point.** The existing 20 bps flat cost means our current Sharpes are not
  flattered by zero-cost fills — improvements are likely to *survive*, not evaporate.
- **The capacity curve is a concrete, fundable, business-relevant deliverable**, not open-ended
  research.

### Weakness 1 — The square-root law's constant `Y` is not identifiable from OHLC alone
We cannot measure *our own* impact from history where we never traded (the counterfactual problem).
OHLC contains no order-flow signing.
**Fix:** Adopt `Y` and the exponent from Almgren (2005) / Bouchaud as a **prior**, then *bound* it
using the ~1.5 months of live fills as they accumulate (compare realized
`avg_order_price − decision_price` against model prediction). Treat `Y` as a calibrated-with-
uncertainty parameter and **report the capacity curve as a band, not a line**:
```
I = Y · σ · sqrt(Q/V),   Y ∈ [Y_low, Y_high]  →  Q_max ∈ [ V(I_b/(Y_high σ))² , V(I_b/(Y_low σ))² ]
```

### Weakness 2 — Reference volume `V` is endogenous and regime-dependent
ADV is not constant: it spikes in stress (when we most want to trade) and is U-shaped intraday.
Using a single ADV understates cost in thin periods and overstates it in liquid ones.
**Fix:** Use **interval volume** matched to the execution window, and model `V` as a *distribution*
per time-of-day and volatility regime, not a scalar. Compute cost per historical decision using
*that bar's* realized volume, then aggregate. Stress-test capacity using the **low-volume quantile**
(e.g. 20th-percentile interval volume), not the mean.

Q: Isn't V something we can simply read by the amount of orders at a specific moment?
Or take the 15min candles instead of the daily candles? I guess that is what was suggested.
Stress test means testing our way of doing things vs the 20% rule?

A: Three parts:
- **V is traded volume over a horizon — yes, we read it straight from the candles** (`volume`). But
  note it's the *flow* (how much actually traded during the window), not the *book depth* (the resting
  limit orders sitting at the bid/ask at one instant). "Amount of orders at a moment" = depth, which
  needs quote/order-book data we don't have; the √-law wants flow, which we do have.
- **Yes — using the 15-min candle volume instead of daily ADV is exactly the fix.** For a 15-min
  decision the relevant V is that bar's volume. We already have it.
- **Stress test ≠ the 20% rule.** It means: compute capacity using a *low-volume* scenario (e.g. the
  20th-percentile thin bar/day) instead of the average, so we don't overstate how much we can trade —
  volume dries up in stress, exactly when we most want to trade (Block 4 did this with the p10 "thin"
  column). The "trade ≤10–20% of volume" participation cap is a *separate* lever (a bound on Q/V);
  "stress" here is about which *V* you plug in, not that cap.

### Weakness 3 — Price-guard execution relocates cost; especially harmful for momentum
A limit-price guard converts measurable impact into invisible **opportunity cost** (unfilled
shares), and is *adversely selected* exactly when price trends — i.e. precisely when a
momentum-style signal fires.
**Fix:** Decide and document whether each strategy is mean-reverting or momentum.
- Mean-reversion → passive/limit execution is fine; the guard helps.
- Momentum → bias toward **aggressive/marketable** execution; account for the *full* round-trip in
  the cost model and **charge unexecuted shares at the IS benchmark** (decision-price-relative) so
  opportunity cost is never hidden.

Q: I do not understand this difference. Explain in detail why this is.

A: A **price-guard** is a limit cap on your order ("buy, but pay no more than X"). It has two effects:
- **Opportunity cost:** if the market runs past X before you fill, you simply don't get the shares.
  There's no trade to measure, so the cost is *invisible* — but real, because you wanted the position,
  didn't get it, and miss the P&L it would have earned.
- **Adverse selection:** the guard fails *exactly* when price trends away from you. Buying with a cap,
  you get filled when price *doesn't* rise and miss out when it *does* — you keep the bad fills and
  lose the good ones.
Now the strategy type decides whether that helps or hurts:
- **Momentum** buys *because* price is rising → price trends in the direction that defeats your
  buy-limit → you systematically miss fills right when the signal is correct. The guard hides a big
  opportunity cost. Momentum should trade **aggressively/marketable** (cross the spread, guarantee the
  fill, pay the impact), and the cost model must **charge the missed shares at the decision price** so
  that hidden cost becomes visible.
- **Mean-reversion** buys *because* price dropped (expecting a bounce) → price drifts *toward* your
  limit → passive limits fill nicely at good prices. The guard *helps*; passive execution is fine.
Same guard, opposite verdict — which is why each strategy must declare its type.

### Weakness 4 — Delay cost from 1-min bars is not the same as fill-path cost
The 1-min forward return measures the market's *natural* drift over the fill window, not our path of
fills, and it ignores intra-minute structure.
**Fix:** Use the 1-min drift as the **delay/timing** term only (it is legitimately that), and keep
**impact** strictly in the modeled √-term. Do **not** try to read impact off forward returns — that
conflates our push with market drift. Revisit with finer data only if the capacity tail demands it.

Q: It doesn't seem like the suggested fix actually gives us something useful when it comes to price difference
due to delays. Am I seeing this correctly?

A: The conclusion actually flips. The fix gives us *exactly* the delay price-difference — that's the
whole of Block 3 (~17 bps/min). The 1-min forward return measures the market's natural drift over the
wait, which **is** the delay cost for the period you're waiting to fill. What the caveat withholds is
**impact** (your own footprint): you must not read that off the same forward return, because it would
conflate market drift with your own push. So delay → from the 1-min drift (useful, and done); impact →
from the modeled √-law (Block 4, kept separate). You're right that it *limits* what we extract — but
the thing it does give (the timing distribution) is genuinely useful, and it's the entire Block 3
result.

### Weakness 5 — Permanent impact accumulates under 15-min repeated trading
Trading every 15 min means our own permanent footprint can carry into the next decision; per-slice
or per-trade impact caps do not control *aggregate* cost.
**Fix:** At the capacity tail, model the order as worked over `N` bars and use the **Almgren–Chriss**
mean-variance objective to choose `N`:
```
minimize  E[cost]  +  λ · Var[cost]
          \____impact, ↓ with slower___/     \__timing risk, ↑ with slower__/
```
Add a **no-trade band / hysteresis** in the signal layer so noisy 15-min flips don't trigger
repeated round-trip costs.

### Weakness 6 — Leveraged-ETF plumbing at the high end
At \$100M+ the binding liquidity is the *underlying* (QQQ / Nasdaq futures via creation/redemption),
not just TQQQ's own book; the single-name √-law underestimates cost there.
**Fix:** Flag the capacity curve as **valid up to ~\$X** (to be determined, likely tens of millions
for TQQQ) and annotate the \$100M+ region as "requires underlying-liquidity modeling — out of scope
for v1." Better an honest bounded curve than a falsely precise one.

### Weakness 7 — Single point estimate hides risk
A single net-Sharpe-vs-AUM line invites false confidence.
**Fix:** Deliver the capacity curve with **uncertainty bands** (from Weakness 1) and a **stress
scenario** (low-volume + high-σ regime, from Weakness 2). The decision-relevant output is *"at AUM A,
net Sharpe is in [s_low, s_high], and degrades to s_stress in stress"* — not a single number.

Q: Explain why this is a problem in detail.

A: A lone "net Sharpe = 0.8 at $X AUM" reads like a fact, but it rests on (1) Y — adopted, with a ~12×
band; (2) which volume regime you assume (normal vs stress); (3) the strategy's own sampling noise.
The honest output is a *range*, and that range can be wide enough to flip the decision ("viable at
$50M" vs "edge gone at $50M"). The harm is concrete: someone reads the single number, sizes capital to
it, and finds in a stress month the real net Sharpe was 0.2 — capacity decisions fail precisely in the
regime that matters. A single line also hides the Y uncertainty we *know* is irreducible and can't show
stress degradation. The fix (bands + a stress scenario) reframes the output as "at AUM A, net Sharpe ∈
[s_low, s_high], degrading to s_stress" — i.e. a bet with a distribution, which is what a
capital-allocation decision actually needs.

---

## 8. The one-line ask for the meeting

> Execution is solved by brokers; we are building the *pre-trade cost model and capacity curve* that
> tell us **how much capital our TQQQ/SQQQ strategies can run before slippage eats the edge**, and
> that make our backtests honest. Methods are established; the calibration to our instruments is the
> value. First concrete output: a spread estimate (no new data) and a net-Sharpe-vs-AUM curve with
> uncertainty bands.
