# Findings 03 — Delay / timing cost

**Date:** 2026-06-24 · **Tickers:** TQQQ, SQQQ, QQQ · **Method:** signed forward return from each
15-min decision point to a fill `k` minutes later, from 1-min closes (within session, gap-filled
grid). Full history 2010→2026.

## Headline

**Timing risk dwarfs the spread by ~17× for the leveraged ETFs.** Even a **1-minute** fill delay
carries **~17 bps (1σ)** of timing risk for TQQQ/SQQQ vs the ~1 bp half-spread from Block 1. The
cost grows with delay: ~36 bps at 5 min, ~59 bps at 15 min. For these instruments, **execution
latency — not spread — is the dominant cost driver**, and it is the thing the flat backtest cost
cannot represent. QQQ (1×) is ~⅓ of that (5.5 / 12 / 20 bps), exactly tracking its lower volatility.

Q: What is the meaning of 1σ in ~17 bps (1σ)? I say this in the plots as well, and it is not clear.
What would be the difference if we take 2 sigma?

A: σ (sigma) = the standard deviation of the delay cost. The move is random with mean ≈ 0, so
"1σ = 17 bps" means a *typical* fill drifts about ±17 bps from the decision price, and ~68% of fills
land within ±17 bps. **2σ = 34 bps** is the ~95% band — only ~1 fill in 20 drifts more than ±34 bps.
So 1σ is the everyday magnitude; 2σ is the "bad-but-not-rare" tail you'd use for risk budgeting (we
also report p05/p95 for the same reason). Nothing about the estimate changes with 2σ — it's the same
number ×2, just a wider confidence band.

Crucially this is a **risk, not a drag**: the mean delay cost is ≈0 (−0.16 to +0.42 bps across
horizons), so it does not bias mean P&L — it inflates **return variance** and creates **tail
slippage**. A backtest that fills at the decision price with a flat cost understates that variance.

Q: You say "≈0 (−0.16 to +0.42 bps across horizons)", this means the probability distribution for drag
is not constant. Would it be possible to see a plot with the probability distribution here?

A: Yes — see `results/delay_distribution.png` (produced by `build_03`). But first the
conceptual point, because the premise is half-right: the distribution **is** not constant across
horizons — but what changes is its **width, not its centre.** As the fill delay grows, the spread
of the distribution grows (σ: 17 → 36 → 59 bps at 1/5/15 min, the √t law), while the centre stays
pinned at ≈0. The `−0.16 → +0.42 bps` wobble in the *mean* is not a varying drag — it's mean-zero
to within noise: each of those means sits **~1–2 standard errors from zero** (SE ≈ σ/√n with
n≈10⁵ decisions, so ≈0.05–0.2 bps) and is **two orders of magnitude smaller than the σ it lives
inside.** And for a directional signal its sign would flip anyway (momentum adverse, mean-reversion
favourable, W3), so a symmetric measurement shouldn't be read as a systematic drag at all.

The plot shows exactly this in two panels:
- **Left — overlaid densities** for the 1/5/15-min horizons. All three are centred on 0 (dotted
  line) and **symmetric**; the only thing that changes is the width — the 1-min curve is tall and
  narrow, the 15-min curve short and wide. Note the **sharp central peak + fat tails**
  (leptokurtic, not Gaussian): most fills barely move, but the occasional fill lands far out — those
  tails are the "tail slippage" of the next question.
- **Right — the mean ± standard error inside the ±1σ band.** The mean (orange) is visually glued to
  zero across every horizon, while the blue ±1σ cone fans out to ±59 bps. That contrast *is* the
  answer: the horizon inflates the **risk** (the cone), not the **expected cost** (the line).

Q: What is meant by inflates return variance and creates tail slippage?

A: Because the mean is ≈0, the delay cost doesn't change your *average* return — but every trade's
realized P&L gets a random ±17 bps wobble. That extra per-trade noise raises the **variance** (std)
of returns, and Sharpe = mean/std, so more std lowers Sharpe even with the same mean. "Tail slippage"
= the occasional 2σ+ fill (e.g. −50 bps) that goes badly against you; rare, but those are the trades
that hurt live. A backtest that fills at the decision price with a flat cost shows returns that are
too smooth — it misses both the variance and those bad-tail fills.

## Strategy, mathematics

### The strategy (what we measure and why)

Our signal decides at a 15-min bar close (the **decision price**), but the order actually fills
some minutes later (the **fill price**). The gap between the two — the **delay/timing cost** — is
the price drifting while we wait. We can't see our future fills, but the market's *natural* drift
over a `k`-minute window is exactly that distribution, and we can read it straight off historical
**1-min closes**: take every 15-min decision point and look at where price is `k` minutes later.
This isolates timing from impact (our own footprint, modelled separately in Block 4) — here we
trade nothing, so the only thing moving the price is the market.

### The mathematics (as implemented in `slippage/delay.py`)

For each decision time `t` and fill horizon `k` (minutes), the signed delay cost in bps is the
forward return:

```
delay_bps(t, k) = ( P_{t+k} / P_t − 1 ) · 1e4
   buy:  cost = (P_fill − P_decision)/P_decision      sell: the negative of that
```

computed on a **per-session, gap-filled 1-min grid** (re-indexed to a contiguous minute grid and
forward-filled, so `shift(−k)` means *k minutes*, never *k bars*, and a window never crosses the
overnight gap). The reported **timing risk** is the cross-sectional standard deviation per horizon:

```
σ_delay(k) = std_t [ delay_bps(t, k) ]          (the 1σ in the tables)
mean_t [ delay_bps(t, k) ] ≈ 0                   → a RISK, not a drag
```

Because the move is driven by price **diffusion**, risk grows with the square root of time. Fitting
`log σ` on `log k` over the six horizons gives a clean power law:

```
σ_delay(k) ≈ 17.6 · k^0.45  bps      (TQQQ/SQQQ; R²=0.999)
```

— slightly **below** the pure random-walk `√k` (γ=0.45<0.5), the signature of mild intraday
mean-reversion. The same diffusion model extrapolates sub-minute, `σ(τ) ≈ σ_1min·√(τ/60s)`, with a
floor at the bid-ask bounce. For the cost model the term is `σ_1min·√k`, `σ_1min ≈ 17 bps`
(TQQQ/SQQQ), `~5.5` (QQQ), entered as **variance**, not mean drag.

### Where it bites

The mean is ≈0 **only if the trade direction is uncorrelated with the drift** — true for the raw
measurement, false for a directional signal (momentum fills adversely → signed positive cost;
mean-reversion favourably). So these magnitudes are the timing-*risk* size; the *sign* of any
expected cost is strategy-specific (W3) and not yet pinned. Regime scales it (×~1.5 stress, ×~0.6
calm), measured cleanly here from realized vol with **no estimator bias** (unlike Block 1's spread).

## Numbers — timing risk (std of delay cost, 1σ bps)

| fill delay | TQQQ | SQQQ | QQQ |
|---|---|---|---|
| 1 min  | 17.4 | 17.0 | 5.5 |
| 2 min  | 24.0 | 24.6 | 7.7 |
| 3 min  | 29.4 | 27.7 | 9.2 |
| 5 min  | 35.7 | 35.2 | 11.9 |
| 10 min | 48.8 | 48.2 | 16.3 |
| 15 min | 59.0 | 58.4 | 19.7 |

Context: spread floor ~1 bp; flat backtest exit 15 bps. **A TQQQ fill delayed even ~1 minute has a
1σ timing risk exceeding the entire flat 15 bps exit cost.**

Q: What is flat 15 bps exit cost here?

A: It's the slippage baked into the existing TQQQ/SQQQ backtest engine: a flat **5 bps on entry /
15 bps on exit**, charged on every trade regardless of size or conditions (see SLIPPAGE_PLAN /
CLAUDE.md). It's the size-blind assumption this whole project is replacing. The comparison's point:
a 1-min delay's timing risk (~17 bps) already exceeds that entire flat 15 bps exit, so the flat
number can't represent the real timing cost.

**√-time scaling holds, slightly sub-linear:** actual TQQQ std at 5/15 min (35.7 / 59.0) is below the
pure-random-walk √t prediction (38.9 / 67.4) — evidence of mild intraday mean-reversion (negative
short-horizon serial correlation). Risk still grows steeply, just a touch slower than √t.

Q: What do we have then? Is it c√t for some constant c or t^{gamma} for some power gamma?
Can we figure this out?

A: Yes — fitted it. A pure random walk would give c·√t (γ=0.5). Regressing log(std) on log(t) over
all six horizons gives **γ ≈ 0.45, c ≈ 17.6**, i.e. **std ≈ 17.6 · t^0.45** (bps, t in min; R²=0.999).
So it's a clean power law slightly *below* √t — the mild intraday mean-reversion shows up as the
exponent being 0.45 rather than 0.50. (Per-ticker fits available on request; SQQQ ≈ TQQQ, QQQ has the
same exponent at ~⅓ the constant.)

**By volatility regime (TQQQ, 1σ bps)** — strong, clean monotonic ordering (directly-measured
realized vol, *no* estimator bias, unlike Block 1's spread regime):

Q: Volatility regimes where never defined. How do you define them? What are the regimes?

A: A day's regime = the tercile of its **trailing 20-day realized volatility** (the std of daily
close-to-close returns over the prior 20 trading days). Split the ticker's whole history into three
equal groups by that value: bottom third = **calm**, middle = **normal**, top third = **stress**. So
"stress" means the day's recent 20-day vol sits in the ticker's top 33%. It's defined per ticker in
`vol_regime()` (build_03 / diagnose_resolution), and it's the same definition used for Block 1's
regime split and Block 4's stress column.

| delay | calm | normal | stress |
|---|---|---|---|
| 1 min  | 10.6 | 13.6 | 24.7 |
| 5 min  | 22.4 | 28.4 | 50.2 |
| 15 min | 37.0 | 47.4 | 82.8 |

In stress, a 15-min TQQQ delay is ~83 bps 1σ — when you most want to trade, timing risk is worst.

## Caveats

- **Direction matters and we measured the symmetric case.** Mean≈0 assumes the trade direction is
  uncorrelated with short-horizon drift. It isn't for a directional signal: **momentum** fills
  adversely (drift goes against you → positive expected cost), **mean-reversion** favourably. The
  numbers here are the raw timing-risk magnitude; the *sign* of the expected cost is strategy-specific
  (SLIPPAGE_PLAN Weakness 3). Resolving this needs the strategy's signal aligned to these windows.
- **Fill-window assumption.** "Delay = k minutes" models a fill completed k minutes after the
  decision. Real latency depends on order size/urgency (seconds for small marketable orders, minutes
  for worked orders). The horizon curve *is* the parameterisation — pick k to match execution style.
- **Decision price = the 1-min close at the 15-min boundary**, a proxy for the actual 15-min bar
  close. Confirming the engine's fill-timing assumption (Stage 1) would tighten this.

## Implication for the cost model

Add a **delay term** to `cost(Q)` driven by `σ_1min · √k`, where `k` is the expected fill latency:
`delay_risk_bps ≈ σ_1min(bps) · √k`, with `σ_1min ≈ 17 bps` (TQQQ/SQQQ), `~5.5 bps` (QQQ), scaled by
regime (×~1.5 stress, ×~0.6 calm). It enters the model as a **variance** contribution, not a mean
drag (until the momentum/mean-reversion sign is pinned down). This term is **~1–2 orders of magnitude
larger than the spread** for the 3× ETFs and is the more important honesty fix for the backtests.

## Sub-minute extrapolation (live latency)

Real live fills complete in seconds, not minutes. We have no sub-minute data, but the timing risk
comes from price diffusion, which scales as √time, so we can extrapolate: `risk(τ) ≈ σ_1min·√(τ/60s)`.

| latency | TQQQ | SQQQ | QQQ |
|---|---|---|---|
| 1 s  | 2.3 | 2.2 | 0.7 |
| 2 s  | 3.2 | 3.1 | 1.0 |
| 5 s  | 5.0 | 4.9 | 1.6 |
| 10 s | 7.1 | 6.9 | 2.3 |

(1σ bps.) **Defensible because** Stage 3's own fine end already follows √t almost exactly (TQQQ
1/2/3-min 17.4/24.0/29.4 vs √t 17.4/24.6/30.1), so the diffusion model holds down to 1-min. **Caveats:**
(1) it's extrapolation, not measurement — validating below 1-min needs tick/second data; (2) it does
*not* go to zero as τ→0 — there's a floor at ~the bid-ask bounce (half-spread), small here since
diffusion dominates even at 2s (3.18→3.33 with the floor); (3) don't double-count — at the seconds
scale the drift and the spread are partly the same phenomenon (the bounce).

**Live-slippage estimate, fast small fill (~2 s), TQQQ:** ~1 bp half-spread to cross (certain) + ~3 bps
timing (1σ, mean≈0) ≈ a few bps total; QQQ ~2 bps. **The danger is not seconds-scale latency** — for
fast small fills slippage is genuinely small and spread+tiny-timing dominated. It is (a) **minutes**-
scale latency (queued/worked orders jump back onto the steep part of the curve — 5-min TQQQ ≈ 36 bps),
and (b) **impact at size** (Stage 4). The decision-relevant question for live is: *what is the real
decision-to-fill latency?* Seconds → negligible; minutes → material.

## Reproduce

```
/Users/.../envs/quant/bin/python research/03_delay_cost/build_03_delay_cost.py
/Users/.../envs/quant/bin/python -m pytest tests/test_delay.py -v
```
Outputs in `results/` (gitignored): `delay_summary.csv`, `delay_by_regime.csv`, `delay_cost.png`,
`delay_distribution.png`.
