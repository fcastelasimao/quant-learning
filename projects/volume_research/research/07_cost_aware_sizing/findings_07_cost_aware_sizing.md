# Findings 07 — Cost-aware sizing with λ (Stage 7)

**Date:** 2026-06-25 · **Tickers:** TQQQ, SQQQ · **Method:** choose the per-trade deployed
fraction `f` that maximises a mean–variance utility on return-on-AUM, given the Stage-5 size-aware
cost. Stage 6 deployed all-in regardless of size; Stage 7 sizes down so the marginal trade keeps
positive net edge.

## Strategy, mathematics

### The strategy (size to the market, not just to confidence)

Stage 6 showed an all-in strategy's Sharpe goes negative once the trade is too big for the market.
The fix isn't to *stop* at that AUM — it's to **deploy less of it per trade**. The strategy: at each
AUM pick the fraction `f` of capital to trade that maximises a mean–variance utility on
return-on-AUM, balancing the edge against the (convex) cost and the variance. Because impact is
convex, this naturally caps the *traded dollar size* and leaves the rest of large AUM idle — and it
composes with the strategy's existing `1 − p_severe` confidence signal as a second, independent
shrink factor.

### The mathematics (as implemented in `build_07`)

Per trade, in return-on-AUM units:

```
U(f) = f·μ_edge  −  f·c(f·AUM)  −  λ·f²·(σ_trade² + τ²(f·AUM))
       └ edge ──┘   └ cost drag ┘   └──── risk penalty (λ) ────┘
```

- `μ_edge` = gross expected return per trade (TQQQ 107 bps, SQQQ 104 bps)
- `c(Q)` = round-trip mean cost rate (spread+impact, 15-min fill) — **convex** in size (√-law)
- `σ_trade` = per-trade gross return std (TQQQ 2.78%, SQQQ 4.18%); `τ` = timing 1σ (independent)
- `λ` = **sizing** risk-aversion — the same mean–variance knob as execution, one layer up

Maximise on a grid `f ∈ (0, 1]`, then read off the trade size and the net metrics:

```
f* = argmax_f U(f);          Q* = f*·AUM
Sharpe(f*) = (μ_edge − c(Q*)) / σ_trade            # f cancels — scale-invariant
ret_on_AUM = f* · (μ_edge − c(Q*)) · turnover      # what idle capital costs you
final_fraction = min( 1 − p_severe , f* )          # compose with the confidence signal
```

### Two structural facts (the derivations)

1. **A uniform `f` leaves Sharpe unchanged** — it scales mean and std together,
   `Sharpe = f(μ−c)/(fσ) = (μ−c)/σ`. So cost-aware sizing does **not** lift Sharpe by shrinking; it
   protects the *per-deployed-$* edge by **not over-trading**. What it gives up is **return on total
   AUM** (idle capital). The right capacity metric here is therefore return-on-AUM, not Sharpe.
2. **The cost-optimal trade size saturates.** Because cost is convex (`c ∝ √Q`), beyond a threshold
   AUM the optimum holds the *traded notional* `Q*=f*·AUM` roughly constant and lets `f*` fall as
   `1/AUM`. The rule reduces to: **"trade a fixed dollar size; deploy the fraction that size implies."**

## Results (central Y=0.5, mean-σ headline)

**TQQQ** (gross Sharpe 4.92):

| AUM | λ=0  f* / Q* / net Sh / ret-on-AUM | λ=5 | λ=20 |
|---|---|---|---|
| $1M | 1.00 / $1.0M / 3.40 / 133% | 0.79 / $0.8M / 3.52 / 109% | 0.24 / $0.2M / 3.99 / 38% |
| $10M | 0.68 / $6.8M / 1.52 / 41% | 0.35 / $3.5M / 2.39 / 33% | 0.16 / $1.6M / 3.10 / 19% |
| $100M | 0.07 / **$6.7M** / 1.54 / 4% | 0.06 / $6.2M / 1.65 / 4% | 0.05 / $4.7M / 2.03 / 4% |
| $1B | 0.007 / ~$7.5M / 1.4 / 0.4% | … | … |

**SQQQ** (gross Sharpe 2.81): same shape, lower saturation — `Q*` ≈ **~$3.6M**, net Sharpe holds
~0.9 (λ=0) instead of collapsing.

**Reading it:**
- **λ=0 (cost-convexity only):** deploy all-in to ~$3M; past that the optimal trade size pins at
  **~$6.8M (TQQQ) / ~$3.6M (SQQQ)** and the deployed fraction falls. Net Sharpe holds **~1.5 / ~0.9
  flat** at high AUM — *rescued* from Stage 6's all-in collapse to negative. Return-on-AUM decays
  (133% → 0.4%) as more capital sits idle: **that decay is the true capacity cost.**
- **Higher λ → smaller f*, smaller Q*, higher Sharpe, lower return-on-AUM.** The mean–variance
  trade-off, explicit: λ buys safety/Sharpe by leaving money on the table.
- **W6 dissolves under sizing:** `Q*` stays ~$6.8M ≪ the $50M ETF ceiling, so the underlying-liquidity
  concern never binds when you size cost-optimally.

*(σ = mean, the expected-cost headline; the median-σ typical-day basis saturates ~15% higher.)*

## Common λ values / using 2–3

This is the **sizing-layer** λ (mean–variance on the trade's return), so its scale differs from the
execution λ of Stage 6 — here `λ·σ²` must be return-comparable, so meaningful values are larger
(O(1)–O(20) for these per-trade variances). λ=0 is risk-neutral (cost-convexity caps size on its
own); λ≈5 is moderate; λ≈20 is conservative. As in Stage 6, **the right presentation is a small grid
{0, 5, 20}**, not a single pinned value — λ is a preference input from the strategy owner. The
*direction* is robust regardless of the exact number; only the level of conservatism moves.

## Integration with the signal (`1 − p_severe`)

The strategy already sizes by confidence `1 − p_severe ∈ [0,1]`. Cost-aware sizing is a **second,
independent shrink factor**; compose by taking the binding one:

```
final_fraction = min( 1 − p_severe ,  f*(AUM, λ) )
```

- At **small AUM**, `f*≈1` so the **confidence signal binds** — sizing is unchanged from today.
- At **large AUM**, `f*≪1` so the **cost cap binds** — you cannot deploy full confidence without
  eating the edge. The two layers answer different questions (when to trust the signal vs. how much
  the market can absorb) and do not double-count.

## Caveats

- Inherits the **Y band** (W1) and **per-trade / not-aggregate** impact (W5) from Stages 4–6.
- `f*` uses the 15-min-cadence (View-1) execution cost; it does not re-optimise execution speed
  jointly with size (that would couple the two λ's — out of scope, low marginal value).
- Tiny non-monotonicity at $1B (net Sharpe ticks up) is F-grid floor granularity, deep in the
  invalid >$50M region; ignore.
- Sizing down means **idle capital** — the analysis assumes the un-deployed cash earns ~0. A real
  book would allocate it elsewhere; the capacity statement is specifically about *this* strategy.

## Reproduce

```
/Users/.../envs/quant/bin/python research/07_cost_aware_sizing/build_07_cost_aware_sizing.py
/Users/.../envs/quant/bin/python -m pytest tests/test_cost_aware_sizing.py -v
```
Outputs in `results/` (gitignored): `cost_aware_sizing_{TQQQ,SQQQ}.csv`, `cost_aware_sizing.png`.
