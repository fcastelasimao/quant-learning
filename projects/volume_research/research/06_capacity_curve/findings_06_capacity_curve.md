# Findings 06 — Net-Sharpe-vs-AUM capacity curve (Stage 6, the headline)

**Date:** 2026-06-25 · **Tickers:** TQQQ, SQQQ · **Method:** take the strategy's *gross*
(pre-slippage) returns and re-charge execution with the Stage-5 size-aware cost model, sweeping
AUM $100k → $1B. Per-trade notional grows with AUM → modelled cost grows → Sharpe decays.

## Inputs (reconstructed / measured — nothing re-fitted)

- **Gross returns:** TQQQ/SQQQ canonical full-history trade logs, reconstructed from
  `decision_price → exit_decision_price` (strips the flat 5/15 bps the backtest baked in; verified
  the stripped cost is exactly 20.2 bps round-trip).
- **Turnover:** TQQQ **≈169 round-trips/yr** (2343 trades / 3492 bdays), SQQQ **≈140/yr**.
- **Per-trade size:** notional ≈ **0.95 · AUM** (measured; ~5% cash buffer) — the strategy is all-in.
- **Gross Sharpe (constant-notional, daily):** **TQQQ 4.92, SQQQ 2.81.**
- **σ, ADV, half-spread:** from findings_04. σ normal is reported as a pair (right-skewed series):
  **headline = mean** (TQQQ 370 bps/d, SQQQ 372) — the expected-cost moment; **median** (339 / 341)
  carried as a typical-day sensitivity (findings_09). ADV $4.9B / $2.8B. **Y adopted from literature**
  (band [0.3,1.0], central 0.5) — never fitted.

## Strategy, mathematics

### The strategy (make a fixed backtest size-aware)

The backtest charges a **flat 20 bps** round-trip regardless of size — so its Sharpe is the same
whether you run $100k or $1B. The capacity question needs the opposite: a cost that **grows with
size**. The strategy is to take the strategy's *gross* (pre-cost) returns, then re-charge each trade
with the Stage-5 `cost(Q)` at a per-trade notional that scales with AUM, and watch the Sharpe decay.
Two channels enter differently: **impact + spread lower the mean** (a drag); **timing raises the
variance** (a risk). That split is the whole reason λ matters (View 2).

### The mathematics (as implemented in `build_06`)

**Gross returns**, reconstructed from the trade log's *decision* prices (stripping the baked-in
5/15 bps):

```
r_gross = (exit_decision_price − decision_price) / decision_price
```

**Per-trade notional** at a given AUM: `Q = deploy_frac · AUM` (`deploy_frac ≈ 0.95`, all-in).

**Net daily Sharpe** from the constant-notional daily PnL series (mean `μ_d`, var `v_d`, `D` business
days, `N` trades), given a per-trade round-trip drag and timing variance:

```
Sharpe_net = ( μ_d − drag · N/D ) / √( v_d + timing_var · N/D ) · √252
```

The drag lowers the numerator; the (mean-0, independent) timing variance raises the denominator;
`N/D` scales per-trade quantities onto the daily series. The two **views** differ only in how the
round-trip cost is computed:

```
WITHOUT λ:  fill pinned to the 15-min cadence → participation = (Q/ADV)·(day/15)
            drag = 2·(spread + impact),  timing_var = 0   (timing held out as a stress band)

WITH λ:     speed chosen by A–C → participation = optimal_participation(Q, λ)
            drag = 2·(spread + impact),  timing_var = 2·(timing_1σ)²   (entry+exit independent)
```

(The ×2 is entry + exit; the round-trip timing 1σ adds in quadrature, hence the factor-2 variance.)

### Where it bites

View 1 (15-min-pinned, expected-cost) is **λ-free and operative**; View 2 lets the fill stretch
arbitrarily, which is unrealistic for a 15-min-cadence strategy, so it's a *mechanism illustration*.
Every level carries the adopted-`Y` band; capacity is valid only to ~$50M (W6) and is per-trade, not
aggregate (W5).

## View 1 — WITHOUT λ (the robust capacity curve)

Execution pinned to the strategy's **15-min decision cadence** (`participation = (Q/ADV)·26`).
Only the **expected** cost (spread + impact) is charged, as a **mean drag** on returns; timing is
held out as a separate stress channel. This is the λ-free, decision-relevant curve.

Net Sharpe, **headline = mean σ** (expected-cost), central Y=0.5 (band = Y∈[0.3,1.0]), + stress
(high-σ + thin volume):

| AUM | TQQQ net Sh | band | round-trip cost | TQQQ stress | SQQQ net Sh | SQQQ stress |
|---|---:|---|---:|---:|---:|---:|
| $100k | 4.47 | [4.09, 4.62] | 10 bps | 4.17 | 2.46 | 2.07 |
| $1M | 3.65 | [2.44, 4.13] | 28 bps | 2.69 | 1.81 | 0.59 |
| $3M | 2.76 | [0.67, 3.60] | 47 bps | 1.11 | 1.12 | −1.00 |
| $10M | 1.03 | [−2.78, 2.56] | 85 bps | −1.99 | −0.23 | −4.10 |
| $30M | −1.76 | [−8.37, 0.88] | 145 bps | −7.00 | −2.42 | −9.12 |
| $100M+ | infeasible — 15-min fill needs >100% of interval volume (clamped); also past the $50M ETF ceiling |

**Typical-day (median σ) sensitivity** — the published median-based figures, ~9% lower σ → a bit
more headroom: TQQQ net Sharpe **4.50 / 3.75 / 2.94 / 1.35 / −1.21** at $100k/$1M/$3M/$10M/$30M;
SQQQ **2.48 / 1.89 / 1.26 / 0.02 / −1.99**. Both bases sit inside the Y-band, so the verdict holds
either way.

**Capacity verdict (normal conditions, central Y; mean-σ headline, median-σ typical-day in ⟨⟩):**
- **TQQQ:** keeps ~80% of gross Sharpe to **~$1M**, half of gross (≈2.5) to **~$4M** ⟨~$4–5M⟩, edge
  gone (Sharpe→0) by **~$14M** ⟨~$20M⟩.
- **SQQQ:** lower — half of gross (≈1.4) by **~$2M** ⟨~$2.5M⟩, edge gone by **~$9M** ⟨~$10M⟩ (thinner ADV).
- **Stress roughly halves these** ($1M-scale TQQQ already at Sharpe 2.7 vs 3.65 normal).
- The **Y band is wide and decision-relevant**: at $10M TQQQ, net Sharpe is anywhere in
  [−2.8, 2.6] depending on the true (unmeasurable) Y — far wider than the mean-vs-median σ shift.
  This is the irreducible W1 uncertainty.

## View 2 — WITH λ (what optimal execution would do)

Here execution speed is **chosen** by the Almgren–Chriss optimum `optimal_participation(Q, λ)`:
impact at that speed is a **mean drag** (numerator), the timing 1σ there is **independent variance**
(denominator). λ is the risk-aversion knob that picks the speed.

Net Sharpe (central Y), λ grid:

(mean-σ headline)

| AUM | TQQQ λ=0 | λ=1 | λ=3 | SQQQ λ=0 | λ=1 | λ=3 |
|---|---:|---:|---:|---:|---:|---:|
| $100k | 4.03 | 3.18 | 2.11 | 2.28 | 1.70 | 0.87 |
| $1M | 3.55 | 2.01 | −0.04 | 2.07 | 0.83 | −0.58 |
| $10M | 1.97 | −0.17 | −3.84 | 1.24 | −0.66 | −3.17 |
| $100M | 0.70 | −3.69 | −7.90 | 0.46 | −3.14 | −4.84 |

**The key lesson (this is *why* λ matters):** for a **~1-day-holding** strategy, **patient
execution wins** — net Sharpe is *highest at λ=0* (trade slowly, pay minimum impact) and *falls as
λ rises*. Cranking λ buys speed to suppress timing variance, but that variance is tiny next to the
strategy's own daily P&L variance, so you end up paying real impact drag to kill a risk that barely
moved the Sharpe. **The risk-averse (high-λ) execution is self-defeating here.** A high-frequency or
intraday strategy, whose alpha decays in minutes, would flip this — there λ should be large and fast
execution is correct (this couples to the still-undeclared momentum/mean-reversion type, W3).

**Caveat on View 2:** the optimiser is allowed to *stretch the fill arbitrarily*. At λ=0, large
orders imply multi-hour/multi-day fills — inconsistent with a strategy that re-decides every 15 min
(you would no longer be in the position you decided on). So **View 1 (15-min-pinned) is the operative
capacity curve; View 2 is the mechanism illustration.** The two agree at small size.

## Common λ values / can we use 2 or 3?

Our `λ` is a **reduced-form, dimensionless weight** on the *1σ timing term vs. expected cost in bps*
(not the literature's dimensional $⁻¹ A–C coefficient). On that scale, O(1) is the natural range:
- **λ = 0** — risk-neutral: minimise expected cost only → slowest fill.
- **λ = 1** — weight a 1σ timing move equally with the expected cost (the moderate anchor used in
  findings_05).
- **λ = 2–3** — progressively risk-averse → faster fills. Above ~3 you are essentially demanding
  near-instant execution.

**Yes — running a small grid {0, 1, 3} is exactly the right presentation** (a fan of curves, the way
the Y range gives a band). It brackets risk-neutral → quite risk-averse and shows the sensitivity
without pretending we can pin one true λ. Pinning a single λ is a *preference* input from the
strategy owner (or, equivalently, a target execution horizon), not something measurable from data.

## Caveats

- **W1 — Y band.** Sharpe levels carry the full 12× Y uncertainty; reported as a band, not a line.
- **W6 — $50M ETF ceiling.** Beyond ~$50M the binding liquidity is the underlying QQQ/futures, not
  TQQQ's book; curve annotated invalid there. At $100M+ the 15-min participation also clamps at 100%
  (can't physically fill) — both say the same thing: this size is infeasible on this cadence.
- **W5 — per-trade, not aggregate.** Single-order temporary impact; repeated 15-min trading
  accumulates permanent footprint not modelled here → true capacity is somewhat lower.
- **Base strategy, not the overlay.** Uses the canonical full-history base; the p_severe + rule
  overlays (TQQQ_SQQQ_analysis item 18) shift the *level* but the AUM-*sensitivity* (curve shape) is
  the point. Cost-aware sizing on top is Stage 7.

## Next (Stage 7)

Cost-aware sizing with λ: choose the per-trade deployed fraction `f*(AUM, λ)` that maximises net
risk-adjusted return, and compose it with the signal's `1 − p_severe` confidence (take the binding
one). This converts the capacity *ceiling* above into a *sizing rule* — at large AUM you deploy less
per trade rather than letting net Sharpe go negative.

## Reproduce

```
/Users/.../envs/quant/bin/python research/06_capacity_curve/build_06_capacity_curve.py
/Users/.../envs/quant/bin/python -m pytest tests/test_capacity.py -v
```
Outputs in `results/` (gitignored): `capacity_no_lambda_{TQQQ,SQQQ}.csv`,
`capacity_with_lambda_{TQQQ,SQQQ}.csv`, `capacity_curve.png`.
