# Findings C02 — Capacity curve refresh (RAN LAST)

**Date:** 2026-07-08 (numbers refreshed 2026-07-09, audit fixes #1+#2) · **Tickers:** TQQQ, SQQQ ·
**Question:** re-run S06's capacity chain with everything learned — the envelope band (C01), extra
risk metrics, an edge-sensitivity row, and the scheduler's chosen horizon instead of the 15-min
pin (E03/E04). **Answer: capacity extends moderately, not dramatically — TQQQ's central-Sharpe-zero
crossing moves from ~$14–25M (S06, 15-min-pinned) to ~$65M (central; envelope ~$13M–$300M+) once
the scheduler's longer horizons are used, but only because the alpha-forfeiture and interruption
cost those horizons actually incur (E04's caution) is charged for, not ignored.**

> **2026-07-09 audit refresh:** re-run after fixes #1 (E01 stretched-exp `g(h)`) and #2 (removed
> the objective's alpha double-count). Both lowered the alpha-forfeiture add-on the scheduler pays
> for a longer horizon (e.g. TQQQ's `alpha+interrupt` at $30M fell 20.1→11.1 bps), so the central
> zero-crossing roughly doubled (pre-fix ~$30M → now ~$65M). The *shape* and the "moderate, not
> dramatic; capped well short of $1B" conclusion are unchanged.

## Headline

A first pass of this stage costed trades at the scheduler's chosen horizon using **execution
cost alone** and found capacity apparently extending past $300M — a bug, not a finding: it
silently banked the benefit of a longer horizon (lower participation, lower impact) without
paying the alpha-forfeiture and interruption cost that E01/E02/E04 established is real, and that
E04 showed can even *reverse* the benefit at large size. **Fixed**: every cost number in this
stage's headline is `execution(h*) + alpha_forfeit(h*) + interruption(h*)` — matching
`schedule.py`'s own `_objective`, not a partial view of it. With that fix (and the 2026-07-09
audit corrections to the alpha terms), TQQQ's central Sharpe crosses zero between $30M (1.40) and
$100M (−0.75), i.e. ~$65M — real capacity extension versus S06's ~$14–25M, but a moderate one,
consistent with E04's finding that the scheduler's benefit shrinks (and can reverse) as size grows.

## Strategy and mathematics

Four additions to S06's chain, per the plan:

1. **Envelope band** (C01): `min/max{sqrt@Y=0.3, sqrt@Y=1.0, Almgren}` — computed at the
   *same* `h*` as the central estimate, with the alpha-forfeiture/interruption add-on (which
   doesn't depend on the impact model) applied identically to every band edge.
2. **Added metrics**: Sortino and max drawdown from the **full daily net-P&L path**
   (`(1+daily_net).cumprod()`, tracked drawdown from the running peak) — S06 only ever kept the
   mean/variance, discarding the path. Return-on-AUM is the annualized mean daily net return
   (already expressed per the ~0.95·AUM deployed notional, matching S06's own convention).
3. **Edge-sensitivity**: the same chain re-run with the daily gross leg scaled by 0.5 (**not**
   the cost side) — an independent Sharpe column at every AUM.
4. **Execution basis**: `h* = schedule.py::_choose_horizon(notional, ..., edge_bps)`, using the
   strategy's **own measured average edge** (`trades['r_gross'].mean()*1e4` — 107.1 bps/trade
   TQQQ, 103.9 bps/trade SQQQ — not E03/E04's illustrative 50 bps) in place of the fixed 15-min
   cadence pin.

**Daily net-series construction** (expected-cost basis, no simulated timing-noise realization —
consistent with S06's own no-λ convention): `daily_net = daily_gross − drag_per_trade × n_trades_that_day`,
where `drag_per_trade` is the **total** (execution+alpha+interruption) cost at that AUM's `h*`.

## Numbers

**TQQQ** (edge = 107.1 bps/trade measured; tier: **Modeled** composite of Measured trades +
Calibrated/adopted impact + Modeled alpha/interruption):

| AUM | h* (min) | alpha+interrupt (bps) | Sharpe (env mid) | [env lo, env hi] | Sortino | maxDD | ret/AUM | Sharpe (50% edge) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| $100k | 17 | 0.9 | 4.47 | [4.77, 4.13] | 10.86 | −16.0% | 163.8% | 4.02 |
| $1M | 30 | 2.5 | 3.92 | [4.62, 3.07] | 9.06 | −17.4% | 142.8% | 2.88 |
| $10M | 54 | 6.8 | 2.56 | [4.20, 0.52] | 5.49 | −27.5% | 93.0% | 0.14 |
| $30M | 75 | 11.1 | 1.40 | [3.79, −1.54] | 2.71 | −70.3% | 50.9% | −2.08 |
| $100M | 90 | 14.8 | −0.75 | [3.12, −5.05] | −1.23 | −99.8% | −27.8% | −5.60 ⚠>$50M |

**SQQQ** (edge = 103.9 bps/trade measured):

| AUM | h* (min) | alpha+interrupt (bps) | Sharpe (env mid) | [env lo, env hi] | Sortino | maxDD | ret/AUM | Sharpe (50% edge) |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| $100k | 23 | 1.4 | 2.49 | [2.69, 2.25] | 6.60 | −19.8% | 127.8% | 2.16 |
| $1M | 35 | 2.9 | 2.07 | [2.58, 1.46] | 5.35 | −22.2% | 106.2% | 1.32 |
| $10M | 75 | 10.5 | 1.15 | [2.23, −0.20] | 2.83 | −52.9% | 58.6% | −0.54 |
| $30M | 90 | 14.0 | 0.27 | [1.96, −1.83] | 0.63 | −88.7% | 13.7% | −2.25 |

**Capacity verdict**: TQQQ half-Sharpe (≈2.1) around $15M–$20M (was ~$4–5M, S06) — the cheaper
(audit-corrected) alpha cost lets the scheduler extend horizon even at moderate size; Sharpe-zero
~$65M (was ~$14–25M) — the real extension, driven by longer `h*` (75→90 min across $30M→$100M)
cutting participation-driven impact faster than it adds alpha-forfeiture. **The envelope band is
wide enough that the "gone" threshold ranges from ~$13M (env hi, pessimistic) to ~$300M+ (env lo,
optimistic)** — the uncertainty is, if anything, larger than S06's Y-band alone, because it now
also spans the Almgren/sqrt divergence documented in C01. Note the central crossing sits right at
the **$50M ETF structural ceiling**, beyond which the single-name model is flagged invalid anyway.

**50%-edge sensitivity**: roughly halves the AUM at which Sharpe crosses zero (TQQQ: ~$10M — the
50%-edge Sharpe is 0.14 at $10M — vs ~$65M at full edge) — confirming the plan's own expectation
that the headline is edge-sensitive, not just cost-sensitive, and the in-sample gross Sharpe (4.9)
deserves a discount in any real capital-allocation decision.

## Caveats

- **This is a moderate extension, not a dramatic one** — the headline story ("capacity is
  ~$4–25M for the 3× ETFs") is not overturned by this stage; it is refined to "central ~$65M,
  roughly $13–300M depending on band edge and how much you trust the scheduler's alpha-forfeiture
  discipline," per E04's own caution about that discipline degrading at large size.
- **`h*` here uses a flat "typical bin" volume proxy** (matching E03's worked examples, not a
  loaded `VolumeProfile`) and a single representative regime ("normal") — not the trade-by-trade
  real state E04's replay used. This stage is a *curve*, not a replay; E04 remains the
  ground-truth check.
- **maxDD/Sortino are computed on the expected-cost daily series** (no simulated timing-risk
  path) — a documented simplification matching S06's own no-λ convention, not a full Monte Carlo.
- **The old S06 table is untouched** (`research/06_capacity_curve/`) and remains the "previous,
  15-min-pinned" reference, per instruction — this stage supplements, does not replace it.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/capacity/02_capacity_refresh/build_02_capacity_refresh.py
```
Outputs in `results/` (gitignored): `capacity_refresh_{SYM}.csv`, `capacity_refresh.png`.
