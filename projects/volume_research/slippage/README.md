# `slippage` — a pre-trade cost model you can drop into a backtest

A small (numpy + pandas only) library that replaces ad-hoc, flat slippage assumptions
(e.g. "20 bps round-trip") with a **size-aware** cost: a spread floor, square-root market
impact, and delay/timing risk, calibrated to your instrument from OHLCV.

It models the **structural** cost a broker can't remove (the spread you cross + the impact of
your size). It does **not** model execution scheduling — that's the broker's job (VWAP/POV/IS
algos). The question this answers is *"how much will my size cost, and how much can I run?"* —
not *"how do I fill this order?"*

## Install / pass around

```bash
pip install -e .          # from this directory (volume_research/)
```

Or just copy the `slippage/` folder anywhere on your path — it only needs numpy + pandas.

Q: this directory = volume_research/slippage, right?
Also, should we start using uv for the projects?

## Quickstart

```python
from slippage import calibrate, CostModel

# 1. Calibrate from your own data (you bring the DataFrames).
#    daily:  columns 'close','volume', chronological.
#    15-min: columns 'high','low', DatetimeIndex.
cal = calibrate(daily_ohlcv, intraday_15min)

# 2. Build the model (all components on; toggle with spread=/impact=/delay=).
model = CostModel(cal.normal)

# 3. Charge a trade. roundtrip = entry + exit, filled within the 15-min cadence.
rt = model.roundtrip(notional_usd=2_000_000)
print(rt.expected_slippage_bps)   # spread + impact, a MEAN drag (bps)
print(rt.timing_bps)          # 1σ timing RISK (mean≈0), NOT a drag
```



## Using it in a backtest (replace your flat bps)

```python
model = CostModel(cal.normal)

def net_trade_return(gross_return, notional_usd):
    rt = model.roundtrip(notional_usd)
    # impact + spread reduce the return (mean drag):
    net = gross_return - rt.expected_slippage_bps / 1e4
    # timing is a RISK, not a drag — add it to your variance, don't subtract it:
    #   per-trade extra variance = rt.timing_var_frac2
    return net
```

The two channels are **not** additive into one number, and the library keeps them apart on
purpose: **impact + spread = expected cost (subtract from return); timing = variance (add to
risk).** If you collapse them you will either double-count or misprice the tails.

## Report a band, not a line (important)

The impact constant `Y` in `I = Y·σ·√(Q/V)` is **adopted from the literature, not fittable from
OHLC** (you never traded, so there's no counterfactual). Every impact number is therefore a
band. Don't quote a single cost — quote the range:

```python
band = model.roundtrip_band(2_000_000)          # {0.3: ..., 0.5: ..., 1.0: ...}
lo, mid, hi = (band[y].expected_slippage_bps for y in (0.3, 0.5, 1.0))
```

Also run `CostModel(cal.stress)` for the high-σ / thin-volume regime — capacity roughly halves
in stress, exactly when a signal most wants to fire.

## Cost-aware sizing (recipe)

The library is the *cost model*; sizing is your application. The pattern (from Stage 7): pick the
per-trade deployed fraction `f` that maximises a mean–variance utility, then compose with your
own confidence signal. Because impact is convex (√-law), this caps the optimal *traded notional*
and you deploy a shrinking fraction of large AUM.

```python
import numpy as np

def cost_aware_fraction(aum, mu_edge, sigma_trade, model, lam=0.0, grid=None):
    """f* maximising  f·μ − f·c(f·AUM) − λ·f²·σ²  on a grid in (0,1]."""
    f = np.linspace(0.005, 1.0, 400) if grid is None else grid
    q = f * aum
    cost = np.array([model.roundtrip(qi).expected_slippage_bps / 1e4 for qi in q])
    tvar = np.array([model.roundtrip(qi).timing_var_frac2 for qi in q])
    util = f * mu_edge - f * cost - lam * f**2 * (sigma_trade**2 + tvar)
    return float(f[int(np.argmax(util))])

# Compose with your signal — the binding (smaller) factor wins:
final_fraction = min(1.0 - p_severe, cost_aware_fraction(aum, mu_edge, sigma_trade, model))
```

`λ` (risk aversion) is a **preference**, not data-fittable — run a small grid (e.g. {0, 5, 20})
rather than pinning one value; the shape of the answer is robust, only the conservatism moves.

## Planning an execution (how fast / how to slice)

Given a decided order, get the recommended participation rate, fill horizon, slice plan, and a cost
forecast — i.e. the *urgency* to hand a broker POV/IS algo. This is a pre-trade **planner**, not a
live order router (the broker still slices and routes).

```python
from slippage import calibrate, plan_execution
cal = calibrate(daily_ohlcv, intraday_15min)

plan = plan_execution(5_000_000, cal.normal, lam=1.0)   # lam = urgency (0 patient … higher = faster)
print(plan)   # "Trade ~6.3% of volume over ~6 min in 6 slice(s) of ~$0.83M.
              #  Expected cost ~43 bps (band 26–86); timing ±43 bps (1σ)."
plan.participation, plan.horizon_min, plan.n_slices, plan.expected_slippage_bps, plan.feasible
```

The speed is the Almgren–Chriss optimum (faster → more impact, slower → more timing), capped at the
decision cadence (`horizon_cap_min`, default 15). An order too big to fill within the cap even at
100% of volume is returned with `feasible=False`. **Note:** the cost number is meaningful from ~$1M
up — at retail size the √-law over-extrapolates impact (real cost there is spread + timing), and
the honest plan is "just cross."

**Entry style (momentum).** Pass your passive-chase drag to get a cross-vs-rest recommendation:

```python
plan = plan_execution(5_000_000, cal.normal, entry_drag_bps=14.0)   # findings_08 momentum entry
plan.cross_entry   # True -> cross now; False -> rest a passive limit; None if entry_drag_bps=0
```

The recommendation compares the **spread you'd cross** against the **drift you'd chase** — impact is
common to both styles, so it cancels (hence the compare is to `half_spread`, not `expected_slippage_bps`,
which over-extrapolates impact at retail). For a narrow-spread name like TQQQ, ~14 bps of chase drift
dwarfs the ~0.7 bp spread → **cross** (exactly what the clean market-order sells in findings_08 show);
it only flips to *rest a limit* when the spread is wider than the drift.

## Spread estimator (CS default; EDGE opt-in)

`calibrate` measures the half-spread with **Corwin–Schultz** by default (`spread_method="cs"`) —
the validated 15-min floor. The **EDGE** estimator (Ardia–Guidotti–Kroencke 2024; uses all four
OHLC prices, unbiased under sparse trading) is also available (`spread_method="edge"`, plus
`edge()` / `edge_intraday()`), and the port matches the authors' reference exactly. **But do not
use EDGE at 15-min:** with ~26 bars/session it is underpowered (reads ~0) and it is overnight-gap
sensitive across sessions — its domain is daily bars / long gap-free samples. See
`research/10_edge_vs_cs/findings_10_edge_vs_cs.md`. Either way the spread is a small, non-binding
floor here, dominated by impact + timing.

## Validity / caveats

- **Single-name √-law valid to ~$50M for 3× ETFs.** Above that the binding liquidity is the
underlying (QQQ / futures via creation-redemption), and this model *underestimates* cost.
- **Per-trade, not aggregate.** It's one order's temporary impact; trading repeatedly accumulates
permanent footprint not modelled here, so true session capacity is somewhat lower.
- **Delay is symmetric by default.** If your execution is signal-correlated (momentum fills
adversely), the delay distribution shifts: its *mean* is a signed drag, on top of the symmetric
1σ variance. Pass `CostModel(..., entry_drag_bps=X)` to charge that mean once per round trip
(entry only; clean market exits add ≈0). Live TQQQ limit-chase entries measured **~+14 bps**
(findings_08) — the dominant cost at retail size. It's a strategy input from live fills, not
fittable from OHLC, so `calibrate` never sets it; you declare it.



## API surface


| Object                                                                  | Use                                                              |
| ----------------------------------------------------------------------- | ---------------------------------------------------------------- |
| `calibrate(daily, intraday_15min, spread_method="cs")`                  | measure σ / ADV / half-spread → `Calibration`. Normal σ = window **mean** (expected-cost; used by `normal`); also returns `sigma_daily_median_bps` (typical-day). `spread_method="edge"` opt-in (not for 15-min) |
| `corwin_schultz` / `corwin_schultz_intraday` · `edge` / `edge_intraday` | proportional spread `S` estimators (CS = high/low; EDGE = OHLC). `half_spread_bps(S)` → one-way bps |
| `CostModel(params, spread=, impact=, delay=, entry_drag_bps=)`          | the facade; toggle components; signed momentum entry drag        |
| `model.roundtrip(notional, horizon_min=15, Y=0.5, impact_model="sqrt")` | round-trip cost at a fixed fill horizon. `impact_model="almgren"` uses Almgren et al. (2005)'s fitted temporary term instead of the Y-band sqrt-law (C01); default stays `"sqrt"` |
| `model.roundtrip_band(notional, Ys=(.3,.5,1.))`                         | the Y-band (report this, not a single line)                      |
| `model.roundtrip_optimal(notional, lam=1.0, impact_model="sqrt")`       | round-trip at the Almgren–Chriss optimal speed                   |
| `plan_execution(notional, params, lam=1.0, entry_drag_bps=0.)`          | urgency / slice plan + cross-vs-rest entry call (`.cross_entry`) |
| `RoundTripCost.expected_slippage_bps` / `.timing_bps` / `.timing_var_frac2` | mean drag / risk 1σ / variance                                   |
| `predict_slippage(notional, side, order_type, state, price)`            | predict a fill's slippage: `mean_bps` + non-Gaussian `p50/p90/p95_bps` band, itemized `components`. `impact_model="sqrt"\|"almgren"\|"envelope"` (P03, C01) |
| `estimate_state(ts, symbol, ...)` → `MarketState`                       | market-state input for `predict_slippage` — volume forecast, vol nowcast, regime, spread (P01) |
| `alpha_forfeit_frac(h_min, symbol)`                                     | fraction of a trade's eventual edge forfeited by delaying entry `h_min` minutes (E01) |
| `interruption_hazard(h_min, state)` / `interruption_cost(h_min, phi, mode, symbol)` | mid-fill trailing-stop/signal-flip hazard + a simple cost model (E02) |
| `schedule_order(notional, side, state, price, edge_bps=, pov_cap=0.10, mode="cancel")` | the centerpiece: picks the horizon `h*` (trading off execution cost vs. E01's alpha-forfeiture vs. E02's interruption risk), then slices VWAP-shaped and POV-capped → `Schedule(slices, horizon_min, expected_slippage_band_bps, alpha_forfeit_bps, interruption_summary, feasible)` (E03) |


Lower-level building blocks (`corwin_schultz`, `impact_bps`, `almgren_temporary`, `capacity`,
`expected_slippage_bps`, `optimal_participation`, …) remain exported for direct use.