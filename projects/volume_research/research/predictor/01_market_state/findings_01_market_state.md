# Findings P01 — Market state: volume profile, predictability, vol nowcast, spread state

**Date:** 2026-07-08 · **Tickers:** TQQQ, SQQQ, QQQ · **Question:** what does "depending on the
state of the market / market volume" concretely mean as an input to the predictor (P03) and
scheduler (E03)? **Answer: four measured curves — an intraday volume-share profile, its
out-of-sample predictability, a volatility nowcast consistent with S03's regime labels, and an
intraday spread curve — packaged behind one pure function, `estimate_state()`.**

## Headline

The U-shape is real and large (open bin carries **4x** the volume of the midday lull), but
**bin-level volume is only modestly predictable out-of-sample** (R² 0.09–0.24 from a simple
time-of-day × trailing-EWMA baseline) — enough to be a useful multiplier, not enough to promise
precision. The **intraday vol nowcast agrees with S03's daily regime label only ~51% of the time**
(vs. ~33% chance) — expected, since they measure genuinely different things (today's first two
hours vs. a trailing 20-day window), not a bug. The **spread curve exactly reproduces S01's
published intraday numbers** (2.53 / 2.76 / 1.19 bps at the open for TQQQ/SQQQ/QQQ), which is the
strongest validation in this stage.

## Strategy and mathematics

**1. Intraday volume profile.** For each 15-min session bin (09:30…15:45), `share = bin_volume /
day_total_volume`, aggregated across ~2,300–4,100 trading days into mean/median/p10/p20/std per
bin. The p10/p20 are the **thin-tape floors** — how low a bin's share can plausibly go, used
separately from the central forecast to flag thin-tape risk.

**2. Interval-volume predictability.** `predicted_volume(bin, day) = bin_share_mean(bin, TRAIN) ×
trailing_daily_EWMA(day)`, where the EWMA (span 10 days) is evaluated **strictly before** the
target day (no leakage) and `bin_share_mean` is fit on a train split (first 80% of days
chronologically). Evaluated out-of-sample (last 20% of days, held out): `R² = 1 - SS_res/SS_tot`.

**3. Volatility nowcast.** `sigma_now_bps = std(recent 1-min log returns) · √(minutes_per_day) ·
1e4` — a trailing-window realized-vol estimate scaled to the same daily-bps units as
`MarketParams.sigma_daily_bps`. Classified against **fixed tercile bounds reproduced exactly from
`research/03_delay_cost/build_03_delay_cost.py::vol_regime`** (20-day trailing daily realized vol,
full-sample terciles) — same q1/q2 cutpoints, so a "stress" label here means the same thing as a
"stress" label in S03's tables. Validated by comparing a first-**120-minute** nowcast each session
against that day's own S03-style daily label.

**4. Spread state.** The CS-15min half-spread by time-of-day bin (identical method to S01, just
grouped by bin instead of averaged over the whole sample). CS pairs consecutive bars and is
indexed by the **second** bar of each pair, so **09:30 never appears** — 09:45 is the earliest bin
with an estimate. Live-facing use should prefer a fresh NBBO read (S11's pattern) via
`estimate_state(live_spread_bps=...)`; this curve is the historical/fallback estimate, and S11
showed CS reads ~20–25% low vs. real SIP NBBO for TQQQ/SQQQ.

## Numbers

**Volume-share profile** (mean, open vs. midday; tier: **Measured**):

| ticker | 09:30 (open) | 12:30 (midday) | ratio | 15:45 (close) |
|---|--:|--:|--:|--:|
| TQQQ | 10.76% | 2.57% | 4.2x | 6.81% |
| SQQQ | 10.56% | 2.52% | 4.2x | 7.97% |
| QQQ  |  8.94% | 2.59% | 3.5x | 10.24% |

Textbook U-shape, both ends elevated — QQQ's close bump is the largest of the three (10.24% >
its own open share), consistent with index-rebalance / MOC flow concentrating late.
Thin-tape floors (p10) at 12:30 are **~0.8–1.2%** of the day's volume (TQQQ 1.19%, SQQQ 0.84%,
QQQ 1.23%) — a fifth of the mean, the number that matters for a "thin tape, be careful" flag.

**Interval-volume predictability** (tier: **Modeled**, OOS): TQQQ R²=0.241 (821 held-out days),
SQQQ R²=0.174, QQQ R²=0.089. Real signal, far from precise — daily total volume swings on news/vol
days beyond what a 10-day EWMA captures. Sufficient as the scheduler's volume-shaping multiplier;
not sufficient to promise a tight per-bin forecast.

**Volatility nowcast vs. S03 daily regime** (tier: **Modeled**): TQQQ 51.1% agreement (q1=250.1,
q2=358.5 bps; 4,091 days), SQQQ 50.3% (q1=249.4, q2=359.5), QQQ 52.0% (q1=87.0, q2=128.2). All
comfortably above the ~33% chance baseline for a 3-class label, confirming the nowcast tracks the
same underlying concept — but the two measures are legitimately different (today's first 2 hours
vs. a trailing 20-day window), so ~50% is the expected ceiling, not a bug to chase.

**Spread state** (tier: **Measured**, reproduces S01 exactly):

| ticker | 09:45 (earliest bin) | 12:30 (midday) |
|---|--:|--:|
| TQQQ | 2.53 bps | 0.62 bps |
| SQQQ | 2.76 bps | 0.99 bps |
| QQQ  | 1.19 bps | 0.65 bps |

## Caveats

- **Predictability R² is a simple baseline**, not a ceiling — a richer model (news/vol-day
  features, day-of-week, options-expiry calendar) would likely do better; out of scope here.
- **Nowcast/daily-regime agreement (~51%) should not be read as "the nowcast is unreliable"** —
  it's measuring a faster-reacting, current-day signal against a slow trailing-window label by
  design. The synthetic tests (`tests/test_state.py`) directly validate `classify_regime`'s
  boundary logic and `sigma_now_bps`'s scaling, which is the actual correctness bar for this code.
- **CS spread is the historical/fallback curve** (S11: ~20–25% low for TQQQ/SQQQ vs. real NBBO,
  ~2x high for QQQ). `estimate_state` supports a live override; nothing here regenerates S11's
  NBBO work.
- **`estimate_state` is a pure combinator** — it does not fetch data itself. The research stage
  measures the reference curves from history (`VolumeProfile`, `SpreadCurve`, `VolRegimeBounds`);
  a caller (P03, E03) is responsible for keeping a live 1-min return window and a trailing daily
  volume EWMA and passing them in.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/predictor/01_market_state/build_01_market_state.py
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python -m pytest tests/test_state.py -v
```
Outputs in `results/` (gitignored): `volume_profile_{SYM}.csv`, `spread_curve_{SYM}.csv`,
`market_state.png`.
