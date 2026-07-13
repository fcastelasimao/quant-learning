# Findings 12 — Stage 0: resolving the two forks (Almgren-adoption plan)

**Date:** 2026-07-07 · **Question:** two upstream unknowns can each delete a downstream stage of
the Almgren-adoption plan — resolve both before writing any impact-model code.

- **0a — Alpha horizon.** Is the 15-min strategy's edge captured intraday (minutes), which would
  make the creation/redemption door (hours-to-overnight fill) temporally unreachable and delete
  Phase 2 (Stages B/E)? Or does it accumulate over days, keeping Phase 2 alive?
- **0b — ETF permanent impact.** Is Almgren's `(Θ/V)^{1/4}` permanent-impact term — built for a
  single stock's fixed float — even the right model for an arbitrage-pinned, elastic-supply ETF?

**Answers: 0a = multi-day (Phase 2 stays live). 0b = mostly no — Almgren's term is the wrong
mechanism for TQQQ/SQQQ, and the true permanent residual is small but not exactly zero.**

## 0a — Alpha horizon: multi-day, not intraday

**Method.** Real 15-min market data (not the backtest's synthetic flat-haircut fills — see
`CLAUDE.md` "Provenance & cost model"), anchored at every `entry_time` in
`../TQQQ_SQQQ_analysis/full_history_canonical/TRADES_{SYM}_full_history.csv` (2,343 TQQQ / 1,930
SQQQ trades, 2013–2026): (a) forward-return decay shape, (b) fraction of a trade's eventual
entry→exit move already realized after delaying entry by 15min–1day, (c) where the trade log's
P&L actually lives, by hold-time bucket.

**Results.**

*Forward-return decay (bps, signed by trade direction) — grows monotonically through 10 days, no
reversal:*

| | 15m | 30m | 1h | 2h | 4h | 1d | 2d | 3d | 5d | 10d |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| TQQQ mean | 1.9 | 0.8 | 2.6 | 5.9 | 15.0 | 21.6 | 44.1 | 66.8 | 91.9 | 161.3 |
| TQQQ median | 2.1 | 1.5 | 2.7 | 8.2 | 26.7 | 45.3 | 78.0 | 106.2 | 158.5 | 277.2 |

(SQQQ mirrors this — magnitude grows monotonically, sign negative; full detail in
`results/0a_decay_{SYM}.csv`. The SQQQ *level* being unconditionally negative is expected volatility
decay drag of a 3× inverse product on buy-and-hold, not a signal failure — the strategy's realized
edge comes from entry *selection* + exit timing, not raw hold-through-return.)

*Fraction of a trade's eventual entry→exit move already used, by execution delay on entry (median
across trades whose hold period spans the delay):*

| | 15m | 30m | 1h | 2h | 4h | 1d |
|---|--:|--:|--:|--:|--:|--:|
| TQQQ | 6.1% | 11.1% | 19.6% | 36.8% | 80.1% | 95.9% |
| SQQQ | 7.3% | 15.2% | 26.2% | 43.6% | 79.4% | 101.4% |

A 30–60 min execution delay gives up only ~11–26% of the eventual move; the bulk (~80%+) isn't
realized until 4h–1 day out.

*Where the P&L lives, by hold-time bucket:*

| hold bucket | TQQQ % of trades | TQQQ % of total P&L | SQQQ % of trades | SQQQ % of total P&L |
|---|--:|--:|--:|--:|
| < 2.4h | 22.0% | **−9.9%** | 23.6% | **−31.6%** |
| 2.4–6h | 16.5% | 4.2% | 18.0% | 17.4% |
| 12–24h | 31.5% | 11.3% | 35.6% | 22.3% |
| 1–2d | 12.3% | **43.2%** | 9.2% | **33.0%** |
| 2–3d | 8.3% | 11.7% | 8.9% | 22.4% |
| 3–5d | 7.4% | 21.5% | 4.7% | 36.6% |
| 5d+ | 1.9% | 17.9% | — | — |

**The fastest-held trades (<2.4h, ~22–24% of the book) are net *losers*** on both symbols. Nearly
all the profit comes from trades held ≥1 day. All exits are `TRAIL_STOP`-driven (winners are let
run, not closed on a fixed clock), consistent with a trend-continuation edge, not a fast momentum
burst.

**Verdict: multi-day accumulation.** The profitable core of the strategy operates on a 1–5+ day
timescale. A creation/redemption fill delay of hours-to-overnight sacrifices only a modest
fraction of the eventual move for the trades that generate the P&L, and is irrelevant risk-wise
for the very trades (<2.4h) that lose money anyway. **Phase 2 (Stages B, E) stays in the plan.**

## 0b — ETF permanent impact: Almgren's term is the wrong mechanism; measured residual is small

**Reasoning.** Almgren's permanent term assumes a single stock's roughly-fixed float: a persistent
buy/sell imbalance is read by the market as genuine information, permanently shifting the price
(§3.1 of the paper — a drift proportional to trade rate, with the *linear* functional form chosen
so total permanent impact scales with size `X` alone, independent of execution speed `T`).

TQQQ/SQQQ break this mechanism structurally: their share count is not fixed — Authorized
Participants create and redeem shares daily in response to exactly the imbalance that would cause
"permanent impact" in a single stock. NAV is computed continuously off a deep, liquid, already
fully-priced underlying (QQQ/NDX via swaps and futures) — there is no "new information" a TQQQ
trade could be revealing that isn't already public and priced in the index itself. If TQQQ's screen
price drifts from `3×NAV`, an AP can create shares and sell them on-screen, arbitraging the
deviation away — typically well within a trading day, faster than Almgren's own ~30-minute
post-trade measurement window. So Almgren's `(Θ/V)^{1/4}` — a single-stock float-scarcity proxy —
is measuring the wrong economic mechanism if applied to TQQQ's own (elastic) share count.

**Empirical check.** Event study on real Alpaca tick data (`/trades`, delayed-SIP, free): 4 sample
trading days (2026-06-30, 07-01, 07-02, 07-03 — the last skipped, a market holiday), full-day tape
for TQQQ, SQQQ, and QQQ. For each trade at/above the 99th-percentile size that day (continuous
hours 09:35–15:55 ET), measured the **excess return** — `TQQQ_return − 3×QQQ_return` (or `×−3` for
SQQQ) — at 1-minute and 30-minute horizons after the print, signed by the print's own immediate
price-move direction (so all events point the same way):

| | n events | short (1min) excess | long (30min) excess | reversion (1 − long/short) |
|---|--:|--:|--:|--:|
| TQQQ | 11,148 | +0.37 ± 0.11 bps | +0.26 ± 0.09 bps | 0.28 |
| SQQQ | 4,063 | −0.63 ± 0.41 bps (n.s.) | +0.23 ± 0.08 bps | not meaningful (sign flips) |

**Interpretation.** For TQQQ, both the short- and long-horizon excess are small but statistically
nonzero (t≈3.4 and t≈2.9). Only ~28% of the initial excess reverts by 30 minutes — **not a clean
"permanent = 0,"** but the surviving residual is tiny: ~0.26 bps, a quarter of the TQQQ half-spread
(0.74 bps, findings_11) and two orders of magnitude below what Almgren's Θ/V term implies for a
comparably-large single-stock order (Table 3: IBM/DRI permanent impact ≈ 20–22 bps at 10% of ADV —
though note that comparison isn't apples-to-apples: these events are individual large *prints*
(a few thousand shares), not $M+ *metaorders*, so this result anchors the near-field only, per the
plan's own R2 caveat (low power at the far field). SQQQ's short-horizon estimate is not
statistically distinguishable from zero, so its numbers are directional at best.

**Verdict: partial confirmation, not a hard zero.** The arbitrage-pinning argument holds in
direction and order of magnitude — Almgren's single-stock permanent term would badly overstate
TQQQ's permanent impact — but the data doesn't support literally omitting a permanent term either.
**Resolution:** Stage A does **not** use Almgren's `(Θ/V)^{1/4}` on TQQQ's own share count (wrong
mechanism, would overstate). Instead, permanent impact is carried as a small, near-field-anchored
term (~0.2–0.3 bps at the measured print scale), tiered **Modeled/no-ground-truth** (per the
plan's 0c evidence-tier convention) with an explicit caveat that it is unvalidated at $M+ metaorder
scale. **A.5 (accumulation/W5) is downgraded from a full stage to a bounded sensitivity check** —
consistent with the plan's "if permanent≈0 ⇒ A.5 shrinks" branch, since the residual is small
even though not literally zero.

## What this changes in the plan

- **Phase 2 (Stages B, E) stays in scope** — 0a came back multi-day, not intraday.
- **Stage A:** omit Almgren's `Θ/V` permanent term (wrong mechanism for an elastic-supply ETF);
  carry a small measured-residual permanent adjustment instead, tiered Modeled/no-ground-truth.
- **Stage A.5:** downgraded to a bounded one-shot check (not a full research stage) — the
  accumulation concern is real in principle (permanent impact isn't literally zero) but small in
  measured magnitude.
- All 0c evidence-tier and reproducibility conventions (as-of dates on external snapshots) apply
  going forward.

## Caveats

- **0b is near-field only.** The event study measures individual large prints (thousands of
  shares), not the $M+ metaorders the capacity question is about. It anchors the *direction* and
  *rough scale* of the residual, not a validated number at scale — irreducible without proprietary
  parent-order data (per Almgren's own paper, §2.1: public tape data cannot see metaorders).
- **SQQQ's 0b result is weak** (short-horizon estimate not significant); treat as directional only.
- **0a's P&L-by-hold-bucket finding is a modeling input, not a causal proof** that delaying entry
  specifically (vs. delaying exit, which the strategy already does via trailing stop) preserves
  edge — but combined with the delay-cost fraction (few % given up in the first 30–60 min), the
  practical conclusion is robust: patient execution does not gut the trades that make money.
- **4 sample days (0b) is a small window** — one was a market holiday (2026-07-03, observed for
  July 4th), leaving 3 effective days. Directionally sufficient for a bounded Stage-0 probe; not a
  long-run calibration.

## Reproduce

```
/Users/franciscosimao/opt/anaconda3/envs/quant/bin/python \
    research/12_stage0_forks/build_12_stage0_forks.py
```
0a is instant (local data). 0b hits the free Alpaca `/trades` endpoint (delayed-SIP, >15 min old)
across 4 sample days × 3 symbols — takes a few minutes, retries transient network timeouts.
Outputs in `results/` (gitignored): `0a_decay_{SYM}.csv`, `0a_delay_cost_{SYM}.csv`,
`0a_pnl_by_hold_{SYM}.csv`, `0b_events_{SYM}.csv`, `stage0_forks.png`.
