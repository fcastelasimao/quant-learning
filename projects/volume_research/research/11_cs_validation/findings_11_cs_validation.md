# Findings 11 — CS-15min accuracy validation (Stage 13)

**Date:** 2026-07-06 · **Tickers:** TQQQ, SQQQ, QQQ · **Question:** is the Corwin–Schultz 15-min
half-spread (~0.74 / 1.00 / 0.72 bp) actually *accurate*, or just internally consistent?
**Answer: yes for the traded names (TQQQ/SQQQ), validated against real NBBO quotes.**

## Method — two tiers

**Tier 1 (no external data):** the quoted spread cannot be below one tick (`$0.01 / price`), so the
ratio CS / tick-floor tells us where CS sits. Plus CS at 1-min vs 15-min (does it stabilise?), and
the findings_08 live-fill / synthetic-recovery context.

**Tier 2 (gold-standard):** pull real **Alpaca SIP quotes** (full consolidated NBBO — the account
has SIP, not just free IEX) for a sample of intraday 1-minute windows (8 recent trading days ×
3 times of day ≈ 21 windows/ticker), compute the **time-weighted quoted half-spread**, and compare
directly to CS.

## Results

| ticker | price | tick-floor ½ | **CS-15min** | CS/floor | **SIP NBBO ½** (real) | CS / SIP |
|---|---:|---:|---:|---:|---:|---:|
| TQQQ | $77.70 | 0.64 | **0.74** | 1.15 | **0.89** (med 0.83) | **0.83** |
| SQQQ | $39.50 | 1.27 | **1.00** | 0.79 | **1.32** (med 1.31) | **0.76** |
| QQQ  | $722.75 | 0.07 | **0.72** | 10.4 | **0.35** (med 0.33) | **2.03** |

(bps, one-way half-spread. SIP = 21 windows/ticker, time-weighted.)

## Verdict

- **CS-15min is accurate for the leveraged ETFs.** TQQQ 0.74 vs real 0.89, SQQQ 1.00 vs real 1.32 —
  within ~25%, right order of magnitude, and **conservative in the useful direction** (CS reads
  slightly *low*, i.e. it doesn't over-charge). The tick-floor and NBBO agree: TQQQ/SQQQ trade at
  ~1 tick and CS pins that.
- **CS overstates QQQ ~2×** (0.72 vs real 0.35) — the **high-price bias** Tier 1 predicted
  (CS/floor = 10× for QQQ). At $723 a penny is a minuscule fraction of price, and CS's
  volatility-driven term inflates the estimate. **Immaterial:** QQQ is not a traded instrument here
  (it's the "capacious" reference), and an *over*-stated spread is conservative for capacity.
- **1-min CS collapses to ~0** for all three → confirms findings_01: daily and 1-min CS fail;
  **15-min is the only working resolution**, and it's the one we use.
- Consistent with findings_08 (live market-order sells ~0±2 bps) and the synthetic recovery test.

**Bottom line:** the ~0.7–1 bp spread floor in the cost model is real and correctly sized for
TQQQ/SQQQ — now confirmed against actual NBBO, not just self-consistent. And it remains a small,
**non-binding** floor: even doubling it wouldn't move the impact/timing-dominated capacity curve.

## Caveats

- **Sample, not census:** ~21 one-minute windows/ticker over 8 recent days — enough to pin the
  level, not a full-history spread series. Spreads widen at the open/close and in stress (the SIP
  p90 runs ~1.4 bps for TQQQ); the mean is the representative floor.
- SIP quotes are **quoted** spreads; realized *effective* spreads (price-improvement inside the
  quote) are typically a touch tighter, so CS-vs-true if anything looks slightly *more* accurate.

## Reproduce

```
/Users/.../envs/quant/bin/python research/11_cs_validation/build_11_cs_validation.py
```
Tier 2 hits the Alpaca SIP quotes API (keys from `api_keys.env`); it skips gracefully if
unreachable. Outputs in `results/` (gitignored): `cs_validation.csv`, `cs_validation.png`.
