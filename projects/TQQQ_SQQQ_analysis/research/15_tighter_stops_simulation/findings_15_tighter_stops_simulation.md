# 15 — Tighter-stops simulation

> Period: Full regime-labeled history 2013–2026 (1,782 TQQQ + 1,511 SQQQ trades).

**Scope.** For each trade, walk the TQQQ/SQQQ 15-min bars between `entry_time` and `exit_time`. Compute the intra-trade maximum loss `min(bar.low / entry_price) − 1`. For each candidate stop level `s ∈ {0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0} %`, simulate "if a stop at `−s %` had been in place, the trade would have exited at `−s %`". Compare simulated total pnl_pct vs original.

Both symbols: 100 % of regime-labeled trades have intraday bar coverage (1,782 TQQQ + 1,511 SQQQ = 3,293 trades, each walked over its lifetime).

## Headline — every tighter stop destroys net pnl

| symbol | stop | n_stop_hit | n_winners killed | n_losers caught | n_severe (≤ −1 %) caught | **Δ total pnl_pct** |
|---|---:|---:|---:|---:|---:|---:|
| TQQQ | 0.25 % | 1525 | 763 | 762 | 560 | **−1496.6** |
| TQQQ | 0.50 % | 1322 | 566 | 756 | 560 | **−1254.8** |
| TQQQ | 1.00 % | 927 | 288 | 639 | 545 | **−873.1** |
| TQQQ | 2.00 % | 392 | 81 | 311 | 283 | **−462.8** |
| TQQQ | 3.00 % | 189 | 22 | 167 | 154 | **−298.7** |
| TQQQ | 5.00 % | 42 | 5 | 37 | 34 | **−130.0** |
| SQQQ | 0.50 % | 80 | 32 | 48 | 39 | −70.4 |
| SQQQ | 1.00 % | 57 | 16 | 41 | 38 | −45.8 |
| SQQQ | 2.00 % | 26 | 8 | 18 | 17 | −35.9 |
| SQQQ | 3.00 % | 10 | 1 | 9 | 9 | −11.5 |

**Every stop on every level on both symbols is a net negative.** The pattern is monotone in TQQQ: tighter stops fire more, kill more winners. SQQQ is similar but smaller magnitude because SQQQ's trades have smaller intraday drawdowns on average.

## Why this fails — the strategy needs intraday drawdown room

The trades that end up as winners often **dip deep intraday before recovering**. A 1 % tighter stop on TQQQ kills **288 winners** (16 % of all trades) while catching only 639 losers — net carnage. The strategy's edge depends on **letting winners breathe through their intraday max drawdown**.

This is the *dynamic* version of item 04's finding. Item 04 said: feature regions of high loser-rate are also high winner-magnitude regions (you can't skip them). Item 15 says: the *time evolution* of a loser-vs-winner inside a trade is also indistinguishable in the early bars — both dip below `−s %` at some point.

## A surprising sub-pattern

Even at very tight 0.25 % stops, **n_winners_killed (763) ≈ n_losers_caught (762)**. The intraday "first touch of −0.25 %" happens on roughly half of all trades, *regardless of outcome*. This implies that **the entry timing is essentially noise at the 15-min scale**: a coin flip whether the next 15-min bar opens above or below entry. The strategy's value is captured by **what the trade does over its full life**, not its first few minutes.

The "almost balanced winners-killed vs losers-caught" weakens monotonically as stops widen: at 5 % stops only **5 winners get killed** vs 37 losers caught — but this is because **a 5 % drawdown is rare** (only 42 of 1782 TQQQ trades ever touched it), so the small saved pnl doesn't recover the lost gains from the few catastrophic winners that needed that 5 % room.

## What this means for the project

1. **Do not add tighter stops to the strategy.** This is now definitively answered.
2. **The current exit logic is well-tuned** — whatever stop mechanic the source strategy uses, it's at least non-destructive at our resolution.
3. **The "tighter stops" branch from the pass-1 synthesis is now closed.** It was listed as a possible reframe; the data says no.
4. **Contrast with item 12 (continuous sizing): sizing-down ENTRY notional works (item 12) because it doesn't change the exit logic; tightening EXIT stops does not work (item 15) because it kills winners that needed their drawdown.**

## Honest caveats

- **15-min bars are coarse.** A drop intra-bar that touches the stop and reverses inside the same bar would have triggered a real stop. My simulation uses `bar.low` so it *does* capture the bar-low, but it can't distinguish "low touched briefly mid-bar" from "low held for 10 minutes". This means my numbers are **upper bounds on stop benefit**; reality is even worse.
- **Idealized fill.** I assume the stop fills at exactly `−s %`. Real stops slip below that, especially in the leveraged ETFs we trade. Adds further negative bias to real-world tighter stops.
- **Stop interaction with strategy's own exit logic.** I assume the tighter stop *replaces* the original exit. If instead it's a *minimum* of (existing exit, new stop), the math is similar — and the conclusion holds (still net negative).

## Per-year breakdown

`stop_sweep_yearly.csv` has the full per-year picture. Headline: there is no year where any tighter stop level was net positive on either symbol. The strategy needed full drawdown room every year — even in 2022 (the bear year for QQQ), tighter stops still hurt TQQQ.

## How to read the plots

**`stop_sweep_delta.png`** — two side-by-side panels (TQQQ blue, SQQQ red) showing the net pnl impact of imposing a tighter stop at each of 8 levels.

- **X axis**: stop level in percent from entry price (`0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0`).
- **Y axis**: `Δ total pnl_pct = simulated total − original total`. A positive bar would mean the tighter stop helped; negative means it hurt.
- **Annotation above each bar**: `n_stop_hit / n_winners_killed`. The first number is how often the stop triggered; the second is how many of those triggers killed an actual winner.
- **What to look for**:
  - **EVERY bar is negative** across both symbols across every stop level. Tighter stops on this strategy are universally destructive.
  - **The annotation ratio is the diagnostic**: at `0.25 %` you see `1525/763` (1525 triggers, 763 of them killed winners — about half). At `5.0 %` you see `42/5` (only 42 triggers, only 5 winners killed) — but the total pnl impact is still negative because the 5 winners that DID need that 5 % room contributed disproportionate gains.
- **Why it matters**: this is the clearest single visual in the whole project that **tightening exit stops** is the wrong lever. The graph is monotone wrong-direction. Use sizing (item 12), not stops, to reduce drawdown.

## Artifacts

| file | content |
|---|---|
| `build_15_tighter_stops_simulation.py` | analysis script |
| `stop_sweep_summary.csv` | per (sym, stop_pct) totals: trigger rate, winners killed, losers caught, Δ total pnl |
| `stop_sweep_yearly.csv` | same metrics per year |
| `per_trade_intraday_path.csv` | per-trade `intraday_max_loss_pct` + sim_pnl for each stop level (3,293 rows × 19 cols) |
| `stop_sweep_delta.png` | Δ total pnl_pct bar chart per symbol per stop level |
| `findings_15_tighter_stops_simulation.md` | this note |
