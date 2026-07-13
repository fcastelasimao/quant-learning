# volume_research — slippage / market-impact & capacity

Understand how execution cost scales with order size for the TQQQ / SQQQ 15-min decision
strategies, and package it into a small, drop-in **slippage library** (`slippage/`) that any
backtest can use to replace flat, size-blind cost assumptions.

**Headline:** TQQQ keeps half its edge to ~\$4M per trade, and the edge is gone by ~\$14M — all
figures ± the adopted-`Y` impact band (not fittable from our OHLC data alone; see ROADMAP.md).

## Where to look

- **`ROADMAP.md`** — the master map: what's done, what's not, and the two paths forward. Start here.
- **`FINDINGS.md`** — the meeting-ready results summary.
- **`BUILD_LOG.md`** — chronological log of decisions and stages.
- **`slippage/README.md`** — how to drop the library into a backtest.
- **`research/NN_*/findings_*.md`** — per-stage detail and reproduce instructions.
- **`plans/`** — forward plans (active execution track + paused capacity-refinement track).
- **`docs/history/`** — superseded planning docs, kept for scoping rationale.

## Run

Uses the **quant** conda env.

```bash
~/opt/anaconda3/envs/quant/bin/python -m pytest tests/ -q
```
