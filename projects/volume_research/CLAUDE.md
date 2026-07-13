# CLAUDE.md — volume_research

Guidance for Claude Code when working in `volume_research/`.

## Mission

Understand **slippage / market impact** and how it scales, then build a **simple, passable
slippage (Implementation-Shortfall) library** that colleagues can drop into their backtests to
*replace* whatever ad-hoc cost handling they use today. See `docs/history/research_goals.md` for
the original questions.

The point is **not** to measure cost on the current (tiny-scale) trades — at retail size cost is
spread-dominated and nearly constant. The point is the **capacity question**: how does cost grow,
and how much can a strategy trade, across order sizes from \$100k → \$500k → \$1M → … → \$1B?
The headline deliverable is a **net-Sharpe-vs-AUM capacity curve** and the size-aware cost
function behind it.

## What we want the library to cover (modular — caller chooses what's on)

- **Spread** — half-spread crossed per side; the always-on, size-independent floor.
- **Permanent impact** — equilibrium price shift, ~linear in total size; persists.
- **Temporary impact** — book depletion from trading *fast*; reverts after you stop. Depends on
  participation rate.
- **Delay / timing cost** — drift between the decision price and the actual fill (the gap between
  the 15-min decision bar and execution). Volatility × time, signless in expectation but a real
  variance/cost driver for 3× ETFs.
- **Commissions** — out of scope for now (assume zero-commission broker), but leave a slot.

Anchor model: the **square-root law** `I = Y·σ·√(Q/V)` (Bouchaud; Almgren 2005 found temporary
impact closer to a 3/5 power). Execution-schedule theory: **Almgren–Chriss** (impact-vs-timing-risk
trade-off). Benchmark framework: **Implementation Shortfall** (Perold 1988).

## Data

- **TQQQ/SQQQ trade logs** live in `../TQQQ_SQQQ_analysis/` — they are *backtest* output from a
  separate execution engine running on **FMP OHLC bars**, with a **flat 5 bps entry / 15 bps exit**
  slippage baked in (size-independent). They therefore contain **no** information about impact at
  scale. Do not try to extract a capacity curve from them directly.
- **FMP** is available for any additional data we need (intraday bars, volume). For 15-min
  decisions, the relevant volume for impact is *interval* volume, not daily ADV.
- Live fills exist only from **May 2026** onward (~1.5 months as of 2026-06-17) — note it, but it is
  too short to calibrate against; treat as noise for now.

## Working style

- Literature first, our own research second. Papers are the backbone; forums (Reddit, Wilmott,
  Elite Trader, Quant SE) are color for hypotheses, never sources of constants.
- Verify any constant against FMP data before it enters a model.
- Keep the eventual library small and dependency-light so it is easy to pass around.

## Working mode

Normal fast-mode: this is a **work project, build it quickly**. Claude writes the code. Track
progress and decisions in `BUILD_LOG.md`. (Teach-mode is intentionally *not* active here.)

---

_Behavioral rules (Karpathy) now live globally in `~/.claude/CLAUDE.md` and apply automatically._
