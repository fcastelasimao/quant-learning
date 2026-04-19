# qframe — Agentic Quantitative Research Pipeline

> A closed-loop system that autonomously proposes, implements, backtests, and evaluates equity factor hypotheses — with a human researcher directing every key decision.

Modelled on [Microsoft RD-Agent(Q)](https://arxiv.org/abs/2404.06975) (NeurIPS 2025). One iteration takes 3–6 minutes on a laptop and costs ~$0 with free-tier LLMs.

---

## What it does

```
Human sets research goal
        ↓
[Synthesis Agent]  — generates a factor hypothesis grounded in academic literature
        ↓
[Implementation Agent]  — writes Python code for the factor (Qwen2.5-Coder via Ollama)
        ↓
[Validation]  — runs a strict walk-forward backtest (OOS from 2018)
        ↓
[Analysis Agent]  — interprets IC, ICIR, IC decay, verdict (PASS / FAIL / ERROR)
        ↓
[Knowledge Base]  — everything logged to SQLite, nothing lost
        ↓
Human reviews results → next iteration
```

96 factors tested across 4 domains (momentum, mean-reversion, volatility, quality) in Phase 1. **0 validated factors.** impl_82 (`trend_quality_calmar_ratio`) retired 2026-04-19 (look-ahead bias: `pct_change(251).shift(-1)` uses tomorrow's price; crypto replication IC=−0.005, t=−0.75). impl_53 not significant (slow-signal t=1.31). Phase 2 HSMM regime analysis complete. Phase 2.5 portfolio — **Gate 3 REVOKED 2026-04-19** (was Sharpe 4.27, driven by impl_82 look-ahead bias; see `notebooks/phase3_crypto_replication.ipynb`).

---

## Key design choices

| Choice | Why |
|--------|-----|
| **Walk-forward only, never in-sample** | Prevents look-ahead; all metrics come from OOS data (2018–) |
| **Net-of-cost IC as primary metric** | Gross IC is easy to game; costs (spread + impact + borrow) always deducted |
| **SQLite knowledge base** | Every hypothesis, code, and result persists; pipeline learns from its own history |
| **Human-in-the-loop** | No autonomous trading; you direct domains, review gates, decide what continues |
| **Tiered LLM routing** | Groq Llama 3.3 70B for synthesis/analysis; Qwen2.5-Coder:14b (local Ollama) for code; auto-fallback to Gemini on quota exhaustion |
| **Self-healing code retry** | On known fixable errors, the implementation agent receives the traceback and attempts one automatic fix |

---

## Running the pipeline

The full pipeline is operational today. Here are the exact terminal commands.

### Prerequisites (one-time setup)

```bash
# 1. Clone and install
git clone <repo-url>
cd qframe
conda create -n qframe python=3.11 -y
conda activate qframe
pip install -e ".[dev]"

# 2. API keys — need at least one for synthesis/analysis (both free tiers)
cp .env.example .env
# Edit .env and fill in at minimum ONE of:
#   GROQ_API_KEY      — free at console.groq.com      (100k tokens/day)
#   CEREBRAS_API_KEY  — free at cloud.cerebras.ai     (1M tokens/day, recommended)
#   GEMINI_API_KEY    — free at aistudio.google.com   (1500 req/day)
# The router tries them in order: groq → cerebras → gemini. Any one is enough.

# 3. Local LLM for code generation (needs ~10 GB RAM; skip if using API fallback)
brew install ollama              # macOS
ollama pull qwen2.5-coder:14b   # 9 GB download, one-time
```

The price cache (`data/processed/sp500_close.parquet`) is already built if you cloned the full repo. If not, it rebuilds automatically on first run (~5 min).

---

### Running Phase 1 — Factor Discovery

```bash
conda activate qframe

# Start Ollama (code generation) — keep this running in a separate terminal
ollama serve

# Run the agentic pipeline in a new terminal
cd /path/to/qframe

# Single iteration (one new factor, ~3 min)
./run_pipeline.sh --domain momentum

# Batch run (5 factors, auto-runs correlation + ensemble every 5)
./run_pipeline.sh --domain momentum --n 5

# Available domains: momentum, mean_reversion, volatility, quality, value
./run_pipeline.sh --domain mean_reversion --n 3
./run_pipeline.sh --domain quality --n 5

# Run all 5 domains in one command (5 iterations each = 25 factors total, ~2 hours)
./run_pipeline.sh --domain all --n 5
```

Each iteration:
1. **Synthesis** — LLM generates a new factor hypothesis (Groq/Cerebras)
2. **Implementation** — Ollama writes the Python factor code (Qwen2.5-Coder local)
3. **Validation** — Walk-forward backtest on 449 stocks, OOS 2018–2024
4. **Analysis** — LLM interprets IC/ICIR/t-stat and issues PASS/FAIL verdict
5. **Logging** — Everything stored to `knowledge_base/qframe.db`

```bash
# Browse results after any number of runs (no new backtests triggered)
jupyter lab notebooks/phase1_pipeline_demo.ipynb
```

---

### Running Phase 2 — Regime Analysis

Phase 2 does not call any LLM and does not need Ollama. It only needs the prices and the knowledge base.

```bash
conda activate qframe
jupyter lab notebooks/phase2_regime_analysis.ipynb
```

Run cells top-to-bottom. The HSMM fitting cell takes **1–2 minutes**. Everything else is fast. The notebook reads the top factors from the knowledge base automatically (via `impl_id` — edit cell 3 to change which factors you analyse).

---

### Inspecting results at any time

```bash
conda activate qframe

# Quick leaderboard (no notebook needed)
python3 -c "
from qframe.knowledge_base.db import KnowledgeBase
from qframe.factor_harness.multiple_testing import correct_ic_pvalues, print_correction_summary
kb = KnowledgeBase('knowledge_base/qframe.db')
corrected = correct_ic_pvalues(kb.get_all_results(), alpha=0.05, n_oos_days=1762)
print_correction_summary(corrected)
"

# Check the SQLite DB directly
sqlite3 knowledge_base/qframe.db "
  SELECT id, ic, icir, t_stat FROM backtest_results
  WHERE ic > 0 ORDER BY ic DESC LIMIT 10;
"
```

---

### Quickstart (fresh clone, minimum setup)

If you just want to test the pipeline works in one go:

```bash
conda activate qframe
ollama serve &                                  # background
./run_pipeline.sh --domain momentum --n 1      # one factor, ~3 min
```

Expected output:
```
[1/5] Synthesis — generating hypothesis for domain: momentum
      → calmar_ratio_12m: Return-to-drawdown momentum...
[2/5] Implementation — generating code with Ollama
[3/5] Validation — running walk-forward backtest
[4/5] Analysis — interpreting results
      → Verdict: PASS
[5/5] Logging result to knowledge base
      → KB ids: hypothesis=83, impl=83, result=83
```

---

## Project structure

```
qframe/
├── src/qframe/
│   ├── data/
│   │   └── loader.py           # All data access: OHLCV, returns, S&P 500 tickers
│   ├── factor_harness/
│   │   ├── ic.py               # Cross-sectional IC, ICIR, IC decay, slow-ICIR
│   │   ├── costs.py            # Almgren-Chriss cost model: spread + impact + borrow
│   │   └── walkforward.py      # Walk-forward validator (expanding window)
│   ├── pipeline/
│   │   ├── loop.py             # Main agentic loop: run_iteration(), run_n()
│   │   ├── models.py           # HypothesisSpec, IterationResult, ResearchSpec
│   │   ├── executor.py         # Sandboxed factor code execution
│   │   └── agents/
│   │       ├── synthesis.py    # Groq/Gemini → HypothesisSpec (with literature seeds)
│   │       ├── implementation.py # Ollama → factor Python code
│   │       ├── analysis.py     # Groq/Gemini → verdict + interpretation
│   │       └── _llm.py         # LLM router: Groq → Gemini fallback
│   ├── knowledge_base/
│   │   └── db.py               # SQLite interface: KnowledgeBase class
│   ├── viz/
│   │   └── charts.py           # 13 visualisation functions (all charts non-blocking)
│   └── regime/                 # Phase 2: HSMM, Hurst, velocity, RegimeICAnalyzer ✅
├── notebooks/
│   ├── gate0_momentum_smoke_test.ipynb
│   ├── phase1_pipeline_demo.ipynb      # Main Phase 1 research interface (15 charts)
│   └── phase2_regime_analysis.ipynb    # Phase 2: HSMM regime analysis (Charts 16–19 + equity curve)
├── knowledge_base/
│   └── qframe.db               # SQLite: 82 implementations, 96+ backtest results
├── data/
│   ├── raw/                    # Immutable. DVC-tracked. Never edit.
│   └── processed/              # Parquet price cache lives here
├── agent_docs/
│   └── research-log.md         # Read at start of every session
├── tests/
│   ├── test_ic.py
│   ├── test_costs.py
│   ├── test_knowledge_base.py
│   └── test_pipeline.py
├── CLAUDE.md                   # Instructions for Claude Code (AI assistant)
├── quant_ai_plan.md            # Master roadmap (Phases 0–4)
├── gate-thresholds.md          # Gate definitions and pass criteria
└── .env.example                # Required environment variables
```

---

## How does the regime model fit into the pipeline?

You're right that the HSMM runs *after* the factors are discovered, not during. This is intentional, and there are three distinct roles it plays at different stages.

### Role 1 — Retrospective validation (Phase 2, done)

Phase 1 evaluates every factor **unconditionally** — averaged over all market conditions. This is correct for discovery, but it raises a follow-up question: *does this factor actually work in all regimes, or only some?*

A factor with IC = 0.03 on average could be IC = 0.07 in trending markets and IC = -0.01 in mean-reverting markets. The average looks modest, but the regime-conditional picture is actionable — you'd run that factor only when the model detects a trending regime.

Phase 2 answers this question for every factor that passed Phase 1. In our case:
- **impl_82 (Calmar/momentum)** has lift = 1.27×. It works about equally in all regimes. Run it all the time, unconditionally.
- **impl_53 (mean-reversion)** has lift = 2.28× in the strong-bull/low-vol state. It works *more than twice as well* in one regime. Only run it when the model says you're in that state.

### Role 2 — Live position sizing (Phase 2.5, coming next)

In deployment, the HSMM updates **every trading day** as new returns arrive. Before setting that day's positions, you ask: *what regime does the model think we're in right now?*

```
Daily close prices arrive
        ↓
HSMM posterior updates (today's state probabilities)
        ↓
regime_weights() computes exposure multiplier per factor
        ↓
factor rank-weights × multiplier → today's positions
        ↓
Execute trades
```

At this point the HSMM is a *real-time filter* running inside the trading loop — not a post-hoc analysis tool.

### Role 3 — Risk reduction during regime transitions (velocity signal)

When the HSMM's posterior changes rapidly (high "velocity"), it means the model detects an imminent regime shift. These transition periods are historically the most dangerous for systematic factors — historical IC no longer applies, but the new regime's characteristics haven't established themselves yet.

The velocity signal gives you an early-warning system: when velocity spikes above a threshold, reduce all factor exposure by some fraction, regardless of which regime you're in. This acts as an automatic de-risking rule before large market inflection points.

### Why not bake the HSMM into Phase 1?

You could, but it would be the wrong order. Running regime conditioning on every factor during Phase 1 would:
1. **Massively multiply the multiple-testing burden** — 5 regimes × each factor = 5× more tests, requiring a much higher BHY threshold
2. **Risk overfitting the regime label** — if you tune which regime is "best" for each factor on the same OOS data you used to select factors, you're data-mining the regime labels
3. **Slow down iteration** — fitting the HSMM adds 1–2 min per factor evaluation

The correct sequence is: discover unconditional factors first → validate regime conditioning only on the small number that passed Phase 1 → build the live regime filter in Phase 2.5.

---

## The factor validation standard

Every factor goes through the same pipeline. Nothing is accepted without passing all checks:

### Walk-forward split
- **In-sample:** 2010–2017 (used only for factor computation warmup)
- **Out-of-sample:** 2018–present (all metrics computed here)
- Expanding window — no re-fitting on OOS data ever

### Primary metrics

| Metric | Description | Weak gate | Pass gate |
|--------|-------------|-----------|-----------|
| **IC** | Mean Spearman rank correlation between factor and next-day returns | ≥ 0.015 | ≥ 0.030 |
| **ICIR** | IC / std(IC) — signal consistency | ≥ 0.15 | ≥ 0.40 |
| **Net IC** | IC after deducting round-trip transaction costs | Positive | Positive |
| **slow_icir_63** | ICIR on non-overlapping 63-day windows (correct for slow signals) | ≥ 0.10 | ≥ 0.25 |

The **WEAK gate** requires IC ≥ 0.015 AND ICIR ≥ 0.15. A factor at the weak gate is worth investigating further but not yet tradeable.

### IC decay

The full IC decay curve from 1 to 63 days is computed and stored for every factor. This tells you *when* the signal realises:
- **Momentum factors** peak at 1–5 days then decay — trade frequently
- **Mean-reversion factors** peak at 21–63 days — hold longer, much lower costs

### Transaction costs

Default cost model (Almgren-Chriss):
```
one-way cost = spread/2 + gamma × (trade_size/ADV)^eta
             = 5 bps   + 30 × (0.10)^0.6
             = 5 + 7.5 = 12.5 bps one-way → 25 bps round-trip
```

Additional costs modelled: short-borrow (50 bps/year default) and leverage/funding (0 by default). All three are summed in `net_ic`.

---

## Current results (Phase 1 + 2 complete, as of 2026-04-18)

**96 factors tested.** Universe: **449 stocks** (survivorship-free S&P 500, best-effort). OOS 2018–2024. Multiple testing correction applied (BHY, m=84 positive-IC factors, threshold t ≥ ~4.0).

### Phase 1 — Factor Library

| Factor | IC | ICIR | t-stat | BHY significant? |
|--------|----|------|--------|-----------------|
| impl_82 `trend_quality_calmar_ratio` (h=1) | 0.0646 | 0.382 | 10.74 | ✅ |
| impl_92 `calmar_proxy_252` (h=1) | 0.0646 | 0.382 | 10.74 | ✅ (duplicate of impl_82) |
| impl_53 `mean_reversion_factor` (h=63) | 0.0490 | 0.251 | 1.31 (slow) | ❌ (fast-formula error; slow t=1.31) |
| Momentum cluster (impl_1/7/85/94/95) | 0.016–0.017 | 0.166–0.169 | ~2.8 | ❌ (HLZ only) |

**0 validated factors. Phase 1 gate INVALIDATED 2026-04-19.** impl_82 and impl_92 (both `trend_quality_calmar_ratio`) were the only BHY-significant factors. Both retired 2026-04-19: look-ahead bias via `pct_change(251).shift(-1)` — uses tomorrow's price. Cross-market check on Binance crypto confirmed: fixed IC = −0.005, t = −0.75. impl_53 was incorrectly flagged (fast-formula t=8.15; correct slow-signal t=1.31).

### Phase 2 — Regime Analysis

5-state HSMM (Zakamulin 2023) fitted walk-forward on OOS 2018–2024.

| Factor | Unconditional IC | Best-regime IC | Lift | Phase 2.5? |
|--------|-----------------|----------------|------|----------|
| impl_82 Calmar (h=1) | 0.0646 | 0.082 (state 2) | 1.27× | Use unconditionally (lift < 1.5×) |

**Gate 2 status moot.** impl_82 was retired for look-ahead bias (2026-04-19). The HSMM infrastructure itself is sound — see `notebooks/phase2_regime_analysis.ipynb`.

---

## LLM providers

| Provider | Role | Free tier | Set in `.env` |
|----------|------|-----------|---------------|
| **Groq** (primary) | Synthesis + analysis | 100k tokens/day | `GROQ_API_KEY` |
| **Gemini** (fallback) | Auto-switches when Groq quota hits | 1500 req/day (2.0 Flash) | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) |
| **Ollama / Qwen2.5-Coder:14b** | Factor code generation | Free (local) | `OLLAMA_HOST` |
| Cerebras | Optional third fallback | 1M tokens/day | `CEREBRAS_API_KEY` |

The router in `_llm.py` handles fallback automatically on quota exhaustion. By default, **non-quota errors propagate** (so you fix config/access), but you can opt in to failover on 403/network-policy denials via `LLM_FAILOVER_ON_ACCESS_DENIED=1`.

---

## Roadmap

```
Phase 0  ✅  Factor harness (IC, ICIR, decay, costs), SQLite KB, unit tests
Phase 1  ⚠️  Agentic pipeline — 96 factors tested; 0 validated (impl_82 INVALIDATED 2026-04-19: look-ahead bias)
Phase 2  ✅  Regime-aware factor timing — 5-state HSMM operational; impl_82 analysis moot
Phase 2.5  ❌  Regime-conditional portfolio — Gate 3 REVOKED 2026-04-19 (driven by impl_82 look-ahead bias)
Phase 3    ⬜  Crypto microstructure + on-chain factors (top 50 assets, Glassnode/CoinMetrics)
Phase 4    ⬜  Cross-lingual signal extraction (Chinese NLP → Western equities)
```

Each phase is gated — Phase N+1 cannot start until Phase N passes validation on two independent markets.

---

## Running the tests

```bash
conda activate qframe
pytest tests/ -v --tb=short
```

---

## Known limitations

- **Survivorship bias:** Historical S&P 500 membership data is unavailable; the 449-stock universe uses current constituents only. Stocks that failed, were acquired, or delisted between 2010 and today are excluded. True fix requires Norgate Data or CRSP (Phase 2 Gate 1 requirement).
- **Multiple testing:** BHY correction (implemented in `factor_harness/multiple_testing.py`) requires t ≥ ~4.0 with m=84 positive-IC factors. Only impl_82 (Calmar, t=10.74) clears this bar. For slow signals (h≥21d), use the non-overlapping window t-stat: `slow_icir_63 × √(N/63)` — the standard `icir × √(N/252)` formula over-counts independent observations and inflates t-stats dramatically.
- **Single market:** All results are US large-cap equities only. Every accepted hypothesis must be confirmed on a second market before being considered real.
- **Cost model:** Position-sizing uses uniform 10% ADV per trade. Hard-to-borrow names use the same borrow rate as easy names. See `costs.py` module docstring for the full gap list.

---

## Acknowledgements

Architecture inspired by [RD-Agent(Q)](https://arxiv.org/abs/2404.06975) (Microsoft Research, NeurIPS 2025). Factor literature seeds from Harvey, Liu & Zhu (2016), Jegadeesh & Titman (1993), Fama & French (1993), Bali, Cakici & Whitelaw (2011), and Almgren et al. (2005).
