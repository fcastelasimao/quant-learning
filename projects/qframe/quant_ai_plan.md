# Quant AI Research Pipeline — Master Plan
*Francisco | April 2026 | Living Document*

---

## Vision

Build an agentic research pipeline that generates, tests, and evaluates quantitative alpha hypotheses faster than a human researcher could alone — with human creative direction at every key decision point. The pipeline is the infrastructure. The four alpha directions are the research programme that runs on top of it.

---

## Guiding Principles

- **Human-in-the-loop always.** No autonomous execution. You are the creative director and critic.
- **Simplest thing first.** Build what is known and proven before building what is novel. Each layer must earn its place through walk-forward validation before the next is added.
- **Walk-forward validation is non-negotiable.** Every factor, every strategy. No exceptions.
- **Quality over quantity.** 20 genuinely uncorrelated factors beat 200 redundant ones.
- **Build to learn, extend to ship.** Rebuild core components from scratch as learning exercises, then plug into proven open-source infrastructure (Qlib + RD-Agent) for production.
- **Persistent memory from day one.** Every hypothesis, result, and failure is logged. The pipeline learns from its own history.
- **Tiered LLM architecture.** Route by task complexity. Don't pay frontier model prices for every call.
- **Ideas are cheap, validation is expensive.** Every idea goes into the backlog. Only one thing is in active development at a time.
- **Every result needs a second market.** Any hypothesis validated on US equities must also be tested on at least one independent market (European equities, Japanese equities, or crypto) before being accepted as real. Single-market results are hypotheses, not findings.

---

## Sequence

```
Phase 0: Foundations (now)
    └── Rebuild factor harness from scratch (with transaction cost model)
    └── Set up infrastructure (Qlib, RD-Agent, SQLite knowledge base)
    └── Set up hardware (M1 + Portugal PC + Tailscale)

Phase 1: Agentic Pipeline (4)  ← ✅ COMPLETE
    └── Core loop: Hypothesis → Code → Backtest → Evaluate → Store  ✅ LIVE
    └── 96 factors tested; 1 BHY-significant (impl_82 t=10.74); impl_82/impl_92 are duplicates; impl_53 t=1.31 (slow-signal formula — not significant)
    └── Universe: 449-stock S&P 500 (survivorship-free best-effort)
    └── Multiple testing: BHY correction implemented (factor_harness/multiple_testing.py)
    └── Auto-update: research-log.md written after every run_n() call
    └── Crowding monitor as risk overlay (reflexivity-aware)  ← BACKLOG

Phase 2: Regime-Aware Factor Timing (1)  ← ✅ COMPLETE
    └── 5-state HSMM (Zakamulin 2023) — walk-forward OOS 2018–2024
    └── Gate 1 ✅: HSMM fitted; 5 states with economically interpretable characteristics
    └── Gate 2 ✅: impl_82 lift = 1.27× (used unconditionally; regime-gating threshold 1.5×)
    └── Phase 2 notebook: sections A–D with equity curve analysis

Phase 2.5: Regime-Conditional Portfolio Construction  ← ✅ COMPLETE
    └── Gate 3 ✅ PASSED: Sharpe 4.27, MaxDD −10.3%, worst year +17.8%, cost efficiency 99.8%
    └── IC-proportional blend + regime posterior blend (Chart 22)
    └── Turnover 883%/yr is operational note (not a gate criterion); cost efficiency gate (≥90%) replaces raw turnover cap
    └── See notebooks/phase25_portfolio.ipynb for full Gate 3 analysis

Phase 3: Crypto Microstructure & On-Chain (3)
    └── On-chain factor library via Glassnode / CoinMetrics
    └── Order book microstructure features
    └── Cross-sectional model across top 50 liquid crypto assets

Phase 4: Cross-Lingual Signal Extraction (2)
    └── Chinese-language NLP pipeline (luxury goods / China sector)
    └── Multilingual LLM sentiment and event extraction
    └── Map signals to tradeable Western-listed universe
```

*Note: Phases 3 and 1 may be swapped based on interest — crypto data is free and feedback loops are fast.*

---

## Phase 1: Agentic Research Pipeline (Foundation)

### What It Is
A closed-loop system that autonomously proposes, implements, backtests, and evaluates factor hypotheses — with you directing research goals and validating promising outputs. Modelled on Microsoft's RD-Agent(Q) (NeurIPS 2025), which achieved ~2x returns over classical factor libraries at under $10 per optimisation cycle.

### Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AGENTIC LOOP                       │
│                                                     │
│  Specification → Synthesis → Implementation         │
│       ↑                              ↓              │
│  Analysis ←── Validation ←── Backtesting            │
│                                                     │
│  [Knowledge Base: SQLite — every result stored]     │
└─────────────────────────────────────────────────────┘
         ↑
  Human Director (Francisco)
  injects domain knowledge,
  sets research goals,
  validates survivors
```

**Five agents (matching RD-Agent architecture):**

| Agent | Role | LLM Tier |
|---|---|---|
| Specification | Define market universe, factor domain, constraints | Tier 3 (Sonnet executor) |
| Synthesis | Generate factor hypotheses from theory + knowledge base | Tier 1/2 (Flash/local) |
| Implementation | Translate hypotheses to Python code | Tier 1 (Ollama/Flash) |
| Validation | Run walk-forward backtest, compute IC/ICIR/Sharpe/decay | No LLM — pure Python |
| Analysis | Interpret results, update knowledge base, schedule next loop | Tier 3 (Advisor Strategy) |

### Advisor Strategy Implementation

The Advisor Strategy (launched April 9, 2026, currently in beta) formalises the tiered LLM architecture inside a single API call. Sonnet 4.6 acts as the executor running the full agentic loop; Opus 4.6 acts as the advisor, consulted only when the executor encounters a decision too complex to resolve alone. The executor decides when to escalate — no manual orchestration required.

**When Opus gets consulted in the qframe loop:**
- Should this factor be accepted or rejected given the IC decay curve?
- Is this backtest result a genuine edge or an overfitting artefact?
- Which hypothesis class should the next iteration pivot to?
- Does this regime state posterior suggest a meaningful factor timing opportunity?

**When Sonnet handles it alone (most of the time):**
- Code generation and scaffolding
- Data ingestion and cleaning
- Standard backtest execution
- Knowledge base writes and reads
- Routine hypothesis generation

**Implementation:**

```python
import anthropic

client = anthropic.Anthropic()

tools = [
    {
        "type": "advisor_20260301",
        "name": "advisor",
        "model": "claude-opus-4-6",
        "max_uses": 3   # cap Opus consultations per loop iteration
    },
    # ... code execution, SQLite logging, data fetch tools
]

response = client.beta.messages.create(
    model="claude-sonnet-4-6",           # executor
    max_tokens=8096,
    betas=["advisor-tool-2026-03-01"],
    tools=tools,
    messages=messages
)
```

**Cost control:** `max_uses=3` caps Opus at 3 consultations per iteration — each returning ~400–700 tokens. Track advisor token usage via the separate `usage` block in the response. Set a per-run budget and alert if it is exceeded.

**Access:** currently in beta requiring the `advisor-tool-2026-03-01` header. Check whether Pro plan access is available or contact Anthropic. If not yet accessible, the manual tiered routing in the LLM tiering section above is the fallback until access is granted.

### Key Components to Build

**Factor Evaluation Harness (rebuild from scratch first):**
- Input: any signal series
- Output: IC, ICIR, Sharpe, turnover, factor decay curve, correlation to existing factors
- Cross-sectional framework (rank assets at each timestep, not time-series)
- Walk-forward with no lookahead, no survivorship bias
- **Transaction cost model (non-optional):** Almgren-Chriss market impact for equities; separate model for crypto (fees, slippage, funding rates). IC after costs is the real metric. A factor with positive gross IC but negative net IC is not a factor.

**Knowledge Base (SQLite → PostgreSQL later):**
```sql
-- Core tables
hypotheses (id, description, rationale, created_at)
implementations (id, hypothesis_id, code, created_at)
backtest_results (id, implementation_id, ic, icir, sharpe, 
                  turnover, decay_rate, regime, created_at)
factor_correlations (factor_a, factor_b, correlation, period)
```

**Crowding Monitor (reflexivity-aware risk overlay):**
- Measures correlation of active factors to known institutional flows
- Tracks Days-ADV proxy for factor crowding
- Scales down position sizing when crowding signal is elevated
- *This is a risk management layer, not a primary alpha source*

### Infrastructure Stack

| Component | Tool | Where |
|---|---|---|
| Factor harness | Python (built from scratch) | M1 MacBook |
| Pipeline orchestration | Raw Python first, Qlib+RD-Agent later | M1 MacBook |
| Knowledge base | SQLite → PostgreSQL | Portugal PC (CPU server) |
| Data ingestion jobs | Scheduled Python scripts | Portugal PC |
| LLM: high-volume tasks | Ollama (Qwen2.5-Coder 7B) | M1 MacBook |
| LLM: mid-tier reasoning | Gemini 2.5 Flash API | Cloud |
| LLM: top-tier synthesis | Claude Sonnet API | Cloud |
| Remote access | Tailscale + VS Code Remote SSH | M1 → Portugal |
| Power monitoring | Tapo P110 smart plug | Portugal PC |

### LLM Tiering

```
Tier 1 — Local / Free (bulk volume)
├── Ollama: Qwen2.5-Coder:7B on M1 MacBook
└── Gemini 2.5 Flash (generous free tier)
    Use for: hypothesis generation, code scaffolding,
             data parsing, simple classification

Tier 2 — Cheap API (medium reasoning)
├── Gemini 2.5 Flash (paid)
└── DeepSeek V3 API (~$0.27/M tokens)
    Use for: hypothesis refinement, backtest interpretation,
             factor scoring and ranking

Tier 3 — Frontier (Advisor Strategy: Sonnet executor + Opus advisor)
├── Claude Sonnet 4.6 — executor: runs the agentic loop cheaply
└── Claude Opus 4.6  — advisor: consulted only for hard decisions
    Use for: novel factor ideation, research synthesis,
             strategy evaluation, final validation
```

*Rule: Run 200 Tier 1 calls to screen. Pass survivors to Tier 2 for refinement. Only top candidates reach Tier 3 — and within Tier 3, use the Advisor Strategy rather than running Opus end-to-end.*

**Advisor Strategy (Phase 1+):** Sonnet runs the agentic loop cheaply and escalates to Opus only for genuinely hard decisions — factor evaluation, hypothesis pivots, strategy synthesis. Opus reads full context, returns 400–700 tokens of strategic guidance, Sonnet continues. One API call, no extra round trips. Early benchmarks: +2.7pp quality lift and −11.9% cost versus Sonnet alone. Beta header required: `advisor-tool-2026-03-01`. See Phase 1 for implementation details.

---

## Phase 2: Regime-Aware Factor Timing

### What It Is
Most factors have wildly different performance across market regimes. The edge is knowing *when* to deploy each factor. This phase builds regime awareness in strict order of complexity — each step only proceeds if the previous one demonstrates measurable value in walk-forward validation.

**HSMM = Hidden Semi-Markov Model.** The "semi" means the time spent in each state follows its own duration distribution, rather than the memoryless geometric distribution that standard HMMs assume. This duration dependence — the longer you've been in a regime, the more likely it is to end — is the key structural improvement over standard HMMs.

### Experimental Gates (build in order, stop when value stops)

**Gate 0 — Baseline: Does regime filtering help at all?**
Use your existing standard HMM (already built). Test whether switching factor weights based on regime state improves walk-forward Sharpe over a single fixed-weight portfolio. If not, regime timing doesn't work for your factor set and further elaboration is pointless.

**Gate 1 — Five-state HSMM: Does richer structure help?**
Replace the standard HMM with a five-state Hidden Semi-Markov Model. Empirically validated by Zakamulin (2023) on 124 years of US equity data — five states outperform two or three. States are numbered 1–5; their character is described by their empirical statistics, not by pre-assigned names.

The five states that emerged empirically:
| State | Return profile | Volatility | Typical factor behaviour |
|---|---|---|---|
| 1 | High positive | Low | Momentum, quality work well |
| 2 | Moderate positive | Elevated | Momentum still works, more noise |
| 3 | High positive, unsustainable | Low then spiking | Mean-reversion dangerous |
| 4 | Negative | Moderate | Carry, low-vol defensive |
| 5 | Sharply negative | Very high | Most factors fail; cash/vol |

The HSMM outputs: (a) current state posterior probabilities, (b) transition matrix, (c) duration distribution per state. All three are inputs to the factor timing decision.

**⟶ Paper Trading Gate (mandatory before Gate 2)**
After Gate 1 passes in backtest, the strategy must run on Alpaca paper trading for a minimum of 3 months before Gate 2 begins. This is not optional. The paper trading period:
- Validates that the backtest translates to live-ish conditions
- Catches implementation bugs that walk-forward validation misses
- Provides real evidence of regime detection latency and rebalancing friction
- Must also be validated on one independent market (European or Japanese equities) before Gate 2 is unlocked

Results are logged in the knowledge base with the same rigour as backtests.

**Gate 2 — Transition velocity: Does regime dynamics help beyond the label?**
Add the first derivative of the state posterior as an explicit signal:
```
velocity_t = P(state_t | data) - P(state_{t-1} | data)
```
Test whether this improves IC over the state label alone. This is the novel research contribution — components exist in the literature but this specific fusion does not appear in published work. If it adds nothing, drop it cleanly.

**Gate 3 — Unsupervised discovery: Does data-driven clustering find structure the HSMM misses?**
Only attempt if Gates 0–2 have all passed. Apply Wasserstein k-means to rolling return distributions — no predefined labels, states numbered i=1,...,k. The output is purely statistical: empirical return/volatility/correlation profile per cluster, and transition probabilities between clusters. No naming required. Compare discovered structure to the HSMM five-state labels. The research question: does the unsupervised method find regimes that the HSMM misses, and do those additional regimes improve factor timing?

**Gate 4 — LLM semantic layer: Does narrative context add signal beyond statistics?**
Only if Gate 3 passes. Feed current macro/news summaries to an LLM and ask whether the narrative is consistent with the current statistical regime. This adds a qualitative layer that statistical models cannot access. Highest implementation complexity; lowest certainty of value.

### Factor Library to Build (Gate 0 prerequisite)
- Momentum (12-1 month)
- Value (book-to-price, earnings yield)
- Carry (dividend yield, short rate differential)
- Low-volatility
- Quality (ROE, debt-to-equity)
- Crowding-adjusted versions of the above (from Phase 1 crowding monitor)

### Key Papers (read before each gate)
| Gate | Paper |
|---|---|
| 0–1 | Zakamulin (2023) — "Not all bull and bear markets are alike" *Risk Management* |
| 0–1 | Zakamulin & Giner (2024) — optimal trend-following under semi-Markov *Journal of Asset Management* |
| 1 | Filardo (1994) — time-varying transition probabilities (statsmodels replication in Python exists) |
| 1 | Cortese, Kolm & Lindström (2024) — statistical jump models vs. HMM |
| 3 | Horvath, Issa & Muguruza (2024) — Wasserstein k-means *Journal of Computational Finance* |
| 4 | Alpha-R1 (Dec 2025, arXiv) — LLM reasoning for context-aware factor screening |

### Skill Gaps
| Skill | Status |
|---|---|
| HMM theory + implementation | ✅ Strong |
| Factor construction + backtesting | ✅ Have infrastructure |
| Semi-Markov / duration-dependent models | ⚠️ Extension of existing — read Zakamulin |
| Time-varying transition probabilities | ⚠️ New but close — statsmodels available |
| LLM API integration | ⚠️ Acquirable quickly |
| Financial factor theory (Fama-French, AQR) | ⚠️ Read the papers |
| Wasserstein k-means implementation | ❌ Gate 3 only — not needed yet |
| NLP / text preprocessing | ❌ Gate 4 only — not needed yet |

---

## Phase 2.5: Regime-Conditional Portfolio Construction

### What It Is
Signal generation (finding good factors) and portfolio construction (combining them into positions) are separate problems that interact in non-obvious ways under regime switching. This phase addresses the second problem explicitly, building on your existing risk parity work from the All Weather project.

### The Problem with Naive Combination
Risk parity and equal-weighting assume a roughly stationary covariance matrix. They do not. In HSMM State 5 (crash/correction), cross-asset correlations spike toward 1 — diversification collapses precisely when you need it most. A portfolio construction model that ignores regime is implicitly assuming you'll always be in State 1.

### What to Build
**Regime-conditional covariance estimation:**
- Estimate a separate covariance matrix for each of the five HSMM states using historical data within each state
- At each rebalancing point, weight the state-conditional covariance matrices by the current HSMM state posterior probabilities
- This gives a blended covariance that reflects regime uncertainty rather than assuming you're definitely in one state

**Regime-aware position sizing:**
- Factor weights are not binary (on/off per regime) — they are continuous functions of the state posterior
- In State 1 (probability 0.8), apply full momentum weight
- In a mixed posterior (0.4 State 1, 0.4 State 4), blend appropriately
- This makes the portfolio smooth across regime transitions rather than lurching

**Connection to All Weather work:**
The risk parity optimisation you already built (SLSQP, covariance-based weights) is the foundation. The extension is making the covariance input regime-conditional. The objective function stays the same; the inputs become state-aware.

### Why This Is Phase 2.5 and Not Later
Portfolio construction is tightly coupled to regime detection — they cannot be evaluated independently. The moment Gate 1 passes, you need a portfolio construction method to actually trade the strategy. Keeping them coupled means the paper trading period tests both together, which is the right unit of validation.

### Prerequisite
Gate 1 (five-state HSMM) passes in backtest.

---

## Phase 3: Crypto Microstructure & On-Chain Signals

### What It Is
Crypto markets have two layers of inefficiency unavailable in traditional finance: undermodelled order book microstructure, and on-chain data (wallet flows, miner behaviour, DeFi liquidity) with no traditional analog.

### On-Chain Factor Library
| Factor | Source | Economic rationale |
|---|---|---|
| Exchange net flows | Glassnode / CoinMetrics | Sell pressure signal |
| MVRV ratio | Glassnode | Overvaluation / mean reversion |
| NVT ratio | CoinMetrics | Network value vs. transaction volume |
| Miner outflows | Glassnode | Miner capitulation signal |
| DeFi TVL flows | DefiLlama | Risk appetite / liquidity |
| Whale wallet activity | Glassnode | Smart money positioning |

### Order Book Features
- Bid-ask spread (liquidity proxy)
- Order imbalance (directional pressure)
- Depth asymmetry (buy vs. sell wall)
- Trade size distribution (retail vs. institutional)

### Key Reading
- Kyle (1985) — foundational microstructure paper
- Glosten-Milgrom (1985) — adverse selection in markets
- *Both are accessible given your physics background*

### Skill Gaps for This Phase
| Skill | Status |
|---|---|
| Time series modelling | ✅ Strong |
| Statistical hypothesis testing | ✅ Strong |
| Crypto domain knowledge | ⚠️ 4-6 weeks focused reading |
| On-chain data APIs | ❌ Gap |
| Microstructure theory | ❌ Real gap — discipline unto itself |
| Execution modelling (slippage) | ⚠️ Gap |

---

## Phase 4: Cross-Lingual Signal Extraction

### What It Is
Markets systematically misprice assets when relevant information exists in a language the dominant investor base cannot process. Chinese-language consumer sentiment, regulatory filings, and industry commentary on luxury goods, EVs, and industrial commodities is the clearest example. Western analysts work from delayed, filtered English summaries.

### Target Pipeline
```
Chinese data sources (Weibo, Caixin, Sina Finance, CSRC)
    ↓
Multilingual LLM (GPT-4o / mBERT fine-tuned)
    ↓
Sentiment scores + event flags + entity relationships
    ↓
Signal mapped to Western-listed universe
(LVMH, Kering, Richemont, etc.)
    ↓
Backtest: does signal predict weekly returns 
          with appropriate lag controls?
```

### Why This Phase Is Last
Data sourcing is the hardest barrier — Chinese platforms are hostile to scraping, multilingual NLP has real failure modes, and the pipeline requires the full agent infrastructure from Phase 1 to operate efficiently.

### Skill Gaps for This Phase
| Skill | Status |
|---|---|
| Statistical modelling / signal testing | ✅ Strong |
| Web scraping + data ingestion | ❌ Real gap |
| Multilingual NLP | ❌ Significant gap |
| SQL / data pipeline management | ⚠️ Gap — build in Phase 1 |
| Market microstructure (signal decay) | ⚠️ Acquirable |
| Chinese language | ❌ Unless acquired separately |

---

## Theoretical Framework: Markets as Open Systems

This section captures a foundational perspective that emerged from first principles and is validated by multiple research traditions. It shapes how all regime and factor work is conceptualised.

### The Core Observation

The stock market is not a closed system. Prices are a noisy, partial projection of a much higher-dimensional reality. Everything that drives prices but is not directly observable — economic fundamentals, investor psychology, geopolitical dynamics, institutional flows, policy decisions, social dynamics, supply chains — belongs to the external environment of the open system. Historical price data alone is an incomplete description of the system's state.

This is not a metaphor. It has precise formal consequences:

- Models trained only on prices will systematically underfit the true dynamics
- Regime transitions can be *endogenous* (generated internally) or *exogenous* (triggered by external shocks) — these have different precursor signatures and require different responses
- The excess volatility puzzle — why prices move more than news justifies — is explained by endogenous feedback amplification, not external shocks alone
- The five HSMM states are not just statistical clusters; they are different *coupling modes* between internal market dynamics and the external information environment

### Four Research Traditions That Formalise This

**1. Econophysics (open systems / dissipative structures)**
Markets as Prigogine-type dissipative structures — far-from-equilibrium open systems that maintain temporary ordered states (regimes, trends) through continuous information inflow, undergoing phase transitions when that flow changes qualitatively. The mathematics of critical phenomena, bifurcation theory, and Hawkes self-excitation processes apply directly. Key result: crashes have endogenous (self-organised) and exogenous (shock-triggered) varieties with measurably different precursor signatures.

**2. Latent factor models (finance)**
The formal finance formalisation of the same observation. Observed returns are driven by latent (unobservable) factors that named factors (Fama-French, momentum) imperfectly proxy. Lettau and Pelger's RP-PCA recovers weak latent factors with Sharpe ratios twice as large as conventional PCA — factors invisible to standard methods because they explain little variance while carrying high risk premia. IPCA allows factor loadings to be time-varying, capturing how the market's coupling to underlying risks changes across states.

**3. Critical slowing down (complex systems)**
As a system approaches a bifurcation, recovery from small perturbations slows. In price data this manifests as rising cross-asset correlation, rising autocorrelation, and rising variance — measurable early warning signals of regime transitions that appear *before* the transition, using prices alone. No external data required.

**4. Endogenous/exogenous decomposition**
Formal decomposition of market volatility into components driven by external shocks versus internal self-excitation. Hawkes processes model the cascade of internally-generated reactions following external perturbations. Key empirical finding: markets generate a substantial fraction of their own volatility endogenously at all timescales.

### Practical Implications for the Pipeline

**Implication 1 — Regime states are coupling modes, not just clusters.**
The five HSMM states represent different regimes of sensitivity to external information. State 1 (low-vol bull) is internally stable — the market absorbs shocks with minimal amplification. State 5 (crash) is near a critical point — small shocks trigger large endogenous cascades. Factor performance differences across states are partly explained by how much endogenous amplification is occurring, not just by economic conditions.

**Implication 2 — Critical slowing down as early warning (backlog item).**
Before the HSMM label changes, the system shows measurable precursors: rising cross-asset correlation, rising return autocorrelation, rising variance. These are computable from prices alone and give advance warning of HSMM state transitions. This is the most actionable near-term addition from the open systems framework.

**Implication 3 — Latent factor recovery alongside regime detection (backlog item).**
Running RP-PCA or IPCA alongside the HSMM gives a richer description of the current system state — not just which regime, but which latent risk dimensions are currently active. The formal answer to: "what is the hidden system that generates the prices I observe?"

**Implication 4 — Endogenous vs. exogenous crash classification (backlog item).**
Crashes triggered by internal dynamics (1987) have different precursor signatures than crashes triggered by external shocks (2001, 9/11). Correctly classifying which type is approaching changes the appropriate defensive response.

### Why This Background Is Particularly Well-Suited

The mathematics of open systems, dissipative structures, phase transitions, critical phenomena, bifurcation theory, and Hawkes processes are all natural territory for a mathematical physicist. The conceptual leap from particle physics (gauge theory, fiber bundles) to financial market dynamics (information geometry, factor spaces) is smaller than it appears — both deal with symmetry, invariants, and the geometry of state spaces. The instinct to ask "what is the system actually, and what are we missing?" is exactly what the physics tradition trains and what decades of quant finance has undervalued.

---

## Reflexivity as Risk Overlay

Reflexivity (Soros) describes how strategies change the markets they predict, invalidating their own assumptions. This is measurable and should be incorporated — but as **risk management**, not as a primary alpha source.

### What to Measure
- Factor correlation to institutional 13F holdings (quarterly, lagged)
- Short interest utilisation differential (long vs. short leg)
- Return comovement within factor baskets (crowding signature)
- Days-ADV proxy (how long to exit a crowded position)
- Live-vs-backtest drift (early decay detection)

### How to Use It
- High crowding signal → reduce factor position sizing
- Sustained drift from backtest behaviour → pause strategy, investigate
- Hawkes process criticality metric → regime risk flag

### Why Not a Primary Strategy
- Historical reflexive episodes are sparse and structurally heterogeneous
- The measurement problem is recursive (crowding signals get crowded)
- Data requirements (full 13F, order flow) are expensive and lagged
- At current scale, the observer effect on your own trades is negligible — revisit when AUM is meaningful

---

## Hardware Setup

```
MacBook Air M1 16GB (London) — Primary Interface
├── Daily coding and research (VS Code)
├── Ollama: Qwen2.5-Coder:7B for local LLM inference
├── API calls to Claude, Gemini, DeepSeek
└── SSH/Tailscale tunnel to Portugal

Portugal PC (CPU Server Role Only)
├── CPU: Intel i5 7th gen
├── GPU: GTX 970 (3.5GB VRAM — too limited for LLM)
├── RAM: 16GB
├── Role: PostgreSQL database, scheduled data jobs, 
│         CPU-bound backtesting runs
├── Power: ~210-250W under load, ~80-100W idle
├── Electricity cost: ~€6-12/month at €0.20/kWh
└── Monitoring: Tapo P110 smart plug

Potential GPU Upgrade (when income is stable)
└── RTX 3060 12GB (~€200-250 secondhand in Portugal)
    → Enables 13B models comfortably, 30B with quantisation
```

---

## Open Source Infrastructure

### Primary: Microsoft Qlib + RD-Agent
- **Qlib:** AI-oriented quant platform covering full pipeline from data to execution
- **RD-Agent(Q):** Agentic R&D loop accepted at NeurIPS 2025 — directly relevant
- **Strategy:** Build factor harness from scratch first (learning), then plug into Qlib/RD-Agent
- **GitHub:** github.com/microsoft/qlib | github.com/microsoft/RD-Agent

### Reference: AgentQuant
- Open-source individual implementation powered by Gemini 2.5 Flash
- Detects market regimes (Bull/Bear/Crisis) via VIX + Momentum
- Generates regime-aware strategy parameters, runs walk-forward validation
- Useful as a reference architecture at accessible scale

### Key Papers to Read

**Pipeline foundations:**
- RD-Agent-Quant (NeurIPS 2025) — the methodological foundation
- Alpha-R1 (Dec 2025) — LLM reasoning for context-aware factor screening

**Regime dynamics (Phase 2 core):**
- Zakamulin (2023) — "Not all bull and bear markets are alike" — 5-state HSMM empirics
- Zakamulin & Giner (2024) — optimal trend-following under semi-Markov
- Filardo (1994) — time-varying transition probabilities (statsmodels replication available)
- Cortese, Kolm & Lindström (2024) — statistical jump models vs. HMM

**Microstructure and factor theory:**
- Kyle (1985), Glosten-Milgrom (1985) — microstructure foundations
- Fama-French (1993), AQR factor papers — factor theory foundations

**Open systems / latent factors / critical transitions:**
- Lettau & Pelger (2020) — RP-PCA: estimating latent asset-pricing factors (*Journal of Econometrics*) — free on Pelger's Stanford page
- Kelly, Pruitt & Su (2019) — IPCA: characteristics are covariances (*Journal of Financial Economics*)
- Scheffer et al. (2009) — "Early-warning signals for critical transitions" (*Nature*) — foundational paper on critical slowing down
- Johansen, Ledoit & Sornette — JLS model of log-periodic power laws before crashes
- Mantegna & Stanley — *Introduction to Econophysics* (Cambridge) — accessible entry point to the physics tradition

---

## Data Infrastructure

### The Survivorship Bias Problem — Non-Negotiable to Understand

Before anything about specific sources: survivorship bias is the single most common way backtests produce fictional results. Delisted companies — those that went bankrupt, were acquired, or were removed from indices — disappear from most standard datasets. A factor strategy built without them looks better than it really is, because all the companies that went to zero are missing from the sample.

The magnitude is not trivial: survivorship bias can inflate annual backtest returns by 4–6%, and in one documented case a strategy's win rate fell from 80% to 52% when delisted tickers were properly included. Every data source discussion below must be read with this in mind.

---

### TradingView — Ruled Out as Data Source

TradingView has no official programmatic API for stock data. What exists is a patchwork of unofficial libraries (`tvDatafeed`, `tradingview-ta`, `tradingview-screener`) that scrape internal endpoints, violate terms of service, and break periodically. It is a charting platform designed for discretionary visual traders — not a data infrastructure tool.

**Decision:** Skip entirely as a data source. Revisit only as a monitoring and visualisation layer after the strategy is live, if visual chart monitoring is useful at that point.

---

### Data Source Strategy by Phase

The principle: use free data until the research demands better. Don't pay for premium data before you have something worth validating.

**OpenBB as the unifying abstraction layer (use from day one)**

OpenBB is free, open-source, and Python-native. It standardises data access across ~100 providers under a single interface. You write:

```python
obb.equity.price.historical("AAPL", provider="yfinance")
```

And later swap `provider="polygon"` or `provider="tiingo"` without changing any downstream code. This is the right engineering choice — your factor harness never touches a provider directly; it always speaks OpenBB. Provider upgrades become zero-code changes.

---

### By Phase

| Phase | What you need | Source | Cost | Survivorship bias |
|---|---|---|---|---|
| **Phase 0** | Daily OHLCV, broad universe for testing harness | yfinance via OpenBB | Free | Biased — acceptable for learning |
| **Phase 2 Gates 0–1** | Index-level price/vol for HMM/HSMM training | yfinance via OpenBB | Free | Not relevant — modelling index, not stocks |
| **Phase 2 Gates 1+ (factor library)** | Cross-sectional stock universe — point-in-time | **Norgate Data** | ~£25/month | **Bias-free — non-optional at this stage** |
| **Phase 2 fundamentals** | P/E, book value, ROE for value/quality factors | OpenBB + yfinance or FMP | Free–$15/month | Partial |
| **Macro / rates** | Fed funds rate, GDP, CPI, yield curve | FRED via OpenBB | Free | Not applicable |
| **Phase 3 crypto OHLCV** | Price history for top 50 liquid assets | CoinGecko API | Free | Not applicable |
| **Phase 3 on-chain** | Exchange flows, MVRV, NVT, miner outflows | Glassnode free tier / CoinMetrics | Free to start | Not applicable |
| **Phase 4** | Chinese-language financial news | Caixin, Sina Finance (scraping) | Free but difficult | Not applicable |

---

### Source Profiles

**yfinance** — Start here. Free, no API key, pandas-native, massive community support. Reliability issues occasionally (breaks 1–2x per year, usually fixed within days). Survivorship-biased — delisted tickers return empty DataFrames silently. Adequate for Phase 0 and index-level work. Never use for cross-sectional factor research.

**Norgate Data** — The correct upgrade for cross-sectional factor research. Point-in-time, survivorship-bias-free, covers all US and Australian equities including delisted tickers. Includes index membership history (S&P 500, Russell 3000) at any historical date — critical for constructing honest backtests. Python API available. ~£25/month. This is the minimum acceptable data standard for Phase 2 factor work.

**CRSP (Center for Research in Security Prices)** — The academic gold standard. Survivorship-bias-free back to the 1920s. Used in virtually every published academic finance paper. Expensive for individuals. **Check whether your PhD gives you alumni library access** — many universities provide CRSP via WRDS (Wharton Research Data Services) to alumni. If yes, use it. It is significantly better than Norgate for historical depth.

**OpenBB** — Not a data provider itself, but the abstraction layer over all others. Install and use from day one. Free.

**FRED (Federal Reserve Economic Data)** — Free, authoritative, massive. All macro data: Fed funds rate, yield curve, CPI, GDP, unemployment. Accessible via OpenBB (`obb.economy.fred_series`). Use for building macro-aware factors and for the LLM semantic layer input.

**Glassnode** — On-chain data for Phase 3. Free tier covers MVRV ratio, exchange net flows, and a handful of other signals with a 24-hour lag. Paid tier (~$29/month) unlocks real-time and the full signal library. Start free.

**CoinGecko** — Crypto OHLCV data. Free API, no key required for basic usage, covers 14,000+ assets. Use for Phase 3 price data. CoinMetrics is higher quality for on-chain analytics.

---

### Universe Selection

**For HSMM regime work (Phases 2 Gates 0–1):** S&P 500 index (SPY or ^GSPC via yfinance). Simple, clean, sufficient.

**For cross-sectional factor research (Phase 2 Gate 1+):** Russell 3000 with point-in-time membership via Norgate Data. Better factor dispersion than S&P 500, less survivorship bias in index membership history.

**For factor research starting point before Norgate:** S&P 500 current constituents via yfinance. Acknowledge the bias, treat results as directional hypotheses only, upgrade to Norgate before claiming anything real.

**For crypto (Phase 3):** Top 50 liquid assets by 90-day average volume. Exclude stablecoins. CoinGecko for universe construction.

---

### On the Decision Table

| Item | Decision | Reason |
|---|---|---|
| TradingView as data source | ❌ Ruled out | No official API; unofficial tools fragile and ToS-violating |
| TradingView as visualisation | 📋 Backlog | Possible monitoring tool once strategy is live — not now |
| yfinance | ✅ Phase 0 only | Free, adequate for learning; survivorship-biased |
| OpenBB | ✅ Use from day one | Abstraction layer — future-proofs all data switching |
| Norgate Data | ✅ Phase 2 Gate 1+ | Non-optional for cross-sectional factor research |
| CRSP via WRDS | ✅ Check alumni access | Gold standard; free if university access available |
| FRED | ✅ Use from Phase 2 | Free macro data via OpenBB |
| Glassnode free tier | ✅ Phase 3 start | On-chain signals; upgrade to paid when needed |
| CoinGecko | ✅ Phase 3 | Free crypto OHLCV |



Ideas are captured here in full, but not acted on until the active work earns them. The structure prevents good ideas from being lost and prevents the queue from driving the work.

### How It Works
- **Active** — one thing being built right now
- **Validated queue** — passed a gate, next in line
- **Backlog** — worth exploring, not yet prioritised
- **Parked** — considered and deliberately set aside, reason recorded

Parked ideas are recorded with reasons so they are not re-evaluated from scratch later.

---

### Active
*Phase 0: Foundations — hardware setup, factor harness rebuild, knowledge base schema*

---

### Validated Queue (in priority order)
1. Phase 1: Agentic pipeline core loop
2. Phase 1 upgrade: Advisor Strategy (Sonnet executor + Opus advisor) — add once pipeline is running
3. Phase 2 Gate 0: Baseline HMM regime filter + factor timing
4. Phase 2 Gate 1: Five-state HSMM
5. Phase 2.5: Regime-conditional portfolio construction (runs in parallel with paper trading gate)

---

### Backlog

| Idea | Rationale | Prerequisite |
|---|---|---|
| Phase 2 Gate 2: Transition velocity signal | Novel — ΔP/Δt of HSMM posterior as factor timing input. Clean falsifiable hypothesis. | Gate 1 passes |
| Critical slowing down early warning | Rising cross-asset correlation + autocorrelation + variance as regime transition precursor. Computable from prices alone. No external data needed. | Gate 1 passes |
| Latent factor recovery (RP-PCA / IPCA) | Recover weak latent factors invisible to standard PCA. Sharpe ratios 2x larger in published results. Run alongside HSMM for richer state description. | Gate 1 passes |
| Endogenous vs. exogenous crash classification | 1987-type (internal) vs. 2001-type (external shock) crashes have different precursor signatures and require different defensive responses. | Gate 1 passes |
| Phase 2 Gate 3: Unsupervised Wasserstein k-means | Data-driven regime discovery — no predefined labels, states 1..k described by empirical statistics + transition matrix | Gates 0–2 pass |
| Phase 2 Gate 4: LLM semantic regime layer | Narrative consistency as qualitative gate on statistical regime signal | Gate 3 passes |
| Phase 3: Crypto on-chain + microstructure | On-chain factors (MVRV, exchange flows, miner outflows) + order book features. Free data, fast feedback loops. | Phase 1 complete |
| Phase 4: Cross-lingual signal extraction | Chinese-language NLP pipeline for luxury/EV sector information asymmetry | Phase 1 complete + NLP skills built |
| Reflexivity crowding monitor (full build) | 13F-based crowding signal for position sizing. Currently using cheaper proxy. | Access to 13F data |
| Hawkes process criticality metric | Physics-grounded measure of market self-excitation as a regime risk flag | Gate 1 passes |
| Regime-transition momentum factor | Factor that explicitly exploits drift between HSMM sub-states | Gate 2 passes |
| Statistical jump model alternative | Non-parametric regime detection — shown to outperform HMM in some settings | Gate 1 passes |
| Synthetic data generation | Diffusion models / GANs calibrated to financial stylised facts — generate thousands of synthetic paths with known regime labels to improve statistical quality of HSMM validation | Gate 1 passes |
| Publication: transition velocity paper | Gate 2 result (positive or negative) is publishable. Clean hypothesis, novel combination, walk-forward validated. Target: SSRN working paper then journal submission | Gate 2 complete |
| Publication: open systems + critical slowing down paper | Endogenous/exogenous crash classification combined with HSMM precursor signals — gap in existing literature | Gate 1 complete + critical slowing down validated |

---

### Parked

| Idea | Reason parked |
|---|---|
| Fully autonomous execution | No published evidence of reliable performance; every failure mode traces here. Revisit never. |
| HFT / ultra-high-frequency strategies | Infrastructure costs and competition make it inaccessible at this scale. |
| RL trading agents | Academically interesting, practically fragile. High implementation cost, unclear live performance. |
| Naming unsupervised regime clusters | Unnecessary — states 1..k with empirical descriptions are sufficient. LLM naming adds hallucination risk. |
| LLM as primary alpha generator | LLMs predict markets poorly on their own. Value is in research acceleration, not direct signal generation. |

---

## What Was Discarded vs. Kept vs. Improved

| Item | Decision | Reason |
|---|---|---|
| Fully autonomous execution | ❌ Parked permanently | No evidence this works reliably |
| HFT / ultra-high-frequency | ❌ Parked permanently | Inaccessible at this scale |
| RL trading agents | ❌ Parked | Fragile in practice |
| Predefined bull/bear/sideways labels | 🔧 Replaced | Five empirical states from HSMM are richer and data-fitted |
| Regime naming via LLM | ❌ Parked | States described by statistics only — naming adds no value |
| Agentic research pipeline | ✅ Kept | Validated by RD-Agent(Q) at institutional level |
| Walk-forward OOS validation | ✅ Kept | Non-negotiable — primary defence against overfitting |
| Human-in-the-loop design | ✅ Kept | Every success story retains human oversight |
| Regime-aware factor timing | ✅ Kept | Universally adopted by serious funds |
| Five-state HSMM | ✅ Added | Peer-reviewed empirical validation; outperforms 3-state HMM |
| Advisor Strategy (Sonnet + Opus) | ✅ Added | Formalises tiered LLM architecture; +2.7pp quality, −11.9% cost vs Sonnet alone |
| Transition velocity signal | 📋 Backlog | Novel contribution — awaiting Gate 1 validation |
| Transaction cost model | ✅ Added | Non-optional part of factor harness from Phase 0 — IC after costs is the real metric |
| Independent market validation | ✅ Added as principle | Every result tested on US equities must be replicated on one independent market |
| Paper trading gate | ✅ Added as mandatory | Gate 1 must run 3+ months on Alpaca before Gate 2 is unlocked |
| Regime-conditional portfolio construction | ✅ Added as Phase 2.5 | Extends existing risk parity work; tightly coupled to regime detection |
| Synthetic data generation | 📋 Backlog | Improves statistical quality of HSMM validation; useful once Gate 1 is established |
| Publication: transition velocity | 📋 Backlog | Gate 2 result is publishable either way; SSRN → journal pathway |
| Publication: open systems / critical slowing down | 📋 Backlog | Genuine gap in literature; pursue after Gate 1 + critical slowing down validated |
| Critical slowing down signal | 📋 Backlog | Early warning of regime transitions from prices alone; awaiting Gate 1 |
| Latent factor recovery (RP-PCA/IPCA) | 📋 Backlog | Recover hidden drivers invisible to standard PCA; awaiting Gate 1 |
| Endogenous/exogenous crash classification | 📋 Backlog | Different crash types need different responses; awaiting Gate 1 |
| All Weather backtesting engine | 🔧 Extended | Needs cross-sectional layer + IC metrics |
| Build pipeline from scratch | 🔧 Modified | Factor harness from scratch (learning), Qlib+RD-Agent for production |
| Reflexivity as alpha source | 🔧 Repositioned | Risk overlay only; full build awaiting 13F data access |
| Persistent knowledge base | 🔧 Elevated | Baked in from day one |

---

## Tooling Stack

A small, well-chosen set of specialised tools. Each does one job without trying to do everything. Don't use one tool for everything — use the right tool for each category.

### Category 1: Code Editor + AI Coding Assistant

| Tool | Role | Cost | Status |
|---|---|---|---|
| **Cursor** | Primary IDE — AI-native editor with multi-file agentic coding, full codebase context | $20/month | ⬜ Install |
| **Claude Code** | Terminal-native agent for complex multi-file tasks, whole-codebase understanding | Included with Claude Pro | ⬜ Install |
| **VS Code Remote SSH** | Connect to Portugal PC from M1 — edit remote files as if local | Free | ⬜ Configure |
| **JupyterLab** | Exploratory research, factor visualisation, interactive prototyping | Free | ⬜ Install |

**Setup notes:**
- Cursor is a VS Code fork — import existing VS Code settings directly, extensions mostly transfer
- Create a `.cursorrules` file in each project root: include preferred libraries (`hmmlearn`, `scipy`, `pandas`, `yfinance`), coding conventions, and project structure. This dramatically improves suggestion quality and is underused
- If Cursor $20/month feels premature before income is stable: **Windsurf** has the most generous free tier of any AI-native IDE and is a valid bridge
- JupyterLab runs locally on M1 for light work; route heavier notebooks to Portugal PC over SSH

---

### Category 2: Knowledge Management

| Tool | Role | Cost | Status |
|---|---|---|---|
| **Obsidian** | Research notes, hypothesis tracking, knowledge graph, project management | Free (commercial licence removed in 2025) | ⬜ Install |
| **Zotero** | Paper library, PDF annotation, bibliography generation | Free | ⬜ Install |

**Setup notes:**
- Obsidian stores everything as plain Markdown files — Git-trackable, portable forever, readable by any editor. Critical for research that may feed into publications
- Install the **Zotero Integration** plugin in Obsidian: reads PDF annotations from Zotero directly into linked Obsidian notes. This is the core academic workflow — read and annotate in Zotero, think and connect in Obsidian
- Install the **Obsidian Tasks** plugin: turns Markdown checkboxes into a queryable task system. The backlog in this document becomes a live task list
- Install the **Dataview** plugin in Obsidian: query your notes like a database — useful for tracking which hypotheses have been tested, which papers have been read, etc.
- Install the **Zotero Browser Connector**: one-click save any paper from arXiv, SSRN, or any journal page directly into Zotero with full metadata
- Add papers to Zotero immediately as you encounter them — the reading list in this document should be the Zotero library

**Zotero reading list to add immediately:**
- Zakamulin (2023) — five-state HSMM
- Zakamulin & Giner (2024) — semi-Markov optimal trend-following
- Filardo (1994) — time-varying transition probabilities
- Lettau & Pelger (2020) — RP-PCA
- Kelly, Pruitt & Su (2019) — IPCA
- Horvath, Issa & Muguruza (2024) — Wasserstein k-means
- Cortese, Kolm & Lindström (2024) — statistical jump models
- Scheffer et al. (2009) — critical slowing down (*Nature*)
- RD-Agent-Quant (NeurIPS 2025)
- Alpha-R1 (Dec 2025, arXiv)
- Mantegna & Stanley — *Introduction to Econophysics* (book)

---

### Category 3: Experiment Tracking + Reproducibility

| Tool | Role | Cost | Status |
|---|---|---|---|
| **MLflow** (self-hosted) | Log every experiment run: parameters, metrics, data version, code version, results | Free | ⬜ Install on Portugal PC |
| **DVC** | Version large datasets alongside Git — market data, feature matrices | Free | ⬜ Install |
| **Git** | Code versioning — already in use | Free | ✅ Active |

**Setup notes:**
- Run MLflow tracking server on Portugal PC — it has the CPU for it and is always on. M1 logs experiments to it over Tailscale. Setup takes ~30 minutes
- Every factor backtest, every HSMM fit, every pipeline run logs to MLflow: parameters, IC/ICIR/Sharpe, data hash, git commit. This is what makes research reproducible for publication and personal sanity
- DVC sits alongside Git: Git tracks code and lightweight `.dvc` pointer files, DVC tracks actual datasets (too large for Git) in a separate remote (local folder or Portugal PC)
- The combo: DVC ensures data + pipeline determinism, MLflow preserves experiment history. Together they give complete reproducibility
- **This should be set up in Phase 0**, not later. The cost is near-zero, the benefit compounds

---

### Category 4: Task Management

Obsidian handles this via the Tasks plugin. The backlog in this document is the single source of truth. No separate project management tool needed.

---

### Total Cost Summary

| Tool | Monthly cost |
|---|---|
| Cursor | $20 |
| Claude Pro (includes Claude Code) | $20 |
| Everything else | £0 |
| **Total** | **~$40/month** |

All data remains local or on your own hardware. Nothing proprietary is stored in third-party clouds.

---

## Claude Skills

### What They Are

Agent Skills are reusable, filesystem-based resources — a folder with a `SKILL.md` file — that give Claude Code domain-specific expertise: workflows, conventions, and best practices that load automatically when relevant. Unlike a system prompt pasted at the start of each session, a Skill persists across all sessions and is version-controlled alongside the code it supports.

Technically: a Skill equals a structured prompt template + conversation context injection + optional Python scripts and reference documents. Claude Code discovers and loads them automatically from `~/.claude/skills/` or from a project-local `.claude/skills/` folder.

Two types exist. **Anthropic's pre-built Skills** (DCF models, earnings reports, comparable company analyses) target institutional investment banking — not relevant here. **Custom Agent Skills** are what matter. A broad ecosystem exists for quant finance (agiprolabs, JoelLewis, tradermonty repositories), but most of it solves different problems — technical indicator backtesting, DeFi analytics, equity research. The right approach for this project is one custom Skill built specifically for it.

---

### The One Skill Worth Building: `qframe-research`

Rather than installing community skills, build a single custom Skill that encodes the conventions of this project. Every Claude Code session in the `qframe` root then automatically inherits the full context without repeating it.

**What it encodes:** everything Claude Code needs to know that would otherwise have to be re-stated each session — the factor harness API, the knowledge base schema, the gate thresholds, the validation criteria, and the research loop conventions.

**Project structure:**

```
qframe/                               ← project root
├── .claude/
│   └── skills/
│       └── qframe-research/
│           ├── SKILL.md                      ← Core — loaded automatically
│           ├── references/
│           │   ├── factor-harness-api.md     ← Factor harness conventions + API
│           │   ├── knowledge-base-schema.md  ← SQLite schema + logging format
│           │   ├── gate-thresholds.md        ← IC/ICIR/Sharpe gates, validation criteria
│           │   └── backlog-format.md         ← How hypotheses and results are logged
│           └── scripts/
│               └── check_environment.py      ← Verify dependencies before a session
├── src/
│   └── qframe/
│       ├── factor_harness/
│       ├── regime/
│       ├── pipeline/
│       └── data/
├── experiments/                      ← MLflow-tracked runs
├── data/                             ← DVC-tracked
├── notebooks/
├── tests/
└── knowledge_base/                   ← SQLite knowledge base
```

**What goes in `SKILL.md`:**

```yaml
---
name: qframe-research
description: >
  Quantitative factor research pipeline for the qframe project.
  Load when working on factor construction, HSMM regime detection,
  backtesting, knowledge base logging, or the agentic research loop.
allowed-tools: ["Bash", "Read", "Write", "Edit", "Python"]
---
```

Followed by the project conventions that would otherwise be pasted at the start of every session:

- Project philosophy: gates must pass before next gate; every result logged; walk-forward only
- Factor harness conventions: cross-sectional framework, IC/ICIR/Sharpe as primary metrics, what "valid output" looks like
- Knowledge base schema: table names, field names, what gets written after each experiment
- Gate definitions and the numeric thresholds that unlock each phase
- Coding conventions: preferred libraries (`hmmlearn`, `scipy`, `pandas`, `openbb`), file structure, test patterns
- What to do when a backtest fails vs. when it succeeds

References loaded on demand from `/references/` keep the core file lean. The gate-thresholds reference should include worked examples of IC calculation, not just threshold values — following the JoelLewis pattern of bundling runnable Python alongside documentation.

**What this replaces:** without the skill, every Claude Code session starts cold. With it, Claude Code opens in `qframe/`, loads the skill automatically, and already knows the factor harness API, that IC must be computed after transaction costs, that every result gets logged to SQLite with a git hash, and what Gate 1 pass criteria look like. The work starts immediately.

**When to build it:** Phase 0, alongside the factor harness rebuild. Write the SKILL.md as conventions are being established — not retrofitted later. It is documentation that executes.

---

### Most Impressive Community Patterns (for reference, not adoption)

**Backtest stagnation detection** (tradermonty/claude-trading-skills): A skill that monitors when a backtesting loop hits a local optimum and automatically proposes structurally different strategy pivots using four deterministic triggers (improvement plateau, overfitting proxy, cost defeat, tail risk) and three pivot types (assumption inversion, archetype switch, objective reframe). This is the kind of meta-awareness the agentic pipeline needs — detecting when to stop iterating on parameters and propose a different strategy class entirely. Worth studying; worth adapting if the agentic loop shows stagnation.

**Skill chaining for research pipelines** (agiprolabs): The architecture where skills call other skills — data fetch → indicator compute → backtest → portfolio analytics — mirrors the five-agent RD-Agent loop exactly. The pipeline is a chain of composable, context-aware steps, not a monolithic script. The structural insight is directly applicable even if the implementation is not.

---

### Decision Table

| Item | Decision | Reason |
|---|---|---|
| Anthropic Financial Services Skills (DCF, comps, coverage) | ❌ Skip | Enterprise investment banking product — wrong domain |
| Community quant skill collections (62-skill, 81-skill packs) | ❌ Skip | Technical indicator / discretionary analysis paradigm — noise, not signal |
| Custom `qframe-research` skill | ✅ Build in Phase 0 | Encodes project conventions once; eliminates per-session context overhead |
| Backlog stagnation-detection pattern (tradermonty) | 📋 Study | Directly relevant to agentic loop; adapt if stagnation observed |
| Skill chaining pattern (agiprolabs) | 📋 Study | Structural insight matches RD-Agent loop architecture |

---

## Gaps Identified: What Physics Focus Caused Us to Miss

A systematic audit of the entire framework against the broader quant literature revealed six areas that were underweighted because no physics analogy pointed toward them. Each is practically important.

### Gap 1 — Transaction Costs and Alpha Decay Rate ⚠️ Critical

**This is the most dangerous gap.** A strategy that looks excellent in backtest can be completely unprofitable after transaction costs. Every signal has a decay rate — the half-life over which its predictive power diminishes. If the signal decays faster than you can execute, you pay costs to capture alpha that has already evaporated.

**What needs to be fixed:**

Every factor in the factor library needs an estimated IC decay curve. The optimal trading speed is approximately $\lambda/(2c)$ where $\lambda$ is the signal decay rate and $c$ is the per-unit transaction cost. Faster-decaying signals require faster (and costlier) trading; this must be in the objective function from day one, not bolted on later.

**Smart rebalancing rule:** Only rebalance when the expected IC gain from the current signal exceeds the round-trip transaction cost. Calendar-based rebalancing is lazy and destroys alpha in high-turnover factors. This applies to all phases.

Known decay rates for reference:
- Momentum: fast decay (~6 months), mean-reverting after ~2 years
- Value: slow decay, builds over 12+ months
- Quality: slow and persistent
- Low-vol: moderate, regime-dependent

**Action:** Factor harness rebuild must include IC decay measurement and net-of-cost IC as a primary evaluation metric alongside gross IC.

---

### Gap 2 — Options Market as Information Source

The options market contains forward-looking information not present in price data: expected volatility, tail risk, and informed trader positioning.

**Three implementable signals:**

*Realised–Implied Volatility Spread (RVol-IVol):* The gap between historical realised volatility and implied volatility captures the variance risk premium. Stocks where implied volatility substantially exceeds realised volatility tend to underperform — the market is pricing in too much risk. Well-documented, reproducible, actionable.

*Put-Call Implied Volatility Spread:* Expensive puts relative to calls at the same strike proxy for informed short-selling. When sophisticated investors are buying downside protection, this is a bearish signal with lead time of several days.

*Implied Volatility Surface Shape:* The slope across strikes (skew) and across maturities (term structure) contains regime information. High skew signals crash risk; inverted term structure signals near-term stress. These are leading indicators for the HSMM regime label.

**Why this matters for the regime model:** Options signals can tell you what regime the market is pricing *before* that regime appears in returns. This makes them natural leading indicators for the HSMM — inputs to the semantic layer that improve transition detection.

**Practical start:** Yahoo Finance provides basic options chain data for free — sufficient to compute RVol-IVol spread and put-call skew as regime inputs. No OptionMetrics subscription needed to get started.

---

### Gap 3 — Earnings Revision and Surprise Factors

The factor library (momentum, value, carry, quality, low-vol) omits **earnings revision** and **earnings surprise drift** — consistently among the strongest factors in the academic literature, distinctly uncorrelated with price-based factors, and genuinely less crowded.

**Earnings revision (analyst estimate changes):** Persistent at 3–6 month horizons. Captures information asymmetry between analysts and the market as analysts update their models. Strong economic rationale: analysts revise when they receive new private information from management channels.

**Post-earnings announcement drift (PEAD):** Stocks continue to drift in the direction of an earnings surprise for 60+ days. One of the most replicated anomalies. The mechanism is behavioural: investors underreact to earnings news and prices take time to fully reflect it.

**Regime interaction:** These factors behave very differently across HSMM states. In a low-vol bull market, revision momentum is strong. In a crash/correction, it collapses because all correlations go to one. This regime-conditional variation is exactly what the Phase 2 framework is designed to capture.

**Data:** Quarterly earnings data available via OpenBB (Financial Modeling Prep provider, free tier). Analyst estimates available via Estimize (integrated in ExtractAlpha, some free data) or Yahoo Finance.

---

### Gap 4 — Portfolio Construction Layer

The framework discusses alpha generation extensively but the **transformation of signals into portfolio weights** is barely addressed. This is where enormous value is either captured or destroyed.

**Missing elements that need explicit design:**

*Factor exposure control:* Naively overweighting a factor in a favourable regime accumulates unintended exposures to correlated factors. The portfolio construction layer must neutralise unwanted exposures while preserving the intended signal.

*Turnover budgeting:* Enforce turnover limits consistent with Gap 1 — the transaction cost structure of each factor should determine the maximum rebalancing frequency.

*Regime-conditional covariance:* Already partially addressed in Phase 2.5 (risk parity with state-conditional covariance). This is the right approach and needs to be explicitly connected to the factor allocation.

*Grinold-Kahn law:* The information ratio equals IC × √(breadth). For the strategy: are more weak signals (high breadth) or fewer strong ones better? The answer depends on factor correlations across regimes — compute this explicitly.

**Connection to All Weather work:** The risk parity optimisation already built (SLSQP, covariance-based weights) is the foundation. Repurpose it for factor weights within each regime rather than asset class weights across regimes. This is the concrete bridge between existing work and the new framework.

---

### Gap 5 — Microstructure as Daily Aggregated Signal

Order flow imbalance (OFI) was discussed for the crypto phase but not integrated into the equity factor framework. For daily-frequency strategies, the relevant signal is **daily aggregated OFI** — the net direction of institutional trades over a full day — which persists for several days.

The regime connection: OFI's memory and forecasting power is regime-dependent. In trending regimes (H > 0.5), institutional flow autocorrelation is high and OFI is predictive. In mean-reverting regimes (H < 0.5), it is near-random. OFI therefore serves as a latent-variable proxy that reveals regime information before it appears in returns.

**Practical implementation:** Lee-Ready algorithm on CRSP data (expensive) or approximation using TAQ quote data. For crypto, directly available from exchange APIs. Add as a feature to the regime detection input alongside price and volume data in Phase 3.

---

### Gap 6 — Causal Mechanism Scoring

Many discovered factors are spurious — they correlate with a true causal factor in historical data but decay rapidly out-of-sample when the correlational relationship breaks. The physics training instinct to ask "what is the mechanism?" is the right instinct to apply here.

**Action:** Add a `mechanism_score` field to the knowledge base for every factor hypothesis (1–5 scale, qualitative). Factors with strong causal stories — momentum from investor underreaction, value from fundamental mean reversion, crowding-adjusted factors from liquidity risk — get higher weight when regime data is limited. Factors discovered purely through data mining get lower weight regardless of backtest IC. This is not about ranking hypotheses before testing them — it's about how much evidence is required before accepting a result as real.

---

### What Is NOT Missing

To be complete: the following areas were covered and do not represent gaps.
- Regime detection: HSMM + Hurst + Wasserstein
- LLM-based signals: semantic regime labelling (Phase 2) + cross-lingual (Phase 4)
- Crypto alpha: on-chain signals (Phase 3)
- Alternative data: covered in initial research survey
- Crowding/reflexivity: covered as risk overlay
- Walk-forward validation: non-negotiable from day one
- Survivorship bias: addressed explicitly in data infrastructure

---

## Regime Dynamics: Velocity Metrics and the Hurst Exponent

### Multiple Velocity Metrics for Discrete Systems

The regime posterior $\pi_t \in \Delta^{K-1}$ is a probability vector observed discretely. There is no unique "velocity" in a discrete system — different definitions capture different aspects of regime dynamics and should be tested as separate features.

**Five candidates, each measuring something genuinely different:**

| Metric | Formula | What it captures |
|---|---|---|
| First-order finite difference | $v_t^{(1)} = \pi_t - \pi_{t-1}$ | Raw change in probability mass |
| Second-order (acceleration) | $a_t = \pi_t - 2\pi_{t-1} + \pi_{t-2}$ | Change in the change — the "second derivative" intuition |
| Exponentially weighted | $v_t^{(\alpha)} = (1-\alpha)\sum_s \alpha^s(\pi_{t-s} - \pi_{t-s-1})$ | Smooth version, reduces noise at cost of lag |
| Geodesic velocity (Fisher metric) | $v_t^{\text{geo}} = \|\log(\pi_t/\pi_{t-1})\|_{\pi_{t-1}}$ | Geometrically correct for probability simplex — small changes in already-small probabilities count more |
| KL divergence | $v_t^{\text{KL}} = D_{KL}(\pi_t \| \pi_{t-1})$ | Scalar surprise measure — how much information was gained by updating the belief |

**Economic interpretation of the KL metric:** Zero means the posterior didn't change. A value of 0.1 nats means a meaningful belief update. A value of 0.5+ nats means a sharp regime signal arrived. This is directly interpretable as "how much did we learn about the regime today?"

**The geodesic metric is the most principled** for a probability simplex: it weights changes in small probabilities more than changes in large ones, which is economically sensible. A posterior moving from 0.01 to 0.03 in the crash state is more significant than moving from 0.50 to 0.52 in the bull state, even though the absolute change is the same.

**Practical approach:** Include all five as separate features in the factor allocation function. Run IC analysis on each. They will be partially orthogonal — some may add independent information — and the data should determine which combination matters.

---

### The Hurst Exponent: Definition, Phase Transitions, and Integration

#### Mathematical Definition

The Hurst exponent $H \in (0,1)$ characterises the long-range dependence structure of a time series. The rescaled range (R/S) estimator:

$$\mathbb{E}\left[\frac{R(n)}{S(n)}\right] \sim cn^H$$

where $R(n)$ is the range of cumulative deviations from the mean over $n$ observations, and $S(n)$ is the standard deviation. More robust: the **Detrended Fluctuation Analysis (DFA)** estimator:

$$F(n) = \sqrt{\frac{1}{N}\sum_{k=1}^{N}[X_k - \hat{X}_k(n)]^2} \sim n^H$$

where $\hat{X}_k(n)$ is a local polynomial fit over a window of size $n$.

**Interpretation:**
- $H = 0.5$: random walk, no long-range dependence, no predictability from past
- $H > 0.5$: persistent, trending — past large moves predict future large moves of the same sign
- $H < 0.5$: anti-persistent, mean-reverting — past large moves predict reversals

#### Connection to Phase Transitions

This is a genuine and deep connection, not a superficial analogy. In statistical physics, phase transitions occur when a control parameter crosses a critical threshold. Near criticality, the system exhibits scale-free behaviour — correlations diverge, fluctuations appear at all scales, the system becomes self-similar.

$H$ is precisely a critical exponent of this type. The boundary $H = 0.5$ is the critical point. Above it (ordered, persistent phase), momentum strategies dominate. Below it (anti-ordered, mean-reverting phase), contrarian and value strategies dominate.

| Physics analogy | Market equivalent |
|---|---|
| Disordered phase ($H = 0.5$) | Efficient market, random walk, no predictability |
| Ordered phase ($H > 0.5$) | Trending regime, herding, momentum works |
| Anti-ordered phase ($H < 0.5$) | Mean-reverting, stat arb works, contrarian profitable |
| Critical point ($H \approx 0.5$) | Maximum uncertainty, regime boundary, factor strategies weakest |

Markets tend toward the critical point (self-organised criticality) — which is why trading is hard. But **departures from the critical point are where factor alpha lives**, and the Hurst exponent measures those departures directly.

#### Why It Should Have Been Included From the Beginning

Three reasons it deserves early inclusion rather than later:

**It captures temporal structure the HSMM misses.** The HSMM classifies the distributional state — what does the return distribution look like now? The Hurst exponent captures the *scaling structure across time* — how does the market behave across different horizons? A high-volatility bull market (HSMM state) can be either trending ($H > 0.5$, momentum works) or choppy ($H < 0.5$, momentum fails). The HSMM alone cannot distinguish these.

**It directly implies which factor to deploy.** The Hurst exponent is not just a regime label — it tells you which strategy paradigm works. This is a step closer to factor allocation decisions than the HSMM posterior alone.

**It is computationally trivial.** Rolling DFA on 250-day windows runs in milliseconds and needs no model fitting, no EM algorithm, no hyperparameters. It can be computed daily alongside the HSMM from day one.

#### Practical Implementation

Add rolling Hurst exponent (DFA method, 250-day window) as a sixth input to the factor allocation function:

$$\mathbf{w}_t^{\text{factors}} = g\!\left(\hat{\pi}_t,\; \dot{\hat{\pi}}_t,\; H_t,\; \text{regime}_t,\; H_t^{\text{crypto}}\right)$$

**Factor routing logic based on $H_t$:**
- $H_t > 0.6$: overweight momentum, underweight mean-reversion factors
- $H_t < 0.4$: overweight value and quality, underweight momentum
- $H_t \in [0.4, 0.6]$: near-critical zone, reduce overall factor exposure, prioritise quality and low-vol

Python implementation: `nolds` library provides DFA estimator; alternatively implement from scratch as a learning exercise. Fast enough for daily updates on any machine.

---

## Claude Code Organisation

### The Core Problem Claude Code Solves (and Its Failure Mode)

Claude Code loses all context between sessions. Without a deliberate system, every session starts cold — you re-explain conventions, re-describe the architecture, re-state what you were working on. The community has converged on a solution: a layered context system where Claude reads only what it needs, when it needs it.

**The primary failure mode is over-stuffing CLAUDE.md.** Frontier models can follow approximately 150–200 instructions with reasonable consistency. Past that, instructions degrade. Context degradation — not lack of capability — is why most Claude Code setups underperform.

---

### The Three-Layer System

**Layer 1 — CLAUDE.md (the permanent brain, always loaded)**

Placed at the project root. Short — under 150 lines. Contains only what is universally true for every session: the directory map, the Python environment command, the non-negotiables (walk-forward only, every result logged to SQLite), and a list of agent_docs with descriptions telling Claude when to read each one.

Does NOT contain: factor definitions, backtest thresholds, schema details, coding conventions. Those live in agent_docs.

**Layer 2 — agent_docs/ (the library, loaded on demand)**

A folder of markdown files, each with a self-descriptive name. Claude reads these when the task requires them — not on every session start. This keeps the active context lean.

**Layer 3 — .claude/skills/qframe-research/ (domain knowledge, auto-activated)**

A skill directory with a SKILL.md that Claude loads automatically when working on factors, regimes, or backtesting. Can bundle Python helper scripts and reference files. Only the YAML frontmatter description (~100 tokens) loads at startup; the full SKILL.md loads when the task matches.

---

### Recommended Directory Structure

```
qframe/
│
├── CLAUDE.md                          ← SHORT (<150 lines). Always loaded.
├── CLAUDE.local.md                    ← Personal/machine overrides. Gitignored.
│                                        (Python env path, Portugal PC alias, etc.)
│
├── .claude/
│   ├── skills/
│   │   └── qframe-research/
│   │       ├── SKILL.md              ← Auto-loads when working on factors/regimes
│   │       └── references/
│   │           ├── factor-harness-api.md
│   │           ├── knowledge-base-schema.md
│   │           ├── gate-thresholds.md
│   │           └── validation-criteria.md
│   └── commands/                     ← Custom slash commands
│       ├── new-hypothesis.md         ← /new-hypothesis
│       ├── run-backtest.md           ← /run-backtest
│       └── gate-check.md            ← /gate-check
│
├── agent_docs/                       ← Read on demand, not on every session start
│   ├── project-overview.md           ← What qframe is and why
│   ├── factor-library.md             ← Current factor definitions and IC status
│   ├── regime-model.md               ← HSMM architecture, five states, velocity metrics
│   ├── data-sources.md               ← OpenBB providers, Norgate, yfinance notes
│   ├── coding-conventions.md         ← Preferred libs, style, test patterns
│   └── research-log.md              ← WHERE WE LEFT OFF. Update every session end.
│
├── src/
│   └── qframe/
│       ├── factor_harness/
│       ├── regime/
│       ├── pipeline/
│       └── data/
│
├── data/
│   ├── raw/                         ← Immutable. DVC-tracked. Never edit.
│   ├── processed/
│   └── intermediate/
│
├── experiments/                     ← MLflow-tracked runs
│   └── YYYY-MM-DD_description/
│
├── notebooks/
│   ├── exploratory/
│   └── final/
│
├── tests/
├── knowledge_base/                  ← SQLite
└── docs/                           ← Architecture decisions, reading notes
```

---

### The research-log.md Pattern (Most Important Habit)

`agent_docs/research-log.md` is the session continuity mechanism. At the end of every working session, update it with one paragraph: what was done, what is next, what is broken or uncertain. Claude Code reads this at the start of the next session and picks up without you re-explaining.

Without this file, every session starts cold. With it, continuity is maintained across days of interrupted work.

---

### What CLAUDE.md Should Contain

```markdown
# qframe — Quantitative Research Pipeline

## What this project is
Regime-aware multi-factor research pipeline. Goal: walk-forward-validated alpha
on US equities with extension to crypto. Human-in-the-loop always.

## Environment
conda activate qframe   # always run this before executing any Python

## Directory map
- src/qframe/           — all Python source code
- data/raw/             — immutable raw data (DVC-tracked, never edit directly)
- experiments/          — MLflow-tracked backtest runs
- knowledge_base/       — SQLite: hypotheses, implementations, results

## Non-negotiables
- Walk-forward validation only — no in-sample optimisation
- Every backtest result logged to SQLite before session ends
- Data access via OpenBB wrappers in src/qframe/data/ only
- Net-of-cost IC is the primary metric, not gross IC

## Agent docs (read when relevant, not by default)
- agent_docs/research-log.md        — WHERE WE LEFT OFF. Read this first.
- agent_docs/factor-library.md      — current factor definitions and IC status
- agent_docs/regime-model.md        — HSMM architecture and gate thresholds
- agent_docs/coding-conventions.md  — preferred libraries and style
- agent_docs/data-sources.md        — data provider notes

## Active skill
Load qframe-research skill when working on factors, regime detection,
backtesting, or knowledge base logging.
```

---

### Obsidian's Role (Separate From Claude Code)

Obsidian lives **outside the qframe repo** as a personal knowledge vault:
- Paper library with Zotero integration (all papers from the reading list)
- Reading notes on Zakamulin, AQR papers, HMM literature
- Ideas backlog and personal research journal
- Anything written for yourself, not for Claude

It does NOT belong inside the Claude Code context system. The rule: Obsidian owns personal knowledge, agent_docs/ owns project context that Claude needs to do its job. Never duplicate the same content in both.

---

### Key Operational Rules

**Do:**
- Run `/compact` at roughly 50% context fill. Run `/clear` when switching tasks entirely.
- Update `agent_docs/research-log.md` at the end of every session — this is the continuity mechanism.
- Build `qframe-research` SKILL.md alongside the factor harness as conventions emerge — not retrofitted later.
- Use `CLAUDE.local.md` for machine-specific paths (Python env, Portugal PC SSH alias) — gitignored.
- Add a sub-directory CLAUDE.md inside `src/qframe/regime/` for HSMM-specific conventions — loaded only when Claude works in that directory.

**Don't:**
- Use `@path/to/file.md` imports that embed entire files on every run — defeats Progressive Disclosure.
- Put more than ~150 lines in CLAUDE.md — context degradation is the primary failure mode.
- Duplicate content between Obsidian and agent_docs/ — pick one owner.
- Auto-generate CLAUDE.md with `/init` and leave it unedited — use `/init` as a starting draft only. CLAUDE.md is the highest-leverage file; write it intentionally.

---

## Immediate Next Steps

**Tooling setup (Phase 0 — do these first):**
- [ ] **Decide on project name** — `qframe` recommended; finalise before creating the repo
- [ ] Create `qframe/` project root with directory structure above
- [ ] Install Cursor — import VS Code settings, create `.cursorrules` for the project
- [ ] Install Obsidian — create vault, install Zotero Integration + Tasks + Dataview plugins
- [ ] Install Zotero — install browser connector, add all papers from reading list above
- [ ] Install DVC — initialise in project repo alongside Git
- [ ] Set up MLflow on Portugal PC — tracking server, test connection from M1 over Tailscale
- [ ] Install JupyterLab on M1
- [ ] **Write `qframe-research` SKILL.md** — draft as factor harness conventions are established; build `/references/` files alongside the code they describe

**Infrastructure setup:**
- [ ] Set up Tailscale on Portugal PC
- [ ] Install Tapo P110 smart plug in Portugal
- [ ] Install Ollama on M1, pull Qwen2.5-Coder:7B
- [ ] Configure VS Code Remote SSH for Portugal PC connection

**Research and coding:**
- [ ] Clone and study Qlib + RD-Agent repositories
- [ ] Rebuild factor harness from scratch (cross-sectional, IC-based, **with transaction cost model**) — *this is the active coding task*
- [ ] Set up SQLite knowledge base schema
- [ ] Get Anthropic API key + Google AI Studio key
- [ ] **Check Advisor Strategy beta access** — confirm whether available on Pro plan; if yes, implement in Phase 1 from the start
- [ ] Install OpenBB — set this up as the data abstraction layer before writing any data ingestion code
- [ ] Read RD-Agent-Quant NeurIPS 2025 paper (add to Zotero first)
- [ ] Read Zakamulin (2023) — first paper for Gate 1 (add to Zotero first)
- [ ] Read Lettau & Pelger (2020) — RP-PCA, latent factor recovery
- [ ] Check PhD alumni status for CRSP/WRDS access — contact university library
- [ ] Sign up for Glassnode free tier (for Phase 3 groundwork)

---

*Last updated: April 2026 — Added gaps audit; velocity metrics; Hurst exponent; Claude Code organisation system (CLAUDE.md hierarchy, agent_docs/, skills, research-log pattern, Obsidian role clarification)*
*This is a living document — update after each major decision or direction change*
