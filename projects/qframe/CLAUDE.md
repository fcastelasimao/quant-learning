# qframe — Quantitative Research Pipeline

## What this project is
Regime-aware multi-factor research pipeline. Goal: walk-forward-validated alpha
on US equities, with extension to crypto. Human-in-the-loop always — no
autonomous execution of trades or irreversible operations.

## Non-negotiable process rule
**Never deviate from `quant_ai_plan.md` without explicit user approval.** If an implementation decision differs from the plan (model choice, parameter, architecture, number of states, etc.), stop and ask before proceeding. Document any approved deviations in `agent_docs/research-log.md` with the rationale.

## Environment
```bash
conda activate qframe   # always run before executing any Python
```

## Directory map
| Path | Purpose |
|---|---|
| `src/qframe/factor_harness/` | IC, ICIR, IC decay, costs, walk-forward harness |
| `src/qframe/pipeline/` | Agentic loop: models, executor, agents, loop |
| `src/qframe/pipeline/agents/` | synthesis.py, implementation.py (Ollama), analysis.py, _llm.py (Groq/Gemini router) |
| `src/qframe/data/` | Data loaders — all data access goes through here |
| `src/qframe/knowledge_base/` | SQLite interface (KnowledgeBase class) |
| `src/qframe/viz/` | Visualisation helpers — 22 chart functions; Charts 16–19 = regime; Charts 20–22 = Phase 2.5 portfolio (no plt.show() — returns Figure) |
| `src/qframe/config.py` | API key loading from .env |
| `src/qframe/regime/` | HSMM, Hurst exponent, velocity metrics (Phase 2) |
| `data/raw/` | Immutable raw data. DVC-tracked. Never edit directly. |
| `data/processed/` | Cleaned data ready for factor computation (parquet cache here) |
| `experiments/` | MLflow-tracked backtest runs |
| `knowledge_base/qframe.db` | SQLite: hypotheses, implementations, results, factor_correlations |
| `notebooks/` | gate0_momentum_smoke_test.ipynb, phase1_pipeline_demo.ipynb (Charts 1–15), phase2_regime_analysis.ipynb (Charts 16–19 + equity curve + batch regime loop), phase25_portfolio.ipynb (Phase 2.5 combined strategy + Gate 3) |
| `tests/` | Unit tests: test_ic.py, test_costs.py, test_knowledge_base.py, test_pipeline.py |
| `agent_docs/` | Context library — read when relevant (see below) |
| `.env` | API keys + LLM config (gitignored) — see `.env.example` for all keys |
| `run_pipeline.sh` | CLI runner: `./run_pipeline.sh --domain momentum --n 3` |

## Non-negotiables
- **Walk-forward only.** No in-sample optimisation. No look-ahead.
- **Net-of-cost IC is the primary metric**, not gross IC.
- **Every backtest result logged to SQLite** before the session ends.
- **Data access via `src/qframe/data/` only.** Never call yfinance or OpenBB directly in research code.
- **One active experiment at a time.** New ideas go to the backlog in the knowledge base.

## Agent docs (read when relevant — not on every session start)
Always read `agent_docs/research-log.md` first to understand where we left off.

| File | Read when... |
|---|---|
| `agent_docs/research-log.md` | **Every session — read this first** |
| `agent_docs/factor-library.md` | Working on factors, IC analysis, factor additions |
| `agent_docs/regime-model.md` | Working on HSMM, Hurst, velocity, regime detection |
| `agent_docs/gate-thresholds.md` | Evaluating whether a backtest passes a gate |
| `agent_docs/data-sources.md` | Data ingestion, provider questions, schema questions |
| `agent_docs/coding-conventions.md` | Writing new code, reviewing code style |

---

## Pipeline implementation notes (updated 2026-04-13)

### Implementation agent (Qwen2.5-coder:14b via Ollama)
- `scipy.stats` is available as `stats` in the execution namespace
- 10 explicit bug anti-patterns in `_PROMPT_TEMPLATE` with correct alternatives; covers groupby-on-DataFrame, ~float, operand alignment, IntCastingNaN, shape mismatch, ndarray attribute errors
- Self-healing retry: on fixable errors (listed in `_FIXABLE_ERRORS` in `loop.py`) the loop calls `implementation.fix(code, error)` for one automatic retry before logging failure
- `num_predict=768` (was 512)
- `fix()` method added to `ImplementationAgent` — sends error message + original code to model with targeted fix prompt

### LLM router (_llm.py) — updated 2026-04-14
- Fallback chain: `groq → cerebras → deepseek → together → mistral → gemini → openrouter`
- Each provider raises `QuotaExhaustedError` on daily quota exhaustion or missing key — caught by `generate()` which tries the next configured provider
- `openai>=1.30` SDK shared by all 5 new providers (lazy import, no install required to use Groq/Gemini)
- `_openai_compat_generate(prompt, *, base_url, api_key, model, provider_name, extra_headers)` — shared OpenAI-compatible backend
- Cerebras API key configured; model: `qwen-3-235b-a22b-instruct-2507` (Qwen3 235B/22B active MoE)
- Set `LLM_PROVIDER=cerebras` in .env to use Cerebras as primary
- Available: groq (llama-3.3-70b), cerebras (qwen-3-235b), deepseek (deepseek-chat), together (llama-3.3-70b), mistral (mistral-large-latest), gemini (gemini-2.0-flash), openrouter (llama-3.3-70b)

### Knowledge base — schema additions (2026-04-13)
- `hypotheses.factor_name TEXT` — short snake_case identifier stored alongside description
- `backtest_results.slow_icir_21 REAL` — ICIR computed on non-overlapping 21-day windows
- `backtest_results.slow_icir_63 REAL` — ICIR computed on non-overlapping 63-day windows
- `backtest_results.ic_decay_json TEXT` — full 63-point IC decay curve as JSON string
- All additions are migration-safe: `init_schema()` runs `ALTER TABLE ADD COLUMN` if missing

### Factor harness — metrics added (2026-04-13)
- `compute_slow_icir(factor_df, returns_df, horizon, oos_start)` in `ic.py` — honest ICIR for slow signals using non-overlapping windows; returns `np.nan` if insufficient periods
- `compute_ic_by_period(factor_df, returns_df, oos_start, period_years, horizon)` in `ic.py` — temporal stability diagnostic; splits OOS IC into consecutive 2-year blocks; returns DataFrame with mean_ic, std_ic, icir, t_stat, n_days per block
- IC decay curve now uses all 63 daily horizons (was 5 sparse points); stored as `ic_decay_json`
- `WalkForwardResult` now includes `slow_icir_21` and `slow_icir_63` fields
- `src/qframe/factor_harness/multiple_testing.py` (NEW) — BHY, Bonferroni, HLZ corrections; `correct_ic_pvalues(results, alpha, n_oos_days)` → corrected DataFrame; `print_correction_summary()` → formatted output

### Cost model — additions (2026-04-13)
- `CostParams` has two new fields: `short_borrow_bps_annual` (default 50) and `funding_cost_bps_annual` (default 0)
- `AGGRESSIVE_COST_PARAMS` added (20 bps spread, 50 bps impact, 150 bps borrow, 550 bps funding)
- `net_ic()` now models 3 cost components: trading + borrow + funding
- `compute_short_fraction(weights)` added — used internally by `net_ic()`
- `cost_summary(turnover_mean, horizon, ...)` added — returns human-readable cost breakdown
- Module docstring has full "NOT MODELLED" table and live deployment checklist

### Synthesis agent — literature seeds (2026-04-13)
- `_LITERATURE_SEEDS` dict with 5 canonical factors per domain (momentum, mean_reversion, volatility, quality, value)
- Synthesis prompt instructs the model to implement seeds first before exploring freely
- Recent archetypes shown explicitly to avoid repetition

### Agentic loop — additions (2026-04-13)
- `run_n()` now calls `run_correlation_analysis()` + `run_ensemble_check(top_n=3)` automatically every 5 total results
- `run_correlation_analysis()` — pairwise Spearman rank correlations between all positive-IC factors; logs to `factor_correlations` table
- `run_ensemble_check(top_n=3)` — IC-weights top N factors, backtests combined signal, logs as `ensemble_topN` hypothesis

### Visualisations (2026-04-13, extended 2026-04-17)
- `src/qframe/viz/charts.py` — 19 chart functions; all return `matplotlib.Figure`, never call `plt.show()`
- Chart 14: `plot_ic_by_period(kb_path, impl_id, prices_path, period_years, horizon)` — temporal stability bar chart (IC per 2-year block, error bars, ICIR annotation)
- Chart 15: `plot_multiple_testing(kb_path, alpha, n_oos_days)` — BHY/HLZ/Bonferroni significance chart with t-stat bars and threshold lines
- Chart 16: `plot_regime_timeline(proba_df, market_returns, oos_start, title)` — HSMM posterior stacked bands + market return
- Chart 17: `plot_regime_ic(by_state, unconditional_ic, factor_name, horizon)` — per-state IC bar chart with unconditional baseline
- Chart 18: `plot_velocity(velocity_raw, velocity_smooth, hard_labels, title)` — regime transition velocity (raw + smoothed)
- Chart 19: `plot_hurst_rolling(hurst_series, hard_labels, oos_start, title)` — rolling DFA Hurst exponent with regime shading
- Chart 20: `plot_combined_equity(equity_curves, benchmark, initial_investment, title)` — 2-panel: equity curves + primary drawdown (Phase 2.5)
- Chart 21: `plot_rolling_sharpe(ic_series, window, title, threshold)` — rolling 12-month Sharpe with Gate 3 threshold shading (Phase 2.5)
- Chart 22: `plot_blend_weights(blend_weights, hard_labels, title)` — stacked area chart of time-varying per-factor blend weights (Phase 2.5b); right-edge mean annotations; optional regime strip
- `notebooks/phase1_pipeline_demo.ipynb` Section A2 now has 15 charts + correction summary table; all key cells have explanatory markdown
- `notebooks/phase2_regime_analysis.ipynb` — Sections A–D with Charts 16–19 + equity curve (Section D); batch regime loop for ALL BHY-significant factors; regime results logged to KB
- `notebooks/phase25_portfolio.ipynb` — Phase 2.5 combined strategy (historical reference only — **Gate 3 REVOKED 2026-04-19**: driven by impl_82 look-ahead bias); Section D2 = Phase 2.5b dynamic posterior blend with Chart 22
- `notebooks/gate0_momentum_smoke_test.ipynb` — enriched with explanatory markdown on pipeline phases, metric definitions, and Gate 0 criteria

### Bug fixes and changes applied 2026-04-19
- **impl_82 look-ahead bias discovered and retired.** `pct_change(251).shift(-1)` uses tomorrow's price. Crypto replication (25 Binance USDT pairs, 2020–2024) confirmed IC = −0.005, t = −0.75. Phase 2.5 Gate 3 REVOKED. Phase 1 gate INVALIDATED.
- **Six new pipeline guards added:** look-ahead boundary check (`executor.py`), signal novelty filter ρ>0.70 (`loop.py`), pre-gate 2012–2016 (`walkforward.py`), BHY runtime gate with rolling m (`loop.py`), Deflated Sharpe Ratio (Bailey & López de Prado 2014, `multiple_testing.py`), per-stock ADV impact cost (`costs.py`).
- **New crypto data loader:** `src/qframe/data/crypto.py` — loads OHLCV from Binance via `ccxt`; caches to parquet; filters by `min_history_days`.
- **Test suite:** 137 → 177 passing.

### Bug fixes and changes applied 2026-04-18
- `knowledge_base/db.py`: `get_bhy_significant()` filters ensemble prefixes (`phase25_`, `ensemble_`, `combined_`) and deduplicates by `factor_name`; `log_regime_result()` upserts (DELETE+INSERT) instead of plain INSERT; `get_regime_results()` uses MAX(id) subquery to deduplicate
- `phase25_portfolio.ipynb`: fixed `az.fit()` missing `is_end`; fixed `ValueError: Invalid frequency: YE` (try YE, except A); suppressed `RuntimeWarning: Mean of empty slice` in `_rank_weights` calls; fixed silent `.mul()` alignment bug (MaxDD=0%) with explicit `columns.intersection()` pattern; fixed turnover display `:.0%` → correct `883%/yr`; Gate 3 turnover criterion replaced with cost-efficiency (net IC/gross IC ≥ 90%); Phase 2.5b (Section D2) added with Chart 22 (dynamic posterior blend weights)
- `multiple_testing.py`/correction: impl_53 slow t-stat = 1.31 (not 8.15) — fast-signal formula `icir × √(N/252)` overcounts observations for h=63d signals; correct formula: `slow_icir_63 × √(N/63)` with N/63 non-overlapping windows

### Bug fixes applied 2026-04-14
- `ic.py`: `fillna(0)` removed from forward return computation — NaN returns now correctly propagate; `min_periods=window` in ICIR warm-up; `min_stocks` param added to `compute_slow_icir`
- `multiple_testing.py`: `slow_icir_63=0.0` falsy check fixed (was using fast formula incorrectly); HLZ threshold floor = 3.0 for m=1
- `costs.py`: `CostParams.__post_init__` validates all parameters on construction
- `data/loader.py`: `UserWarning` when ffill fills NaN price gaps
- `pipeline/executor.py`: NaN threshold 95%→50%; constant cross-sectional variance check added
- `pipeline/models.py`: `pd.notna()` for NaN check; IndexError fixed for empty error traceback
- `pipeline/agents/implementation.py`: Ollama `timeout` kwarg now passed to both `generate()` and `fix()` calls
- `pipeline/agents/synthesis.py`: JSON retry on parse failure; mechanism_score clamped to [1,5]
- `viz/charts.py`: leaderboard deduplicates by factor_name (best IC per factor); slow-signal detection uses Option A (same-sign only: |IC@63|>|IC@1| AND same sign)

## Active skill
The `qframe-research` skill encodes project conventions, factor harness API,
and validation criteria. Load it manually with `/qframe-research` if needed.
