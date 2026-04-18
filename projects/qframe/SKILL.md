---
name: qframe-research
description: >
  Quantitative factor research conventions for the qframe project.
  Load when working on: factor construction, IC evaluation, HSMM regime
  detection, Hurst exponent, regime velocity metrics, walk-forward backtesting,
  knowledge base logging, or the agentic research loop. Trigger phrases:
  "factor", "IC", "ICIR", "backtest", "regime", "HSMM", "Hurst", "knowledge base",
  "gate", "alpha", "signal".
allowed-tools: ["Bash", "Read", "Write", "Edit"]
---

# qframe-research Skill

## Project Philosophy

- Walk-forward validation is non-negotiable. Every claim about factor performance must come from OOS data (2018 onwards).
- Net-of-cost IC is the primary metric. Gross IC is informational only.
- One active experiment at a time. New ideas go to the knowledge base backlog.
- Every backtest result gets logged to SQLite before the session ends — no exceptions.
- Every result must eventually be confirmed on a second market before being accepted as real.
- **Multiple testing correction is required when m ≥ 10 factors have been tested.** Use BHY. Never report raw IC gates as "significant" without running `correct_ic_pvalues()`.
- **Universe must be ≥ 500 stocks** before trusting any IC result. Results on 48-stock cache should be treated as exploratory only.

## Factor Harness API

The factor harness lives in `src/qframe/factor_harness/`.

```python
from qframe.factor_harness.ic import (
    compute_ic, compute_icir, compute_ic_decay,
    compute_slow_icir, compute_ic_by_period, estimate_ic_halflife,
)
from qframe.factor_harness.costs import CostParams, DEFAULT_COST_PARAMS, AGGRESSIVE_COST_PARAMS, cost_summary
from qframe.factor_harness.walkforward import WalkForwardValidator, WalkForwardResult

# Cross-sectional IC at a given forward horizon
ic_series = compute_ic(factor_df, returns_df, horizon=1, min_stocks=10)
# → pd.Series indexed by date; NaN during warmup

# Rolling ICIR
icir_series = compute_icir(ic_series, window=63)

# IC decay curve (all 63 horizons for smooth curve)
decay_df = compute_ic_decay(factor_df, returns_df, horizons=list(range(1, 64)))
# → pd.DataFrame with columns [horizon, mean_ic, icir, n_obs]

# Honest ICIR for slow signals (non-overlapping windows)
slow_icir = compute_slow_icir(factor_df, returns_df, horizon=63, oos_start='2018-01-01')
# → float; np.nan if insufficient data (<8 periods)

# Temporal stability diagnostic: IC in consecutive OOS sub-periods
period_df = compute_ic_by_period(
    factor_df, returns_df,
    oos_start='2018-01-01',
    period_years=2.0,   # ~2-year blocks
    horizon=1,
)
# → pd.DataFrame with columns: period_label, period_start, period_end,
#   mean_ic, std_ic, icir, t_stat (within-period), n_days
# Use for stability diagnostics ONLY — does not improve the main t-stat.

# Estimated IC half-life in days (exponential decay fit)
halflife = estimate_ic_halflife(decay_df)

# Full walk-forward validation
validator = WalkForwardValidator(
    factor_fn=my_factor,      # Callable(prices: DataFrame) -> DataFrame
    oos_start='2018-01-01',
    cost_params=DEFAULT_COST_PARAMS,
    min_stocks=20,
)
result = validator.run(prices)   # → WalkForwardResult
metrics = result.summary()       # → dict ready for kb.log_result()
```

**WalkForwardResult fields:**
- `ic_series`, `icir_series`, `net_ic_series` — daily time series
- `decay_df` — IC vs horizon DataFrame
- `slow_icir_21`, `slow_icir_63` — honest ICIR at natural holding periods
- `weights` — portfolio weights (dates × tickers)
- `turnover` — daily one-way turnover series

**Input format for all harness functions:**
- `factor_df`: `pd.DataFrame`, index=date, columns=ticker, values=factor score; NaN = stock excluded
- `returns_df`: same shape, values=simple daily returns
- All data must be point-in-time (no look-ahead)

## Cost Model API

```python
from qframe.factor_harness.costs import CostParams, DEFAULT_COST_PARAMS, cost_summary

# Default parameters (S&P 500 large-cap, unlevered)
# spread=10bps, gamma=30, eta=0.6, borrow=50bps/yr, funding=0
params = DEFAULT_COST_PARAMS

# Pessimistic / sensitivity check
from qframe.factor_harness.costs import AGGRESSIVE_COST_PARAMS
# spread=20bps, gamma=50, eta=0.6, borrow=150bps/yr, funding=550bps/yr

# Cost breakdown for a given strategy
s = cost_summary(turnover_mean=0.05, horizon=1)
# Returns: one_way_cost_bps, round_trip_cost_bps, trading_drag_bps_day,
#          borrow_drag_bps_day, funding_drag_bps_day, total_drag_bps_year, etc.
```

**Key numbers to remember:**
- Daily-rebalancing factor (5% turnover): ~341 bps/year total cost drag
- Monthly-rebalancing factor (5% turnover): ~40 bps/year total cost drag
- Short borrow (S&P 500 easy-to-borrow): 50 bps/year = 0.19 bps/day
- Hard-to-borrow names: 500–2000 bps/year

## Data Loader API

```python
from qframe.data.loader import load_ohlcv, load_returns, load_sp500_tickers
from qframe.data.loader import load_sp500_historical_tickers, load_survivorship_free_prices

# OHLCV (yfinance, Phase 0)
ohlcv = load_ohlcv(['AAPL', 'MSFT'], start='2010-01-01', end='2024-12-31')
# Returns MultiIndex DataFrame columns: (field, ticker)

# Simple returns
returns = load_returns(['AAPL', 'MSFT'], start='2010-01-01', end='2024-12-31', freq='D')
# Returns (date × ticker) DataFrame of simple returns

# S&P 500 tickers (current — survivorship biased)
tickers = load_sp500_tickers()

# S&P 500 tickers as of a given date (best-effort point-in-time)
historical = load_sp500_historical_tickers(as_of_date='2015-01-01', cache_path='data/sp500_changes.csv')

# Survivorship-bias-reduced universe (slow first run ~10 min, cached after)
prices = load_survivorship_free_prices(start='2010-01-01', end='2024-12-31')
# Returns (date × ticker) close prices with quality filter
```

## Knowledge Base API

```python
from qframe.knowledge_base.db import KnowledgeBase

kb = KnowledgeBase('knowledge_base/qframe.db')
kb.init_schema()  # idempotent, migration-safe

hyp_id  = kb.add_hypothesis(description, rationale, mechanism_score, status, factor_name)
impl_id = kb.add_implementation(hypothesis_id, code, git_hash, notes)
res_id  = kb.log_result(impl_id, metrics_dict)        # metrics_dict from result.summary()
kb.update_hypothesis_status(hyp_id, 'passed')         # or 'failed', 'retired'

all_results  = kb.get_all_results()                   # full JOIN ordered by IC DESC; includes hypothesis_id and code
correlations = kb.get_factor_correlations()

# Phase 2+ methods
bhy = kb.get_bhy_significant(alpha=0.05)              # BHY-corrected significant factors (list of dicts with t_stat, hypothesis_id, code)
impl = kb.get_implementation(hypothesis_id=84)        # most recent implementation for a hypothesis
rid = kb.log_regime_result(                           # log Phase 2 decomposition to regime_results table
    hypothesis_id=84, n_states=5, best_state=2,
    lift=2.28, best_state_ic=0.054, unconditional_ic=0.024,
    go_verdict=1, by_state_json=decomp.by_state.to_json(),
)
rr = kb.get_regime_results()                          # all regime results joined with factor names
```

## Pipeline API

```python
from qframe.pipeline.loop import PipelineLoop
from qframe.pipeline.models import ResearchSpec

loop = PipelineLoop(prices=close, kb_path='knowledge_base/qframe.db', oos_start='2018-01-01')

# Single iteration
result = loop.run_iteration(ResearchSpec(factor_domain='momentum'))
result.print_summary()

# N iterations (auto-runs correlation + ensemble every 5)
results = loop.run_n(ResearchSpec(factor_domain='momentum'), n=5)

# Manual analysis
loop.run_correlation_analysis()    # pairwise Spearman, logs to factor_correlations
loop.run_ensemble_check(top_n=3)   # IC-weighted combo of top 3 factors
```

## Visualisations API

```python
from qframe.viz.charts import (
    plot_leaderboard, plot_ic_decay_curves, plot_ic_decay_heatmap,
    plot_ic_vs_icir, plot_ic1_vs_ic63, plot_cumulative_ic,
    plot_slow_icir_comparison, plot_turnover_scatter,
    plot_correlation_heatmap, plot_sharpe_histogram,
    plot_domain_breakdown, plot_error_rate, plot_net_vs_gross_ic,
    plot_ic_by_period, plot_multiple_testing,
    # Phase 2 regime charts (Charts 16–19):
    plot_regime_timeline, plot_regime_ic,
    plot_velocity, plot_hurst_rolling,
    # Phase 2.5 portfolio charts (Charts 20–22):
    plot_combined_equity, plot_rolling_sharpe,
    plot_blend_weights,
)

fig = plot_leaderboard('knowledge_base/qframe.db', top_n=20)
# All functions: return plt.Figure, never call plt.show()

# Chart 14 — temporal stability (requires prices on disk):
fig = plot_ic_by_period(
    'knowledge_base/qframe.db',
    impl_id=1,                    # implementation ID from SQLite
    prices_path='data/processed/sp500_close.parquet',
    oos_start='2018-01-01',
    period_years=2.0,             # ~2-year blocks
    horizon=1,
)
# Bar chart: mean IC per 2-year period ± 95% CI, with per-period ICIR annotation.
# Use horizon=63 for slow mean-reversion signals.

# Chart 15 — multiple testing significance:
fig = plot_multiple_testing('knowledge_base/qframe.db', alpha=0.05, n_oos_days=1762)
# Shows t-stats for all positive-IC factors with BHY / HLZ / Bonferroni threshold lines.

# Chart 16 — regime timeline (Phase 2):
fig = plot_regime_timeline(az.proba_df, market_returns, oos_start='2018-01-01')
# Stacked posterior probability bands + market return with regime shading.

# Chart 17 — per-state IC bar chart (Phase 2):
fig = plot_regime_ic(decomp.by_state, unconditional_ic=decomp.unconditional,
                     factor_name='impl_82 Calmar', horizon=1)
# Bar per state with unconditional IC dashed baseline.

# Chart 18 — velocity (Phase 2):
fig = plot_velocity(az.velocity_raw.loc[oos:], az.velocity_smooth.loc[oos:],
                    hard_labels=hard_labels_oos)

# Chart 19 — rolling Hurst DFA (Phase 2):
fig = plot_hurst_rolling(az.hurst_series.loc[oos:], hard_labels=hard_labels_oos)

# Chart 20 — combined equity curve + drawdown (Phase 2.5):
fig = plot_combined_equity(
    {'Combined (IC-blend)': combined_eq, 'Factor A': eq_a},
    benchmark=bm,
    initial_investment=10_000,
    title='Phase 2.5 Combined Strategy',
)
# 2-panel: top = equity curves, bottom = drawdown from peak of first curve.

# Chart 21 — rolling 12-month Sharpe (Phase 2.5):
fig = plot_rolling_sharpe(wf_result.net_ic_series, window=252, threshold=0.5)
# Highlights periods above Gate 3 threshold (green) and below (red).

# Chart 22 — dynamic blend weights stacked area (Phase 2.5b):
fig = plot_blend_weights(
    blend_weights=blend_weights_df,   # DataFrame: dates × factor_names, rows sum to 1
    hard_labels=az.hard_labels(oos_start='2018-01-01'),  # optional regime strip
    title='Per-Regime IC-Weighted Blend Weights',
)
# Stacked area chart; right-edge annotations show mean weight per factor.
```

## Regime Module API (Phase 2)

```python
from qframe.regime import RegimeICAnalyzer

az = RegimeICAnalyzer(
    n_states=5,
    hurst_window=252,
    velocity_window=21,
    velocity_ewm_hl=21,
    hsmm_window=504,
    hsmm_step=63,
    min_state_days=30,
)
az.fit(market_returns, is_end='2017-12-31')

# Posterior probabilities (T × n_states DataFrame)
az.proba_df

# Regime IC decomposition
decomp = az.regime_ic_decomposition(factor_df, prices, oos_start='2018-01-01', horizon=1)
# decomp.unconditional  — float
# decomp.by_state       — DataFrame with columns: ic, icir, t_stat, n_days, pct_time, mean_ret_ann
# decomp.best_state     — int (state label)
# decomp.lift           — float (best IC / unconditional IC)

# Regime-conditional exposure multiplier
multiplier = az.regime_weights(factor_df, oos_start='2018-01-01', ic_by_state=decomp.by_state['ic'].values)
# pd.Series; multiply into rank weights before portfolio construction

# Hard regime labels (OOS)
hard_labels = az.hard_labels(oos_start='2018-01-01')  # pd.Series of int state labels

# Phase 2.5b — time-varying per-factor blend weights
# ic_by_state_dict: {factor_name: np.ndarray of shape (n_states,)} — per-state IC for each factor
blend_weights_df = az.regime_blend_weights(
    ic_by_state_dict={'factor_a': ic_arr_a, 'factor_b': ic_arr_b},
    oos_start='2018-01-01',
    shrinkage=0.0,    # 0 = pure posterior; 0.3–0.5 recommended if few obs per state
)
# Returns DataFrame (dates × factor_names), rows sum to 1.
# Degrades to IC-proportional weights when posterior is near-uniform.
# Negatives clipped to 0 before normalisation (never inverts a factor).
```

## Multiple Testing Correction API

```python
from qframe.factor_harness.multiple_testing import (
    correct_ic_pvalues,
    print_correction_summary,
    compute_t_stat,          # for fast signals (h=1)
    compute_slow_t_stat,     # for slow signals (non-overlapping windows)
)

# Run on all KB results
from qframe.knowledge_base.db import KnowledgeBase
kb = KnowledgeBase('knowledge_base/qframe.db')
results = kb.get_all_results()

corrected = correct_ic_pvalues(results, alpha=0.05, n_oos_days=1762)
print_correction_summary(corrected)
# Returns DataFrame with columns: factor_name, ic, t_stat, p_raw, bhy_significant, hlz_sig
```

**Required t-thresholds with current m=84 positive-IC factors:**
- BHY (recommended): t ≥ ~4.0  (c(84) ≈ 5.0)
- Harvey-Liu-Zhu (2016): t ≥ 3.03 = √(2·ln(84))
- **1 factor clears BHY:** `trend_quality_calmar_ratio` / impl_82 (t=11.03). impl_92 is a duplicate.
- impl_53 (`mean_reversion_factor`) was erroneously listed as BHY-significant (t=8.15 used fast formula).
  Correct slow-signal t-stat = 1.31 (27 non-overlapping windows at h=63). Not significant.

**IMPORTANT — slow-signal t-stat formula:** for factors run at horizon h > 1, use:
  `t = slow_icir_63 × √(N_oos_days / 63)` not `icir × √(N_oos_days / 252)`.
  The `correct_ic_pvalues()` function handles this automatically if `slow_icir_63` is populated.

**Fast signal t-stat:** `t = sharpe × √(N_OOS/252)` = sharpe × 2.645 for 7-year OOS
**Slow signal t-stat:** `t = slow_icir_63 × √(N_OOS/63)` = slow_icir_63 × 5.29

## Gate Thresholds

See `gate-thresholds.md` for full definitions.

| Gate | Test | Pass criteria |
|---|---|---|
| 0 | Factor harness working | IC != 0, no NaN errors, SQLite logging works | ✅ PASSED |
| **Factor weak gate** | Cross-sectional alpha | IC ≥ 0.015 AND ICIR ≥ 0.15 | ✅ PASSED (impl_82, impl_53) |
| **Factor pass gate** | Strong alpha | IC ≥ 0.030 AND ICIR ≥ 0.40 | ✅ PASSED (impl_82) |
| 1 | HSMM regime detection | 5 states recovered, Zakamulin (2023) reproduced | ✅ PASSED |
| 2 | Regime-conditional IC | Best-regime IC ≥ 1.5× unconditional for ≥1 factor | ✅ PASSED (impl_53 lift=2.28×) |
| 3 | Combined strategy | Sharpe ≥ 0.5, MaxDD < 25%, worst year > −15%, cost efficiency ≥ 90% | ✅ PASSED (Sharpe 4.27, MaxDD −10.3%, efficiency 99.8%) |
| 4 | Second market confirmation | Same direction on STOXX 600 or crypto | ⬜ Future |

## LLM Providers

| Provider | Role | Free tier |
|----------|------|-----------|
| Groq (default) | Synthesis + analysis | 100k tokens/day |
| Gemini (fallback) | Auto on Groq TPD exhaustion | 1500 req/day (2.0 Flash) |
| Ollama Qwen2.5-Coder:14b | Factor code generation | Free (local) |
| Cerebras | Optional 3rd fallback | 1M tokens/day |

## Coding Conventions

- Type hints on all public functions
- Docstrings with Args/Returns/Raises
- No magic numbers — constants go in `src/qframe/config.py`
- Tests in `tests/`, mirroring `src/` structure
- All data access through `src/qframe/data/loader.py` — never import yfinance directly

```bash
# Run tests
pytest tests/ -v --tb=short

# Run pipeline
./run_pipeline.sh --domain momentum --n 5
```

## When a Backtest Fails

1. The loop already logs the failure to SQLite automatically (`status='failed'`)
2. Check for look-ahead bias first — most spurious results come from this
3. Check IC decay — is the signal too slow or too fast for the assumed cost?
4. Check `ic_horizon_63 >> ic_horizon_1` — if so, it's a slow signal; re-evaluate with `slow_icir_63`
5. Update `agent_docs/research-log.md` with what was learned

## When a Backtest Passes a Gate

1. The loop logs the result to SQLite with `passed_gate=1`
2. Update `agent_docs/research-log.md`
3. Do NOT move to the next gate until the result is confirmed on the second market
4. Run `loop.run_correlation_analysis()` to check if the new factor is orthogonal to existing ones

## Common Failure Modes

| Symptom | Likely cause |
|---------|-------------|
| IC spike at horizon 0 or 1 | Look-ahead bias — check return calculation shift |
| IC > 0.05 consistently | Check for survivorship bias or data error |
| IC drops to 0 in OOS | Overfitting in IS — use simpler factors |
| ICIR negative but IC positive | Noisy/inconsistent signal — IC@63d may be better |
| IC@63d >> IC@1d | Slow mean-reversion signal — evaluate with `slow_icir_63` not standard ICIR |
| Sharpe collapses after 2020 | Factor crowding — check correlation to recent top factors |
| Implementation error rate > 30% | Add more anti-patterns to `_PROMPT_TEMPLATE` in `implementation.py` |
