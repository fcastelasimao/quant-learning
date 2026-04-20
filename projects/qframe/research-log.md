# Research Log

*Update this file at the end of every working session. Claude Code reads this at the start of the next session. Also update `agent_docs/research-log.md` with the technical detail.*

---

## Session: 2026-04-20 (auto — domain=value)

**Done:** ran 5 iteration(s): 0 PASS / 0 FAIL / 5 SKIP / 0 ERROR

- `price_to_52w_range` → **SKIP** (execution error)
- `price_to_5y_median_deviation` → **SKIP** (execution error)
- `price_to_5y_median_deviation` → **SKIP** (execution error)
- `ff_momentum_12_1` → **SKIP** (execution error)
- `ff_momentum_12_1` → **SKIP** (execution error)


## Session: 2026-04-20 (auto — domain=quality)

**Done:** ran 5 iteration(s): 0 PASS / 1 FAIL / 2 SKIP / 2 ERROR

- `trend_quality_r2`: IC=-0.0059 ICIR=-0.0834 → **FAIL**
- `drawdown_duration` → **SKIP** (execution error)
- `recovery_ratio` → **SKIP** (execution error)
- `return_skewness_63day` → **ERROR** (execution error)
- `downside_deviation_sortino` → **ERROR** (execution error)


## Session: 2026-04-20 (auto — domain=volatility)

**Done:** ran 5 iteration(s): 0 PASS / 1 FAIL / 2 SKIP / 2 ERROR

- `realized_skewness` → **ERROR** (execution error)
- `max_daily_return` → **ERROR** (execution error)
- `volatility_of_volatility`: IC=-0.0005 ICIR=-0.1126 → **FAIL**
- `downside_volatility` → **SKIP** (execution error)
- `bab_frazzini_pedersen` → **SKIP** (execution error)


## Session: 2026-04-20 (auto — domain=mean_reversion)

**Done:** ran 5 iteration(s): 0 PASS / 3 FAIL / 2 SKIP / 0 ERROR

- `bollinger_mean_reversion` → **SKIP** (execution error)
- `bollinger_mean_reversion`: IC=-0.0086 ICIR=-0.1345 → **FAIL**
- `long_term_reversal_debondt_thaler`: IC=-0.0105 ICIR=0.0551 → **FAIL**
- `rsi_contrarian`: IC=0.0033 ICIR=0.0267 → **FAIL**
- `williams_percent_r_proxy` → **SKIP** (execution error)


## Session: 2026-04-20 (auto — domain=momentum)

**Done:** ran 5 iteration(s): 0 PASS / 3 FAIL / 1 SKIP / 1 ERROR

- `intermediate_momentum_novy_marx`: IC=0.0094 ICIR=0.0321 → **FAIL**
- `industry_momentum_moskowitz_grinblatt`: IC=0.0125 ICIR=-0.0439 → **FAIL**
- `residual_momentum_blitz` → **ERROR** (execution error)
- `aqr_value_momentum` → **SKIP** (execution error)
- `short_term_reversal_jegadeesh`: IC=0.0018 ICIR=-0.0795 → **FAIL**


## Session: 2026-04-20 (auto — domain=value)

**Done:** ran 10 iteration(s): 0 PASS / 1 FAIL / 9 ERROR

- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_zscore_reversal_velocity`: IC=-0.0169 ICIR=-0.0817 → **FAIL**
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_zscore` → **ERROR** (execution error)


## Session: 2026-04-20 (auto — domain=quality)

**Done:** ran 10 iteration(s): 0 PASS / 0 FAIL / 10 ERROR

- `rolling_sharpe_ratio_proxy` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)
- `rolling_sharpe_ratio_252` → **ERROR** (execution error)


## Session: 2026-04-20 (auto — domain=volatility)

**Done:** ran 10 iteration(s): 0 PASS / 0 FAIL / 10 ERROR

- `garch_proxy_skew_adjusted_21` → **ERROR** (execution error)
- `downside_volatility_skew_ratio_126` → **ERROR** (execution error)
- `value_from_volatility_regime_deviation` → **ERROR** (execution error)
- `value_from_realized_volatility_regime` → **ERROR** (execution error)
- `garch_proxy_regime_adjusted` → **ERROR** (execution error)
- `value_from_downside_volatility_regime` → **ERROR** (execution error)
- `garch_proxy_regime_shift` → **ERROR** (execution error)
- `downside_volatility_skew_ratio_252` → **ERROR** (execution error)
- `garch_proxy_asymmetry_ratio` → **ERROR** (execution error)
- `value_from_volatility_regime_deviation` → **ERROR** (execution error)


## Session: 2026-04-20 (auto — domain=mean_reversion)

**Done:** ran 10 iteration(s): 0 PASS / 0 FAIL / 10 ERROR

- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_reversion_to_5_year_median` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)
- `value_long_term_zscore_reversal` → **ERROR** (execution error)
- `value_long_run_zscore_reversal` → **ERROR** (execution error)


## Session: 2026-04-20 (auto — domain=momentum)

**Done:** ran 10 iteration(s): 0 PASS / 5 FAIL / 5 ERROR

- `industry_momentum_surprise`: IC=0.0086 ICIR=0.1243 → **FAIL**
- `industry_momentum_surprise_acceleration` → **ERROR** (execution error)
- `industry_relative_short_term_reversal` → **ERROR** (execution error)
- `industry_relative_52_week_high_growth` → **ERROR** (execution error)
- `residual_momentum_orthogonal_to_52w_high_growth_acceleration`: IC=0.0083 ICIR=-0.0100 → **FAIL**
- `residual_momentum_orthogonal_to_52w_high_growth` → **ERROR** (execution error)
- `residual_momentum_orthogonal_to_52w_high_growth`: IC=0.0082 ICIR=0.1175 → **FAIL**
- `residual_momentum_orthogonal_to_52w_high_growth_industry`: IC=0.0148 ICIR=0.1688 → **FAIL**
- `industry_momentum_surprise_rescaled` → **ERROR** (execution error)
- `industry_momentum_surprise_persistence`: IC=-0.0082 ICIR=-0.0798 → **FAIL**


## Session: 2026-04-19 — impl_82 look-ahead discovered; Phase 1 invalidated; six new guards active; crypto replication added

**Done:**

- impl_82 (`trend_quality_calmar_ratio`, IC 0.0646, t 10.74, the only BHY-significant factor) found to use forward-looking data via `pct_change(251).shift(-1)`. Evidence: (a) the new look-ahead boundary guard triggers for 100% of stocks; (b) on S&P 500 the fixed factor IC = 0.0138, t = 2.34 (fails BHY at m=140); (c) on 25 Binance USDT pairs (2020–2024) the fixed factor IC = −0.005, t = −0.75 — zero alpha on an independent market.
- Phase 2.5 Gate 3 (Sharpe 4.27) **REVOKED**: ensemble inherited bias from impl_82. See `notebooks/phase3_crypto_replication.ipynb`.
- Six new guards now active in the pipeline: look-ahead boundary check (`executor.py`), signal novelty filter ρ>0.70 (`loop.py`), pre-gate 2012–2016 (`walkforward.py`), BHY runtime gate with rolling m (`loop.py`), Deflated Sharpe Ratio (`multiple_testing.py`), per-stock ADV impact cost (`costs.py`).
- Synthesis agent now receives a "high-IC avoid list" + domain-rotation hint.
- Test suite: 137 → 177 passing.

**Gate summary:**

| Gate | Status |
|------|--------|
| Gate 0 (factor harness) | ✅ PASSED |
| Gate 1 (HSMM) | ✅ PASSED |
| Gate 2 (regime-conditional IC) | ✅ PASSED (impl_82 moot, but infrastructure valid) |
| Gate 3 (net-of-cost Sharpe) | ❌ REVOKED 2026-04-19 |

**Next:** run the pipeline with new guards to find a genuinely novel factor.

---

## Session: 2026-04-18 (auto — domain=value)

**Done:** ran 5 iteration(s): 0 PASS / 3 FAIL / 2 ERROR

- `five_year_zscore_reversal_acceleration`: IC=-0.0169 ICIR=-0.0817 → **FAIL**
- `five_year_range_position_relative_to_trend`: IC=0.0136 ICIR=0.1014 → **FAIL**
- `five_year_range_position_zscore` → **ERROR** (execution error)
- `five_year_range_position_momentum_adjusted`: IC=0.0129 ICIR=0.1587 → **FAIL**
- `distance_from_52w_high_acceleration` → **ERROR** (execution error)


## Session: 2026-04-18 (auto — domain=quality)

**Done:** ran 5 iteration(s): 0 PASS / 0 FAIL / 5 ERROR

- `trend_quality_r_squared_63_adj` → **ERROR** (execution error)
- `trend_quality_hurst_rolling_window` → **ERROR** (execution error)
- `trend_quality_r_squared_residual` → **ERROR** (execution error)
- `trend_quality_r_squared_rolling_stability` → **ERROR** (execution error)
- `trend_quality_hurst_residual` → **ERROR** (execution error)


## Session: 2026-04-18 (auto — domain=volatility)

**Done:** ran 5 iteration(s): 0 PASS / 2 FAIL / 3 ERROR

- `downside_volatility_skew_ratio_21`: IC=0.0020 ICIR=0.1079 → **FAIL**
- `garch_proxy_skew_adjusted` → **ERROR** (execution error)
- `downside_volatility_persistence` → **ERROR** (execution error)
- `downside_volatility_skew_ratio_63`: IC=0.0023 ICIR=0.0474 → **FAIL**
- `downside_volatility_regime_shift` → **ERROR** (execution error)


## Session: 2026-04-18 (auto — domain=mean_reversion)

**Done:** ran 5 iteration(s): 0 PASS / 3 FAIL / 2 ERROR

- `bollinger_band_width_reversal`: IC=-0.0017 ICIR=-0.3036 → **FAIL**
- `rsi_14_residual_vs_long_term`: IC=-0.0076 ICIR=-0.1547 → **FAIL**
- `price_deviation_reversal_speed` → **ERROR** (execution error)
- `long_term_residual_reversal` → **ERROR** (execution error)
- `price_zscore_reversal_acceleration`: IC=-0.0154 ICIR=-0.0431 → **FAIL**


## Session: 2026-04-18 (auto — domain=momentum)

**Done:** ran 5 iteration(s): 0 PASS / 1 FAIL / 4 ERROR

- `industry_momentum_acceleration_relative_to_market` → **ERROR** (execution error)
- `residual_momentum_orthogonal_to_short_term_reversal` → **ERROR** (execution error)
- `52_week_high_breakout_acceleration`: IC=-0.0167 ICIR=-0.0706 → **FAIL**
- `residual_momentum_orthogonal_to_52w_high_growth` → **ERROR** (execution error)
- `industry_relative_momentum_acceleration` → **ERROR** (execution error)


## Session: 2026-04-15 (auto — domain=quality)

**Done:** ran 5 iteration(s): 0 PASS / 3 FAIL / 1 ERROR

- `trend_quality_hurst`: IC=0.0002 ICIR=0.0416 → **FAIL**
- `trend_quality_r_squared` → **ERROR** (execution error)
- `trend_quality_hurst_exponent_proxy`: IC=0.0002 ICIR=0.0416 → **FAIL**
- `trend_persistence_ratio`: IC=-0.0013 ICIR=-0.1929 → **FAIL**
- `trend_quality_calmar_ratio`: IC=0.0646 ICIR=0.3823 → **WEAK**


## Session: 2026-04-15 (auto — domain=volatility)

**Done:** ran 5 iteration(s): 0 PASS / 5 FAIL / 0 ERROR

- `downside_volatility_ratio`: IC=-0.0055 ICIR=-0.0099 → **FAIL**
- `volatility_of_volatility_ratio`: IC=-0.0008 ICIR=-0.1350 → **FAIL**
- `volatility_regime_stability`: IC=-0.0011 ICIR=-0.0480 → **FAIL**
- `volatility_asymmetry`: IC=0.0103 ICIR=0.1891 → **FAIL**
- `garch_proxy_ratio`: IC=-0.0041 ICIR=-0.2012 → **FAIL**


## Session: 2026-04-14 (auto — domain=mean-reversion)

**Done:** ran 5 iteration(s): 0 PASS / 1 FAIL / 4 ERROR

- `mean_reversion_to_historical_mode` → **ERROR** (execution error)
- `mean_reversion_to_historical_median`: IC=0.0041 ICIR=0.0477 → **FAIL**
- `price_acceleration_autocorrelation` → **ERROR** (execution error)
- `price_acceleration_skewness` → **ERROR** (execution error)
- `price_reversal_ratio` → **ERROR** (execution error)


## Session: 2026-04-14 (auto — domain=momentum)

**Done:** ran 5 iteration(s): 0 PASS / 3 FAIL / 2 ERROR

- `short_term_residual_momentum`: IC=0.0106 ICIR=0.1419 → **FAIL**
- `price_momentum_52_week_high_ratio`: IC=0.0032 ICIR=0.0436 → **FAIL**
- `trend_persistence` → **ERROR** (execution error)
- `residual_momentum_52_week` → **ERROR** (execution error)
- `industry_momentum_acceleration`: IC=0.0025 ICIR=0.1297 → **FAIL**


## Current Status

**Phase:** 1 — Agentic Pipeline (running)

**Gate status:**
- Gate 0 (infrastructure smoke test): ✅ PASSED
- Gate 1 (factor library): 🔄 IN PROGRESS
- Gate 2+ (HSMM regime detection): ⬜ NOT STARTED

**Knowledge base:** 26 hypotheses, 9 backtest results, 3 factor correlations.

**Passed-gate factors:**
  - None yet

**Top-5 by IC (OOS):**
  | Factor | IC | ICIR | slow_icir_63 |
  |--------|-----|------|----------|
  | industry_momentum_moskowitz_grinblatt | 0.0125 | -0.0439 | 0.0144 |
  | ensemble_top1 | 0.0125 | -0.0439 | 0.0144 |
  | intermediate_momentum_novy_marx | 0.0094 | 0.0321 | -0.2510 |
  | rsi_contrarian | 0.0033 | 0.0267 | 0.5918 |
  | short_term_reversal_jegadeesh | 0.0018 | -0.0795 | 0.5473 |
---

## Session: 2026-04-13 (part 2 — completion + live-trading review)

**Done:**

### Visualisations
- Created `src/qframe/viz/charts.py` with **13 chart functions**:
  1. `plot_leaderboard` — IC and Sharpe horizontal bars for all factors
  2. `plot_ic_decay_curves` — per-factor lines (solid = slow mean-reversion, dashed = normal decay)
  3. `plot_ic_decay_heatmap` — orange outline where IC@63d > IC@1d (slow signal anomaly)
  4. `plot_ic_vs_icir` — efficiency frontier scatter (dot size = turnover, colour = domain)
  5. `plot_ic1_vs_ic63` — scatter with diagonal; above = mean-reversion, below = momentum
  6. `plot_cumulative_ic` — IC building across horizons for top factors
  7. `plot_slow_icir_comparison` — standard vs slow-ICIR (21d, 63d) grouped bars
  8. `plot_turnover_scatter` — with break-even cost line
  9. `plot_correlation_heatmap` — pairwise factor rank correlations (empty until `run_correlation_analysis()` called)
  10. `plot_sharpe_histogram` — distribution of IC Sharpe
  11. `plot_domain_breakdown` — pass/fail counts + mean IC per domain
  12. `plot_error_rate` — hypothesis outcomes over time (shows prompt engineering improvement)
  13. `plot_net_vs_gross_ic` — cost drag gap per factor
- Added **28 new cells** to `notebooks/phase1_pipeline_demo.ipynb` Section A2; all charts verified working

### Live trading cost model (costs.py)
- Added `short_borrow_bps_annual` to `CostParams` (default 50 bps/yr — typical S&P 500 easy-to-borrow)
- Added `funding_cost_bps_annual` to `CostParams` (default 0; set to ~550 bps for leveraged books)
- Added `AGGRESSIVE_COST_PARAMS` (20 bps spread, 50 bps impact, 150 bps borrow, 550 bps funding)
- `net_ic()` now models **3 components**: trading + borrow + funding
- Added `compute_short_fraction()` — uses actual portfolio weights for borrow calculation
- Added `cost_summary()` — human-readable cost breakdown in bps/day and bps/year
- Added full "NOT MODELLED" table and live deployment checklist to module docstring
- Walk-forward harness updated to pass `weights` to `net_ic()` (so borrow cost uses actual short fraction)

### Key cost numbers (DEFAULT params, 5% daily turnover):
| Horizon | Trading drag | Borrow drag | Total drag/year |
|---------|-------------|-------------|-----------------|
| 1 day   | 316 bps/yr  | 25 bps/yr   | **341 bps/yr**  |
| 21 days | 15 bps/yr   | 25 bps/yr   | **40 bps/yr**   |

Daily-rebalancing factors lose ~341 bps/year to costs. Slow factors (horizon=21+) are far cheaper to operate.

### Documentation
- Created `agent_docs/research-log.md` (technical detail for Claude Code)
- Created `README.md` (public-facing GitHub README)
- Updated `research-log.md` (this file), `CLAUDE.md`, `knowledge-base-schema.md`

---

## Session: 2026-04-13 (part 1 — pipeline improvements)

**Done:**
- Ran volatility (6×), quality (5×), momentum (5×), mean_reversion (3×) — **28 total backtested results, 60 hypotheses** in KB
- Fixed major implementation agent bugs (error rate dropped from ~60% → ~20%):
  - `scipy.stats` (as `stats`) added to executor namespace
  - 10-bug anti-pattern list in implementation prompt with correct alternatives
  - Self-healing one-shot retry on fixable errors
  - `num_predict` bumped 512 → 768
- LLM router: Groq TPD → Gemini fallback; `_parse_retry_delay` handles `Xm Ys` format
- Multi-horizon ICIR: `compute_slow_icir()` with non-overlapping windows; `slow_icir_21` + `slow_icir_63` stored in every result
- IC decay curve: changed from 5 sparse points to **all 63 days** stored as `ic_decay_json`
- Survivorship bias: `load_sp500_historical_tickers()` and `load_survivorship_free_prices()` added
- Factor correlation: `run_correlation_analysis()` + `run_ensemble_check()` added to `PipelineLoop`; run automatically every 5 iterations
- Literature seeds: `_LITERATURE_SEEDS` dict added to synthesis agent (5 canonical factors per domain from Harvey/Liu/Zhu 2016, Jegadeesh-Titman 1993, etc.)
- `factor_name` stored in KB and shown in synthesis context for better deduplication

**Leaderboard (top 5, OOS 2018–2024):**

| # | Factor | IC | ICIR | Net IC | TO/day | IC@63d |
|---|--------|----|------|--------|--------|--------|
| 1 | Risk-adjusted momentum (vol-normalized) | +0.0169 | +0.169 | +0.0167 | 4.5% | -0.019 |
| 2 | 12-1 Jegadeesh-Titman momentum          | +0.0157 | +0.166 | +0.0156 | 4.1% | -0.014 |
| 3 | Mean-reversion z-score                  | +0.0124 | +0.031 | +0.0122 | 6.6% | +0.002 |
| 4 | EWMA returns                            | +0.0101 | +0.090 | +0.0100 | 5.9% | +0.001 |
| 5 | Return skewness                         | +0.0092 | +0.113 | +0.0092 | 2.0% | +0.026 |

Notable: `price_level_autocorrelation` has IC = +0.060 at 63-day horizon (genuine slow mean-reversion signal). `slow_icir_63` will be computed on next run to confirm.

**LLM provider status (end of 2026-04-13):**
- Groq: daily TPD limit (100k tokens) reached. Resets on rolling 24h window.
- Gemini: free tier exhausted. Consider enabling billing on Google AI Studio.

---

## Session: 2026-04-12

**Done:**
- Phase 0 complete: factor harness, data loader, SQLite KB, 48 unit tests. Gate 0 PASSED (12-1 momentum IC=0.0157).
- Phase 1 pipeline live: Synthesis → Implementation → Validation → Analysis → SQLite. All agents working.
- 10 factors logged across momentum and mean_reversion domains.
- Prices cached to `data/processed/sp500_close.parquet`.

---

## What to run next (when Groq/Gemini reset)

```bash
# Priority 1: more slow mean-reversion signals (most promising direction)
./run_pipeline.sh --domain mean_reversion --n 5

# Priority 2: push momentum over the weak gate (still worth exploring)
./run_pipeline.sh --domain momentum --n 5

# Priority 3: under-explored domains
./run_pipeline.sh --domain quality --n 5
./run_pipeline.sh --domain value --n 5
```

After every 5 new results `run_n()` automatically:
- Runs `run_correlation_analysis()` and `run_ensemble_check(top_n=3)`
- Updates `research-log.md` with the latest KB stats (the "Current Status" block + new session entry)

---

## Session: 2026-04-19 — Phase 0 cost stress test & pipeline improvement (senior review)

**Done:**
- **Cost stress test added** (`phase25_portfolio.ipynb` Section I): return-space net equity
  reveals that IC-space IR (4.27) overstates portfolio Sharpe (~1.4×). True return-space Sharpe:
  - Gross: 3.05   Default costs: 2.73   Aggressive (unlevered): 2.43   Aggressive (levered): 1.79
  - All four scenarios PASS Gate 3. Real cost drag under Default = 273 bps/yr (2.7%/yr); under
    Aggressive no-funding = 522 bps/yr (5.2%/yr). Use return-space Sharpe for live decisions.
  - IC-space cost efficiency (99.8%) overstates true return-space efficiency (88% under Default).
- **Committed** all Phase 1–2.5 source + notebook changes to git (`53fb740`).
- **KB triage**: No `status='error'` entries in DB — pipeline uses `status='failed'` for all
  non-passing runs including execution errors. 112 failed, 5 passed, 23 active (2 incomplete:
  IDs 128, 138 with NULL factor_name from interrupted runs). Recent high error rates were due to
  Groq/Gemini quota exhaustion falling through to weaker fallback models — fix: A5 (Groq primary).

**In progress / next:** Phase A–B pipeline improvements:
- A1: Dedup 7× phase25_combined KB rows + UNIQUE constraint
- A2: Fix `except Exception: pass` signal-cache swallowing
- A3: slow-ICIR regression test
- A4: HOLDOUT_START = "2024-06-01" sealed hold-out
- A5: Switch implementation agent to Groq llama-3.3-70b-versatile
- B1–B7: novelty filter, pre-gate, parallelism, BHY-in-gate, per-stock ADV

---

## Known Issues / Uncertainties

- **Survivorship bias:** Universe is current S&P 500 constituents only. `load_survivorship_free_prices()` partially mitigates. True fix requires Norgate Data or CRSP (Phase 2 Gate 1 requirement). All current results should be treated as optimistic by ~10–30% IC.
- **slow_icir_63 NULL for results 1–24, 26–28:** These predate the harness upgrade. Result id=25 was backfilled. Run the slow-ICIR investigation script on any other promising factors manually.
- **impl_1 naming bug:** Code defines `momentum_factor` instead of `factor` — excluded from correlation analysis. Low priority.
- **Momentum cluster redundancy:** impl_7, impl_50, impl_10 are ρ=0.79–0.88 correlated (effectively the same signal). Synthesis agent should be prompted to avoid 12-1 momentum variants.
- **Universe expanded to 449 stocks (2026-04-13):** New parquet cached at `data/processed/sp500_close.parquet` (overwriting the 48-stock cache) and `data/processed/sp500_close_expanded.parquet`. 503 tickers attempted; 2 failed yfinance download (Q, SNDK — delisted); 52 dropped by quality filter (min 252 days history). All prior backtest results in SQLite were computed on the 48-stock cache — they remain in the DB for tracking but numbers are now known to be inflated.
- **Multiple testing correction implemented:** `src/qframe/factor_harness/multiple_testing.py` — BHY, Bonferroni, Harvey-Liu-Zhu. On 449-stock universe: 0/14 BHY-significant, 1/14 HLZ-significant (impl_1). `price_level_autocorrelation` slow_icir_63 dropped from 0.251 → 0.190 on honest universe — not significant.
- **Survivorship bias still present:** Historical ticker membership could not be fetched for any year (data source unavailable). `load_survivorship_free_prices()` fell back to current S&P 500 constituents for all years. The 449-stock universe is still 100% survivorship-biased — better than 48 stocks, but not a true fix.
- **Portugal PC offline:** Everything on M1. Revisit for scheduled data jobs and PostgreSQL.

---

## Backlog (parked, not active)

- MLflow full integration (Part B of Phase 1)
- Transfer entropy from news sentiment to returns (Phase 4)
- RP-PCA latent factor recovery (after HSMM Gate 1 passes)
- Options signals: RVol-IVol spread, put-call skew (Phase 2 Gate 2+)
- Critical slowing down early warning signals
- Earnings revision factor via OpenBB + FMP
- PEAD (post-earnings announcement drift)
- Advisor Strategy (Sonnet executor + Opus advisor) — needs Anthropic API beta access
- Cerebras as third LLM provider (1M TPD free — highest of all free providers)
