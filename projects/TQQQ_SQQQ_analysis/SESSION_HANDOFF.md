# Session handoff

**Read this first if you're resuming a session.** This is operational state, not the research deep dive. For the research summary, read `research/SYNTHESIS.md`.

---

**Last updated**: 2026-06-10.

**Last meaningful work**: completed research items 01–20 including feature selection (item 06 permutation importance), multi-threshold sizing (items 12/17/19), combined strategy (item 18), cross-symbol signal (item 20), and created the `deployable_strategies/` package. 65 tests pass.

## Where The Project Is

- Research items 01–20 exist under `research/`. All findings files have `> Period:` date-range headers.
- **Strict-prior daily context** is enforced everywhere: `context_date < entry_date`. Fixed 2026-06-09.
- `deployable_strategies/` is the production-ready package:
  - `metrics.py` — `constant_notional_metrics()`
  - `continuous_sizing/sizing.py` — sizing function library + `SIZING_FUNCTIONS` dict
  - `continuous_sizing/p_severe_scorer/` — walk-forward scorer with annual JSON artifacts (moved from root-level `p_severe_scorer/`)
  - `focus_rules/sqqq_rsi_atr_skip.py` — SQQQ RSI×ATR skip rule mask
  - `regime_rules/tqqq_sideways_skip.py` — TQQQ sideways_lowvol skip rule mask
  - `README.md` — colleague-facing integration guide
- `research/_walkforward.py` — shared `fit_predict_walkforward_logit`, `equity_metrics`, `sizing_functions`, `extended_sizing_functions`.
- Current test suite: **65 passed** in the `quant` conda env.

## Active Recommendation (Validated)

The combined strategy (item 18):

| symbol | config | Sharpe | Max DD | Calmar |
|---|---|---:|---:|---:|
| TQQQ | p_severe linear_skip @ −1% + sideways_lowvol rule | 4.27 | −10.0% | 10.43 |
| SQQQ | p_severe linear_skip @ −2% + RSI×ATR rule | 2.63 | −15.2% | 5.69 |

vs baselines TQQQ (Sharpe 4.14, MaxDD −18.3%) and SQQQ (Sharpe 2.56, MaxDD −18.1%).

Zero overlap between rule fires and p_severe > 0.5 in both symbols.

## Decisions Made

- Same-day daily context is forbidden for intraday entries: `context_date < entry_date`.
- TQQQ target: `is_severe_loss @ −1%` (linear_skip). SQQQ target: `is_severe_loss @ −2%` (linear_skip or sqrt_skip).
- TQQQ and SQQQ remain separate models.
- `p_severe_scorer/` lives under `deployable_strategies/continuous_sizing/p_severe_scorer/`. No backward-compat shim; old `p_severe_scorer/` root directory is deleted.
- `aggressive_2x` and `step_skip_at_50` sizing functions degrade SQQQ Sharpe — do not use.
- `HYG_LQD_ratio` has negative permutation importance for both symbols — remove from enriched models.
- Cross-symbol models (item 20) confirm that own-symbol beats cross: keep TQQQ/SQQQ separate.

## Files Not To Touch Casually

- `archive/full_history_feature_scan.py` — load-bearing original pipeline.
- `full_history_canonical/TRADES_*_full_history.csv` — derived canonical tables.
- `full_history_canonical/trades_backtest/` — source CSVs from the producing backtest.
- `archive/` — legacy project docs and outputs (read-only).
- `deployable_strategies/continuous_sizing/p_severe_scorer/artifacts/` — pre-trained JSON checkpoints.

## Next Priorities

1. External backtest validation: run the combined strategy on live paper-trading or a separate backtest engine using the yearly checkpoint loop.
2. Annual checkpoint update: when 2026 trades are complete, retrain and export `model_2027.json`.
3. Calibration: review Brier score and decile calibration before treating `p_severe` as an absolute probability.
4. Optionally: evaluate whether the SQQQ −1% `sqrt_skip` (Calmar 6.60) or −2% `sqrt_skip` (Sharpe 2.62) is preferable given the actual strategy mandate.
5. Optionally: investigate intraday context (item 14) once ^VIX 15-min FMP history extends before 2023-09-25.

## Operational Facts

- Conda env: `~/opt/anaconda3/envs/quant/bin/python`
- Test command (from repo root): `~/opt/anaconda3/envs/quant/bin/python -m pytest TQQQ_SQQQ_analysis/tests -q`
- Market-context DBs: `/Users/franciscosimao/Documents/QuantFinance/data/`
- `pnl_pct` is in percentage points: `2.5` means +2.5%
- TQQQ and SQQQ are always analyzed separately
- `regime_entry` is missing for early rows; research items filter with `.notna()`
- `high_water_mark_entry` equals `avg_order_price` everywhere — do not use

## Current Test Coverage (65 tests)

The suite covers:
- Structural invariants and source-data semantics
- Headline reproducibility: items 04, 05, 06, 07, 08, 11, 12, 13, 17, 18, 19, 20
- Strict-prior context checks for enriched rows
- `p_severe_scorer` feature contract failures and artifact metadata
- Annual model look-ahead safety and reproduction
- Calibration output existence
- Yearly checkpoint example execution
- `deployable_strategies`: sizing function unit tests, rule mask correctness + NaN safety, metrics reproduction, import correctness

## Living Docs Map

| file | role |
|---|---|
| `SESSION_HANDOFF.md` | This file: operational state for resumption |
| `SESSION_LOG.md` | Append-only session history |
| `README.md` | Human-facing orientation |
| `research/SYNTHESIS.md` | Current research-state summary (items 01–20) |
| `deployable_strategies/README.md` | Integration guide for the production components |
| `FEATURE_DICTIONARY.md` | Column semantics, pruned feature sets |
| `CLAUDE.md` | Collaboration and operational guidance |
