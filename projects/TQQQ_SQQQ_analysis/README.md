# TQQQ / SQQQ Loss-Region Research

Quantitative research on a 15-min intraday trading strategy that produces six CSV trade logs for TQQQ (3× long QQQ) and SQQQ (3× inverse QQQ) covering 2013–2026. The goal of this repo is to **find patterns in losing trades that let us either skip them or down-size them**, ultimately to improve risk-adjusted return.

The research pass under `research/` (items 01–20) produced a validated combined strategy: strict-prior `p_severe` continuous sizing + crisp skip rules for each symbol. See `research/SYNTHESIS.md` for the current state and `deployable_strategies/` for the production-ready components.

## Where things live

| path | role |
|---|---|
| `full_history_canonical/TRADES_<SYM>_full_history.csv` | Stable joined per-symbol trade table. Overwritten on every pipeline run. |
| `full_history_canonical/trades_backtest/` | Source — six CSVs from the producing backtest. Read-only here. |
| `archive/full_history_feature_scan.py` | Original research pipeline. Generates the canonical files, plus a numbered-folder research pack under `full_history_research/<run-id>/`. |
| `full_history_research/<run-id>/` | One per run of the original pipeline. Each contains numbered subfolders 00–06. |
| **`research/`** | **The newer exploratory analysis pass (items 01–20). Where most of the recent work lives.** |
| `research/SYNTHESIS.md` | Headline conclusions across all 20 directions. Read first. |
| `research/_walkforward.py` | Shared walk-forward helpers used by items 12, 17, 18, 19, 20. |
| `deployable_strategies/` | **Self-contained production components.** Continuous sizing (`p_severe_scorer`), sizing functions, SQQQ focus rule, TQQQ regime rule. Copy to your project. |
| `archive/` | Paused / legacy work — the RSI leverage-overlay project. Read-only. |

## Living docs (do not duplicate content across them)

| file | role |
|---|---|
| `README.md` | This file — orientation for a new reader. |
| `PROJECT_PLAN.md` | Active spec for full-history pattern discovery (Steps 1–7). |
| `FINDINGS.md` | Dated lab notebook for the active project. Stale entries are **deleted, not buried**. |
| `FEATURE_DICTIONARY.md` | What each *data column* means (source features, derived features, research-pass output columns, known data caveats). |
| `GLOSSARY.md` | What each *concept* means (statistical tests, classification metrics, multivariate methods, predictive models, validation conventions, performance metrics). Cheat sheet for reading findings prose. |
| `CLAUDE.md` | Collaboration directives and operational facts for future sessions. |

## How to run the pipeline

The conda env: `~/opt/anaconda3/envs/quant/bin/python`. Matplotlib runs headless (`Agg`).

```bash
# Original pipeline (creates a new numbered run folder + updates canonicals)
~/opt/anaconda3/envs/quant/bin/python archive/full_history_feature_scan.py

# Optional flags
~/opt/anaconda3/envs/quant/bin/python archive/full_history_feature_scan.py \
    --run-id my_label \
    --input-dir full_history_canonical/trades_backtest \
    --output-root full_history_research
```

Default `--run-id` is `YYYYMMDD_HHMM_feature_scan` so intra-day reruns don't collide. The canonical CSVs at `full_history_canonical/` are **overwritten every run by design**.

### Re-running individual research directions

Each direction under `research/<NN>_<name>/` has its own `build_<NN>_<name>.py`. Run from project root:

```bash
~/opt/anaconda3/envs/quant/bin/python research/01_data_diagnostics/build_01_data_diagnostics.py
~/opt/anaconda3/envs/quant/bin/python research/06_context_enrichment/build_06_context_enrichment.py
# ...etc
```

Item 06 onward consumes daily bars from the shared `quantcore` SQLite store at `/Users/franciscosimao/Documents/QuantFinance/data/`. If those DBs aren't up to date, refresh first:

```bash
# Refresh daily data (all 17 default tickers)
quantcore-ingest --intervals 1d

# Refresh intraday for items 14 and 15
quantcore-ingest --intervals 15min --symbols QQQ ^VIX TQQQ SQQQ
```

## The `research/` directions in order

Read `research/SYNTHESIS.md` first for the headline. Then in roughly logical order:

| # | Direction | What it does |
|---|---|---|
| 01 | Data diagnostics | Sample sizes, missingness, multicollinearity audit → curated 13-feature set |
| 02 | Univariate signal | Per-feature signal vs `is_loser` / `is_severe_loss` / `pnl_pct` |
| 03 | Multivariate structure | PCA / PLS-DA / LDA loadings |
| 04 | Loss-region models | Depth-4 tree + L1-logit + GBM, with walk-forward |
| 05 | Capital normalization | Constant-notional Sharpe / Sortino / Calmar / DD |
| 06 | Context enrichment | 21 strict-prior daily QQQ / SPY / VIX / credit / curve features. Modest L1-logit severe-loss lift after causality correction. |
| 07 | Severity threshold sweep | Predictability rises sharply at −2 % and beyond |
| 08 | Focus rule recheck | The previous `rsi_x_atr_cell_3_1` survivor passes the harder block bootstrap |
| 09 | Validation redesign | IS / RESEARCH_OOS / **EMBARGO 2026** split |
| 10 | Regime paper review | Zakamulin 5-state HSMM — decided not to implement |
| 11 | Regime-conditional rules | TQQQ severe-in-`sideways_lowvol` → 10/10 OOS precision, +10.6 pp |
| 12 | Continuous sizing simulation | Linear `size = 1 - p_severe` using curated features cuts MaxDD with roughly unchanged Sharpe. |
| 13 | Within-CSV compounding | Auxiliary CAGR view (59 % TQQQ, 43 % SQQQ) |
| 14 | Intraday context | Intraday features add ~0 AUC over daily — VIX intraday deferred (FMP coverage limit) |
| 15 | Tighter stops simulation | Every tighter stop destroys net pnl — do not add stops |
| 16 | Calendar / FOMC features | Day-of-week, FOMC distance, OPEX, seasonality — redundant with daily context, but FOMC distance carries small univariate signal worth monitoring |
| 17 | Sizing with enriched features | Re-runs item 12 with strict-prior item-06 daily context. TQQQ best: linear_skip @ −1% (Sharpe 4.22, MaxDD −10%); SQQQ best: sqrt_skip @ −2% (Sharpe 2.62). |
| 18 | Combined strategy | p_severe sizing + crisp skip rules together. TQQQ Sharpe 4.27, SQQQ Sharpe 2.63. Zero rule/p_severe overlap — components are genuinely complementary. |
| 19 | SQQQ target exploration | Grid: 3 targets × 5 sizing functions. Best Calmar: −1% sqrt_skip (6.60). Best Sharpe: −2% sqrt_skip (2.62). `aggressive_2x` and `step_skip` degrade Sharpe. |
| 20 | Cross-symbol signal | Own-symbol AUC 0.593/0.581 beats cross-symbol 0.576/0.548. Keep TQQQ/SQQQ as separate models. |

Each direction has:
- `build_<NN>_<name>.py` — the reproducer script
- `findings_<NN>_<name>.md` — 1-page conclusions + plot explanations (read this first)
- one or more CSVs and PNGs

## Load-bearing data semantics

If anything below drifts, `FINDINGS.md` § "Data semantics" is authoritative.

- `pnl_pct` is in **percentage points**, not fractions. `2.5` means +2.5 %. Divide by 100 to use in a return formula.
- **TQQQ and SQQQ are always analyzed separately.** No pooling.
- **`capital_before` resets** at the six source-CSV boundaries. Compounded equity across those resets is broken (old CAGR figures of 285 % / 180 % are wrong). Use constant-notional or within-CSV-compounded — see item 13.
- **In-sample window**: `entry_time ≤ 2020-12-31`. After item 09: research-OOS is 2021–2025, **2026 is the embargoed holdout**.
- **Effective IS for regime-labeled subset**: 2015–2020 (the 2013–2014 rows are unlabeled and dropped in items 01+).
- **Loser target**: `pnl_pct < 0`. **Current scorer candidate severe-loss target**: `pnl_pct <= -1.0`. Item 07 showed the `-2%` tail is more separable for skip-rule research, but the strict-prior sizing candidate still uses `-1%` because the probabilities move size enough to affect drawdown.
- **Daily context rule**: for intraday candidate trades, joined market context must satisfy `context_date < entry_date`. Same-day daily bars are look-ahead leakage.

## Conventions

- No new code in `archive/full_history_feature_scan.py` until the pattern is validated under `research/`.
- Every artifact (CSV, PNG, MD) should be discoverable via the `research/<NN>/findings_<NN>.md` for that direction.
- `FINDINGS.md` is dated. Stale findings are **deleted, not buried.**
- Don't commit CSV / PNG artifacts unless asked. The reproducer script is what gets committed.

## Status as of last research turn (2026-06-10)

Done: research items 01–20, strict-prior item-06/item-17 rebuild, `deployable_strategies/` package (metrics, sizing, focus rules, regime rules, README), shared `research/_walkforward.py`, feature selection via permutation importance (item 06), combined strategy simulation (item 18), SQQQ target grid (item 19), cross-symbol signal check (item 20), 65 passing tests.

The combined strategy (item 18) is the validated result. See `deployable_strategies/README.md` for integration guidance and `research/SYNTHESIS.md` for the full evidence base.
