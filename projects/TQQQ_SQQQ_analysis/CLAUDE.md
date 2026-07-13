# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**If you are resuming work on this project, read `SESSION_HANDOFF.md` (at project root) first.** It captures the operational state - what's been done, what's next, what decisions are settled, what files not to touch. `SESSION_LOG.md` (same dir) is the append-only history of substantive sessions.

**Current state (2026-06-10):** Research items 01–20 are complete. Items 06/17 were corrected for same-day daily-context leakage on 2026-06-09. The combined strategy (item 18) is the validated result: TQQQ Sharpe 4.27, SQQQ Sharpe 2.63, with zero rule/p_severe overlap. `deployable_strategies/` contains the production-ready components. `p_severe_scorer/` has moved to `deployable_strategies/continuous_sizing/p_severe_scorer/`.

## Working style

- Role: **quantitative researcher + data scientist + software developer**. Day-to-day mode is data-scientist — read the trade-log CSVs, hunt for patterns that improve overall PnL (filter losers, size winners, find regime conditions), and validate IS → OOS before recommending anything. Software-developer mode is secondary: keep the pipeline tidy enough that a human can re-run it and audit the outputs.
- **Minimum tokens, minimum code.** Extend existing functions before adding new scripts. Don't refactor opportunistically.
- **Plans will be executed by Sonnet.** Be precise: name the function to add/edit, the numbered output folder, and the `Manifest` entry. No hand-waving.
- Every artifact (CSV, PNG, MD) must be discoverable by a human via `manifest.csv` or `FINDINGS.md`. If you produce a file no index points to, it doesn't exist.
- Ask before any non-trivial action; do not implement without an approved plan.

## Commands

Use the `quant` conda environment:

```
~/opt/anaconda3/envs/quant/bin/python archive/full_history_feature_scan.py
```

Run from the repo root. Optional flags: `--run-id <name>`, `--input-dir <path>`, `--output-root <path>`. Matplotlib runs headless (`Agg`); no display needed.

There is now a test suite:

```
~/opt/anaconda3/envs/quant/bin/python -m pytest TQQQ_SQQQ_analysis/tests -q
```

## Architecture (big picture)

Two project phases.

- **Active:** full-history feature discovery on the six CSVs in `full_history_canonical/trades_backtest/` (2013–2026), driven entirely by `archive/full_history_feature_scan.py`.
- **Paused:** the RSI leverage-overlay work. All paused artefacts live under `archive/` — the original scripts (`step1_eda.py`, `step2_backtest.py`, `step2_5.py`, `prep_trade_logs.py`), the four `STEP*_PLAN.md` handoffs, `LEGACY_RSI_OVERLAY_PLAN.md`, and `LEGACY_FINDINGS.md`. Treat the whole folder as historical; do not edit or rerun.

Pipeline shape inside `archive/full_history_feature_scan.py`: `load_trades` → `add_features` → `audit_schema` + `build_feature_audit` (leakage filter) → `single_variable_tables` + plots → `interaction_tables_and_plots` → `regime_analysis` → `discover_candidate_rules` (frozen on IS) → `validation_outputs` + `traditional_metrics` → `write_memo` + manifest + `update_docs`.

Output convention per run:
```
full_history_research/<run-id>/
  00_data_quality/  01_single_variable/  02_interactions/
  03_regime_analysis/  04_candidate_rules/  05_validation/
  06_reports/  manifest.csv
```
Stable joined per-symbol tables also land in `full_history_canonical/TRADES_<SYM>_full_history.csv`. These canonicals are **overwritten on every run by design**.

Living docs — do not duplicate content across them:

| File | Role |
|---|---|
| `PROJECT_PLAN.md` | Active spec for full-history pattern discovery (Steps 1–7) |
| `FINDINGS.md` | Dated lab notebook for the active project; stale entries are **deleted, not buried** |
| `FEATURE_DICTIONARY.md` | What each *data column* means (source features, derived features, research-pass output columns, known data caveats) |
| `GLOSSARY.md` | What each *concept* means (statistical tests, classification metrics, multivariate methods, predictive models, validation conventions, performance metrics). Cheat sheet for reading findings prose. Not the source of truth for project-specific data semantics — that's `FINDINGS.md`. |
| `archive/` | Everything paused. `LEGACY_RSI_OVERLAY_PLAN.md`, `LEGACY_FINDINGS.md`, `STEP*_PLAN.md`, the legacy `.py` scripts, and the `20260519_small_sample_rsi_overlay/` outputs all live here. Read-only as far as the active project is concerned. |
| `research/SYNTHESIS.md` | Current research state summary (items 01–20). Read this before touching any research item. |
| `deployable_strategies/README.md` | Colleague-facing guide to the deployable package. |

## Data semantics that are easy to get wrong

`FINDINGS.md` is authoritative if anything here drifts.

- `pnl_pct` is in **percentage points** (`2.5` means +2.5 %). Divide by 100 anywhere it enters a return formula.
- The TQQQ canonical chain has **186 external-cash-flow gaps**. Use `pnl / capital_before` per trade for return — never `capital_end / capital_before`.
- **TQQQ and SQQQ are always analyzed separately.** Don't pool.
- **Daily context for intraday entries must be strict-prior**: `context_date < entry_date`. Same-day daily bars are look-ahead leakage.
- In-sample window is hardcoded: `entry_time <= 2020-12-31`. Out-of-sample is 2021–2026. The constant is `IS_END` in the script.
- `RSI_entry` spans roughly `[35, 72]` in this data — rule grids outside that range are vacuous.
- `regime_entry` is populated for the current research-pass subset but has missing early rows. Items 01+ generally filter with `df[df["regime_entry"].notna()]`.
- Loser target: `pnl_pct < 0`. Severe-loss target: `pnl_pct <= -1`.
- **Provenance & cost model.** The `trades_backtest/` CSVs come from a *separate execution engine* that backtests on **FMP OHLC bars** — these are not live fills. Slippage is baked in as a flat, deterministic haircut on `decision_price → avg_order_price`: **+5 bps on entry, −15 bps on exit (~20 bps round-trip), size-independent.** Consequence: these logs **cannot** reveal market-impact or capacity effects — every trade costs the same 20 bps whether it's \$1k or \$1B. Modelling impact/capacity requires a size-aware cost layer (see `volume_research/`). Real live fills exist only from **May 2026 onward** (~1.5 months as of mid-June 2026) — too short to calibrate against yet.
- Original sizing is **all-in / all-out** (full position flip per signal). The item-12 research direction down-scales size by `1 - p_severe`; under that, per-trade Q is a fraction of capital, which pushes the capacity ceiling further out.

## When planning changes

For Sonnet implementing a plan:

- For new research items under `research/`: create `research/NN_name/build_NN_name.py` and `findings_NN_name.md`. Import shared helpers from `research/_walkforward.py`.
- Reuse the existing research helpers: `fit_predict_walkforward_logit`, `equity_metrics`, `sizing_functions`, `extended_sizing_functions` (all in `research/_walkforward.py`).
- For the original pipeline (`archive/full_history_feature_scan.py`): reuse `quantile_bucket`, `summarize_buckets`, `performance_metrics`, `evaluate_rule`, `apply_rule`, `Manifest`, `simple_markdown_table`, `safe_name`.
- Every artifact (CSV, PNG, MD) must be referenced in its item's `findings_NN.md`. New deployable components go under `deployable_strategies/` and are documented in `deployable_strategies/README.md`.
- Findings get updated with actual numeric results from generated CSVs — no placeholders.

## Things to avoid

- **Do not auto-run `archive/full_history_feature_scan.py` without asking.** The default `--run-id` includes `%Y%m%d_%H%M` so intra-day folder collisions are gone, but the canonical CSVs in `full_history_canonical/` are still overwritten every run.
- **Do not revert the run-id format** to date-only — that re-introduces silent overwrites.
- Do not regenerate the paused RSI-overlay outputs.
- Do not re-pull market data here; this project consumes the precomputed CSVs in `full_history_canonical/trades_backtest/`.
- Do not double-scale `pnl_pct`.
- Do not commit CSV/PNG artifacts unless asked.

---

_Behavioral rules (Karpathy) now live globally in `~/.claude/CLAUDE.md` and apply automatically — no longer duplicated per project._
