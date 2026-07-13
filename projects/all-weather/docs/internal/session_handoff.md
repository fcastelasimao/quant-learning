# Session handoff — All Weather

**Status:** active. **Last updated:** 2026-06-10.
**Previous session covered:** completed **Section E in full (E.20–E.25)** — JEPQ benchmark in bundle, rebalance markers on growth chart, tax cost panel, regime comparison panel, threshold sweep heatmap, notebook documentation. 331 tests passing.
**Session 2026-06-09 covered:** Phase 5a (notebook path fixes), Phase 1 (tax-drift equity curve script + artifacts), Phase 5b/5c (two new marimo notebooks), Phase 2 (daily snapshot script + updated launchd installer), Phase 3 (server deployment package). 331 tests still passing.
**This session (2026-06-10) covered:** Section L complete (L.51–L.54) — daily-resolution tax backtest engine, 4 tests, daily-vs-monthly comparison script, F.26 candidate updated from 5.5pp to **6.5pp**. 335 tests passing. Kill criterion FAILED (7pp disqualified); daily engine consistently prefers 6.5pp across all 3 OOS windows. Next actionable: F.26 human gate (now with updated candidate 6.5pp) + launchd install.

---

## 0. If you are a new Claude Code session, read this first

You've just been spun up. Don't wander the repo for 20 minutes trying to figure out what's going on.

**Read order:**

1. **This file end-to-end** (you're here). It contains live state, the active plan, decisions taken, and conventions.
2. `CLAUDE.md` (project root) — repo layout, key constraints, environment.
3. The two **decisions docs**: `research/tax_drift_trigger/findings_alpaca_lot_selection.md` and `docs/data/central_data_manager.md`. Don't reopen these debates without reading them first.
4. `docs/internal/research_log.md` — historical narrative if you need backstory on how we got here.
5. `docs/internal/ToDo.md` is a **historical changelog of completed items**, retained as a record but no longer the active worklist. **Use this file for active work**, not ToDo.md.

**Working with this user:**

- Push back on scope creep. The user explicitly asked for less infrastructure, not more. If a plan you're drafting includes Fargate / ECR / Terraform / CI/CD pipelines for a single-strategy single-user system, you're probably wrong.
- One open question at a time. Close it before opening the next.
- Defer; don't pre-build. The user said: "TQQQ/SQQQ working style" — small focused scripts, decisions before infrastructure.
- The user is Portugal-based but the strategy is modeled as US-taxable for research realism. Don't second-guess this.
- Always use the conda env: `conda run -n allweather python ...` or direct path `/Users/franciscosimao/opt/anaconda3/envs/allweather/bin/python`.

---

## 1. Current verified state (2026-06-10)

| Item | State |
|---|---|
| Test suite | **335 passing**, 2 deselected, 1 deprecation warning (websockets, harmless) — full run ~376s, exit 0. Golden hash MATCH (`64c5753…`). This session added `tests/test_daily_tax_backtest.py` (4 tests). |
| Live execution | Still on legacy `live/_legacy/alpaca_rebalance.py` for the paper accounts in production |
| New pipeline | `live/rebalance.py` ready; first execute pending review |
| Notebooks | **All 5 fixed/created**: `strategy_comparison.py` (paths fixed), `leverage_comparison.py` (paths fixed), `data_explorer.py` (was already OK), `research_overview.py` (NEW — one panel per investigation), `tax_drift_analysis.py` (NEW — drift-trigger deep-dive) |
| Equity comparison | `research/tax_drift_trigger/plot_equity_comparison.py` NEW — produces equity curves + rebalance markers + cumulative tax per policy. Artifacts at `results/equity_comparison/2026-06-09_*`. |
| Daily snapshot | `live/daily_snapshot.py` NEW — price + portfolio-drift monitor, appends to `live/logs/daily_snapshots.csv` daily. Runs with `--no-broker` (no auth) or with broker for actual positions. |
| Launchd installer | `live/scheduler/install_launchd.sh` UPDATED — supports `--snapshot-only`, `--rebalance-only`, `--all`, `--auto-execute`. New template `com.allweather.snapshot.plist.template`. |
| Server package | `server/` NEW — `requirements.txt` (no quantcore/scipy/matplotlib), `strategies.json.example`, `setup_ec2.sh` (bootstrap), `README.md` (step-by-step EC2 deploy guide). |
| Central data store | Up to date through 2026-05-30 for SPY, QQQ, TLT, TIP, GLD, GLDM, GSG, JEPQ daily candles |
| Dividend store (new) | SPY 134 rows back to 1993-03-19; QQQ 87; TLT 285; TIP 198; JEPQ 48; GLD/GLDM/GSG empty (no cash distributions — correct) |
| JEPQ price coverage | 2022-05-04 → 2026-05-29 (1021 daily rows) — newly added to central store |
| `engine/data.py` | Has `fetch_prices()` (via central FMP SQLite) AND `fetch_dividends()` (new) |
| Daily engine | `engine/daily_tax_backtest.py` NEW — daily-resolution tax-aware backtest with 31-day gate. 4 tests in `tests/test_daily_tax_backtest.py`. |
| Daily vs monthly | `research/tax_drift_trigger/daily_vs_monthly_comparison.py` NEW. Artifacts at `results/daily_vs_monthly/2026-06-10_09-23-14_6_asset_rp_baseline/`. Kill criterion FAILED → candidate updated from 5.5pp to 6.5pp, 7pp disqualified. |

---

## 2. Active plan — execute in this order

**Do NOT start a downstream item before its upstream is verified.**

### Section A — Pre-launch (June 1 on client's Alpaca account)

| ID | What | Status |
|---|---|---|
| A.1 | Configure client's account env vars in `QuantFinance/api_keys.env` (`BROKER_ALPACA_<LABEL>_KEY` + `_SECRET`); confirm `strategies.json` allocation | **HUMAN** — pending |
| A.2 | `make healthcheck BROKER=alpaca ACCOUNT=<client> MODE=--live` | **HUMAN** — pending |
| A.3 | `make rebalance-dry-execute BROKER=alpaca ACCOUNT=<client> MODE=--live`; save and review the `live/logs/runs/<timestamp>.json` summary | **HUMAN** — pending |
| A.4 | Enriched preview output (cadence + budget + lots inline) | **DONE 2026-05-30** |
| A.5 | `make rebalance-new-execute BROKER=alpaca ACCOUNT=<client> MODE=--live` | **HUMAN** — pending |
| A.6 | Verify post-execution: `live/logs/runs/<latest>.json`, `live/logs/cadence_*.json`, `live/logs/lots_*.json`, `live/logs/budget_*.json` match Alpaca reality | **HUMAN** — pending |
| A.7 | Verified Alpaca lot-level support — confirmed absent (FIFO at broker, no order-level lot selection) | **DONE 2026-05-30** — see `research/tax_drift_trigger/findings_alpaca_lot_selection.md` |

### Section B — Cleanup

| ID | What | Status |
|---|---|---|
| B.8 | Phase 1A: delete `archive/`, fold `ALPACA_SETUP_MAC.md`, move root MDs to `docs/internal/`, rename legacy rebalancer, refresh `CLAUDE.md` | **DONE 2026-05-30** |
| B.1B | `research/` subpackage reorg (22 flat files → `plotting/`, `reports/`, `validation/`, `compare/`) | **DEFERRED** — pick up only when touching those imports anyway |

### Section C — Data plumbing

| ID | What | Status |
|---|---|---|
| C.9 | `data_manager.py` dividend support: `--with-dividends`, `--dividends-only`, new `dividends` table, JEPQ added to `DEFAULT_SYMBOLS` | **DONE 2026-05-30** |
| C.10 | `engine/data.py` reader: `fetch_dividends()` consumes the central store; price path already used central store | **DONE 2026-05-30** |
| C.11 | `docs/data/central_data_manager.md` contract doc | **DONE 2026-05-30** |

### Section D — Tax model + drift trigger (the high-value research work — START HERE NEXT)

Land the order below. Tests must pass after each. **Add a golden regression test on production weights/Calmar/MDD BEFORE landing D.15 or D.16** so existing numbers reproduce byte-identical with the defaults.

| ID | What | Status |
|---|---|---|
| D.12 | `engine/tax_rates_us.yaml` — year-keyed schedule (ST top, LT top, qualified-div top, NIIT) from Tax Foundation. Bisect-by-date lookup. | **DONE 2026-05-30** — YAML + `engine/tax.py` (`TaxRates`, `TaxSchedule.from_yaml`/`rates_on`, cached `us_tax_schedule()`); 24 tests in `tests/test_tax_schedule.py`. 3 regimes cover 2006→present: 2003 (35/15/15/0), 2013 (39.6/20/20/3.8), 2018 (37/20/20/3.8, incl. OBBBA-2025 permanence through 2026). |
| D.13 | `engine/tax.py` — `TaxRegime`, `compute_tax_on_event(event, lot_ledger, regime)`. Per-ETF distribution classification: SPY/QQQ qualified, TLT/TIP ordinary (interest), GLD collectibles-rate on sale (28% top), GSG partnership/§1256 (60/40 LT/ST mark-to-market). | **DONE 2026-05-30** — added to `engine/tax.py`: `AssetTaxClass`/`DividendCharacter` enums, `DEFAULT_ASSET_TAX_CLASS` map (incl. GLDM), `RealizedLot`/`SaleEvent`/`DividendEvent`, `TaxResult`, `TaxRegime` (`.us()` / `.none()`), `tax_on_sale`/`tax_on_dividend`/`compute_tax_on_event`. Overrides: GLD LT capped at `COLLECTIBLES_LT_TOP=0.28`; GSG 60/40 split ignoring holding period. 29 tests in `tests/test_tax_events.py`. **Signature deviation:** `compute_tax_on_event(event, regime)` — no `lot_ledger` arg; the D.14 selector emits the `RealizedLot` list carried by `SaleEvent`. **Defaults taken (overridable):** NIIT always-on (top bracket); losses offset at marginal rate (`allow_loss_offset=True`). **Deferred to D.16:** §1256 year-end mark-to-market *timing* (emit a Dec-31 deemed-sale `SaleEvent` for GSG); dividend ex-date event emission. (29 tests, not "38".) |
| D.14 | `engine/lot_ledger.py` — pluggable selectors. Default `FIFO` (matches Alpaca broker reality). Optional `tax_optimal` (LT-first then highest basis — **research counterfactual only on Alpaca**, see `research/tax_drift_trigger/findings_alpaca_lot_selection.md`). Optional `HIFO`. Mirror the FIFO shape from `live/lots.py` so backtest and live agree. **Interface contract from D.13:** a selector consumes the held lots for a ticker and a sell quantity and emits a `list[engine.tax.RealizedLot]` (ticker, quantity, proceeds, cost_basis, acquired, disposed) which becomes `SaleEvent.lots`. | **DONE 2026-05-30** — `engine/lot_ledger.py`: `Lot` (mirrors `live/lots.py` shape, re-declared so engine doesn't import live), `LotSelector` enum (`fifo`/`hifo`/`tax_optimal`, `.coerce()` from str), `_order_lots`, and `LotLedger` (`buy`/`sell`/`sell_event` + `total_quantity`/`total_cost_basis`/`average_cost`/`unrealized_gain`/`lots`/`tickers`). FIFO default = Alpaca reality; tax_optimal = LT-first then highest basis (research-only on Alpaca). Oversell raises. 24 tests in `tests/test_lot_ledger.py` incl. end-to-end into `compute_tax_on_event` (FIFO GLD → 28% LT; FIFO-vs-HIFO realize different basis/tax). In-memory per run (no JSON persistence, no 31-day gate — those are live-only). |
| D.15 | Drift trigger in `engine/backtest.py` — add `rebalance_policy: RebalancePolicy` parameter. Default `monthly_unconditional` (current behavior, zero regression). Other modes: `drift_relative(pct)`, `drift_absolute(pp)`, `monthly_check_then_drift(pct)`. Move threshold logic from `research/rebalance_thresholds.py` into the engine; have research script call into engine. | **DONE 2026-05-30 (core); one sub-item deferred)** — `engine/backtest.py` now has a frozen `RebalancePolicy` (factories `monthly_unconditional`/`drift_relative`/`drift_absolute`/`monthly_check_then_drift`, `.should_rebalance()`, `.label`). `run_backtest` gained `rebalance_policy=None` (last param → defaults to `monthly_unconditional`). **Zero-regression proven**: `tests/test_backtest_golden.py` captured the pre-refactor output (synthetic seeded 6-asset series, 228 rows, sha256 `64c5753…`, AW final 21029.91) and the refactored default reproduces it byte-identical (hash MATCH). Transaction cost now only charged on months that actually trade. 16 tests in `tests/test_rebalance_policy.py`. **NB on month-end engine:** `monthly_check_then_drift` ≡ `drift_relative` here (engine resamples to month-end before iterating); kept distinct for manifest intent, would diverge only in a daily engine. **DEFERRED sub-item:** did NOT consolidate `research/rebalance_thresholds.py` into the engine — its `DriftPolicy` has a richer `hybrid` mode + full-on-breach diagnostics/CSVs and a passing test contract; migrating it is orthogonal to the D.18 sweep (which can call `run_backtest` with a `RebalancePolicy` directly). Leave the research module as-is until F.26/H.35 touch it. |
| D.16 | Tax hook in rebalance loop — at each event, selector picks lots → realized gain ST/LT → year-effective rate → debit portfolio value. Dividend cashflows taxed at ex-date. | **DONE 2026-06-03** — built as a SEPARATE engine `engine/tax_backtest.py:run_tax_aware_backtest` (the share-based `run_backtest` is golden-locked; not entangled). Uses `LotLedger`+`compute_tax_on_event`. Taxes realized gains per sale (selector-driven), dividends at ex-date, GSG §1256 year-end MTM (deemed sale, basis reset). Pays tax+cost by pro-rata lot scaling. **Validated:** `TaxRegime.none()`+zero-cost reproduces share engine to 1e-6 (both end 29,684.69 on FMP 2006-24). 11 tests `tests/test_tax_backtest.py`. NB cumulative tax is non-monotonic by design (loss offsets/§1256 credits). |
| D.17 | New artifacts in `production_validation` bundle: `rebalance_events.csv`, `tax_summary.csv`. Extended `manifest.json` with `tax_regime` and `rebalance_policy` blocks. | **DONE 2026-06-03** — `research/production_validation.py:build_tax_addendum` writes `rebalance_events.csv`, `tax_summary.csv`, `tax_monthly_series.csv`, `tax_addendum_manifest.json` (with `tax_regime`/`rebalance_policy`/`lot_selector` blocks) into the bundle. Decoupled from the yfinance comparison build (addendum uses FMP+dividends; skipped with a warning if central store absent). CLI flags `--tax-regime/--lot-selector/--rebalance-policy/--no-tax-addendum`. FIFO default = Alpaca reality. 5 tests `tests/test_production_validation_tax.py`. |
| D.18 | `research/tax_threshold_sweep.py` — sweep drift threshold (relative: 10/15/20/25/30/40%; absolute: 1/2/3/5pp) × tax regime (US, none) × lot selector (FIFO, tax_optimal). Emit `threshold_sweep_summary.csv`. **Kill criterion:** if best drift+tax beats monthly+tax on Calmar by ≥5% on ≥2 of 3 OOS windows, propose new production policy; else tax model stays research-only. | **DONE 2026-06-03 — VERDICT: PASSED (propose new production policy).** Under US tax, EVERY drift policy beat monthly on Calmar on all 3 OOS windows. Best FIFO (Alpaca-achievable): `drift_absolute_5pp` Calmar 0.378/0.399/0.327 vs monthly 0.295/0.299/0.250 (≈+28/33/30%). Effect is tax-deferral (largely vanishes under `none`). 17 configs passed (10 FIFO, 7 tax_optimal). Artifacts + `verdict.json` under `results/tax_threshold_sweep/`. 7 tests `tests/test_tax_threshold_sweep.py` pin the kill-criterion logic. |
| D.19 | Docs: `research/tax_drift_trigger/findings_*.md`. TQQQ/SQQQ style: what was tested, methodology, results, decision. | **DONE 2026-06-03** — all three written with real sweep numbers. |

### Section E — Marimo dashboard (no recomputation in notebooks)

| ID | What | Status |
|---|---|---|
| E.20 | JEPQ as benchmark line in `production_validation` bundle so it appears in `strategy_comparison.py` automatically | **DONE 2026-06-05** — JEPQ added to `DISPLAY_NAMES`, `required_tickers`, `COLORS` (purple `#c678dd`). Series indexed at 100 from first valid date (2022-05-04), no fee applied. Synthetic JEPQ in test fixture. |
| E.21 | Rebalance markers on growth chart — `axvline` per event sized by trade notional; toggle via `mo.ui.checkbox` | **DONE 2026-06-05** — `plot_growth` gained `rebalance_events`/`show_rebalance_events` params. `_plot_rebalance_markers` draws axvlines with alpha 0.15–0.70 scaled by trade notional. Checkbox default off. |
| E.22 | Tax cost panel — annual stacked bars (ST gain tax / LT gain tax / dividend tax) + cumulative tax-paid line | **DONE 2026-06-05** — `plot_tax_cost(tax_summary, tax_monthly)` in `strategy_plotting.py`. Stacked bars (capital gains / dividend / §1256 MTM) + cumulative line. Notebook cell reads `tax_summary.csv` + `tax_monthly_series.csv`. |
| E.23 | Regime comparison panel — same strategy side-by-side under ISA (zero) vs US-taxable | **DONE 2026-06-05** — `build_tax_addendum` now also runs the alternate regime and writes `tax_regime_comparison.csv` (US Value / ISA Value). `plot_regime_comparison` shows both lines + tax drag annotation. |
| E.24 | Threshold sweep heatmap — Calmar vs (drift threshold, lot selector) for chosen tax regime | **DONE 2026-06-05** — `plot_sweep_heatmap(sweep_summary, regime)` in `strategy_plotting.py`. Calmar heatmap (policy × OOS window) faceted by lot selector, annotated cell values. Notebook loads from `results/tax_threshold_sweep/`. |
| E.25 | `docs/notebooks/strategy_comparison.md` describing each cell and its artifact | **DONE 2026-06-05** — written at `docs/notebooks/strategy_comparison.md`. |

### Section F — Decision gate

| ID | What | Status |
|---|---|---|
| F.26 | Read D.18 verdict. If drift+tax wins: update `strategies.json` with `rebalance_policy` block; update `live/rebalance.py` to use drift trigger (currently 31-day cadence only); update `research/rebalance_frequency/findings.md` with new verdict; one paper-month of agreement before flipping live policy. | **READY — D.18 verdict PASSED.** Recommended candidate updated to **`drift_absolute(0.065)` (6.5pp, FIFO)** after daily-engine validation (L.53, 2026-06-10). **7pp is disqualified**: ranks #7/#7/#5 in the daily engine (which matches live system) vs. top-2 in the monthly engine — the 31-day gate interacts poorly with a 7pp threshold. 6.5pp is the consistent #1 across all 3 OOS windows in the daily engine (Calmar 0.398/0.409/0.363 vs. monthly baseline 0.297/0.302/0.256). 5.5pp (avg rank 3.3 in daily engine) is the conservative fallback. **HUMAN gate before flipping live.** Before flip: (a) confirm sweep on yfinance total-return matches FMP (near-identical per CLAUDE.md 2026-05-09); (b) `strategies.json` `rebalance_policy` block; (c) `live/rebalance.py` drift gate alongside 31-day cadence; (d) reopen `research/rebalance_frequency/findings.md` verdict (tax-aware); (e) one paper-month agreement; (f) **daily-engine validation complete — 6.5pp confirmed (L.53)**. **Do NOT auto-flip.** |

### Section K — Fine-grained drift threshold analysis (2026-06-09)

| ID | What | Status |
|---|---|---|
| K.49 | Fine-grained sweep: abs 4pp/5pp/5.5pp/6pp/6.5pp/7pp + rel 40%, FIFO/US tax, OOS 2018. Calmar peak at 6pp (0.3824) in isolation. Per-ETF triggers rejected (step-function approximation of relative drift). | **DONE 2026-06-09** |
| K.50 | Walk-forward across OOS 2018 / 2020 / 2022 (monthly engine): 6pp is NOT consistently best (ranks 1/3/4, avg 2.7). **5.5pp and 7pp were the monthly-engine candidates.** Candidate set to 5.5pp provisionally. **Superseded by L.53**: daily-engine validation (2026-06-10) shows 7pp is disqualified (ranks #7/#7/#5 in daily engine); updated candidate is **`drift_absolute(0.065)` (6.5pp)**. | **DONE 2026-06-09 (superseded by L.53)** |

Full results (FIFO, US tax, OOS 2018–2026-06-09, $10k start):

| Policy | Calmar | CAGR | MDD | Rebalances | Cum. Tax | Final Value |
|---|---|---|---|---|---|---|
| Monthly | 0.2798 | 5.19% | −18.53% | 239 | $6,297 | $23,690 |
| abs 4pp | 0.3417 | 6.07% | −17.77% | 12 | $5,467 | $27,725 |
| abs 5pp | 0.3618 | 6.29% | −17.37% | 9 | $5,681 | $29,173 |
| abs 5.5pp | 0.3804 | 6.54% | −17.20% | 8 | $5,466 | $29,991 |
| **abs 6pp ← candidate** | **0.3824** | 6.57% | **−17.17%** | **7** | $5,959 | $30,629 |
| abs 6.5pp | 0.3761 | 6.70% | −17.82% | 6 | $5,624 | $31,250 |
| abs 7pp | 0.3819 | 6.55% | −17.14% | 6 | $5,497 | $30,131 |
| rel 40% | 0.3614 | 6.32% | −17.48% | 10 | $5,543 | $28,755 |

### Section I — Notebooks + visualisation (2026-06-09)

| ID | What | Status |
|---|---|---|
| I.40 | Fix broken notebook paths (`strategy_comparison.py`, `leverage_comparison.py`) | **DONE 2026-06-09** — BUNDLE_ROOTS updated to point at `results/` (root-level). All 5 notebooks parse and import cleanly. |
| I.41 | `research/tax_drift_trigger/plot_equity_comparison.py` — equity curves + rebalance markers + cumulative tax per drift policy (FIFO/US tax). Artifacts at `results/equity_comparison/`. | **DONE 2026-06-09** — 5 policies, 4 plots (equity_curves.png, rebalance_markers.png, tax_cumulative.png, panel.png), 4 CSVs. drift_abs_5pp: $29,173 final vs monthly $23,690, only 9 rebalances vs 239. |
| I.42 | `notebooks/research_overview.py` — NEW marimo notebook: one panel per investigation, verdict filter, loads from CSV artifacts. | **DONE 2026-06-09** |
| I.43 | `notebooks/tax_drift_analysis.py` — NEW marimo notebook: equity curves, rebalance markers, cumulative tax, Calmar heatmap, F.26 gate summary. | **DONE 2026-06-09** |

### Section J — Automation + deployment (2026-06-09)

| ID | What | Status |
|---|---|---|
| J.44 | `live/daily_snapshot.py` — daily price + portfolio-drift monitor. Appends to `live/logs/daily_snapshots.csv`. Runs with or without broker auth. | **DONE 2026-06-09** |
| J.45 | `live/scheduler/install_launchd.sh` updated: `--snapshot-only`, `--rebalance-only`, `--auto-execute`, new snapshot plist template. | **DONE 2026-06-09** |
| J.46 | Install launchd agents on Mac (rebalance + snapshot). | **HUMAN** — run: `bash live/scheduler/install_launchd.sh --all`, then `launchctl load ~/Library/LaunchAgents/com.allweather.rebalance.plist && launchctl load ~/Library/LaunchAgents/com.allweather.snapshot.plist` |
| J.47 | `server/` package: `requirements.txt` (no quantcore), `setup_ec2.sh`, `strategies.json.example`, `README.md`. | **DONE 2026-06-09** |
| J.48 | EC2 deployment: launch instance, run `server/setup_ec2.sh`, add API keys. | **HUMAN** — see `server/README.md` for step-by-step. |

### Section L — Daily-resolution drift backtest (NEXT — implement before F.26 flip)

**Why this exists:** The backtest (Sections D/K) uses a month-end engine. The live system checks drift every day and enforces a 31-day minimum gate. These are different systems. We must validate the daily-check version before flipping live.

| ID | What | Status |
|---|---|---|
| L.51 | `engine/daily_tax_backtest.py` — daily-resolution engine. `run_daily_tax_backtest(prices_daily, allocation, regime, rebalance_policy, lot_selector, dividends, transaction_cost_pct, min_rebalance_days=31)`. Returns `DailyTaxBacktestResult` with `daily_records` (DataFrame), `rebalance_dates` (list), `monthly` property (month-end resample). Reuses `LotLedger`, `TaxRegime`, `compute_tax_on_event`, `RebalancePolicy`. | **DONE 2026-06-10** |
| L.52 | `tests/test_daily_tax_backtest.py` — 4 seeded synthetic tests: (1) buy-and-hold baseline (drift_absolute(0.999), value = weighted sum), (2) gate blocks rebalance before 31 days, (3) monthly Calmar within 10% of monthly engine on same 10-year series, (4) US tax charged on sale. **335 tests passing.** | **DONE 2026-06-10** |
| L.53 | `research/tax_drift_trigger/daily_vs_monthly_comparison.py` — ran K.50 policy ladder (monthly/4–7pp/rel40%) under both engines, OOS 2018/2020/2022. Artifacts at `results/daily_vs_monthly/2026-06-10_09-23-14_6_asset_rp_baseline/`. **KILL CRITERION: FAILED** — original top-2 (5.5pp, 7pp) NOT in top-3 of daily engine. **New winner: 6.5pp ranks #1 across all 3 OOS windows** (0.398/0.409/0.363). 7pp is disqualified: ranks #7/#7/#5 in daily engine vs. top-2 in monthly engine — the 31-day gate interacts poorly with 7pp threshold. | **DONE 2026-06-10 — VERDICT: FAILED (see F.26 update)** |
| L.54 | Update F.26 gate: candidate updated from 5.5pp to **6.5pp**; 7pp disqualified; item (f) added to checklist. | **DONE 2026-06-10** |

**Constraints for implementation:**
- Reuse (do not rewrite): `engine/lot_ledger.LotLedger`, `engine/tax.TaxRegime`, `engine/tax.compute_tax_on_event`, `engine/backtest.RebalancePolicy`
- `prices` input: same daily DataFrame from `fetch_prices()` — forward-fill NaNs before loop
- Cash pool: dividends + over-target sale proceeds reinvested same day; if tax makes cash negative, scale to zero
- `monthly` property on result: resample `daily_records["Value"]` to month-end so `_calmar_oos()` works unchanged

### Section G — Shadow comparison upgrade

| ID | What | Status |
|---|---|---|
| G.27 | `backtest_shadow.py` consumes both legacy `performance_tracking_*.csv` and new `live/logs/run_summary.jsonl` / `live/logs/runs/*.json` | TODO |
| G.28 | Plot: cumulative actual vs simulated with deviation bands | TODO |
| G.29 | Plot: per-rebalance fill-price vs simulated-price diff (catches Alpaca slippage) | TODO |
| G.30 | `shadow_summary.csv` to bundle for marimo | TODO |
| G.31 | Write `live/logs/WARNINGS.log` when MAE > threshold — no Slack, just a file you grep | TODO |
| G.32 | `docs/research/shadow_comparison.md` | TODO |

### Section H — Deferred (do not start without explicit user ask)

- H.33 EC2 t4g.nano + cron migration — small Lightsail/EC2 box, local log file, optional S3 nightly sync. **No Fargate, no Terraform, no CI/CD.**
- H.34 Top-3 universe spot-check under new policy — only if F.26 changes the policy
- H.35 Repo Phase 1B `research/` subpackage reorg — when touching those imports anyway
- H.36 RSI overlay walk-forward, trade diagnostics, mixed-pair production — leverage track, separate research
- H.37 GBP/EUR currency adjustment, FCA, customer comms — separate workstream
- H.38 Full 100-ETF universe rerun — only if F.26 + H.34 both surprise

---

## 3. Decisions taken — do not re-litigate

1. **Tax regime: US-only first.** ISA / PT_residente excluded. The user is Portugal-based, but the research model uses US-taxable-individual as the standard reference. Cross-border tax reality is a separate model, deferred indefinitely.
2. **AWS plan dropped from blocker list.** First-version Fargate plan was over-engineered. When ready: EC2 t4g.nano + cron + local log file. The user only wanted "live rebalancing on its own," not autonomous research infrastructure.
3. **No Slack notifications.** Logs to file only. `live/notify.py` is silent when env vars are unset — that's the desired default.
4. **Alpaca lot-level support: confirmed absent.** Verified against `alpaca-py 0.43.2` and Alpaca docs. Order placement has no `lot_id` / `cost_basis_method` field. Tax-optimal selector in the backtest is a **research counterfactual only**. Live execution stays FIFO. (`research/tax_drift_trigger/findings_alpaca_lot_selection.md`)
5. **Rebalance trigger semantic: drift-based, not pure calendar.** Production engine currently still calendar-monthly; the closed `research/rebalance_frequency/` verdict was made under transaction-cost-only modelling and **may be reopened** under realistic US tax + tax-optimal lot selection.
6. **Live ticker mapping: only `GLD → GLDM`.** Confirmed against `strategies.json` and the project memory. Don't introduce other substitutions.
7. **No new `--summary` flag for `live/rebalance.py`.** The existing preview/dry-execute output was enriched in place. One review surface.
8. **`engine/data.py` already consumed from central store for the FMP path.** C.10 reduced to adding a `fetch_dividends()` reader. No duplicate fetch logic.
9. **`ToDo.md` is retired in favor of this file.** Kept as a historical changelog of completed items (41 entries through 2026-05-27). All active work tracking happens here.
10. **JEPQ added to central data store.** Price history 2022-05-04 → present, dividend history 2022-06-01 → present. Use `engine.data.fetch_prices_from_fmp_db(['JEPQ'], ...)` and `engine.data.fetch_dividends(['JEPQ'], ...)`.

---

## 4. Operating principles — apply on every change

- **One open question at a time.** Close it before opening the next.
- **Every new research output → CSV/JSON artifact** in `results/<task>/<timestamp>_<strategy>/`. Marimo just reads. Never recompute in the notebook.
- **New investigation → `research/<topic>/findings.md`** documenting what was tested, results, decision. TQQQ/SQQQ project style.
- **Engine changes: default behavior preserved.** New params default-off. **Golden regression test** on production weights/Calmar/MDD must pass byte-identical before landing.
- **Push back on me if scope grows.** The first version of the AWS plan was an example the user explicitly called out.
- **Always use conda env.** `conda run -n allweather python ...` or direct path `/Users/franciscosimao/opt/anaconda3/envs/allweather/bin/python`.
- **Don't commit without being asked.** The user controls commits.

---

## 5. Verification commands

```bash
# Full test suite (~6 min)
/Users/franciscosimao/opt/anaconda3/envs/allweather/bin/python -m pytest tests/ -q

# Targeted (much faster)
/Users/franciscosimao/opt/anaconda3/envs/allweather/bin/python -m pytest tests/test_live_rebalance.py tests/test_broker_protocol.py -q

# Refresh central data (run periodically)
quantcore-ingest --symbols SPY QQQ TLT TIP GLD GLDM GSG JEPQ --with-dividends

# Pre-flight an Alpaca paper account
conda run -n allweather python -m live.rebalance --paper --broker alpaca --dry-execute

# New pipeline preview (will show enriched cadence/budget/lot status)
conda run -n allweather python -m live.rebalance --paper --broker alpaca

# Check dividend data from engine side
python -c "from engine.data import fetch_dividends; \
    df = fetch_dividends(['SPY','TLT','TIP','JEPQ'], '2022-01-01', '2026-06-01'); \
    print(df.groupby('Ticker').size())"
```

---

## 5b. RESOLVED: the data engine is now the versioned `quantcore` package

The downloader/data layer was extracted into a dedicated, version-controlled
package: **`QuantFinance/quantcore/`** (repo `github.com/fcastelasimao/quantcore`).

- Fetcher: `quantcore.ingest` (CLI `quantcore-ingest`). Readers: `quantcore.data`
  (`load_prices`, `load_dividends`, `load_panel`).
- `engine/data.py` is a thin wrapper; `_repo_data_dir()` delegates to
  `quantcore.config.data_dir()` (env `$QUANT_DATA_DIR`, then walk-up, then
  workspace fallback) — no hard-coded paths.
- Installed editable per project: `pip install -e ../../quantcore`.
- The old `QuantFinance/data_manager.py` has been removed; use
  `quantcore-ingest` (`python -m quantcore.ingest`).

The C.9 dividend support (`--with-dividends` / `--dividends-only`, the
`dividends` table, JEPQ in `DEFAULT_SYMBOLS`) now lives in versioned
`quantcore.ingest`, so a fresh clone is reproducible. Contract:
`docs/data/central_data_manager.md`.

---

## 6. Files and where to look

| For | Look at |
|---|---|
| Repo layout | `CLAUDE.md` |
| Engine code | `engine/` |
| Live execution (new pipeline) | `live/rebalance.py`, `live/brokers/` |
| Live execution (legacy, paper still uses this) | `live/_legacy/alpaca_rebalance.py` |
| Tests | `tests/` (15 files, 211 tests, ~6 min full run) |
| Data engine (shared across projects) | `QuantFinance/quantcore/` (CLI `quantcore-ingest`) |
| Data cache (shared) | `/Users/franciscosimao/Documents/QuantFinance/data/DB_<TICKER>_historical_data.db` |
| Central data contract | `docs/data/central_data_manager.md` |
| Alpaca lot-level decision | `research/tax_drift_trigger/findings_alpaca_lot_selection.md` |
| Strategy registry (gitignored) | `strategies.json` |
| Live state files (gitignored) | `live/logs/cadence_*.json`, `live/logs/lots_*.json`, `live/logs/budget_*.json`, `live/logs/run_summary.jsonl`, `live/logs/runs/*.json` |
| Closed research investigations | `research/<topic>/findings.md` |
| Historical research narrative | `docs/internal/research_log.md` |
| Historical completed-item log | `docs/internal/ToDo.md` |

---

## 7. Open questions for the user (next time they're in)

- **Client's Alpaca account label** to use for June 1 — what `--account <X>` should the scripts use?
- **Initial budget** for the client's account — `--budget AMOUNT` or use full equity?
- **First execute date confirmation** — June 1 still on, or has it shifted?
- After D.18 lands: **lot selector default for production reporting** — `fifo` (matches Alpaca reality) or `tax_optimal` (research counterfactual)?
