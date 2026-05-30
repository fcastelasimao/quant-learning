# Session handoff — All Weather

**Status:** active. **Last updated:** 2026-05-30.
**Previous session covered:** repo cleanup (Phase 1A), preview enrichment for June 1 launch, Alpaca lot-level capability check, central data manager dividend support, JEPQ added to central store.

---

## 0. If you are a new Claude Code session, read this first

You've just been spun up. Don't wander the repo for 20 minutes trying to figure out what's going on.

**Read order:**

1. **This file end-to-end** (you're here). It contains live state, the active plan, decisions taken, and conventions.
2. `CLAUDE.md` (project root) — repo layout, key constraints, environment.
3. The two **decisions docs**: `docs/research/alpaca_lot_selection.md` and `docs/data/central_data_manager.md`. Don't reopen these debates without reading them first.
4. `docs/internal/research_log.md` — historical narrative if you need backstory on how we got here.
5. `docs/internal/ToDo.md` is a **historical changelog of completed items**, retained as a record but no longer the active worklist. **Use this file for active work**, not ToDo.md.

**Working with this user:**

- Push back on scope creep. The user explicitly asked for less infrastructure, not more. If a plan you're drafting includes Fargate / ECR / Terraform / CI/CD pipelines for a single-strategy single-user system, you're probably wrong.
- One open question at a time. Close it before opening the next.
- Defer; don't pre-build. The user said: "TQQQ/SQQQ working style" — small focused scripts, decisions before infrastructure.
- The user is Portugal-based but the strategy is modeled as US-taxable for research realism. Don't second-guess this.
- Always use the conda env: `conda run -n allweather python ...` or direct path `/Users/franciscosimao/opt/anaconda3/envs/allweather/bin/python`.

---

## 1. Current verified state (2026-05-30)

| Item | State |
|---|---|
| Test suite | **211 passing**, 2 deselected, 1 deprecation warning (websockets, harmless) |
| Live execution | Still on legacy `live/_legacy/alpaca_rebalance.py` for the paper accounts in production |
| New pipeline | `live/rebalance.py` ready; first execute pending review |
| Central data store | Up to date through 2026-05-30 for SPY, QQQ, TLT, TIP, GLD, GLDM, GSG, JEPQ daily candles |
| Dividend store (new) | SPY 134 rows back to 1993-03-19; QQQ 87; TLT 285; TIP 198; JEPQ 48; GLD/GLDM/GSG empty (no cash distributions — correct) |
| JEPQ price coverage | 2022-05-04 → 2026-05-29 (1021 daily rows) — newly added to central store |
| `engine/data.py` | Has `fetch_prices()` (via central FMP SQLite) AND `fetch_dividends()` (new) |

---

## 2. Active plan — execute in this order

**Do NOT start a downstream item before its upstream is verified.**

### Section A — Pre-launch (June 1 on boss's Alpaca account)

| ID | What | Status |
|---|---|---|
| A.1 | Configure boss's account env vars in `QuantFinance/api_keys.env` (`BROKER_ALPACA_<LABEL>_KEY` + `_SECRET`); confirm `strategies.json` allocation | **HUMAN** — pending |
| A.2 | `make healthcheck BROKER=alpaca ACCOUNT=<boss> MODE=--live` | **HUMAN** — pending |
| A.3 | `make rebalance-dry-execute BROKER=alpaca ACCOUNT=<boss> MODE=--live`; save and review the `logs/runs/<timestamp>.json` summary | **HUMAN** — pending |
| A.4 | Enriched preview output (cadence + budget + lots inline) | **DONE 2026-05-30** |
| A.5 | `make rebalance-new-execute BROKER=alpaca ACCOUNT=<boss> MODE=--live` | **HUMAN** — pending |
| A.6 | Verify post-execution: `logs/runs/<latest>.json`, `logs/cadence_*.json`, `logs/lots_*.json`, `logs/budget_*.json` match Alpaca reality | **HUMAN** — pending |
| A.7 | Verified Alpaca lot-level support — confirmed absent (FIFO at broker, no order-level lot selection) | **DONE 2026-05-30** — see `docs/research/alpaca_lot_selection.md` |

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
| D.12 | `engine/tax_rates_us.yaml` — year-keyed schedule (ST top, LT top, qualified-div top, NIIT) from Tax Foundation. Bisect-by-date lookup. | TODO |
| D.13 | `engine/tax.py` — `TaxRegime`, `TaxSchedule`, `compute_tax_on_event(event, lot_ledger, regime)`. Per-ETF distribution classification: SPY/QQQ qualified, TLT/TIP ordinary (interest), GLD collectibles-rate on sale (28% top), GSG partnership/§1256 (60/40 LT/ST mark-to-market). | TODO |
| D.14 | `engine/lot_ledger.py` — pluggable selectors. Default `FIFO` (matches Alpaca broker reality). Optional `tax_optimal` (LT-first then highest basis — **research counterfactual only on Alpaca**, see `docs/research/alpaca_lot_selection.md`). Optional `HIFO`. Mirror the FIFO shape from `live/lots.py` so backtest and live agree. | TODO |
| D.15 | Drift trigger in `engine/backtest.py` — add `rebalance_policy: RebalancePolicy` parameter. Default `monthly_unconditional` (current behavior, zero regression). Other modes: `drift_relative(pct)`, `drift_absolute(pp)`, `monthly_check_then_drift(pct)`. Move threshold logic from `research/rebalance_thresholds.py` into the engine; have research script call into engine. | TODO |
| D.16 | Tax hook in rebalance loop — at each event, selector picks lots → realized gain ST/LT → year-effective rate → debit portfolio value. Dividend cashflows taxed at ex-date. | TODO |
| D.17 | New artifacts in `production_validation` bundle: `rebalance_events.csv`, `tax_summary.csv`. Extended `manifest.json` with `tax_regime` and `rebalance_policy` blocks. | TODO |
| D.18 | `research/tax_threshold_sweep.py` — sweep drift threshold (relative: 10/15/20/25/30/40%; absolute: 1/2/3/5pp) × tax regime (US, none) × lot selector (FIFO, tax_optimal). Emit `threshold_sweep_summary.csv`. **Kill criterion:** if best drift+tax beats monthly+tax on Calmar by ≥5% on ≥2 of 3 OOS windows, propose new production policy; else tax model stays research-only. | TODO |
| D.19 | Docs: `docs/research/tax_model.md`, `docs/research/drift_trigger.md`, `docs/research/tax_threshold_sweep.md`. TQQQ/SQQQ style: what was tested, methodology, results, decision. | TODO |

### Section E — Marimo dashboard (no recomputation in notebooks)

| ID | What | Status |
|---|---|---|
| E.20 | JEPQ as benchmark line in `production_validation` bundle so it appears in `strategy_comparison.py` automatically | TODO |
| E.21 | Rebalance markers on growth chart — `axvline` per event sized by trade notional; toggle via `mo.ui.checkbox` | TODO |
| E.22 | Tax cost panel — annual stacked bars (ST gain tax / LT gain tax / dividend tax) + cumulative tax-paid line | TODO |
| E.23 | Regime comparison panel — same strategy side-by-side under ISA (zero) vs US-taxable | TODO |
| E.24 | Threshold sweep heatmap — Calmar vs (drift threshold, lot selector) for chosen tax regime | TODO |
| E.25 | `docs/notebooks/strategy_comparison.md` describing each cell and its artifact | TODO |

### Section F — Decision gate

| ID | What | Status |
|---|---|---|
| F.26 | Read D.18 verdict. If drift+tax wins: update `strategies.json` with `rebalance_policy` block; update `live/rebalance.py` to use drift trigger (currently 31-day cadence only); update `failed_strategies/weekly_rebalance/README.md` with new verdict; one paper-month of agreement before flipping live policy. | TODO |

### Section G — Shadow comparison upgrade

| ID | What | Status |
|---|---|---|
| G.27 | `backtest_shadow.py` consumes both legacy `performance_tracking_*.csv` and new `logs/run_summary.jsonl` / `logs/runs/*.json` | TODO |
| G.28 | Plot: cumulative actual vs simulated with deviation bands | TODO |
| G.29 | Plot: per-rebalance fill-price vs simulated-price diff (catches Alpaca slippage) | TODO |
| G.30 | `shadow_summary.csv` to bundle for marimo | TODO |
| G.31 | Write `logs/WARNINGS.log` when MAE > threshold — no Slack, just a file you grep | TODO |
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
4. **Alpaca lot-level support: confirmed absent.** Verified against `alpaca-py 0.43.2` and Alpaca docs. Order placement has no `lot_id` / `cost_basis_method` field. Tax-optimal selector in the backtest is a **research counterfactual only**. Live execution stays FIFO. (`docs/research/alpaca_lot_selection.md`)
5. **Rebalance trigger semantic: drift-based, not pure calendar.** Production engine currently still calendar-monthly; the closed `failed_strategies/weekly_rebalance/` verdict was made under transaction-cost-only modelling and **may be reopened** under realistic US tax + tax-optimal lot selection.
6. **Live ticker mapping: only `GLD → GLDM`.** Confirmed against `strategies.json` and the project memory. Don't introduce other substitutions.
7. **No new `--summary` flag for `live/rebalance.py`.** The existing preview/dry-execute output was enriched in place. One review surface.
8. **`engine/data.py` already consumed from central store for the FMP path.** C.10 reduced to adding a `fetch_dividends()` reader. No duplicate fetch logic.
9. **`ToDo.md` is retired in favor of this file.** Kept as a historical changelog of completed items (41 entries through 2026-05-27). All active work tracking happens here.
10. **JEPQ added to central data store.** Price history 2022-05-04 → present, dividend history 2022-06-01 → present. Use `engine.data.fetch_prices_from_fmp_db(['JEPQ'], ...)` and `engine.data.fetch_dividends(['JEPQ'], ...)`.

---

## 4. Operating principles — apply on every change

- **One open question at a time.** Close it before opening the next.
- **Every new research output → CSV/JSON artifact** in `results/<task>/<timestamp>_<strategy>/`. Marimo just reads. Never recompute in the notebook.
- **New investigation → `docs/research/<topic>.md`** documenting what was tested, results, decision. TQQQ/SQQQ project style.
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
python /Users/franciscosimao/Documents/QuantFinance/data_manager.py \
    --symbols SPY QQQ TLT TIP GLD GLDM GSG JEPQ --with-dividends

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

## 5b. Important: `data_manager.py` is NOT under git

`/Users/franciscosimao/Documents/QuantFinance/data_manager.py` lives one
directory above the `quant-learning` git repo and is **not version-controlled**.
This matters because:

- The C.9 changes (dividend support, `--with-dividends` / `--dividends-only`
  flags, JEPQ in `DEFAULT_SYMBOLS`) live only on this machine's disk.
- A future Claude session checking out a fresh clone of `quant-learning` will
  see All Weather's `engine/data.py` referring to a `dividends` table that
  may not exist in their local `data/` SQLite cache.
- The `docs/data/central_data_manager.md` contract describes the expected
  shape; the actual implementation is unprotected.

**If you need to recover the dividend implementation**, the canonical
implementation is described in `docs/data/central_data_manager.md` (schema
section), and the working version is in
`/Users/franciscosimao/Documents/QuantFinance/data_manager.py` on the user's
machine as of 2026-05-30. Consider whether to:

- (a) leave it as a local script (current state — adequate for single-user)
- (b) move `data_manager.py` *into* the `quant-learning` repo (e.g.
  `quant-learning/shared/data_manager.py`) so it gets versioned alongside
  the consuming projects (preferred long-term, but a non-trivial move)
- (c) just put it in a personal `~/scripts` git that the user controls

The user has not made this decision yet. Don't volunteer to do it without
asking — moving the file would force-update every consuming project's import
paths.

---

## 6. Files and where to look

| For | Look at |
|---|---|
| Repo layout | `CLAUDE.md` |
| Engine code | `engine/` |
| Live execution (new pipeline) | `live/rebalance.py`, `live/brokers/` |
| Live execution (legacy, paper still uses this) | `live/_legacy/alpaca_rebalance.py` |
| Tests | `tests/` (15 files, 211 tests, ~6 min full run) |
| Data fetcher (shared across projects) | `/Users/franciscosimao/Documents/QuantFinance/data_manager.py` |
| Data cache (shared) | `/Users/franciscosimao/Documents/QuantFinance/data/DB_<TICKER>_historical_data.db` |
| Central data contract | `docs/data/central_data_manager.md` |
| Alpaca lot-level decision | `docs/research/alpaca_lot_selection.md` |
| Strategy registry (gitignored) | `strategies.json` |
| Live state files (gitignored) | `logs/cadence_*.json`, `logs/lots_*.json`, `logs/budget_*.json`, `logs/run_summary.jsonl`, `logs/runs/*.json` |
| Closed research investigations | `failed_strategies/<topic>/README.md` |
| Historical research narrative | `docs/internal/research_log.md` |
| Historical completed-item log | `docs/internal/ToDo.md` |

---

## 7. Open questions for the user (next time they're in)

- **Boss's Alpaca account label** to use for June 1 — what `--account <X>` should the scripts use?
- **Initial budget** for the boss's account — `--budget AMOUNT` or use full equity?
- **First execute date confirmation** — June 1 still on, or has it shifted?
- After D.18 lands: **lot selector default for production reporting** — `fifo` (matches Alpaca reality) or `tax_optimal` (research counterfactual)?
