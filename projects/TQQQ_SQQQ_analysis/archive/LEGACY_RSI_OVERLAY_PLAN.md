# LEGACY — RSI Leverage Overlay Plan (paused)

This is the original RSI leverage-overlay project spec. **Paused** as of 2026-05-28 in favor of the full-history pattern discovery work (see `PROJECT_PLAN.md`). Kept here for reference. The Step handoff docs (`STEP1_PLAN.md`, `STEP2_PLAN.md`, `STEP2_5_PLAN.md`, `STEP2_6_PLAN.md`) and the scripts (`step1_eda.py`, `step2_backtest.py`, `step2_5.py`) all belong to this legacy line of work.

---

# RSI Leverage Overlay — Project Plan

Reference document. Captures the full pipeline from data to final figures and the decisions already locked in. **Step 1 has its own detailed handoff doc: `STEP1_PLAN.md`.**

---

## 1. Goal

Take an existing TQQQ/SQQQ Alpaca backtest's trade log and add an RSI-gated leverage sleeve on top. For each trade where `RSI_entry` is below a configurable threshold, deploy an additional **30 % of current portfolio value** into the same trade. Compute risk/return metrics across a sweep of entry-RSI thresholds, compare with baseline (no sleeve), and produce visualizations.

v1 sweeps **entry-RSI only**. v2 (later) adds an **exit-RSI** axis using FMP intraday bars, producing 2-D heatmaps.

---

## 2. Working directory

`/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/RSI_tests/`

---

## 3. Data inventory

### Trade logs

| File | Rows | Span | Source run |
|---|---|---|---|
| `TRADES_TQQQ_canonical.csv` | 1627 | 2020-01-02 13:30 → 2026-05-08 14:00 | `2026-05-11 16:23:17` |
| `TRADES_SQQQ_canonical.csv` | 1572 | 2020-01-03 09:45 → 2026-05-01 12:30 | `2026-05-05 15:18:00` |

Originals retained (used only for the disagreement diagnostic in Step 1):
- `TRADES_TQQQ_backtest_alpaca.csv` — 8829 rows × 51 runs
- `TRADES_SQQQ_backtest_alpaca.csv` — 24914 rows × 110 runs

**Common analysis window:** 2020-01-03 → 2026-05-01 (~6.3 years, ~1590 trading days). TQQQ extends a further week (to 2026-05-08); for the combined-portfolio run we will truncate TQQQ to 2026-05-01.

### Price / rate data (FMP daily, SQLite)

| DB file | Table | Range | Key column |
|---|---|---|---|
| `DB_TQQQ_historical_data.db` | `candles_1d` | 2010-02-11 → 2026-05-19 | `adj_close` (split + dividend adjusted) |
| `DB_SQQQ_historical_data.db` | `candles_1d` | 2010-02-11 → 2026-05-19 | `adj_close` |
| `DB_^IRX_historical_data.db` | `candles_1d` | 2006-05-15 → 2026-05-19 | `close` = 13-wk T-bill annualized yield in % |

Schema notes:
- **TQQQ/SQQQ `close` is already split-adjusted by FMP.** `adj_close` adds dividend reinvestment. Use `adj_close` for B&H benchmark equity curves.
- **^IRX `close` is the annualized yield in percent** (e.g., 3.60 means 3.60 %). `adj_close` is mostly NULL by design (yield index, no dividends/splits). Use `close`.
- ^IRX has 2 fewer trading days than TQQQ/SQQQ in our window (Treasury holidays). Forward-fill on join.

---

## 4. Sleeve mechanics

For each trade in a canonical log:

1. If `RSI_entry < entry_threshold`, deploy a leveraged sleeve **in the same direction and at the same prices** as the underlying trade.
2. **Sleeve notional** = `0.30 × portfolio_value_at_entry`. `portfolio_value_at_entry` = strategy equity right before this trade is opened, chronologically compounded from prior trade outcomes + prior sleeve P/L.
3. **Gross sleeve P/L** = `0.30 × portfolio_value_at_entry × (pnl_pct / 100)`. **NOTE:** `pnl_pct` in the canonical CSVs is in **percentage form** (e.g., `2.33` means 2.33 %), not fractional form. The `/ 100` is mandatory. Verified in Step 1 EDA (residuals collapse to ~6 bps max after %-scaling).
4. **Borrow cost** = `sleeve_notional × annual_rate × days_held / 365`, where `annual_rate = ^IRX_close_pct_on_entry / 100 + tier_spread`. `days_held` = calendar days between `entry_time` and `exit_time`.
5. **Net sleeve P/L** = `gross_sleeve_pnl − borrow_cost`.
6. **Sleeve exits when the underlying trade exits.** Coded as a pluggable `Sleeve` interface so an independent RSI-based exit can drop in later (v2).
7. Underlying-trade P/L is unchanged (already implicit in `pnl_pct × $5,000 × position_size`).

### Tier-spread schedule (Alpaca-style, configurable)

| Equity tier | Spread over ^IRX |
|---|---|
| $0 – $25,000 | 8.0 % |
| $25,000 – $50,000 | 7.0 % |
| $50,000 – $100,000 | 6.0 % |
| $100,000 – $250,000 | 5.0 % |
| $250,000 – $500,000 | 4.5 % |
| $500,000 + | 4.0 % |

Treat as approximate; expose as a config dict and revisit once the user has confirmed actual Alpaca tiers for the historical period.

### What slippage is NOT

Slippage is **already baked** into the trade log via the `decision_price → avg_order_price` gap and `exit_decision_price → exit_avg_order_price` gap. `pnl_pct` is computed off `avg_order_price`, so do NOT add additional slippage modeling. Same fill assumption is used for the sleeve (i.e., the sleeve gets the same per-share return as the underlying trade).

---

## 5. Modes per sweep

Three modes are envisioned for the full project. **Step 2 implements only TQQQ standalone**; the other two are deferred.

- **TQQQ standalone (Step 2 scope)** — single portfolio starting at **$10,000** (the TQQQ canonical's native starting capital), only TQQQ trades.
- **SQQQ standalone (deferred)** — would use a single portfolio starting at ~$5,000 (SQQQ canonical's native), only SQQQ trades.
- **Combined chronological (deferred)** — single shared portfolio, both symbols merged on `entry_time`. Requires resolving the allocation-rule open item in §10 (each canonical strategy deploys 95–99 % of its wallet in isolation, so naive merging would exceed 100 % gross exposure).

---

## 6. Entry-RSI window sweep (2-D grid)

The sleeve fires for a trade when `RSI_entry ∈ [low, high)` (low inclusive, high exclusive — matches Step 1's EDA binning convention).

**Grid (21 cells in the upper triangle):**
- `low ∈ {35, 40, 45, 50, 55, 60}`
- `high ∈ {45, 50, 55, 60, 65, 70}`
- Constraint: `high > low`

The 21 cells are: (35,45), (35,50), (35,55), (35,60), (35,65), (35,70), (40,50), (40,55), (40,60), (40,65), (40,70), (45,55), (45,60), (45,65), (45,70), (50,60), (50,65), (50,70), (55,65), (55,70), (60,70).

Plus a **baseline scenario** (sleeve never fires) as the 22nd row in metrics output, for delta computations.

The 1-D threshold rule `RSI < X` is a special case (`low=0`, `high=X`); the window grid strictly generalizes it. Range chosen because the strategy only entered trades when `RSI_entry ∈ [35, 72]` (Step 1 EDA), so windows outside that range carry no information.

Future (v2): `exit_threshold` axis using FMP 15-min bars. Adds a third dimension to the metrics CSV; presentation code becomes 3-D-aware.

**Step 2.5 additions to the sweep:** Step 2's results plus targeted EDA showed the windowed approach mostly traces an exposure dial. Step 2.5 (see `STEP2_5_PLAN.md`) adds 9 reference scenarios to make the windowed grid interpretable:
- **6 always-on baselines** at sleeve sizes `{5, 10, 15, 20, 25, 30}%` — the "no RSI selection, fixed leverage" comparison.
- **2 targeted-bin sleeves** — `targeted_55_60` (fires when RSI ∈ [55, 60), the workhorse bin) and `targeted_55_575` (fires when RSI ∈ [55, 57.5), the fine workhorse). 30 % sleeve size.
- **1 skip-dead-zone sleeve** — fires when RSI ∈ [40, 60) ∪ [65, 70), dodging the unproductive [60, 65) zone. 30 % sleeve size.

After Step 2.5 the total scenario count is **32** (1 baseline + 21 windows + 6 always-on + 3 targeted + 1 benchmark).

---

## 7. Project steps

### Step 1 — EDA + sanity + edge prior  ✏️ see `STEP1_PLAN.md`

Validate the canonical CSVs, run integrity checks, produce the "edge prior" plot of `pnl_pct` vs `RSI_entry`. No backtest. Output: `step1_outputs/`.

### Step 2 — Backtest engine (TQQQ-only for now)  ✏️ see `STEP2_PLAN.md` (implemented), follow-up in `STEP2_5_PLAN.md`

Pure computation. Reads `TRADES_TQQQ_canonical.csv` + `DB_TQQQ_historical_data.db` (for B&H benchmark) + `DB_^IRX_historical_data.db` (for borrow rate). Writes results to `runs/`.

Outputs:
- `runs/metrics.csv` — one row per `(low, high)` cell + baseline. Columns: every metric in §8 plus identifiers (`scenario`, `low`, `high`).
- `runs/equity_<scenario>.csv` — per-trade equity walk, one row per trade with `trade_id, entry_time, exit_time, RSI_entry, sleeve_triggered, baseline_pnl_dollars, sleeve_gross_pnl, sleeve_borrow_cost, sleeve_net_pnl, equity_before, equity_after, exit_reason`.

Architecture requirements:
- Pluggable `Sleeve` class with `should_enter(trade) → bool` and `exit_event(trade, current_time) → bool`. v1 has `WindowEntryRSISleeve(low, high)` and `NoSleeve` (baseline). v2 plugs in `WindowEntryExitRSISleeve` using intraday RSI.
- Configurable: window grid, borrow spread schedule (§4 tier table), starting capital (default $10,000 for TQQQ standalone), 30 % sleeve sizing constant (make this an input even though it's fixed for v1).
- Single CLI entrypoint that runs the full sweep across all 22 scenarios; no interactive use.
- SQQQ standalone and combined-portfolio modes are NOT in Step 2 scope.

### Step 2.6 — Cross-validation + paired-bootstrap diagnostics  ✏️ see `STEP2_6_PLAN.md`

Robustness check on Step 2.5's headline finding that targeted-bin sleeves outperform always-on baselines on Sharpe.

- **Part A — Time-split validation.** Split trades at the calendar midpoint of the window. Recompute the per-bin contribution decomposition and 8 key scenarios on each half. Does the `[55, 60)` workhorse and `[60, 65)` dead-zone structure replicate in both halves?
- **Part B — Paired-bootstrap on Sharpe and CAGR differences.** 8 scenario pairs; stationary block bootstrap (block length 10, 5000 iterations) on matched daily-return indices. Output: CIs on the *difference* of metrics, not the per-scenario CIs already in `metrics.csv`. The headline pair is `targeted_55_575` vs `always_on_30pct`.
- **Part C — Rolling 12-month Sharpe/CAGR per scenario (optional)**. Confirms the win isn't a single-year artifact.

### Step 2.5 — Targeted scenarios + non-linear EDA  ✏️ see `STEP2_5_PLAN.md`

Two parallel pieces:
- **Part A (Step 1 EDA follow-up)**: quadratic regression, LOWESS, Kruskal-Wallis, Levene tests for variance equality, per-bin contribution decomposition (5-pt and 2.5-pt), counterfactual drop-bin CAGR table, loss-distribution-by-RSI-bin plot. Lives in `step1_outputs/`. Justification in `FINDINGS.md` — Step 1's linear test missed the non-monotone risk-asymmetric pattern.
- **Part B (Step 2 scenario additions)**: extend the `Sleeve` interface with size as a class attribute; add `AlwaysOnSleeve` and `MultiWindowEntryRSISleeve` concrete classes; run 9 additional scenarios (6 always-on at sizes 5–30 %, 2 targeted-bin, 1 skip-dead-zone) and append to `runs/metrics.csv`. Recompute Deflated Sharpe on the expanded 31-strategy grid.

### Step 3 — Presentation

Reads `runs/`. Pure rendering, no computation. Produces:

- **Metric vs threshold lines** — one panel per metric, lines for the three modes plus baseline.
- **Equity-curve overlay** — log scale, baseline + 3-4 representative thresholds per mode.
- **Yearly returns table** — 7 cells (2020, 2021, 2022, 2023, 2024, 2025, 2026 YTD).
- **Monthly heatmap** — year × month grid, 7×12.
- **Top-5 drawdowns table** per (mode, threshold) — peak, trough, recovery, depth, duration.
- **Rolling 1-year Sharpe and rolling 1-year DD lines.**
- **Sub-period breakdown** — 2020-2021 / 2022 (bear) / 2023-2024 (recovery) / 2025-2026.
- **Prior plot** re-rendered for the final report.
- **Benchmark overlay** — B&H TQQQ (primary), B&H SQQQ (secondary).

All figures saved as 150-DPI PNGs in `figures/`.

### Step 4 (later) — Bar-data extension

Pull 15-min TQQQ + SQQQ bars from FMP (the downloader supports it; just change `--intervals` and re-run). Compute intra-trade RSI series. Wire in the exit-RSI axis. Metrics CSV gains an `exit_threshold` column; presentation generates 2-D heatmaps.

---

## 8. Metrics (per mode × threshold)

**Returns:** total return, CAGR, mean monthly return, std monthly return.

**Risk:** Max DD, Max DD duration (calendar days, trading days), Ulcer Index, time underwater %, daily volatility (annualized), VaR-95, CVaR-95.

**Risk-adjusted:** Sharpe (annualized, on daily-resampled equity), Sortino (annualized), Calmar (CAGR / MaxDD), Omega(0).

**Trade-level:** trade count, win rate, profit factor, expectancy per trade, average holding period (days), max consecutive losing streak.

**Sleeve attribution:** sleeve trigger rate (%), sleeve-only P/L isolated, marginal CAGR (= CAGR_with_sleeve − CAGR_baseline), marginal Sharpe.

**Distributional:** skew, excess kurtosis, tail ratio (|p95| / |p5|) of daily returns.

**Robustness:** stationary-block-bootstrap 95 % CI on Sharpe and CAGR (block length ~10); Deflated Sharpe Ratio across the threshold grid (multiple-comparison adjustment).

**Vs benchmark (primary = B&H TQQQ; for SQQQ standalone, secondary = B&H SQQQ):** alpha, beta, Information Ratio.

---

## 9. Decisions log

| # | Question | Decision | Date |
|---|---|---|---|
| 1 | Symbols in Step 2 | **TQQQ standalone only.** SQQQ standalone and combined-portfolio deferred to later phases | 2026-05-19 (revised) |
| 2 | Sleeve return formula | `sleeve_pnl = 0.30 × portfolio_value_at_entry × (pnl_pct / 100)` (pnl_pct is in percentage form, not fraction) | 2026-05-19 (revised after Step 1 EDA) |
| 3 | Sleeve exit (v1) | Exits with underlying trade. Architect for independent exit in v2 | 2026-05-19 |
| 4 | Leverage multiplier | None for v1 (30 % sizing is the lever); reserved as future sweep axis | 2026-05-19 |
| 5 | Borrow cost model | `^IRX_close_pct / 100 + tier_spread`. Tiered Alpaca-style schedule (§4) | 2026-05-19 |
| 6 | Slippage | Already baked into `avg_order_price`; do NOT double-count | 2026-05-19 |
| 7 | Benchmarks | B&H TQQQ for Step 2. (B&H SQQQ deferred with SQQQ standalone mode.) | 2026-05-19 (revised) |
| 8 | Trade-log dedup | Use the two canonical-spine runs (largest single-run trade count) | 2026-05-19 |
| 9 | Disagreement diagnostic | Yes — produced in Step 1 | 2026-05-19 |
| 10 | Combined-portfolio sleeve sizing | Deferred (combined mode not in Step 2) | 2026-05-19 (revised) |
| 11 | Time aggregations | Yearly + monthly. **No weekly** | 2026-05-19 |
| 12 | Sub-period split | 2020-2021 / 2022 / 2023-2024 / 2025-2026 | 2026-05-19 |
| 13 | FMP scope (v1) | Daily TQQQ + SQQQ + ^IRX only. No intraday until v2 | 2026-05-19 |
| 14 | Python env | `quant` conda env at `~/opt/anaconda3/envs/quant` | 2026-05-19 |
| 15 | FMP API key | Eventually env-var `FMP_API_KEY`; currently pasted on line 47 of fetcher | 2026-05-19 |
| 16 | Starting capital (TQQQ standalone) | **$10,000** (TQQQ canonical's native starting capital — discovered in Step 1) | 2026-05-19 (revised after Step 1 EDA) |
| 17 | Sleeve sizing | 30 % of current portfolio value at trade entry | 2026-05-19 |
| 18 | Sleeve trigger rule | **2-D window**: sleeve fires when `RSI_entry ∈ [low, high)`. Sweep grid: 21 upper-triangle cells from `low ∈ {35,40,45,50,55,60}` × `high ∈ {45,50,55,60,65,70}` (§6) | 2026-05-19 (locked after Step 1 EDA) |
| 19 | Step 2 baseline accounting | Use `1 + pnl/capital_before` per trade. Do NOT use `capital_end/capital_before` — the canonical CSV has 186 chain gaps from external cash flows in the original backtest. See `FINDINGS.md` finding 1. | 2026-05-19 (locked after Step 2 review) |
| 20 | Step 2.5 additions | 9 new scenarios in Step 2.5: 6 `always_on_*` at sizes {5,10,15,20,25,30}%, 2 targeted-bin (`targeted_55_60`, `targeted_55_575`), 1 `skip_dead_zone`. Plus Step 1 EDA follow-up with non-linear tests | 2026-05-19 (locked after Step 2 review) |
| 21 | Persistent research log | Create and maintain `FINDINGS.md` as a chronological lab notebook of substantive findings — separate from plan docs | 2026-05-19 |

---

## 10. Open items

- **Confirm exact Alpaca margin tier values** (§4 schedule is approximate).
- ✅ **Self-overlap within a single symbol** — Step 1 confirmed **0 self-overlaps** for both symbols. Step 2's per-symbol portfolio is plain sequential.
- ✅ **`pnl_pct` sanity** — Step 1 confirmed math matches recomputed `exit_avg_order_price / avg_order_price − 1` within ~6 bps after recognizing that `pnl_pct` is in percentage form. See decision 2 above.
- ✅ **Starting capital** — TQQQ standalone uses **$10,000** (Step 1 finding). SQQQ + combined deferred.
- ✅ **Entry-RSI rule** — locked as 2-D window sweep. See decision 18.
- **Deferred to a later phase:** SQQQ standalone backtest, combined-portfolio backtest, combined-portfolio allocation rule. These are not in Step 2 scope.
- **TQQQ `regime_entry` is all NaN** — regime-conditional analysis on TQQQ not possible from canonical data. Step 2 metrics can skip regime breakdown; decision is to live with it for now (Step 1 finding).
- ✅ **Linear-only EDA missed non-linearity** (Step 1) — being remedied in Step 2.5 Part A.
- ✅ **No baseline comparison to always-on leverage** (Step 2) — being remedied in Step 2.5 Part B.
- ⚠️ **`vs_bh_beta` / alpha / IR columns are biased low** due to sparse daily marks in the resampled equity series. Marked approximate; properly resolvable only with intraday bar marks in v2.

---

## 11. Folder structure (target end state)

```
RSI_tests/
├── PROJECT_PLAN.md                              # this file (full spec, decisions log)
├── FINDINGS.md                                  # running research log; current understanding
├── STEP1_PLAN.md                                # Step 1 handoff
│
├── TRADES_TQQQ_canonical.csv                    # input
├── TRADES_SQQQ_canonical.csv                    # input
├── TRADES_TQQQ_backtest_alpaca.csv              # raw (for disagreement diag)
├── TRADES_SQQQ_backtest_alpaca.csv              # raw (for disagreement diag)
│
├── DB_TQQQ_historical_data.db                   # FMP data
├── DB_SQQQ_historical_data.db
├── DB_^IRX_historical_data.db
│
├── /Users/franciscosimao/Documents/QuantFinance/data_manager.py   # shared FMP downloader
│
├── step1_eda.py                                 # Step 1 script
├── step1_outputs/
│   ├── eda_report.md                            # Step 1
│   ├── prior_pnl_vs_rsi.png                     # Step 1
│   ├── prior_pnl_vs_rsi_binned.csv              # Step 1
│   ├── disagreements_TQQQ.csv                   # Step 1
│   ├── disagreements_SQQQ.csv                   # Step 1
│   ├── overlap_report.csv                       # Step 1
│   ├── eda_followup_report.md                   # Step 2.5 Part A
│   ├── contribution_by_rsi_bin.csv              # Step 2.5 Part A
│   ├── contribution_by_rsi_bin.png              # Step 2.5 Part A
│   ├── loss_distribution_by_rsi.png             # Step 2.5 Part A
│   └── polynomial_fit.png                       # Step 2.5 Part A
│
├── STEP2_PLAN.md                                # Step 2 handoff (TQQQ-only)
├── STEP2_5_PLAN.md                              # Step 2.5 handoff (targeted scenarios + non-linear EDA)
├── step2_backtest.py                            # Step 2 script
├── step2_5_backtest.py                          # Step 2.5 Part B script (or extension of step2_backtest.py)
├── runs/                                        # Step 2 + Step 2.5 outputs
│   ├── metrics.csv                              # one row per scenario (32 rows: 1 baseline + 21 windows + 9 new + 1 bh_tqqq)
│   └── equity_<scenario>.csv                    # per-trade equity walks (31 strategy files)
│
├── step3_render.py                              # Step 3 (later)
└── figures/                                     # Step 3 outputs (later)
```
