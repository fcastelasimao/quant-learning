# Step 1 — EDA + Sanity + Edge Prior

Hand-off document for Sonnet. **Read this entire document before writing any code.** Project-wide context and decisions live in `PROJECT_PLAN.md` in the same folder; skim that too if any reference here feels unexplained.

---

## 0. What this step is and is NOT

**Is:** load the two canonical trade logs, validate them, run integrity checks, produce one diagnostic plot ("edge prior"), and write a brief report. Read-only on existing data — no FMP fetches, no modifications to canonical CSVs.

**Is NOT:** the backtest. No equity curves, no Sharpe, no metrics, no sleeve logic, no FMP joining. Those are Step 2.

---

## 1. Project briefing in two paragraphs

We have backtest trade logs for a TQQQ/SQQQ Alpaca strategy. The user wants to add a leverage-sleeve overlay: for each trade where `RSI_entry < threshold`, deploy an extra 30 % of portfolio value into the same trade. Over a sweep of entry-RSI thresholds, we'll compute risk/return metrics and compare. Before any of that, we need to verify the canonical data is clean and — crucially — see whether `RSI_entry` actually has any predictive relationship with `pnl_pct` at all. **If RSI has no edge, no threshold sweep can find one, and that finding is itself the deliverable for this step.**

The canonical CSVs were each derived from a single backtest run (the run that had the largest trade count covering the analysis window). The full raw files contain many additional runs — used in this step only to produce a disagreement diagnostic comparing the canonical spine against other runs that touch the same trades.

---

## 2. Working directory

```
/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/RSI_tests/
```

Use absolute paths in code. Create all outputs under `step1_outputs/`.

---

## 3. Environment

Run all Python in the `quant` conda env:

```bash
conda activate quant
# or, equivalently:
~/opt/anaconda3/envs/quant/bin/python step1_eda.py
```

Required packages (assume available; if not, error clearly): `pandas`, `numpy`, `matplotlib`, `scipy.stats`. Use the non-interactive `Agg` backend for matplotlib (`matplotlib.use("Agg")` before `pyplot` import) so the script runs headless.

---

## 4. Inputs

### Canonical trade logs (primary)
- `TRADES_TQQQ_canonical.csv` — 1627 rows, single run `2026-05-11 16:23:17`, span 2020-01-02 13:30 → 2026-05-08 14:00.
- `TRADES_SQQQ_canonical.csv` — 1572 rows, single run `2026-05-05 15:18:00`, span 2020-01-03 09:45 → 2026-05-01 12:30.

### Raw trade logs (for disagreement diagnostic only)
- `TRADES_TQQQ_backtest_alpaca.csv` — 8829 rows × 51 distinct `run_started_at` values.
- `TRADES_SQQQ_backtest_alpaca.csv` — 24914 rows × 110 distinct `run_started_at` values.

### Schema highlights (51 columns total; only the relevant ones listed)

| Column | Meaning |
|---|---|
| `trade_id` | per-run sequential id; not stable across runs |
| `run_started_at` | wall-clock timestamp when the backtest run started |
| `mode` | always `backtest` |
| `symbol` | `TQQQ` or `SQQQ` |
| `entry_time` | bar timestamp of entry decision (ET) |
| `decision_price` | mid-price at decision |
| `avg_order_price` | actual fill price (slippage already applied) |
| `qty` | shares — per-run, depends on capital and sizing |
| `capital_before`, `capital_after`, `capital_end`, `cumulative_profit`, `pnl` | per-run accounting, **drop for canonical analysis** |
| `RSI_entry` | RSI value at entry — the gating variable for this project |
| `BBP_entry`, `MA*`, `volume_*` | features, optional keep |
| `regime_entry` | strategy's regime classification (e.g., `chop_highvol`) |
| `hour_of_entry` | hour bucket |
| `exit_time` | bar timestamp of exit |
| `exit_decision_price`, `exit_avg_order_price` | exit prices (slippage already applied) |
| `exit_reason` | e.g., `TRAIL_STOP` |
| `pnl_pct` | `(exit_avg_order_price / avg_order_price) − 1` — per-share return |

The CSVs are NOT sorted by `entry_time` — they're sorted by `trade_id` from the source run. Sort in memory before any time-ordered check.

---

## 5. Deliverable

Single Python script `step1_eda.py` that, when run from the working directory, produces every file listed in §6. The script must be re-runnable without manual cleanup. It must print a one-line progress message per major section and exit cleanly on success.

**Do NOT mutate the canonical or raw CSVs.** All operations in memory or to new files under `step1_outputs/`.

---

## 6. Outputs (write all into `step1_outputs/`)

| File | Format | Purpose |
|---|---|---|
| `eda_report.md` | markdown | human-readable summary of every check |
| `prior_pnl_vs_rsi.png` | PNG, 150 DPI | THE headline plot of this step |
| `prior_pnl_vs_rsi_binned.csv` | CSV | data behind the plot |
| `disagreements_TQQQ.csv` | CSV | rows where non-canonical runs disagree with canonical on price/RSI |
| `disagreements_SQQQ.csv` | CSV | same for SQQQ |
| `overlap_report.csv` | CSV | cross-symbol time overlaps (TQQQ trade open while SQQQ trade open) |

---

## 7. Tasks in execution order

### 7.0 Setup
- Create `step1_outputs/` if it doesn't exist.
- Configure matplotlib for `Agg` backend.

### 7.1 Quick FMP DB inventory (sanity, do not block on this)
For each of `DB_TQQQ_historical_data.db`, `DB_SQQQ_historical_data.db`, `DB_^IRX_historical_data.db`:
- Connect via `sqlite3`.
- Count rows in `candles_1d` where `et_datetime BETWEEN '2020-01-03 00:00:00' AND '2026-05-08 23:59:59'`.
- For TQQQ/SQQQ, count rows where `adj_close IS NULL` in that same window (expect 0).
- Report counts in `eda_report.md` under a "Price/rate data availability" section.

### 7.2 Load canonical files
For each canonical CSV:
- Parse `entry_time` and `exit_time` as datetimes.
- Sort by `entry_time` ascending.
- Assert single distinct `run_started_at` and single distinct `symbol`. Fail loudly if not.
- Assert `mode == 'backtest'` everywhere.

### 7.3 Null check on critical columns
Columns: `RSI_entry`, `decision_price`, `avg_order_price`, `exit_decision_price`, `exit_avg_order_price`, `pnl_pct`, `entry_time`, `exit_time`.

For each, count null/NaN. Report counts in `eda_report.md`. If any of these are non-zero, list the first 5 row indices.

### 7.4 Timestamp validation
For each canonical file, count and report:
- Rows where `exit_time < entry_time` (should be 0).
- Rows where `entry_time` is outside NYSE regular hours (09:30–16:00 ET, weekdays). Same for `exit_time`. (Note: timestamps are already in ET, no conversion needed.)
- Rows where `entry_time` minute is not in {00, 15, 30, 45} (15-min bar boundary). Same for `exit_time`.

List the first 5 violations of each.

### 7.5 pnl_pct sanity check
Compute `recomputed = exit_avg_order_price / avg_order_price - 1`.
Compute `diff = abs(recomputed - pnl_pct)`.

Report:
- Count of rows with `diff > 0.001` (10 bps tolerance).
- `diff.max()`, `diff.mean()`, `diff.quantile(0.99)`.
- First 5 rows where `diff > 0.001`, showing both values side by side.

If the bulk of rows match within 1 bp, no further action. If meaningful divergence, just report — don't drop.

### 7.6 Outlier flags (report, do not remove)
- Trades with `abs(pnl_pct) > 0.50`.
- Trades with `(exit_time - entry_time).total_seconds() / 86400 > 5`.

Counts + first 5 rows in `eda_report.md`.

### 7.7 Self-overlap within symbol
After sorting by `entry_time`, walk consecutive rows. Count cases where `entry_time[i+1] < exit_time[i]`. If non-zero, also compute:
- Max overlap depth (how many trades open simultaneously, sweeping events).
- Longest contiguous overlap span.

Report in `eda_report.md`. If non-zero, this affects how Step 2 must handle position sizing.

### 7.8 Cross-symbol overlap report
- Merge TQQQ and SQQQ canonical frames keeping `(symbol, entry_time, exit_time, trade_id)`.
- For each pair (TQQQ_trade, SQQQ_trade), check whether the intervals `[entry_time, exit_time]` overlap.
- For overlapping pairs, write to `overlap_report.csv` with columns:
  `tqqq_trade_id, tqqq_entry, tqqq_exit, sqqq_trade_id, sqqq_entry, sqqq_exit, overlap_seconds`.

Performance note: naive O(N×M) is ~2.5M comparisons — fine in pandas/numpy. If it's slow, sort SQQQ by `entry_time` and binary-search the candidate window.

Summary stats in `eda_report.md`:
- Number of overlapping pairs.
- Fraction of TQQQ trades that overlap at least one SQQQ trade.
- Total overlapping wall-time (sum of `overlap_seconds`).
- Max number of simultaneously-open positions across both symbols.

### 7.9 Disagreement diagnostic
For each symbol:
- Load the raw file (`TRADES_<symbol>_backtest_alpaca.csv`).
- Build a key `(entry_time, exit_time)` from the canonical file → canonical price/RSI fingerprint.
- For each row in the raw file with `run_started_at != canonical_run`, check if its `(entry_time, exit_time)` is in the canonical set.
- If yes, compare its `avg_order_price`, `exit_avg_order_price`, `RSI_entry` to the canonical fingerprint. If ANY of them differ by more than `1e-6`, record the row.
- Write `disagreements_<symbol>.csv` with columns:
  `entry_time, exit_time, canonical_avg, canonical_exit_avg, canonical_RSI, other_run, other_run_trade_count, other_avg, other_exit_avg, other_RSI, diff_avg, diff_exit_avg, diff_RSI`.

Include `other_run_trade_count` so the user can filter out tiny debug runs in analysis.

Summary in `eda_report.md`: count of disagreements per symbol, max abs diff per field, list of `other_run` values that disagreed.

### 7.10 Distribution profiling
For each symbol, summarize in `eda_report.md`:
- `RSI_entry`: mean, median, std, min, max, IQR.
- `pnl_pct`: mean, median, std, win rate (% > 0), skew, excess kurtosis.
- Trade duration in calendar days: median, p90, max.
- `exit_reason`: value_counts.
- `regime_entry`: value_counts.
- `hour_of_entry`: value_counts.

### 7.11 Edge prior — THE most important deliverable

**Goal:** answer whether `RSI_entry` carries any signal about `pnl_pct`.

Combine the two symbols' canonical trades into one frame (keep a `symbol` column).

**Binned table** (`prior_pnl_vs_rsi_binned.csv`):
- Bin `RSI_entry` into 5-point bins: `[0, 5), [5, 10), ..., [95, 100]` (20 bins).
- Group by `(symbol, bin)`. For each group, compute:
  - `n_trades`, `mean_pnl_pct`, `median_pnl_pct`, `std_pnl_pct`, `win_rate`, `sem = std / sqrt(n)`.
- Save table.

**Plot** (`prior_pnl_vs_rsi.png`):
- Figure: 2 subplots side by side, shared y-axis. Left = TQQQ, right = SQQQ.
- Each subplot:
  - Scatter `pnl_pct` vs `RSI_entry`, `alpha=0.25`, `s=10`.
  - Overlay: line at `mean_pnl_pct` per bin, with error bars `±1 sem`.
  - Overlay: linear regression `pnl_pct ~ RSI_entry` using `scipy.stats.linregress`. Plot the fit line over `RSI_entry` range.
  - Title: `"{symbol}: slope={slope:.5f}, p={pvalue:.3f}, R²={rsq:.3f}"`.
  - Y-axis: clip to `[-0.10, 0.10]`. Annotate count of clipped points in a corner text.
  - X-axis: `[0, 100]`.
  - Horizontal reference line at `y=0`.
- DPI 150, save as PNG.

### 7.12 Write `eda_report.md`
Markdown file structured roughly:

```
# Step 1 — EDA Report

## Headline
[Single sentence: did we find an edge or not? Slope + p-value for each symbol.]

## Data availability
[Counts from §7.1]

## Canonical trade logs
[Row counts, span, distinct run]

## Validation results
- Nulls in critical columns: …
- Timestamp validity: …
- pnl_pct math check: …
- Outliers: …

## Trade structure
- Self-overlap (within symbol): …
- Cross-symbol overlap: …
- Disagreements vs other runs: …

## Distributions
[Tables from §7.10]

## Edge prior — pnl_pct vs RSI_entry
[Reproduce binned table compactly; reference the PNG; interpret the slope.]

## Conclusions
[2–4 bullets. Honest read of whether the threshold sweep is worth running, what to watch out for in Step 2, and any anomalies that need investigation.]
```

---

## 8. Acceptance criteria

- Script runs to completion with no uncaught exceptions.
- All 6 files listed in §6 exist after running.
- `eda_report.md` includes a numeric summary for every check.
- Plot has both subplots, regression-line annotations, binned-mean overlays, and is at least 150 DPI.
- For each "report violations" check: if zero, explicitly state "0 violations." Don't omit silently.

---

## 9. Hard constraints / what NOT to do

- Do NOT build the backtest engine. No sleeve logic, no equity curves, no Sharpe/CAGR/MaxDD calculations.
- Do NOT modify the canonical or raw CSVs.
- Do NOT drop any rows based on validation. Just count and report.
- Do NOT pull more data from FMP. All needed data is already in the directory.
- Do NOT compute benchmarks. B&H curves are a Step 2 concern.
- Do NOT generate intermediate "cleaned" CSVs. The canonical files are already the canonical inputs.
- Do NOT add fees, dividends, or slippage modeling on top of what's in `pnl_pct`. Slippage is already baked into `avg_order_price`.

---

## 10. When the user reviews

The user will read `eda_report.md` and look at `prior_pnl_vs_rsi.png`. The two questions they'll be asking are:

1. **Does the data look healthy?** (Few or zero validation violations; sensible distributions.)
2. **Does RSI carry signal?** (Non-trivial slope on the prior plot, or at least monotone binned means.)

If the answer to #2 is "no edge visible," that's important and the user may want to revisit the strategy before pursuing the sweep. Note this honestly in the report — do not soften.

If anything in §7 unexpectedly fails or produces a result that looks substantively wrong (e.g., 30 % of trades have invalid timestamps), STOP and surface it in `eda_report.md` rather than trying to fix it. The user wants to see the truth of the data, not a tidied version.
