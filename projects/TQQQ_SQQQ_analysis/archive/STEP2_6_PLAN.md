# Step 2.6 — Cross-validation and Bootstrap Diagnostics

Hand-off document for Sonnet. **Read this entire document plus `FINDINGS.md` before writing code.** Project spec lives in `PROJECT_PLAN.md`. Step 2 / 2.5 implementations live in `step2_backtest.py` and `step2_5.py` — reuse their engine and helpers, do not duplicate.

---

## 0. What this step is and is NOT

**Is:** robustness testing of the headline findings from Step 2.5.

Step 2.5 found that three "targeted" sleeves (`targeted_55_575`, `targeted_55_60`, `skip_dead_zone`) beat both the windowed cells and the always-on baselines on risk-adjusted metrics. Open question: are those differences **real signal** or **artifacts** of the single 6.3-year sample? This step answers it two ways:

- **Part A — Time-split validation.** Split the 1,627 trades into two halves by entry_time. Recompute the contribution decomposition and the headline scenarios on each half independently. If `[55, 60)` is the workhorse and `[60, 65)` is the dead zone in *both* halves, the structure is real. If the structure shifts, it's a single-period artifact.
- **Part B — Paired-bootstrap diagnostics.** Compute bootstrap CIs on the *difference* of Sharpe (and CAGR) between key scenario pairs. The existing CSV has CIs per scenario, but those CIs are wide because daily returns are noisy — even when a scenario *consistently* beats another. Paired bootstrap on the difference uses the correlation between scenarios' returns and gives a tighter, more honest CI.
- **Part C (optional) — Rolling-window stability.** 12-month rolling Sharpe / CAGR for key scenarios. Confirms results aren't driven by a single year.

**Is NOT:**
- A new set of strategy scenarios. We're testing the existing 32, not adding to them.
- The visualization layer. Step 3 still owns plots. Part C produces one optional diagnostic figure.
- A rerun of the full Step 2.5 backtest. The 32 equity files in `runs/` are reused as inputs.

---

## 1. Briefing

The user's specific question this step answers: *"are the targeted-vs-always-on Sharpe differences real or noise?"*

Step 2.5 produced:
- `targeted_55_575`: Sharpe 1.586, MaxDD 29.9 %, 6.6 % trigger
- `targeted_55_60`: Sharpe 1.575
- `skip_dead_zone`: Sharpe 1.566, near-matching `always_on_30pct` CAGR with better Sharpe
- `always_on_30pct`: Sharpe 1.514
- Baseline: Sharpe 1.540

The differences are 0.05–0.07 in Sharpe units — small in absolute terms but consistent in direction (every targeted scenario has positive marginal Sharpe vs baseline; every always-on scenario has negative). The question is whether 0.05–0.07 is inside or outside the noise floor for this dataset.

The paired-bootstrap test answers it directly: build the daily-return series for two scenarios from their equity CSVs, sample matched daily-index blocks with replacement, recompute both scenarios' Sharpes on the resampled draws, and look at the distribution of (Sharpe_A − Sharpe_B). If the 95 % CI of that difference excludes zero, the gap is signal.

---

## 2. Working directory, env, inputs

- Working dir: `/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/RSI_tests/`
- Env: `quant` conda env. `~/opt/anaconda3/envs/quant/bin/python` or `conda activate quant`.
- Required packages: `pandas`, `numpy`, `scipy.stats`. **No matplotlib in Part A or B.** Part C uses matplotlib (`Agg` backend) for one optional plot.
- Trade log: `TRADES_TQQQ_canonical.csv`.
- Existing equity CSVs: `runs/equity_*.csv` — used as the source of daily-return series for Part B.
- Existing metrics: `runs/metrics.csv` — read for reference, do not modify.
- Reference code: `step2_backtest.py` and `step2_5.py` for engine internals.

---

## 3. Outputs

### Part A (write into `step1_outputs/`)

| File | Format | Purpose |
|---|---|---|
| `timesplit_validation_report.md` | markdown | narrative summary, headline conclusion |
| `contribution_by_rsi_bin_first_half.csv` | CSV | per-bin contribution table (5-pt + 2.5-pt) for first half only |
| `contribution_by_rsi_bin_second_half.csv` | CSV | same for second half |
| `timesplit_metrics_comparison.csv` | CSV | side-by-side metrics for key scenarios on full / first / second half |

### Part B (write into `runs/`)

| File | Format | Purpose |
|---|---|---|
| `paired_bootstrap_sharpe.csv` | CSV | per-pair: scenario_a, scenario_b, sharpe_a, sharpe_b, diff_point_est, diff_ci_low, diff_ci_high, p_value_two_sided, n_iters |
| `paired_bootstrap_cagr.csv` | CSV | same columns for CAGR (`cagr_a`, `cagr_b`, etc.) |

### Part C (write into `step1_outputs/`, optional)

| File | Format | Purpose |
|---|---|---|
| `rolling_stability.csv` | CSV | rolling 12-month metric values per scenario per window |
| `rolling_stability.png` | PNG 150 DPI | rolling Sharpe per scenario as line plot |

---

## 4. Part A — Time-split validation

### A.1 Define the split point

Load `TRADES_TQQQ_canonical.csv`, sort by `entry_time`. The split point is the **calendar midpoint** of the analysis window: `split_date = first_entry_time + (last_exit_time − first_entry_time) / 2`.

- "First half" = trades with `entry_time < split_date`.
- "Second half" = trades with `entry_time >= split_date`.

Report the split point and the trade count per half in the report. Expected: roughly equal counts (~800 each), since trades are fairly evenly distributed across the period.

### A.2 Re-run per-bin contribution on each half

For each half, repeat the Step 2.5 Part A contribution decomposition (5-pt and 2.5-pt bins). Compute the same columns: `n_trades`, `pct_of_total_trades`, `mean_pnl_pct`, `total_pnl_pct_sum`, `pct_of_total_pnl`, `sharpe_per_trade`.

Save to `contribution_by_rsi_bin_first_half.csv` and `contribution_by_rsi_bin_second_half.csv`.

### A.3 Re-run key scenarios on each half

For each half, walk the trades through the backtest engine (use the same `1 + pnl/capital_before` baseline accounting, the same borrow-cost rules, $10 000 starting capital per half) for these **8 key scenarios**:

1. `baseline`
2. `targeted_55_575` (sleeve fires when `RSI_entry ∈ [55, 57.5)`, 30 % size)
3. `targeted_55_60` (sleeve fires when `RSI_entry ∈ [55, 60)`, 30 % size)
4. `skip_dead_zone` (sleeve fires when `RSI_entry ∈ [40, 60) ∪ [65, 70)`, 30 % size)
5. `always_on_30pct`
6. `always_on_15pct`
7. `low50_high60` (the best windowed cell)
8. `low55_high65` (a window that straddles the dead zone — should underperform if our structural hypothesis is right)

For each scenario × each half, compute: `final_equity`, `total_return`, `cagr`, `sharpe_ann`, `sortino_ann`, `max_dd`, `sleeve_trigger_rate`, `sleeve_only_total_pnl`.

Save side-by-side in `timesplit_metrics_comparison.csv` with columns: `scenario`, `metric`, `full_value`, `first_half_value`, `second_half_value`, `pct_diff_first_vs_full`, `pct_diff_second_vs_full`. (Or pivot however you find readable — the goal is one row per scenario × metric.)

### A.4 The narrative `timesplit_validation_report.md`

Structure:

```
# Time-split Validation Report

## Split point
[Date, trade counts, span days per half.]

## Bin structure stability
[Compare [55, 60) and [60, 65) bins across halves. If both halves show
[55, 60) as the top contributor and [60, 65) as the bottom contributor,
the workhorse/dead-zone structure is real. Otherwise flag.]

## Scenario stability
[8 scenarios × 2 halves. For each scenario, does it rank consistently?
Specifically check: is `targeted_55_575` Sharpe in the top 3 in BOTH halves?
Is `low55_high65` (dead-zone-spanning) in the bottom in both?]

## Out-of-sample check
[Imagine you picked the "best scenario by Sharpe" using only first-half data.
Then evaluate that choice on the second half. Same Sharpe rank?
The classic OOS test.]

## Conclusion
[2-3 bullets. Is the Step 2.5 structure real or fragile?]
```

---

## 5. Part B — Paired-bootstrap Sharpe and CAGR diagnostics

### B.1 Build daily-return series per scenario

For each of these 9 scenarios, load `runs/equity_<scenario>.csv` and build a daily-resampled equity series via the same convention as `step2_backtest.py` (last-known equity at each daily timestamp, ffilled). Compute daily returns by `pct_change()`.

Scenarios needed:
- `baseline`
- `targeted_55_575`, `targeted_55_60`, `skip_dead_zone`
- `always_on_30pct`, `always_on_25pct`, `always_on_15pct`, `always_on_5pct`
- `low50_high60`

### B.2 Pair list to test

Compute paired bootstrap for these contrasts:

| # | Pair A | Pair B | What it tests |
|---|---|---|---|
| 1 | `targeted_55_575` | `baseline` | Does the tiny workhorse sleeve add Sharpe vs no leverage? |
| 2 | `targeted_55_60` | `baseline` | Same for the slightly wider workhorse sleeve. |
| 3 | `skip_dead_zone` | `baseline` | Does the dead-zone-avoiding sleeve add Sharpe? |
| 4 | `targeted_55_575` | `always_on_30pct` | **Headline #1**: targeted beats naive max-leverage on Sharpe? |
| 5 | `skip_dead_zone` | `always_on_30pct` | **Headline #2**: dead-zone avoidance beats naive max-leverage? |
| 6 | `targeted_55_60` | `low50_high60` | Targeted beats best-windowed-cell? |
| 7 | `skip_dead_zone` | `always_on_25pct` | Matched-trigger-rate (70 %) comparison: structure vs no-structure. |
| 8 | `always_on_30pct` | `baseline` | Sanity check: does naive max-leverage even add Sharpe vs no leverage? Expected sign: positive Sharpe diff (always-on increases Sharpe slightly OR not at all — Step 2.5 says it slightly DECREASES). |

### B.3 The paired-bootstrap algorithm

Standard stationary block bootstrap on **matched daily-return indices** (preserves correlation between A and B):

```
def paired_bootstrap_sharpe_diff(returns_a, returns_b, n_iters=5000, block_len=10, rng=...):
    assert returns_a.index.equals(returns_b.index)
    T = len(returns_a)
    diffs = []
    sharpe_as = []
    sharpe_bs = []
    for _ in range(n_iters):
        # Stationary block bootstrap: each block length drawn from geometric(1/block_len),
        # blocks tile the original index, wrapping circularly.
        idx = []
        while len(idx) < T:
            start = rng.integers(0, T)
            length = rng.geometric(1/block_len)
            idx.extend([(start + k) % T for k in range(length)])
        idx = np.array(idx[:T])
        ra = returns_a.values[idx]
        rb = returns_b.values[idx]
        sa = ra.mean() / ra.std() * np.sqrt(252)
        sb = rb.mean() / rb.std() * np.sqrt(252)
        sharpe_as.append(sa)
        sharpe_bs.append(sb)
        diffs.append(sa - sb)
    diffs = np.array(diffs)
    point_a = returns_a.mean() / returns_a.std() * np.sqrt(252)
    point_b = returns_b.mean() / returns_b.std() * np.sqrt(252)
    return {
        "sharpe_a": point_a,
        "sharpe_b": point_b,
        "diff_point_est": point_a - point_b,
        "diff_ci_low": np.percentile(diffs, 2.5),
        "diff_ci_high": np.percentile(diffs, 97.5),
        "p_value_two_sided": 2 * min((diffs > 0).mean(), (diffs < 0).mean()),
        "n_iters": n_iters,
    }
```

Use `numpy.random.default_rng(seed=42)` for reproducibility.

Same algorithm for CAGR: replace the Sharpe formula with `(1 + r).prod() ** (252 / T) - 1` (annualized geometric mean).

### B.4 Output schemas

`paired_bootstrap_sharpe.csv`:

```
pair_index, scenario_a, scenario_b, sharpe_a, sharpe_b, diff_point_est, diff_ci_low, diff_ci_high, p_value_two_sided, n_iters, ci_excludes_zero
```

`ci_excludes_zero`: bool. True iff (diff_ci_low > 0 OR diff_ci_high < 0).

`paired_bootstrap_cagr.csv`: same columns with `cagr_a`, `cagr_b` instead of `sharpe_*`.

### B.5 Result interpretation in `timesplit_validation_report.md`

Add a section "Paired-bootstrap Sharpe-difference tests" with a table showing all 8 pairs, their point estimates, CI, and whether the CI excludes zero. Then narrative:

- Which pairs have statistically distinguishable Sharpe differences?
- Headline pair #4 (`targeted_55_575` vs `always_on_30pct`) is the most consequential — what does it say?

---

## 6. Part C — Rolling-window stability (optional but recommended)

For the 4 key scenarios (`baseline`, `targeted_55_575`, `skip_dead_zone`, `always_on_30pct`):

- Compute rolling 12-month Sharpe and rolling 12-month CAGR on each scenario's daily-return series.
- 12-month rolling Sharpe = `mean(R) / std(R) * sqrt(252)` over each 252-trading-day window, step 1 day.
- Same for CAGR.

Output `rolling_stability.csv`: columns `date`, `scenario`, `rolling_sharpe_12m`, `rolling_cagr_12m`.

Output `rolling_stability.png`: two-panel figure, top = rolling Sharpe, bottom = rolling CAGR, 4 lines per panel (one per scenario), x-axis = date, both panels share x-axis. DPI 150.

Note in the report whether any scenario's outperformance is concentrated in one period (e.g., 2022 bear market) vs spread evenly.

---

## 7. Acceptance criteria

- All Part A files exist with non-empty content.
- `timesplit_metrics_comparison.csv` has 8 scenarios × 8 metrics × (full, first, second) entries.
- `paired_bootstrap_sharpe.csv` has exactly 8 rows.
- `paired_bootstrap_cagr.csv` has exactly 8 rows.
- For each row of paired-bootstrap CSVs, `diff_point_est` is consistent in sign with the `metrics.csv` Sharpe / CAGR difference between the two scenarios (sanity check that the daily-return series are correct).
- `n_iters >= 5000` for each row.
- The narrative `timesplit_validation_report.md` is readable and answers the four section prompts.
- Part C files exist if implemented; otherwise note "Part C skipped" in the report.

---

## 8. Hard constraints / what NOT to do

- **No new strategy scenarios.** All 32 stay as-is.
- **Do NOT modify `runs/metrics.csv`** or the existing equity CSVs.
- **Do NOT modify the engine in `step2_backtest.py` or `step2_5.py`.** If you need their helpers, import them.
- **The bootstrap MUST be paired** — same resampled indices for both scenarios in a pair. Independent-sample bootstrap of each scenario's Sharpe (and taking the diff of the CIs) is the wrong test and overstates uncertainty. We're explicitly leveraging the fact that A and B share the same underlying trade sequence and hence have correlated returns.
- **`pnl_pct` is in percent**, `^IRX close` is in percent, `pnl_pct/100` in any formula — Step 1 finding 2 / Step 2.5 reminder.
- **Re-walk the backtest from $10,000 on each half independently** in Part A. Do not use partial slices of the existing `equity_*.csv` files — those compounded through the full period.

---

## 9. When the user reviews

The user will want to know two things, in order:

1. **Is the Step 2.5 "targeted beats always-on" finding statistically robust?** Bootstrap pair #4 (`targeted_55_575` vs `always_on_30pct`) is the headline. If the CI excludes zero, the user has a defensible finding. If it doesn't, the 0.07 Sharpe gap is within noise.
2. **Does the workhorse/dead-zone structure hold across time?** Part A bin tables on each half. If `[55, 60)` is the top contributor in both halves, the structure is real. If it shifts dramatically — e.g., the workhorse moves to `[45, 50)` in the second half — then the Step 2.5 finding was a regime-specific artifact.

Don't pre-write conclusions. The numbers will speak.

If something unexpected fails (e.g., a scenario from Step 2.5 isn't replicable from the daily returns), STOP and surface it. The user prefers correctness over speed.
