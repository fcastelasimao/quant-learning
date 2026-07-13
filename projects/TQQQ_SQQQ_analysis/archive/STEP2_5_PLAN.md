# Step 2.5 — Targeted Scenarios + Step 1 EDA Follow-up

Hand-off document for Sonnet. **Read this entire document plus `FINDINGS.md` before writing code.** Project-wide spec lives in `PROJECT_PLAN.md`. Previous step handoffs live in `STEP1_PLAN.md` and `STEP2_PLAN.md`; the implementations live in `step1_eda.py` and `step2_backtest.py`. Their outputs live in `step1_outputs/` and `runs/`.

---

## 0. What this step is and is NOT

**Is:** a combined step covering two pieces of work the original plan missed:

- **Part A — Step 1 EDA follow-up.** Step 1's prior plot used a linear regression, which missed a non-monotone signal that turned out to matter a lot. This part adds the non-linear tests, conditional-distribution plots, and per-bin contribution decomposition that should have been there from the start.
- **Part B — Step 2 scenario additions.** Step 2's window grid covered RSI-windowed sleeves but didn't include "always-on" baselines or targeted-bin sleeves. This part adds 9 new scenarios that make the existing 22 windowed scenarios interpretable.

**Is NOT:**
- A rerun of Step 1's full validation. The data has been validated; we're only adding tests.
- A redesign of Step 2's engine. The existing `step2_backtest.py` engine is correct; we extend it with new `Sleeve` subclasses, not refactor it.
- The visualization layer. **No plots in Part B.** Part A produces three specific diagnostic figures, but no general heatmaps. Step 3 still owns visualization.

---

## 1. Briefing

Step 1's linear regression of `pnl_pct` on `RSI_entry` returned slope ≈ 0 and p = 0.40, leading us to conclude "no RSI signal." Step 2 ran a 22-cell window-grid sweep and found Sharpe was essentially flat across cells. Post-hoc analysis revealed both conclusions were missing the real structure:

- The relationship between `RSI_entry` and trade outcomes is **non-monotone and risk-asymmetric** — RSI predicts the *shape* of the return distribution (tail risk, variance, contribution density), not the *mean*.
- The strategy has a clear **heavy-lift bin at `[55, 60)`** (14.6 % of trades produce 25.9 % of P/L) and a **dead zone at `[60, 65)`** (15.2 % of trades produce 2.0 % of P/L).
- The Step 2 window cells were carving around this structure imperfectly. We need to test sleeves that target the structure directly.

Part A produces the diagnostic evidence that supports these claims. Part B runs the additional scenarios that test their consequences.

---

## 2. Working directory, environment, inputs

- Working dir: `/Users/franciscosimao/Documents/QuantFinance/personal_projects/projects/RSI_tests/`
- Env: `quant` conda env. `~/opt/anaconda3/envs/quant/bin/python` or `conda activate quant` first.
- Trade log: `TRADES_TQQQ_canonical.csv` (already validated by Step 1).
- Price/rate DBs: `DB_TQQQ_historical_data.db`, `DB_^IRX_historical_data.db` (same as Step 2).
- Reference code: read `step2_backtest.py` for the existing engine, sleeve interface, borrow-cost helper, and metric definitions. Re-use freely — do not duplicate.

---

## 3. Outputs

### Part A (write into `step1_outputs/`)

| File | Format | Purpose |
|---|---|---|
| `eda_followup_report.md` | markdown | summary of all Part A results |
| `contribution_by_rsi_bin.csv` | CSV | per-bin P/L contribution (both 5-pt and 2.5-pt) + counterfactual drop-bin CAGRs |
| `contribution_by_rsi_bin.png` | PNG 150 DPI | bar plot of per-bin %-of-total-P/L (5-pt bins) |
| `loss_distribution_by_rsi.png` | PNG 150 DPI | boxplot or violin of pnl_pct distribution per 5-pt RSI bin — the figure that *shows* the floor |
| `polynomial_fit.png` | PNG 150 DPI | original prior scatter (TQQQ only) with quadratic + LOWESS overlays |

### Part B (write into `runs/`, append to existing files)

| File | Format | Purpose |
|---|---|---|
| `metrics.csv` | CSV | **OVERWRITE** with all 22 existing rows + 9 new rows + `bh_tqqq` row. Same schema as before. Same row ordering convention: `baseline` first, windowed cells, then new scenarios in the order below, then `bh_tqqq` last. |
| `equity_always_on_5pct.csv` | CSV | one of 9 new per-trade equity walks |
| `equity_always_on_10pct.csv` | CSV | … |
| `equity_always_on_15pct.csv` | CSV | … |
| `equity_always_on_20pct.csv` | CSV | … |
| `equity_always_on_25pct.csv` | CSV | … |
| `equity_always_on_30pct.csv` | CSV | … |
| `equity_targeted_55_60.csv` | CSV | sleeve fires when `RSI_entry ∈ [55, 60)`, 30 % size |
| `equity_targeted_55_575.csv` | CSV | sleeve fires when `RSI_entry ∈ [55, 57.5)`, 30 % size |
| `equity_skip_dead_zone.csv` | CSV | sleeve fires when `RSI_entry ∈ [40, 60) ∪ [65, 70)`, 30 % size |

The 22 existing `equity_<window>.csv` files do not need to be regenerated unless you're refactoring the engine. If you keep `step2_backtest.py` unchanged and write a separate `step2_5_backtest.py` that only computes the new 9 scenarios + appends to `metrics.csv`, that's fine. Either approach acceptable; document which you chose at the top of the script.

---

## 4. Part A — EDA follow-up

### A.1 Per-bin P/L contribution

For both **5-pt bins** (range 35–75, edges `[35, 40), [40, 45), …, [70, 75)`) and **2.5-pt bins** (same range, edges `[35, 37.5), [37.5, 40), …, [72.5, 75)`):

Compute per bin:
- `n_trades`
- `pct_of_total_trades`
- `mean_pnl_pct`, `median_pnl_pct`, `std_pnl_pct`
- `total_pnl_pct_sum` (sum of pnl_pct values in the bin)
- `pct_of_total_pnl` (= bin's sum / overall sum × 100)
- `sharpe_per_trade` = mean / std
- `min_pnl_pct`, `max_pnl_pct`

Save to `contribution_by_rsi_bin.csv` with a `granularity` column distinguishing rows (`"5pt"` or `"2.5pt"`).

### A.2 Counterfactual drop-bin CAGR

For each **5-pt bin** in the analysis range:

1. Drop all trades in that bin.
2. Walk the remaining trades using `1 + pnl[i] / capital_before[i]` per trade (same convention as Step 2's engine).
3. Compute resulting final equity and CAGR over the original span (use original first entry_time and last exit_time for span calculation, not the filtered ones).

Append rows to `contribution_by_rsi_bin.csv` with `granularity="drop_bin"`, columns `bin`, `final_equity`, `cagr`, `cagr_delta_vs_full`.

### A.3 Non-linear regression tests

Compute and report in `eda_followup_report.md`:

- **Quadratic regression** `pnl_pct ~ RSI_entry + RSI_entry²`:
  - Coefficients, R², t-statistic on the quadratic term, p-value on the quadratic term.
  - Vertex of the parabola (`-β₁ / (2β₂)`) — the inflection point.
- **LOWESS smoothing curve** of `pnl_pct` vs `RSI_entry` (use `statsmodels.nonparametric.smoothers_lowess.lowess` with `frac=0.3`). Don't tabulate; just show on `polynomial_fit.png`.
- **Kruskal-Wallis test** across 5-pt RSI bins — H-statistic, p-value.
- **Levene's test for variance equality** across 5-pt RSI bins (`scipy.stats.levene`) — statistic, p-value. We expect *variance* differs across bins even if the mean doesn't.

### A.4 Risk-asymmetry analysis

For each 5-pt RSI bin, report in `eda_followup_report.md`:

- `n_losses` (count of trades with `pnl_pct < 0`)
- `min_loss` (worst single trade)
- `loss_std` (std of negative-pnl_pct trades)
- `pct5_loss` (5th percentile of negative trades)

Highlight in the narrative: bins where `min_loss > −2.5 %` indicate a hard floor.

### A.5 Plots

**`contribution_by_rsi_bin.png`** — bar chart, 5-pt bins on x-axis, two bars per bin: `pct_of_total_pnl` and `pct_of_total_trades`. Annotate: "[55, 60) does 26 % of work with 15 % of trades"; "[60, 65) is the dead zone."

**`loss_distribution_by_rsi.png`** — boxplot or violin of `pnl_pct` per 5-pt RSI bin. Sorted by bin. Y-axis clipped to `[-8, 8]` (in %). Annotate the floor: `RSI < 45` losses cap near −2.1 %.

**`polynomial_fit.png`** — like Step 1's `prior_pnl_vs_rsi.png` but TQQQ-only and with three overlays on the scatter: (1) original OLS line in red dashed, (2) quadratic fit in solid green, (3) LOWESS curve in solid orange. Title shows the quadratic vertex and R².

### A.6 Write `eda_followup_report.md`

Structure:

```
# Step 1 EDA Follow-up — Non-linear Tests and Contribution Decomposition

## Why this exists
[One paragraph: Step 1's linear test missed a non-monotone risk-asymmetric pattern.
This document presents the tests that catch it.]

## Per-bin contribution
[5-pt and 2.5-pt contribution tables. Reference the PNG.]

## Counterfactual drop-bin CAGR
[Table of "drop this bin, baseline CAGR becomes …".]

## Non-linear tests
- Quadratic regression: …
- LOWESS curve: …
- Kruskal-Wallis: …
- Levene: …

## Risk asymmetry
[Loss-side stats per bin. Reference the loss-distribution PNG.]

## Conclusions
- Two bullets max. Headline takeaways. Honest tone.
```

---

## 5. Part B — Scenario additions

### B.1 Engine extension

The existing `step2_backtest.py` has a `Sleeve` ABC. Extend the interface and add three new concrete classes.

**Refactor (mandatory):** the existing classes have a hardcoded 30 % sleeve size. Promote sleeve size to a class attribute so different scenarios can use different sizes.

```python
class Sleeve(ABC):
    size: float  # fraction of portfolio used as sleeve notional (e.g. 0.30)

    @abstractmethod
    def should_enter(self, trade_row) -> bool: ...
    @abstractmethod
    def exit_event(self, trade_row, current_time) -> bool: ...
    @property
    @abstractmethod
    def label(self) -> str: ...


class NoSleeve(Sleeve):
    size = 0.0
    label = "baseline"
    def should_enter(self, trade_row): return False
    def exit_event(self, trade_row, current_time): return True


class WindowEntryRSISleeve(Sleeve):
    size = 0.30
    def __init__(self, low: float, high: float):
        self.low, self.high = low, high
    @property
    def label(self): return f"low{int(self.low)}_high{int(self.high)}"
    def should_enter(self, trade_row):
        return self.low <= trade_row.RSI_entry < self.high
    def exit_event(self, trade_row, current_time):
        return current_time >= trade_row.exit_time


# NEW
class AlwaysOnSleeve(Sleeve):
    def __init__(self, size: float):
        self.size = size
    @property
    def label(self): return f"always_on_{int(round(self.size*100))}pct"
    def should_enter(self, trade_row): return True
    def exit_event(self, trade_row, current_time): return current_time >= trade_row.exit_time


# NEW
class MultiWindowEntryRSISleeve(Sleeve):
    """Fires when RSI_entry falls in ANY of the listed [low, high) intervals."""
    size = 0.30
    def __init__(self, windows: list[tuple[float, float]], label: str):
        self.windows = windows
        self._label = label
    @property
    def label(self): return self._label
    def should_enter(self, trade_row):
        rsi = trade_row.RSI_entry
        return any(lo <= rsi < hi for lo, hi in self.windows)
    def exit_event(self, trade_row, current_time): return current_time >= trade_row.exit_time
```

Update the engine's sleeve-notional computation to use `sleeve.size` instead of a hardcoded 0.30.

### B.2 New scenarios

The new scenarios, instantiated in order:

```python
new_scenarios = [
    AlwaysOnSleeve(size=0.05),
    AlwaysOnSleeve(size=0.10),
    AlwaysOnSleeve(size=0.15),
    AlwaysOnSleeve(size=0.20),
    AlwaysOnSleeve(size=0.25),
    AlwaysOnSleeve(size=0.30),
    MultiWindowEntryRSISleeve(windows=[(55.0, 60.0)], label="targeted_55_60"),
    MultiWindowEntryRSISleeve(windows=[(55.0, 57.5)], label="targeted_55_575"),
    MultiWindowEntryRSISleeve(windows=[(40.0, 60.0), (65.0, 70.0)], label="skip_dead_zone"),
]
```

### B.3 Backtest each new scenario

Use the existing walker logic from `step2_backtest.py`. Per trade:

1. `equity_before = current_equity`.
2. `baseline_ratio = 1 + pnl[i] / capital_before[i]`. `baseline_pnl_dollars = equity_before × (baseline_ratio − 1)`.
3. `equity_after_baseline = equity_before × baseline_ratio`.
4. If `sleeve.should_enter(trade)`:
   - `sleeve_notional = sleeve.size × equity_before`
   - `sleeve_gross = sleeve_notional × (pnl_pct[i] / 100)`
   - `days_held = (exit_time − entry_time).total_seconds() / 86400`
   - `irx_rate_pct = ^IRX close on entry_date, forward-filled`
   - `ann_rate = irx_rate_pct / 100 + tier_spread(equity_before)`
   - `borrow_cost = sleeve_notional × ann_rate × days_held / 365`
   - `sleeve_net = sleeve_gross − borrow_cost`
   - `equity_after = equity_after_baseline + sleeve_net`
5. Append row to per-trade DataFrame.

Write each per-trade DataFrame to `runs/equity_<label>.csv` using the existing 15-column schema.

### B.4 Metrics computation

For each new scenario, compute every column in `metrics.csv` using the existing helpers from `step2_backtest.py`. Same daily-resampled-equity approach for risk metrics. Same bootstrap CI (1000 iterations, block length 10). Same `vs_bh_*` formulas.

### B.5 Deflated Sharpe — recompute on full grid

The Deflated Sharpe value should now reflect **31 strategy scenarios** (baseline + 21 window cells + 9 new). Recompute and update every row in `metrics.csv`. The `bh_tqqq` row keeps NaN.

### B.6 Update `metrics.csv`

Two acceptable approaches:
- **Approach 1 (recommended):** rebuild `metrics.csv` from scratch by re-running all 31 scenarios + benchmark. Cheap (~30 seconds compute).
- **Approach 2:** read existing `metrics.csv`, drop `bh_tqqq` row, append 9 new rows, recompute Deflated Sharpe column for all, append `bh_tqqq` last.

Row ordering convention:
1. `baseline`
2. 21 window cells in the existing order
3. 6 `always_on_*` rows in ascending size order
4. 3 targeted rows: `targeted_55_60`, `targeted_55_575`, `skip_dead_zone`
5. `bh_tqqq`

Total: **32 rows**.

---

## 6. Acceptance criteria

- `step1_outputs/` contains 5 new files (4 listed above plus updated `eda_followup_report.md`).
- `runs/metrics.csv` has exactly 32 rows.
- `runs/` contains 9 new `equity_*.csv` files (no need to recreate the existing 22 unless rebuilding).
- The 6 `always_on_*` rows in `metrics.csv` show `sleeve_trigger_rate = 1.0` (sanity check).
- The 3 targeted rows show `sleeve_trigger_rate` matching direct calculation of `mean(condition)` over the canonical trades.
- The `always_on_30pct` row should have the highest `sleeve_only_total_pnl` of any scenario.
- The `targeted_55_575` row should have the highest `sharpe_per_trade_fired` if computed — but this isn't required as a metric column; just verify in script output.

---

## 7. Hard constraints / what NOT to do

- **No plots in Part B.** Step 3 owns visualization.
- **Do NOT change the existing 22 window-cell scenarios' results.** If you refactor the engine, verify the existing scenarios' final equity matches the prior `metrics.csv` to ≤ 1e-6 relative tolerance.
- **Do NOT modify `step2_backtest.py` in a way that breaks its standalone execution.** If you refactor, make sure running `step2_backtest.py` alone still reproduces the original 22-scenario `metrics.csv`. Better: write a separate `step2_5_backtest.py` that imports from `step2_backtest.py`.
- **Do NOT add new scenarios beyond the 9 specified.** No "interesting alternatives" — the design is locked.
- **Do NOT change the sleeve size for the 21 existing window cells.** They stay at 30 %.
- **`pnl_pct` is in percent**. Divide by 100 in the sleeve gross formula.
- **`^IRX close` is in percent**. Divide by 100 in the borrow formula.
- **Use the existing `baseline_ratio = 1 + pnl / capital_before` convention**, not `capital_end / capital_before`. The canonical has chain gaps — see `FINDINGS.md` finding 1.

---

## 8. When the user reviews

The user will want to know:

1. **Did Part A confirm the non-monotone / dead-zone story?** Quadratic R² should be small but t-stat suggestive; Levene should be highly significant (variances differ); contribution table should show the [55, 60) workhorse and [60, 65) dead zone.
2. **Does `always_on_30pct` beat the best windowed cell?** This is the cleanest test of "is selection useful, or just leverage?" If `always_on_30pct` Sharpe ≥ best windowed Sharpe, RSI selection is essentially noise. If a windowed cell beats it on Sharpe (even slightly), there's some selection signal.
3. **What's the Sharpe of `targeted_55_60` and `targeted_55_575`?** Expected: higher than any 5-pt window because they target the workhorse cleanly. If they don't, the "heavy lifter" interpretation is overstated.
4. **What's `skip_dead_zone` vs `low40_high70` or `low35_high70`?** They have similar trigger rates; `skip_dead_zone` should do better if avoiding the dead zone matters.

Don't pre-write these conclusions. Just produce the numbers cleanly and the user will read them.
